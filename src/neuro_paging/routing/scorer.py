"""Context-aware scoring — the brain that re-ranks retrieval candidates.

The deck's formula:

    Score(m | q, c) = α·cos(e_m, e_q)
                    + β·ctxSim(tags_m, c)
                    + γ·log(1 + freq_m)·decay(Δt)

Three terms, three notions of relevance:
  - SEMANTIC (α): is this memory ABOUT what was asked? Cosine similarity
    of the memory's embedding vs the query's. The dominant signal.
  - CONTEXT (β): was this memory created in a situation like NOW? Same
    app, same time-of-day, same location, overlapping tags. This is the
    signal flat RAG throws away — the "context-aware" in the project name.
  - FREQUENCY/RECENCY (γ): is this a memory the user keeps coming back to,
    recently? log(1+freq) rewards repeated access with diminishing returns;
    decay(Δt) discounts memories that have gone cold.

Weights are fixed and hand-tuned (semantic dominates, context refines,
frequency breaks ties). Online per-user learning of α/β/γ from implicit
feedback is documented as future work — the architecture supports it
(the weights are injected, not hard-coded) but it's not implemented here.

This class implements the Scorer protocol (memory.manager.Scorer):
    score(memory, query_text, query_context) -> (float, Provenance)

The query embedding is computed once per query by the caller and passed
in via set_query(); score() then reuses it for every candidate. This
keeps the embedder call out of the per-candidate hot loop.
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass
from datetime import UTC, datetime

import numpy as np

from neuro_paging.context.types import ContextTags
from neuro_paging.memory.types import Memory, Provenance


@dataclass(frozen=True, slots=True)
class ScorerWeights:
    """The α, β, γ weights. Fixed defaults; injectable for experimentation.

    Defaults reflect the design intent:
      - semantic (α) dominates: relevance is mostly "is it about this?"
      - context (β) refines: situation-match nudges the ranking
      - frequency (γ) breaks ties: habits surface over one-offs
    They sum to 1.0 so the final score stays in a predictable [0, 1]-ish range.
    """

    alpha: float = 0.60  # semantic similarity
    beta: float = 0.25  # context match
    gamma: float = 0.15  # frequency × recency decay

    def __post_init__(self) -> None:
        total = self.alpha + self.beta + self.gamma
        if not math.isclose(total, 1.0, abs_tol=1e-6):
            raise ValueError(f"weights must sum to 1.0, got {total}")


# Recency decay half-life: a memory untouched for this long scores 0.5 on
# the decay term. 7 days is a reasonable "warm" horizon for personal memory.
_DECAY_HALF_LIFE_SECONDS = 7 * 24 * 3600


class ContextAwareScorer:
    """Implements the deck's context-aware scoring formula.

    Usage from MemoryManager:
        scorer = ContextAwareScorer(embedder=my_embedder)
        # manager calls set_query() once per query, then score() per candidate
        scorer.set_query("what do I like to eat?", current_context)
        score, prov = scorer.score(memory, query_text, current_context)

    If no embedder is provided, the semantic term falls back to 0 and the
    scorer ranks purely on context + frequency (still useful, just not
    semantic). This keeps it usable in tests without a model.
    """

    def __init__(
        self,
        embedder=None,
        weights: ScorerWeights | None = None,
        decay_half_life_seconds: float = _DECAY_HALF_LIFE_SECONDS,
    ) -> None:
        self._embedder = embedder
        self._weights = weights or ScorerWeights()
        self._half_life = decay_half_life_seconds

        # Cached per-query state, set by set_query()
        self._query_text: str | None = None
        self._query_embedding: np.ndarray | None = None
        self._query_context: ContextTags | None = None

    # ── Per-query setup ──────────────────────────────────────────────────────

    def set_query(self, query_text: str, query_context: ContextTags) -> None:
        """Cache the query embedding + context once, before scoring candidates.

        Computing the query embedding once (not per-candidate) is the whole
        point — embedding is the expensive op, ranking is cheap.
        """
        self._query_text = query_text
        self._query_context = query_context
        if self._embedder is not None:
            # Prefer the query-side encoding if the embedder is asymmetric
            embed_query = getattr(self._embedder, "embed_query", None)
            if callable(embed_query):
                self._query_embedding = embed_query(query_text)
            else:
                self._query_embedding = self._embedder.embed(query_text)
        else:
            self._query_embedding = None

    # ── The three terms ──────────────────────────────────────────────────────

    def _semantic_term(self, memory: Memory) -> float:
        """Cosine similarity of memory vs query embedding, in [0, 1].

        If we have no query embedding (no embedder), returns 0 — the scorer
        then ranks on context + frequency only.

        We re-embed the memory text here. In production the memory's stored
        vector would be reused; for v0 correctness we embed on demand, which
        is fine because the candidate set is already small (post-ANN).
        """
        if self._query_embedding is None or self._embedder is None:
            return 0.0
        mem_vec = self._embedder.embed(memory.text)
        # Both vectors are L2-normalised, so dot == cosine. Clamp to [0,1]:
        # cosine is in [-1, 1]; negative similarity means "unrelated", which
        # we floor to 0 so it doesn't subtract from the other terms.
        cos = float(np.dot(mem_vec, self._query_embedding))
        return max(0.0, min(1.0, cos))

    def _context_term(self, memory: Memory, ctx: ContextTags) -> float:
        """How well the memory's stored context matches the current context.

        A normalised average of field matches in [0, 1]:
          - time bucket match (morning/afternoon/evening/night)
          - same foreground app
          - same location
          - semantic-tag overlap (Jaccard)

        Each component contributes equally. Missing fields (None) don't
        count for or against — they're skipped and the average is over the
        comparable fields only.
        """
        mem_ctx = memory.context
        components: list[float] = []

        # Time-of-day bucket match
        components.append(1.0 if mem_ctx.time_bucket == ctx.time_bucket else 0.0)

        # Foreground app match (only if both known)
        if mem_ctx.foreground_app is not None and ctx.foreground_app is not None:
            components.append(1.0 if mem_ctx.foreground_app == ctx.foreground_app else 0.0)

        # Location match (only if both known)
        if mem_ctx.location is not None and ctx.location is not None:
            components.append(1.0 if mem_ctx.location == ctx.location else 0.0)

        # Semantic-tag overlap (Jaccard), only if either side has tags
        mem_tags = set(mem_ctx.semantic_tags)
        cur_tags = set(ctx.semantic_tags)
        if mem_tags or cur_tags:
            union = mem_tags | cur_tags
            inter = mem_tags & cur_tags
            components.append(len(inter) / len(union) if union else 0.0)

        if not components:
            return 0.0
        return sum(components) / len(components)

    def _frequency_term(self, memory: Memory) -> float:
        """log(1 + access_count) · decay(time since last touch), in [0, 1]-ish.

        - log(1+freq) grows with repeated access but with diminishing returns,
          so a memory accessed 100 times isn't 100× a memory accessed once.
        - decay is exponential on the time since last_touch: a memory touched
          now scores ~1 on decay; one untouched for the half-life scores 0.5.

        We normalise the log term by a soft ceiling (log(1+32)) so a
        well-used memory approaches but doesn't blow past 1.0.
        """
        # Recency decay
        now = datetime.now(UTC)
        last = memory.last_touch
        if last.tzinfo is None:
            last = last.replace(tzinfo=UTC)
        delta_seconds = max(0.0, (now - last).total_seconds())
        decay = math.exp(-delta_seconds / self._half_life)

        # Frequency (log, soft-normalised)
        freq_raw = math.log1p(memory.access_count)
        freq_norm = min(1.0, freq_raw / math.log1p(32))  # ceiling at ~32 accesses

        return freq_norm * decay

    # ── The Scorer protocol ──────────────────────────────────────────────────

    def score(
        self,
        memory: Memory,
        query_text: str,
        query_context: ContextTags,
    ) -> tuple[float, Provenance]:
        """Return (final_score, provenance) for this memory given the query.

        If set_query() wasn't called (or was called with a different query),
        we lazily set it now so the scorer is correct even when used directly.
        """
        # Self-heal: if the cached query doesn't match, recompute.
        if self._query_text != query_text or self._query_context is not query_context:
            self.set_query(query_text, query_context)

        start = time.perf_counter()

        semantic = self._semantic_term(memory)
        ctx_sim = self._context_term(memory, query_context)
        freq = self._frequency_term(memory)

        w = self._weights
        final = w.alpha * semantic + w.beta * ctx_sim + w.gamma * freq

        elapsed_ms = (time.perf_counter() - start) * 1000.0

        prov = Provenance(
            tier=memory.tier,
            raw_relevance=semantic,
            raw_context_sim=ctx_sim,
            raw_frequency=freq,
            weights=(w.alpha, w.beta, w.gamma),
            final_score=final,
            elapsed_ms=elapsed_ms,
        )
        return final, prov

    @property
    def weights(self) -> ScorerWeights:
        return self._weights
