"""MemoryAgent — orchestrates store → retrieve → rank → assemble → generate.

This is the single public entry point that ties the whole system together.
It owns a MemoryManager (the tiered substrate) wired with a real embedder
and the context-aware scorer, and exposes three operations:

    remember(text, context)        -> MemoryId      (write path)
    recall(query, context, k)      -> AssembledContext  (read path)
    respond(query, context)        -> str           (recall + generate)

The read path is where the system's value is visible: recall() retrieves
candidates across all tiers, the scorer ranks them by semantic + context
+ recency relevance, and the result is ASSEMBLED into a context block an
LLM would actually consume — with tier labels and per-memory provenance.

Generation is a plugin. The default _EchoGenerator returns the assembled
context block verbatim, making the system's output fully transparent: you
can see exactly what context the memory system would feed a model. Swap in
a real LLM via the Generator protocol without changing anything here.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

from loguru import logger

from neuro_paging.context.types import ContextTags
from neuro_paging.memory.manager import MemoryManager
from neuro_paging.memory.types import Hit, MemoryId, Tier

# ── Generation plugin ─────────────────────────────────────────────────────────


class Generator(Protocol):
    """The contract a text generator implements.

    Given a user query and an assembled context block, produce a response.
    A real implementation wraps llama.cpp / Qwen / any LLM. The pipeline
    doesn't care which.
    """

    def generate(self, query: str, context_block: str) -> str:
        """Return a response given the query and the assembled memory context."""
        ...


class _EchoGenerator:
    """Default generator: returns the assembled context block, transparently.

    A memory system's core job is to surface the RIGHT context. This
    generator makes that output directly inspectable rather than hiding it
    behind an LLM. It's the honest default — and a perfectly good demo of
    'here is what my memory layer would feed your model'.
    """

    def generate(self, query: str, context_block: str) -> str:
        if not context_block.strip():
            return f"[no relevant memories found for: {query!r}]"
        return (
            f"[Generator stub — this is the context the memory system "
            f"assembled for the query {query!r}. A real LLM would consume "
            f"this block to produce an answer.]\n\n{context_block}"
        )


# ── Assembled context (the read path's output) ───────────────────────────────


@dataclass(frozen=True, slots=True)
class AssembledContext:
    """The output of recall(): ranked hits + the LLM-ready context block.

    `hits` is the raw ranked retrieval (with provenance, for the dashboard).
    `context_block` is those hits formatted into text an LLM would consume.
    `query` and `n_candidates` are kept for observability / display.
    """

    query: str
    hits: list[Hit]
    context_block: str
    n_candidates_considered: int

    @property
    def is_empty(self) -> bool:
        return len(self.hits) == 0


# ── The agent ─────────────────────────────────────────────────────────────────


class MemoryAgent:
    """End-to-end memory agent: remember, recall, respond.

    Construct once at app startup. Owns the substrate + plugins.

    Args:
        data_dir: where the substrate persists (passed to MemoryManager).
        embedder: text→vector. If None, MemoryManager's stub is used (fine
            for tests; for real semantic recall pass a BGESmallEmbedder).
        scorer: the ranking function. If None, MemoryManager's stub scorer
            is used. For real context-aware ranking pass a ContextAwareScorer.
        generator: the response generator. Defaults to the transparent echo
            generator (returns the assembled context block).

    The agent deliberately wires embedder + scorer into the SAME manager so
    stored memories and queries use consistent embeddings, and ranking sees
    real context.
    """

    def __init__(
        self,
        data_dir: Path | str | None = None,
        embedder=None,
        scorer=None,
        generator: Generator | None = None,
    ) -> None:
        self._manager = MemoryManager(
            data_dir=data_dir,
            scorer=scorer,  # None → manager's stub scorer
            embedder=embedder,  # None → manager's stub embedder
        )
        self._generator = generator or _EchoGenerator()
        logger.debug("MemoryAgent initialised")

    # ── Write path ───────────────────────────────────────────────────────────

    def remember(
        self,
        text: str,
        context: ContextTags,
        *,
        tier: Tier = Tier.L1,
    ) -> MemoryId:
        """Store a memory. Returns its id.

        New memories land in L1 and cascade down as tiers fill — the same
        flow the substrate already implements. This is just the friendly
        front door.
        """
        return self._manager.insert(text, context, tier=tier)

    # ── Read path ────────────────────────────────────────────────────────────

    def recall(
        self,
        query: str,
        context: ContextTags,
        *,
        k: int = 5,
        min_tier: Tier = Tier.L3,
    ) -> AssembledContext:
        """Retrieve + rank + assemble. Returns an AssembledContext.

        This is the heart of the system: it pulls candidates across the
        requested tiers, the scorer ranks them by semantic + context +
        recency relevance, and the top-k are formatted into a context block
        an LLM would consume.
        """
        hits = self._manager.query(query, context, k=k, min_tier=min_tier)
        block = self._assemble_context_block(hits)
        return AssembledContext(
            query=query,
            hits=hits,
            context_block=block,
            n_candidates_considered=len(hits),
        )

    @staticmethod
    def _assemble_context_block(hits: list[Hit]) -> str:
        """Format ranked hits into an LLM-ready context block.

        The format is deliberately simple + inspectable: each memory on its
        own line, tagged with its tier and relevance score, most-relevant
        first. A real prompt template would wrap this; the structure is what
        matters.
        """
        if not hits:
            return ""
        lines = ["Relevant memories (most relevant first):"]
        for i, h in enumerate(hits, start=1):
            tier = h.provenance.tier.value if h.provenance else "?"
            lines.append(f"{i}. [{tier} · score {h.score:.2f}] {h.text}")
        return "\n".join(lines)

    # ── Respond (recall + generate) ──────────────────────────────────────────

    def respond(
        self,
        query: str,
        context: ContextTags,
        *,
        k: int = 5,
    ) -> str:
        """Full loop: recall relevant memories, hand the assembled context
        to the generator, return its response.

        With the default echo generator this returns the context block,
        making the memory system's contribution fully transparent. With a
        real LLM generator it returns a generated answer grounded in the
        recalled memories.
        """
        assembled = self.recall(query, context, k=k)
        return self._generator.generate(query, assembled.context_block)

    # ── Passthrough observability ────────────────────────────────────────────

    def get_stats(self):
        """Tier occupancy + rolling metrics (delegates to the manager)."""
        return self._manager.get_stats()

    def close(self) -> None:
        """Persist + close the underlying substrate."""
        self._manager.close()

    @property
    def manager(self) -> MemoryManager:
        """Escape hatch for advanced use (the dashboard reaches in for stats)."""
        return self._manager
