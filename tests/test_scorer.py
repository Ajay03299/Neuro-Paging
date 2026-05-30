"""Tests for ContextAwareScorer.

Network-free: uses a tiny deterministic stub embedder so we never download
a model. Each scoring term is tested in isolation, then the combined score.
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

import numpy as np
import pytest

from neuro_paging import ContextTags, TimeBucket
from neuro_paging.memory.types import Memory, MemoryId, Tier
from neuro_paging.routing.scorer import ContextAwareScorer, ScorerWeights


# ── A tiny deterministic embedder for tests (no network) ─────────────────────


class _StubEmbedder:
    """Deterministic hashed-random embedder. Same text → same vector."""

    def __init__(self, dim: int = 384) -> None:
        self._dim = dim

    def embed(self, text: str) -> np.ndarray:
        seed = abs(hash(text)) % (2**32)
        rng = np.random.default_rng(seed)
        v = rng.standard_normal(self._dim).astype(np.float32)
        v /= np.linalg.norm(v) + 1e-8
        return v


def _make_memory(
    text: str = "hello",
    mid: str = "m1",
    tier: Tier = Tier.L2,
    *,
    time_bucket: TimeBucket = TimeBucket.EVENING,
    location: str | None = None,
    foreground_app: str | None = None,
    semantic_tags: tuple[str, ...] = (),
    access_count: int = 0,
    last_touch: datetime | None = None,
) -> Memory:
    now = datetime.now(timezone.utc)
    ctx = ContextTags.now(
        time_bucket=time_bucket,
        location=location,
        foreground_app=foreground_app,
        semantic_tags=semantic_tags,
    )
    return Memory(
        id=MemoryId(mid),
        text=text,
        embedding_ref=f"emb:{mid}",
        context=ctx,
        tier=tier,
        created_at=now,
        last_touch=last_touch or now,
        access_count=access_count,
        is_consolidated=False,
    )


# ── Weights ──────────────────────────────────────────────────────────────────


class TestWeights:
    def test_default_weights_sum_to_one(self):
        w = ScorerWeights()
        assert abs(w.alpha + w.beta + w.gamma - 1.0) < 1e-6

    def test_invalid_weights_rejected(self):
        with pytest.raises(ValueError):
            ScorerWeights(alpha=0.5, beta=0.5, gamma=0.5)  # sums to 1.5


# ── Semantic term ────────────────────────────────────────────────────────────


class TestSemanticTerm:
    def test_identical_text_scores_high(self):
        emb = _StubEmbedder()
        scorer = ContextAwareScorer(embedder=emb)
        ctx = ContextTags.now()
        scorer.set_query("the quick brown fox", ctx)
        # A memory with the SAME text → cosine ~1
        mem = _make_memory(text="the quick brown fox")
        sem = scorer._semantic_term(mem)
        assert sem > 0.99

    def test_no_embedder_semantic_is_zero(self):
        scorer = ContextAwareScorer(embedder=None)
        ctx = ContextTags.now()
        scorer.set_query("anything", ctx)
        mem = _make_memory(text="something else")
        assert scorer._semantic_term(mem) == 0.0


# ── Context term ─────────────────────────────────────────────────────────────


class TestContextTerm:
    def test_full_context_match_scores_one(self):
        scorer = ContextAwareScorer(embedder=_StubEmbedder())
        ctx = ContextTags.now(
            time_bucket=TimeBucket.MORNING,
            location="office",
            foreground_app="VSCode",
            semantic_tags=("code", "review"),
        )
        mem = _make_memory(
            time_bucket=TimeBucket.MORNING,
            location="office",
            foreground_app="VSCode",
            semantic_tags=("code", "review"),
        )
        assert scorer._context_term(mem, ctx) == 1.0

    def test_no_context_match_scores_low(self):
        scorer = ContextAwareScorer(embedder=_StubEmbedder())
        ctx = ContextTags.now(
            time_bucket=TimeBucket.MORNING,
            location="office",
            foreground_app="VSCode",
            semantic_tags=("code",),
        )
        mem = _make_memory(
            time_bucket=TimeBucket.NIGHT,
            location="home",
            foreground_app="Netflix",
            semantic_tags=("movies",),
        )
        # Every comparable field mismatches → 0
        assert scorer._context_term(mem, ctx) == 0.0

    def test_partial_context_match_is_between(self):
        scorer = ContextAwareScorer(embedder=_StubEmbedder())
        ctx = ContextTags.now(
            time_bucket=TimeBucket.MORNING,
            location="office",
            foreground_app="VSCode",
        )
        mem = _make_memory(
            time_bucket=TimeBucket.MORNING,  # match
            location="home",                 # mismatch
            foreground_app="VSCode",          # match
        )
        sim = scorer._context_term(mem, ctx)
        assert 0.0 < sim < 1.0


# ── Frequency term ───────────────────────────────────────────────────────────


class TestFrequencyTerm:
    def test_recent_frequent_scores_higher_than_stale_rare(self):
        scorer = ContextAwareScorer(embedder=_StubEmbedder())
        now = datetime.now(timezone.utc)

        recent_frequent = _make_memory(
            access_count=20, last_touch=now
        )
        stale_rare = _make_memory(
            access_count=1, last_touch=now - timedelta(days=30)
        )

        assert scorer._frequency_term(recent_frequent) > scorer._frequency_term(
            stale_rare
        )

    def test_zero_access_never_touched_is_low(self):
        scorer = ContextAwareScorer(embedder=_StubEmbedder())
        # access_count=0 → log1p(0)=0 → freq term is 0 regardless of decay
        mem = _make_memory(access_count=0)
        assert scorer._frequency_term(mem) == 0.0

    def test_decay_reduces_score_over_time(self):
        scorer = ContextAwareScorer(embedder=_StubEmbedder())
        now = datetime.now(timezone.utc)
        fresh = _make_memory(access_count=10, last_touch=now)
        old = _make_memory(
            access_count=10, last_touch=now - timedelta(days=14)
        )
        assert scorer._frequency_term(fresh) > scorer._frequency_term(old)


# ── Combined score + provenance ──────────────────────────────────────────────


class TestCombinedScore:
    def test_score_returns_provenance_with_breakdown(self):
        scorer = ContextAwareScorer(embedder=_StubEmbedder())
        ctx = ContextTags.now(time_bucket=TimeBucket.EVENING)
        mem = _make_memory(text="hello world", time_bucket=TimeBucket.EVENING)

        score, prov = scorer.score(mem, "hello world", ctx)

        assert 0.0 <= score <= 1.0
        assert prov.weights == (0.60, 0.25, 0.15)
        # Provenance carries the per-term breakdown for the dashboard
        assert 0.0 <= prov.raw_relevance <= 1.0
        assert 0.0 <= prov.raw_context_sim <= 1.0
        assert 0.0 <= prov.raw_frequency <= 1.0
        assert prov.final_score == score

    def test_semantically_relevant_context_matching_beats_neither(self):
        """A memory matching BOTH query meaning and context should outrank
        one matching neither."""
        emb = _StubEmbedder()
        scorer = ContextAwareScorer(embedder=emb)
        now = datetime.now(timezone.utc)
        ctx = ContextTags.now(
            time_bucket=TimeBucket.MORNING,
            foreground_app="VSCode",
            semantic_tags=("code",),
        )

        # Good: same text as query + same context + recently used
        good = _make_memory(
            text="fix the deployment bug",
            mid="good",
            time_bucket=TimeBucket.MORNING,
            foreground_app="VSCode",
            semantic_tags=("code",),
            access_count=10,
            last_touch=now,
        )
        # Bad: different text, different context, never used
        bad = _make_memory(
            text="grocery shopping list milk eggs",
            mid="bad",
            time_bucket=TimeBucket.NIGHT,
            foreground_app="Notes",
            semantic_tags=("shopping",),
            access_count=0,
            last_touch=now - timedelta(days=30),
        )

        scorer.set_query("fix the deployment bug", ctx)
        good_score, _ = scorer.score(good, "fix the deployment bug", ctx)
        bad_score, _ = scorer.score(bad, "fix the deployment bug", ctx)

        assert good_score > bad_score

    def test_custom_weights_change_ranking(self):
        """With γ=0 weights, frequency stops affecting the score."""
        emb = _StubEmbedder()
        # All weight on semantic, none on context/frequency
        w = ScorerWeights(alpha=1.0, beta=0.0, gamma=0.0)
        scorer = ContextAwareScorer(embedder=emb, weights=w)
        ctx = ContextTags.now()

        high_freq = _make_memory(text="same text", access_count=100)
        low_freq = _make_memory(text="same text", access_count=0)

        scorer.set_query("same text", ctx)
        s_high, _ = scorer.score(high_freq, "same text", ctx)
        s_low, _ = scorer.score(low_freq, "same text", ctx)
        # Same text → same semantic; γ=0 → frequency irrelevant → equal scores
        assert abs(s_high - s_low) < 1e-6