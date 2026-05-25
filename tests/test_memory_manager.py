"""Contract tests for MemoryManager.

These tests pin down the *behavior* of the API. They use the stub
backend today; the same tests will run against the real L1/L2/L3
implementation in Sprint 1+. If they keep passing, the contract held.
"""

from __future__ import annotations

from datetime import UTC

import pytest

from neuro_paging import (
    ContextTags,
    Hit,
    MemoryManager,
    Tier,
    TimeBucket,
)

# ── Fixtures ──────────────────────────────────────────────────────────────────


@pytest.fixture
def mgr(tmp_path) -> MemoryManager:  # type: ignore
    """Fresh manager per test. Uses tmp_path so L2's on-disk state
    is isolated per test."""
    m = MemoryManager(data_dir=tmp_path / "neuro-paging")
    yield m
    m.close()


@pytest.fixture
def ctx_evening_swiggy() -> ContextTags:
    """The deck's '9 PM, in Swiggy' example."""
    return ContextTags.now(
        time_bucket=TimeBucket.EVENING,
        foreground_app="Swiggy",
        location="home",
        semantic_tags=("food", "dinner"),
    )


@pytest.fixture
def ctx_morning_ide() -> ContextTags:
    """The deck's 'Tue 9 AM, IDE open' example."""
    return ContextTags.now(
        time_bucket=TimeBucket.MORNING,
        foreground_app="VSCode",
        location="office",
        semantic_tags=("code", "standup"),
    )


# ── Insert ────────────────────────────────────────────────────────────────────


class TestInsert:
    def test_insert_returns_id(self, mgr, ctx_evening_swiggy):
        mid = mgr.insert("user prefers Italian on weeknights", ctx_evening_swiggy)
        assert mid is not None
        assert isinstance(mid, str)

    def test_insert_defaults_to_l1(self, mgr, ctx_evening_swiggy):
        mid = mgr.insert("test memory", ctx_evening_swiggy)
        mem = mgr.get(mid)
        assert mem is not None
        assert mem.tier == Tier.L1

    def test_insert_to_l2_directly(self, mgr, ctx_evening_swiggy):
        mid = mgr.insert("hot memory", ctx_evening_swiggy, tier=Tier.L2)
        mem = mgr.get(mid)
        assert mem.tier == Tier.L2

    def test_insert_empty_raises(self, mgr, ctx_evening_swiggy):
        with pytest.raises(ValueError):
            mgr.insert("", ctx_evening_swiggy)
        with pytest.raises(ValueError):
            mgr.insert("   ", ctx_evening_swiggy)

    def test_ids_are_unique(self, mgr, ctx_evening_swiggy):
        ids = {mgr.insert(f"memory {i}", ctx_evening_swiggy) for i in range(50)}
        assert len(ids) == 50


# ── Query ─────────────────────────────────────────────────────────────────────


class TestQuery:
    def test_query_empty_store_returns_empty(self, mgr, ctx_evening_swiggy):
        hits = mgr.query("dinner ideas", ctx_evening_swiggy)
        assert hits == []

    def test_query_returns_hits(self, mgr, ctx_evening_swiggy):
        mgr.insert("user likes Italian food", ctx_evening_swiggy)
        mgr.insert("user dislikes spicy food", ctx_evening_swiggy)
        hits = mgr.query("Italian dinner", ctx_evening_swiggy)
        assert len(hits) >= 1
        assert all(isinstance(h, Hit) for h in hits)

    def test_query_respects_k(self, mgr, ctx_evening_swiggy):
        for i in range(20):
            mgr.insert(f"memory number {i} about food", ctx_evening_swiggy)
        hits = mgr.query("food", ctx_evening_swiggy, k=5)
        assert len(hits) == 5

    def test_query_k_zero_returns_empty(self, mgr, ctx_evening_swiggy):
        mgr.insert("memory", ctx_evening_swiggy)
        assert mgr.query("anything", ctx_evening_swiggy, k=0) == []

    def test_hits_carry_provenance(self, mgr, ctx_evening_swiggy):
        mgr.insert("dinner plan", ctx_evening_swiggy)
        hits = mgr.query("dinner", ctx_evening_swiggy)
        assert hits[0].provenance is not None
        assert hits[0].provenance.tier in (Tier.L1, Tier.L2, Tier.L3)
        assert hits[0].provenance.elapsed_ms >= 0
        # The provenance explanation should mention the tier
        assert hits[0].provenance.tier.value in hits[0].provenance.explanation()

    def test_query_increments_access_count(self, mgr, ctx_evening_swiggy):
        mid = mgr.insert("Italian food", ctx_evening_swiggy)
        mgr.query("Italian", ctx_evening_swiggy)
        assert mgr.get(mid).access_count == 1
        mgr.query("Italian", ctx_evening_swiggy)
        assert mgr.get(mid).access_count == 2


# ── Tier movement ─────────────────────────────────────────────────────────────


class TestTierMovement:
    def test_promote_l3_to_l2(self, mgr, ctx_evening_swiggy):
        mid = mgr.insert("cold memory", ctx_evening_swiggy, tier=Tier.L3)
        mgr.promote(mid)
        assert mgr.get(mid).tier == Tier.L2

    def test_promote_l2_to_l1(self, mgr, ctx_evening_swiggy):
        mid = mgr.insert("warm memory", ctx_evening_swiggy, tier=Tier.L2)
        mgr.promote(mid)
        assert mgr.get(mid).tier == Tier.L1

    def test_promote_l1_is_noop(self, mgr, ctx_evening_swiggy):
        mid = mgr.insert("hot memory", ctx_evening_swiggy, tier=Tier.L1)
        mgr.promote(mid)
        assert mgr.get(mid).tier == Tier.L1

    def test_demote_l1_to_l2(self, mgr, ctx_evening_swiggy):
        mid = mgr.insert("hot memory", ctx_evening_swiggy, tier=Tier.L1)
        mgr.demote(mid)
        assert mgr.get(mid).tier == Tier.L2

    def test_demote_l2_to_l3(self, mgr, ctx_evening_swiggy):
        mid = mgr.insert("warm memory", ctx_evening_swiggy, tier=Tier.L2)
        mgr.demote(mid)
        assert mgr.get(mid).tier == Tier.L3

    def test_demote_l3_is_noop(self, mgr, ctx_evening_swiggy):
        mid = mgr.insert("cold memory", ctx_evening_swiggy, tier=Tier.L3)
        mgr.demote(mid)
        assert mgr.get(mid).tier == Tier.L3

    def test_promote_unknown_id_does_not_raise(self, mgr):
        mgr.promote("nonexistent")  # logs a warning, doesn't crash

    def test_forget(self, mgr, ctx_evening_swiggy):
        mid = mgr.insert("doomed memory", ctx_evening_swiggy)
        assert mgr.forget(mid) is True
        assert mgr.get(mid) is None
        assert mgr.forget(mid) is False


# ── Tier ordering (regression test for the L3 >= L1 string-compare bug) ──────


class TestTierOrdering:
    """Cold→hot rank ordering. Was bitten by string comparison once. Never again."""

    def test_rank_ordering(self):
        assert Tier.L3.rank < Tier.L2.rank < Tier.L1.rank

    def test_is_hot(self):
        assert Tier.L1.is_hot
        assert Tier.L2.is_hot
        assert not Tier.L3.is_hot

    def test_at_least_as_cold(self):
        assert Tier.L3.is_at_least_as_cold_as(Tier.L1)
        assert Tier.L3.is_at_least_as_cold_as(Tier.L2)
        assert Tier.L2.is_at_least_as_cold_as(Tier.L1)
        assert not Tier.L1.is_at_least_as_cold_as(Tier.L3)

    def test_query_min_tier_l1_excludes_cold(self, mgr, ctx_evening_swiggy):
        """min_tier=L1 must skip L2 and L3 — the battery-critical path."""
        mgr.insert("hot mem", ctx_evening_swiggy, tier=Tier.L1)
        mgr.insert("warm mem", ctx_evening_swiggy, tier=Tier.L2)
        mgr.insert("cold mem", ctx_evening_swiggy, tier=Tier.L3)
        hits = mgr.query("mem", ctx_evening_swiggy, k=10, min_tier=Tier.L1)
        # Only the L1 memory should show up
        assert len(hits) == 1
        assert hits[0].provenance.tier == Tier.L1

    def test_query_min_tier_l3_includes_all(self, mgr, ctx_evening_swiggy):
        """min_tier=L3 (default) must include every tier."""
        mgr.insert("hot mem", ctx_evening_swiggy, tier=Tier.L1)
        mgr.insert("warm mem", ctx_evening_swiggy, tier=Tier.L2)
        mgr.insert("cold mem", ctx_evening_swiggy, tier=Tier.L3)
        hits = mgr.query("mem", ctx_evening_swiggy, k=10, min_tier=Tier.L3)
        assert len(hits) == 3


# ── Stats ─────────────────────────────────────────────────────────────────────


class TestStats:
    def test_stats_empty(self, mgr):
        s = mgr.get_stats()
        assert s.total_count == 0
        assert s.l1_count == 0 and s.l2_count == 0 and s.l3_count == 0
        assert s.queries_24h == 0

    def test_stats_counts_by_tier(self, mgr, ctx_evening_swiggy):
        for _ in range(3):
            mgr.insert("L1 mem", ctx_evening_swiggy, tier=Tier.L1)
        for _ in range(5):
            mgr.insert("L2 mem", ctx_evening_swiggy, tier=Tier.L2)
        for _ in range(2):
            mgr.insert("L3 mem", ctx_evening_swiggy, tier=Tier.L3)
        s = mgr.get_stats()
        assert s.l1_count == 3
        assert s.l2_count == 5
        assert s.l3_count == 2
        assert s.total_count == 10

    def test_stats_tracks_queries(self, mgr, ctx_evening_swiggy):
        mgr.insert("memory", ctx_evening_swiggy)
        mgr.query("memory", ctx_evening_swiggy)
        mgr.query("memory", ctx_evening_swiggy)
        s = mgr.get_stats()
        assert s.queries_24h == 2

    def test_stats_capacity_defaults(self, mgr):
        s = mgr.get_stats()
        assert s.l1_capacity_bytes == 32 * 1024
        assert s.l2_capacity_bytes == 8 * 1024 * 1024
        assert s.l3_capacity_bytes == 128 * 1024 * 1024


# ── Context types ─────────────────────────────────────────────────────────────


class TestContextTags:
    def test_now_factory(self):
        ctx = ContextTags.now()
        assert ctx.timestamp is not None
        assert ctx.time_bucket is not None
        assert ctx.battery_pct == 100

    def test_with_tags_returns_new_instance(self, ctx_evening_swiggy):
        new_ctx = ctx_evening_swiggy.with_tags("italian", "weeknight")
        assert "italian" in new_ctx.semantic_tags
        assert "weeknight" in new_ctx.semantic_tags
        # original unchanged (frozen)
        assert "italian" not in ctx_evening_swiggy.semantic_tags

    def test_time_bucket_from_datetime(self):
        from datetime import datetime

        # 7 PM → EVENING
        dt = datetime(2026, 5, 21, 19, 0, tzinfo=UTC)
        assert TimeBucket.from_datetime(dt) == TimeBucket.EVENING
        # 9 AM → MORNING
        dt = datetime(2026, 5, 21, 9, 0, tzinfo=UTC)
        assert TimeBucket.from_datetime(dt) == TimeBucket.MORNING


# ── Custom scorer wiring ──────────────────────────────────────────────────────


class TestCustomScorer:
    def test_custom_scorer_used(self, ctx_evening_swiggy):
        """A custom scorer takes precedence over the default stub."""

        class AlwaysOneScorer:
            def score(self, memory, query_text, query_context):
                from neuro_paging.memory.types import Provenance

                prov = Provenance(
                    tier=memory.tier,
                    raw_relevance=1.0,
                    raw_context_sim=1.0,
                    raw_frequency=0.0,
                    weights=(0.5, 0.5, 0.0),
                    final_score=1.0,
                    elapsed_ms=0.1,
                )
                return 1.0, prov

        mgr = MemoryManager(scorer=AlwaysOneScorer())
        mgr.insert("anything", ctx_evening_swiggy)
        hits = mgr.query("anything", ctx_evening_swiggy)
        assert hits[0].score == 1.0
        assert hits[0].provenance.weights == (0.5, 0.5, 0.0)

        # ── L1 cascade integration  ──────────────────────


class TestL1Cascade:
    """L1 is now real (FIFO + byte budget). Verify the cascade-to-L2 path."""

    def test_l1_capacity_cascades_to_l2(self, tmp_path):
        """When L1 fills up, evicted memories should land in L2 automatically."""
        mgr = MemoryManager(data_dir=tmp_path / "np", l1_capacity_bytes=600)
        ctx = ContextTags.now()

        for i in range(10):
            mgr.insert(f"memory number {i} with some filler text", ctx)

        stats = mgr.get_stats()
        # Some memories made it to L2 via cascade
        assert stats.l2_count > 0, "L1 should have cascaded overflow to L2"
        # L1 stayed within budget
        assert stats.l1_bytes <= stats.l1_capacity_bytes
        # All 10 memories accounted for somewhere
        assert stats.total_count == 10
        mgr.close()

    def test_oversized_memory_routes_directly_to_l2(self, tmp_path):
        """A single memory bigger than L1 capacity should skip L1 entirely."""
        mgr = MemoryManager(data_dir=tmp_path / "np", l1_capacity_bytes=512)
        ctx = ContextTags.now()

        # ~10KB memory, L1 is 512B
        mid = mgr.insert("x" * 10_000, ctx)

        mem = mgr.get(mid)
        assert mem is not None
        assert mem.tier == Tier.L2, "Oversized memory should land in L2"

        stats = mgr.get_stats()
        assert stats.l1_count == 0
        assert stats.l2_count == 1
        mgr.close()

    def test_cascade_demotions_counted(self, tmp_path):
        """The demotions_24h counter should track cascade events."""
        mgr = MemoryManager(data_dir=tmp_path / "np", l1_capacity_bytes=600)
        ctx = ContextTags.now()

        for i in range(10):
            mgr.insert(f"memory {i} with filler", ctx)

        stats = mgr.get_stats()
        assert stats.demotions_24h > 0
        mgr.close()

    def test_query_finds_memories_in_cascaded_l2(self, tmp_path):
        """A memory cascaded to L2 must still be findable by query()."""
        mgr = MemoryManager(data_dir=tmp_path / "np", l1_capacity_bytes=600)
        ctx = ContextTags.now()

        target_id = mgr.insert("unique-target-phrase needle", ctx)
        # Push lots of memories to force the target to cascade to L2
        for i in range(20):
            mgr.insert(f"filler memory {i} hay hay hay", ctx)

        hits = mgr.query("needle", ctx, k=10)
        found_ids = {h.memory_id for h in hits}
        assert target_id in found_ids
        mgr.close()

    def test_l1_real_stats_match_underlying(self, tmp_path):
        """get_stats() should reflect the L1WorkingContext's actual state."""
        mgr = MemoryManager(data_dir=tmp_path / "np", l1_capacity_bytes=4096)
        ctx = ContextTags.now()

        for i in range(3):
            mgr.insert(f"memory {i}", ctx, tier=Tier.L1)

        stats = mgr.get_stats()
        assert stats.l1_count == 3
        assert stats.l1_bytes > 0
        assert stats.l1_capacity_bytes == 4096
        mgr.close()
