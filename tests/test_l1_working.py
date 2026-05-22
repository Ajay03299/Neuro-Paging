"""Tests for L1WorkingContext.

Pin down the eviction semantics tightly. L1 is the most-touched tier
in the whole system — if its budget accounting is off or its FIFO
order is wrong, downstream tiers inherit the mess.
"""

from __future__ import annotations

from datetime import UTC, datetime

import pytest

from neuro_paging import ContextTags, TimeBucket
from neuro_paging.memory.l1_working import L1WorkingContext
from neuro_paging.memory.types import Memory, MemoryId, Tier

# ── Helpers ───────────────────────────────────────────────────────────────────


def _ctx() -> ContextTags:
    return ContextTags.now(time_bucket=TimeBucket.EVENING)


def _make_memory(text: str, mem_id: str | None = None) -> Memory:
    """Build a Memory with deterministic id (or auto)."""
    now = datetime.now(UTC)
    mid = MemoryId(mem_id) if mem_id else MemoryId(f"mem-{abs(hash(text)) % 10**9}")
    return Memory(
        id=mid,
        text=text,
        embedding_ref=f"stub-emb:{mid}",
        context=_ctx(),
        tier=Tier.L1,
        created_at=now,
        last_touch=now,
        access_count=0,
        is_consolidated=False,
    )


# ── Construction ──────────────────────────────────────────────────────────────


class TestConstruction:
    def test_default_capacity(self):
        l1 = L1WorkingContext()
        assert l1.capacity_bytes == 32 * 1024

    def test_custom_capacity(self):
        l1 = L1WorkingContext(capacity_bytes=4096)
        assert l1.capacity_bytes == 4096

    def test_zero_capacity_rejected(self):
        with pytest.raises(ValueError):
            L1WorkingContext(capacity_bytes=0)

    def test_negative_capacity_rejected(self):
        with pytest.raises(ValueError):
            L1WorkingContext(capacity_bytes=-1)

    def test_empty_stats(self):
        l1 = L1WorkingContext(capacity_bytes=4096)
        s = l1.stats()
        assert s.count == 0
        assert s.bytes_used == 0
        assert s.capacity_bytes == 4096
        assert s.inserts_total == 0
        assert s.evictions_total == 0
        assert s.rejects_total == 0
        assert s.utilization == 0.0


# ── Basic insert / read ───────────────────────────────────────────────────────


class TestInsertRead:
    def test_insert_single_memory(self):
        l1 = L1WorkingContext()
        mem = _make_memory("hello world", mem_id="m1")
        evicted = l1.insert(mem)
        assert evicted == []
        assert len(l1) == 1
        assert l1.contains("m1")

    def test_get_returns_memory(self):
        l1 = L1WorkingContext()
        mem = _make_memory("hi", mem_id="m1")
        l1.insert(mem)
        assert l1.get("m1") == mem

    def test_get_missing_returns_none(self):
        l1 = L1WorkingContext()
        assert l1.get("nonexistent") is None

    def test_contains_missing(self):
        l1 = L1WorkingContext()
        assert not l1.contains("nope")

    def test_stats_tracks_inserts(self):
        l1 = L1WorkingContext()
        for i in range(5):
            l1.insert(_make_memory(f"memory {i}", mem_id=f"m{i}"))
        s = l1.stats()
        assert s.count == 5
        assert s.inserts_total == 5
        assert s.bytes_used > 0


# ── FIFO eviction ─────────────────────────────────────────────────────────────


class TestFIFOEviction:
    def test_eviction_when_full(self):
        # Tight budget: just enough for ~2 small memories
        l1 = L1WorkingContext(capacity_bytes=600)
        m1 = _make_memory("first", mem_id="m1")
        m2 = _make_memory("second", mem_id="m2")
        m3 = _make_memory("third", mem_id="m3")

        e1 = l1.insert(m1)
        e2 = l1.insert(m2)
        e3 = l1.insert(m3)  # Should evict m1

        assert e1 == []
        assert e2 == []
        assert len(e3) >= 1
        assert e3[0].id == "m1", "FIFO must evict the oldest first"

    def test_evicted_memories_returned(self):
        """Caller (manager) must receive every evicted memory to demote."""
        l1 = L1WorkingContext(capacity_bytes=600)
        ids = [f"m{i}" for i in range(10)]
        all_evicted = []
        for mid in ids:
            evicted = l1.insert(_make_memory(f"text {mid}", mem_id=mid))
            all_evicted.extend(evicted)

        # Every evicted memory should be one we originally inserted
        evicted_ids = {ev.id for ev in all_evicted}
        assert evicted_ids.issubset(set(ids))
        # The newest few should still be in L1
        in_l1_ids = {mem.id for mem in l1}
        assert "m9" in in_l1_ids  # most recent insert

    def test_eviction_order_is_oldest_first(self):
        l1 = L1WorkingContext(capacity_bytes=600)
        l1.insert(_make_memory("oldest", mem_id="oldest"))
        l1.insert(_make_memory("middle", mem_id="middle"))
        evicted = l1.insert(_make_memory("newest", mem_id="newest"))

        # Whatever was evicted, "oldest" must come before "middle"
        if len(evicted) >= 2:
            assert evicted[0].id == "oldest"
            assert evicted[1].id == "middle"
        elif len(evicted) == 1:
            assert evicted[0].id == "oldest"

    def test_bytes_used_stays_within_capacity(self):
        l1 = L1WorkingContext(capacity_bytes=1024)
        for i in range(50):
            l1.insert(_make_memory(f"memory text number {i}" * 3, mem_id=f"m{i}"))
            s = l1.stats()
            assert s.bytes_used <= l1.capacity_bytes, (
                f"L1 exceeded budget at insert {i}: {s.bytes_used} > {l1.capacity_bytes}"
            )

    def test_evictions_total_counter(self):
        l1 = L1WorkingContext(capacity_bytes=600)
        for i in range(10):
            l1.insert(_make_memory(f"m{i} text", mem_id=f"m{i}"))
        assert l1.stats().evictions_total > 0


# ── Oversized memory rejection ────────────────────────────────────────────────


class TestOversizedReject:
    def test_oversized_memory_raises(self):
        l1 = L1WorkingContext(capacity_bytes=512)
        huge = _make_memory("x" * 10_000, mem_id="huge")
        with pytest.raises(ValueError, match="exceeds L1 capacity"):
            l1.insert(huge)

    def test_oversized_does_not_corrupt_state(self):
        l1 = L1WorkingContext(capacity_bytes=512)
        l1.insert(_make_memory("small", mem_id="s1"))
        with pytest.raises(ValueError):
            l1.insert(_make_memory("x" * 10_000, mem_id="huge"))
        # Original small memory is still there
        assert l1.contains("s1")
        assert l1.stats().rejects_total == 1


# ── Update semantics (re-insert same id) ──────────────────────────────────────


class TestReinsertSemantics:
    def test_reinsert_same_id_updates(self):
        l1 = L1WorkingContext()
        mem1 = _make_memory("first version", mem_id="m1")
        mem2 = _make_memory("second version", mem_id="m1")  # same id
        l1.insert(mem1)
        l1.insert(mem2)
        assert len(l1) == 1
        assert l1.get("m1").text == "second version"

    def test_reinsert_moves_to_back(self):
        """Updating an existing memory should refresh its FIFO position."""
        l1 = L1WorkingContext(capacity_bytes=2000)
        l1.insert(_make_memory("text a", mem_id="a"))
        l1.insert(_make_memory("text b", mem_id="b"))
        l1.insert(_make_memory("text c", mem_id="c"))

        # Re-insert "a" — should move to the back
        l1.insert(_make_memory("text a updated", mem_id="a"))

        order = [m.id for m in l1]
        assert order == ["b", "c", "a"]


# ── Touch (move to back of FIFO without re-inserting) ─────────────────────────


class TestTouch:
    def test_touch_existing_moves_to_back(self):
        l1 = L1WorkingContext()
        l1.insert(_make_memory("a", mem_id="a"))
        l1.insert(_make_memory("b", mem_id="b"))
        l1.insert(_make_memory("c", mem_id="c"))

        assert l1.touch("a") is True
        order = [m.id for m in l1]
        assert order == ["b", "c", "a"]

    def test_touch_missing_returns_false(self):
        l1 = L1WorkingContext()
        l1.insert(_make_memory("a", mem_id="a"))
        assert l1.touch("nonexistent") is False

    def test_touch_protects_from_eviction(self):
        """If we touch an old item, the next eviction shouldn't kill it."""
        l1 = L1WorkingContext(capacity_bytes=600)
        l1.insert(_make_memory("old but touched", mem_id="protected"))
        l1.insert(_make_memory("middle", mem_id="middle"))
        l1.touch("protected")  # refresh it
        evicted = l1.insert(_make_memory("newest", mem_id="newest"))
        evicted_ids = {ev.id for ev in evicted}
        assert "protected" not in evicted_ids, "Touched memory should not be the first to evict"


# ── Remove + clear ────────────────────────────────────────────────────────────


class TestRemove:
    def test_remove_existing(self):
        l1 = L1WorkingContext()
        mem = _make_memory("doomed", mem_id="d1")
        l1.insert(mem)
        removed = l1.remove("d1")
        assert removed == mem
        assert not l1.contains("d1")
        assert l1.stats().bytes_used == 0

    def test_remove_missing_returns_none(self):
        l1 = L1WorkingContext()
        assert l1.remove("nonexistent") is None

    def test_clear_returns_all(self):
        l1 = L1WorkingContext()
        for i in range(5):
            l1.insert(_make_memory(f"m{i}", mem_id=f"m{i}"))
        cleared = l1.clear()
        assert len(cleared) == 5
        assert len(l1) == 0
        assert l1.stats().bytes_used == 0


# ── Sizing helper ─────────────────────────────────────────────────────────────


class TestSizing:
    def test_size_includes_text(self):
        small = _make_memory("hi")
        big = _make_memory("hi" * 1000)
        assert L1WorkingContext.memory_size_bytes(big) > L1WorkingContext.memory_size_bytes(small)

    def test_size_is_deterministic(self):
        mem = _make_memory("hello world", mem_id="m1")
        s1 = L1WorkingContext.memory_size_bytes(mem)
        s2 = L1WorkingContext.memory_size_bytes(mem)
        assert s1 == s2

    def test_size_has_overhead(self):
        """Even a tiny memory should account for some overhead."""
        mem = _make_memory("x", mem_id="x")
        assert L1WorkingContext.memory_size_bytes(mem) > 100  # at least the overhead
