"""Tests for L2HotVectorCache — the real L2 tier.

Every test uses tmp_path so disk artifacts don't leak between tests.
Embeddings are deterministic random vectors via a seeded RNG.
"""

from __future__ import annotations

from datetime import UTC, datetime

import numpy as np
import pytest

from neuro_paging import ContextTags, TimeBucket
from neuro_paging.memory.l2_cache import L2HotVectorCache
from neuro_paging.memory.types import Memory, MemoryId, Tier

# ── Helpers ───────────────────────────────────────────────────────────────────


def _make_memory(text: str = "hello", mid: str = "m1", tier: Tier = Tier.L2) -> Memory:
    now = datetime.now(UTC)
    return Memory(
        id=MemoryId(mid),
        text=text,
        embedding_ref=f"emb:{mid}",
        context=ContextTags.now(time_bucket=TimeBucket.EVENING),
        tier=tier,
        created_at=now,
        last_touch=now,
        access_count=0,
        is_consolidated=False,
    )


def _rand_vec(rng: np.random.Generator, dim: int = 384) -> np.ndarray:
    v = rng.standard_normal(dim).astype(np.float32)
    v /= np.linalg.norm(v) + 1e-8
    return v


@pytest.fixture
def l2(tmp_path):
    """Fresh L2 cache per test, automatically cleaned up."""
    cache = L2HotVectorCache(
        data_dir=tmp_path / "l2",
        capacity_bytes=1024 * 1024,  # 1 MB for tests
        max_elements=500,
        dim=384,
    )
    yield cache
    cache.close()


@pytest.fixture
def rng():
    return np.random.default_rng(42)


# ── Construction ──────────────────────────────────────────────────────────────


class TestConstruction:
    def test_creates_data_dir(self, tmp_path):
        path = tmp_path / "fresh_l2"
        cache = L2HotVectorCache(data_dir=path, dim=384)
        assert path.exists()
        cache.close()

    def test_invalid_capacity_rejected(self, tmp_path):
        with pytest.raises(ValueError):
            L2HotVectorCache(data_dir=tmp_path / "x", capacity_bytes=0)

    def test_invalid_max_elements_rejected(self, tmp_path):
        with pytest.raises(ValueError):
            L2HotVectorCache(data_dir=tmp_path / "x", max_elements=0)

    def test_initial_stats(self, l2):
        s = l2.stats()
        assert s.count == 0
        assert s.bytes_estimate == 0
        assert s.inserts_total == 0
        assert s.evictions_total == 0


# ── Insert + read ─────────────────────────────────────────────────────────────


class TestInsertRead:
    def test_insert_then_get(self, l2, rng):
        mem = _make_memory("italian food preference", mid="m1")
        evicted = l2.insert(mem, _rand_vec(rng))
        assert evicted == []
        loaded = l2.get("m1")
        assert loaded is not None
        assert loaded.text == "italian food preference"

    def test_contains(self, l2, rng):
        l2.insert(_make_memory(mid="m1"), _rand_vec(rng))
        assert l2.contains("m1")
        assert not l2.contains("nope")

    def test_len(self, l2, rng):
        for i in range(5):
            l2.insert(_make_memory(mid=f"m{i}"), _rand_vec(rng))
        assert len(l2) == 5

    def test_inserts_total_counter(self, l2, rng):
        for i in range(3):
            l2.insert(_make_memory(mid=f"m{i}"), _rand_vec(rng))
        assert l2.stats().inserts_total == 3

    def test_wrong_dim_raises(self, l2):
        with pytest.raises(ValueError, match="dim"):
            l2.insert(_make_memory(mid="m1"), np.zeros(99, dtype=np.float32))


# ── Query ─────────────────────────────────────────────────────────────────────


class TestQuery:
    def test_empty_cache_returns_empty(self, l2, rng):
        assert l2.query(_rand_vec(rng), k=5) == []

    def test_query_returns_hits(self, l2, rng):
        for i in range(10):
            l2.insert(_make_memory(mid=f"m{i}"), _rand_vec(rng))
        hits = l2.query(_rand_vec(rng), k=5)
        assert len(hits) == 5
        assert all(isinstance(m, Memory) and isinstance(d, float) for m, d in hits)

    def test_query_k_larger_than_cache(self, l2, rng):
        for i in range(3):
            l2.insert(_make_memory(mid=f"m{i}"), _rand_vec(rng))
        hits = l2.query(_rand_vec(rng), k=10)
        assert len(hits) == 3

    def test_query_k_zero(self, l2, rng):
        l2.insert(_make_memory(mid="m1"), _rand_vec(rng))
        assert l2.query(_rand_vec(rng), k=0) == []

    def test_query_finds_exact_match(self, l2, rng):
        """A query vector matching an inserted vector exactly should return it first."""
        target_vec = _rand_vec(rng)
        l2.insert(_make_memory(mid="target"), target_vec)
        for i in range(5):
            l2.insert(_make_memory(mid=f"noise{i}"), _rand_vec(rng))

        hits = l2.query(target_vec, k=1)
        assert len(hits) == 1
        mem, dist = hits[0]
        assert mem.id == "target"
        # Cosine distance for identical vectors is ~0
        assert dist < 0.01

    def test_query_wrong_dim_raises(self, l2):
        with pytest.raises(ValueError, match="dim"):
            l2.query(np.zeros(99, dtype=np.float32), k=5)


# ── Eviction ──────────────────────────────────────────────────────────────────


class TestEviction:
    """L2's two-budget eviction:
    - byte_budget (capacity_bytes): the REAL cap, matches deck's '~8 MB'
    - max_elements: starting slot count; index grows geometrically when full
    """

    def test_byte_budget_caps_total_size(self, tmp_path, rng):
        """Byte budget is the deck's actual cap. L2 must stay within it."""
        # Tight byte budget (~30 KB), generous element count
        cache = L2HotVectorCache(
            data_dir=tmp_path / "l2",
            capacity_bytes=30 * 1024,
            max_elements=500,
            dim=384,
        )
        # Each memory ~1.5KB overhead + text → roughly 20 fit in 30KB
        for i in range(50):
            cache.insert(
                _make_memory(f"memory text number {i} filler", mid=f"m{i}"),
                _rand_vec(rng),
            )

        stats = cache.stats()
        assert stats.bytes_estimate <= stats.capacity_bytes, (
            f"L2 exceeded byte budget: {stats.bytes_estimate} > {stats.capacity_bytes}"
        )
        assert stats.evictions_total > 0, "Byte budget should have triggered eviction"
        cache.close()

    def test_eviction_returns_oldest_first(self, tmp_path, rng):
        """When the byte budget evicts, oldest goes first."""
        import time

        # Tight byte budget to force eviction
        cache = L2HotVectorCache(
            data_dir=tmp_path / "l2",
            capacity_bytes=4 * 1024,  # ~4KB — only ~2 memories fit
            max_elements=500,
            dim=384,
        )
        cache.insert(_make_memory("oldest text content", mid="oldest"), _rand_vec(rng))
        time.sleep(0.02)
        cache.insert(_make_memory("middle text content", mid="middle"), _rand_vec(rng))
        time.sleep(0.02)
        evicted_so_far = []
        evicted_so_far.extend(
            cache.insert(_make_memory("third text content", mid="third"), _rand_vec(rng))
        )
        time.sleep(0.02)
        evicted_so_far.extend(
            cache.insert(_make_memory("fourth text content", mid="fourth"), _rand_vec(rng))
        )

        # At least one eviction happened, and the first eviction was the oldest
        assert len(evicted_so_far) >= 1
        assert evicted_so_far[0].id == "oldest", (
            f"Expected oldest evicted first, got {evicted_so_far[0].id}"
        )
        cache.close()

    def test_index_grows_when_max_elements_exhausted(self, tmp_path, rng):
        """When the HNSW index runs out of physical slots, it grows."""
        cache = L2HotVectorCache(
            data_dir=tmp_path / "l2",
            capacity_bytes=10 * 1024 * 1024,  # 10 MB — byte budget won't trigger
            max_elements=5,  # small starting capacity
            dim=384,
        )
        start_max = cache.stats().max_elements
        # Insert beyond the initial max_elements
        for i in range(12):
            cache.insert(_make_memory(f"memory {i}", mid=f"m{i}"), _rand_vec(rng))

        end_max = cache.stats().max_elements
        assert end_max > start_max, f"Index should have grown: start={start_max}, end={end_max}"
        # Most memories should still be alive — growth, not eviction
        assert len(cache) >= 10
        cache.close()

    def test_byte_eviction_counter_advances(self, tmp_path, rng):
        """The evictions_total counter advances when the byte budget triggers."""
        cache = L2HotVectorCache(
            data_dir=tmp_path / "l2",
            capacity_bytes=4 * 1024,  # tight byte budget
            max_elements=500,
            dim=384,
        )
        # Insert enough to overflow the byte budget several times
        for i in range(20):
            cache.insert(
                _make_memory(f"memory text number {i} with content", mid=f"m{i}"),
                _rand_vec(rng),
            )
        assert cache.stats().evictions_total >= 3
        cache.close()


# ── Update (re-insert same id) ────────────────────────────────────────────────


class TestUpdate:
    def test_reinsert_updates_text(self, l2, rng):
        l2.insert(_make_memory("v1", mid="m1"), _rand_vec(rng))
        l2.insert(_make_memory("v2", mid="m1"), _rand_vec(rng))
        assert l2.get("m1").text == "v2"
        assert len(l2) == 1  # not duplicated

    def test_reinsert_creates_tombstone(self, l2, rng):
        l2.insert(_make_memory(mid="m1"), _rand_vec(rng))
        before = l2.stats().tombstones_total
        l2.insert(_make_memory(mid="m1"), _rand_vec(rng))
        after = l2.stats().tombstones_total
        assert after > before


# ── Touch + remove ────────────────────────────────────────────────────────────


class TestTouchRemove:
    def test_touch_bumps_access_count(self, l2, rng):
        l2.insert(_make_memory(mid="m1"), _rand_vec(rng))
        l2.touch("m1")
        l2.touch("m1")
        assert l2.get("m1").access_count == 2

    def test_touch_missing_returns_false(self, l2):
        assert l2.touch("nonexistent") is False

    def test_remove_existing(self, l2, rng):
        l2.insert(_make_memory(mid="m1"), _rand_vec(rng))
        removed = l2.remove("m1")
        assert removed is not None and removed.id == "m1"
        assert not l2.contains("m1")

    def test_remove_missing_returns_none(self, l2):
        assert l2.remove("nope") is None


# ── Persistence ───────────────────────────────────────────────────────────────


class TestPersistence:
    def test_save_and_reopen(self, tmp_path, rng):
        path = tmp_path / "persist"
        c1 = L2HotVectorCache(data_dir=path, max_elements=500, dim=384)
        vec = _rand_vec(rng)
        c1.insert(_make_memory("durable", mid="m1"), vec)
        c1.save()
        c1.close()

        c2 = L2HotVectorCache(data_dir=path, max_elements=500, dim=384)
        loaded = c2.get("m1")
        assert loaded is not None
        assert loaded.text == "durable"
        # The vector survived too — exact-match query should find it
        hits = c2.query(vec, k=1)
        assert hits[0][0].id == "m1"
        c2.close()

    def test_label_counter_resumes_after_reopen(self, tmp_path, rng):
        path = tmp_path / "labels"
        c1 = L2HotVectorCache(data_dir=path, max_elements=500, dim=384)
        c1.insert(_make_memory(mid="m1"), _rand_vec(rng))
        c1.insert(_make_memory(mid="m2"), _rand_vec(rng))
        c1.close()

        c2 = L2HotVectorCache(data_dir=path, max_elements=500, dim=384)
        # Next insert must get a label > the existing max
        c2.insert(_make_memory(mid="m3"), _rand_vec(rng))
        # _next_label should be at least 3 now
        assert c2._next_label >= 3
        c2.close()
