"""Tests for L3ArchiveCache — the archive tier.

Same pattern as test_l2_cache.py since L3 shares L2's backbone in v0.
Each test gets a tmp_path so on-disk state is isolated.
"""

from __future__ import annotations

from datetime import UTC, datetime

import numpy as np
import pytest

from neuro_paging import ContextTags, TimeBucket
from neuro_paging.memory.l3_archive import L3ArchiveCache
from neuro_paging.memory.types import Memory, MemoryId, Tier


def _make_memory(text: str = "hello", mid: str = "m1", tier: Tier = Tier.L3) -> Memory:
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
def l3(tmp_path):
    cache = L3ArchiveCache(
        data_dir=tmp_path / "l3",
        capacity_bytes=4 * 1024 * 1024,
        max_elements=500,
        dim=384,
    )
    yield cache
    cache.close()


@pytest.fixture
def rng():
    return np.random.default_rng(42)


class TestConstruction:
    def test_invalid_capacity_rejected(self, tmp_path):
        with pytest.raises(ValueError):
            L3ArchiveCache(data_dir=tmp_path / "x", capacity_bytes=0)

    def test_invalid_max_elements_rejected(self, tmp_path):
        with pytest.raises(ValueError):
            L3ArchiveCache(data_dir=tmp_path / "x", max_elements=0)

    def test_initial_stats(self, l3):
        s = l3.stats()
        assert s.count == 0
        assert s.bytes_estimate == 0


class TestInsertRead:
    def test_insert_then_get(self, l3, rng):
        mem = _make_memory("archived memory", mid="m1")
        l3.insert(mem, _rand_vec(rng))
        loaded = l3.get("m1")
        assert loaded is not None
        assert loaded.text == "archived memory"
        assert loaded.tier == Tier.L3

    def test_insert_with_wrong_tier_gets_corrected(self, l3, rng):
        """A memory with tier=L1 inserted into L3 should be stored as L3."""
        mem = _make_memory("test", mid="m1", tier=Tier.L1)
        l3.insert(mem, _rand_vec(rng))
        loaded = l3.get("m1")
        assert loaded is not None
        assert loaded.tier == Tier.L3

    def test_contains(self, l3, rng):
        l3.insert(_make_memory(mid="m1"), _rand_vec(rng))
        assert l3.contains("m1")
        assert not l3.contains("nope")

    def test_len(self, l3, rng):
        for i in range(5):
            l3.insert(_make_memory(mid=f"m{i}"), _rand_vec(rng))
        assert len(l3) == 5

    def test_wrong_dim_raises(self, l3):
        with pytest.raises(ValueError, match="dim"):
            l3.insert(_make_memory(mid="m1"), np.zeros(99, dtype=np.float32))


class TestQuery:
    def test_empty_returns_empty(self, l3, rng):
        assert l3.query(_rand_vec(rng), k=5) == []

    def test_query_finds_exact_match(self, l3, rng):
        target_vec = _rand_vec(rng)
        l3.insert(_make_memory(mid="target"), target_vec)
        for i in range(5):
            l3.insert(_make_memory(mid=f"noise{i}"), _rand_vec(rng))

        hits = l3.query(target_vec, k=1)
        assert len(hits) == 1
        mem, dist = hits[0]
        assert mem.id == "target"
        assert dist < 0.01

    def test_query_wrong_dim_raises(self, l3):
        with pytest.raises(ValueError, match="dim"):
            l3.query(np.zeros(99, dtype=np.float32), k=5)


class TestEviction:
    def test_byte_budget_caps_total(self, tmp_path, rng):
        cache = L3ArchiveCache(
            data_dir=tmp_path / "l3",
            capacity_bytes=30 * 1024,
            max_elements=500,
            dim=384,
        )
        for i in range(50):
            cache.insert(
                _make_memory(f"text {i} content", mid=f"m{i}"),
                _rand_vec(rng),
            )
        stats = cache.stats()
        assert stats.bytes_estimate <= stats.capacity_bytes
        assert stats.evictions_total > 0
        cache.close()

    def test_index_grows_when_max_elements_exhausted(self, tmp_path, rng):
        cache = L3ArchiveCache(
            data_dir=tmp_path / "l3",
            capacity_bytes=10 * 1024 * 1024,
            max_elements=5,
            dim=384,
        )
        start_max = cache.stats().max_elements
        for i in range(12):
            cache.insert(_make_memory(f"memory {i}", mid=f"m{i}"), _rand_vec(rng))
        end_max = cache.stats().max_elements
        assert end_max > start_max
        cache.close()


class TestTouchRemove:
    def test_touch_bumps_access_count(self, l3, rng):
        l3.insert(_make_memory(mid="m1"), _rand_vec(rng))
        l3.touch("m1")
        l3.touch("m1")
        assert l3.get("m1").access_count == 2

    def test_remove(self, l3, rng):
        l3.insert(_make_memory(mid="m1"), _rand_vec(rng))
        removed = l3.remove("m1")
        assert removed is not None and removed.id == "m1"
        assert not l3.contains("m1")


class TestPersistence:
    def test_save_and_reopen(self, tmp_path, rng):
        path = tmp_path / "persist"
        c1 = L3ArchiveCache(data_dir=path, max_elements=500, dim=384)
        vec = _rand_vec(rng)
        c1.insert(_make_memory("durable", mid="m1"), vec)
        c1.close()

        c2 = L3ArchiveCache(data_dir=path, max_elements=500, dim=384)
        loaded = c2.get("m1")
        assert loaded is not None
        assert loaded.text == "durable"
        hits = c2.query(vec, k=1)
        assert hits[0][0].id == "m1"
        c2.close()
