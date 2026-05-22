"""Tests for L2Metadata — the SQLite sidecar.

These tests use a temporary directory per test (pytest's tmp_path
fixture) so each one gets a fresh DB. SQLite is fast — even with
hundreds of tests this is negligible.
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

import pytest

from neuro_paging import ContextTags, TimeBucket
from neuro_paging.memory.l2_metadata import SCHEMA_VERSION, L2Metadata
from neuro_paging.memory.types import Memory, MemoryId, Tier

# ── Helpers ───────────────────────────────────────────────────────────────────


def _make_memory(text: str = "hello world", mid: str = "m1", tier: Tier = Tier.L2) -> Memory:
    now = datetime.now(UTC)
    return Memory(
        id=MemoryId(mid),
        text=text,
        embedding_ref=f"emb:{mid}",
        context=ContextTags.now(
            time_bucket=TimeBucket.EVENING,
            semantic_tags=("food", "italian"),
        ),
        tier=tier,
        created_at=now,
        last_touch=now,
        access_count=0,
        is_consolidated=False,
    )


@pytest.fixture
def db(tmp_path):
    """Fresh L2Metadata per test, automatically cleaned up."""
    path = tmp_path / "test_l2.db"
    md = L2Metadata(path)
    yield md
    md.close()


# ── Construction & schema ────────────────────────────────────────────────────


class TestConstruction:
    def test_db_file_created(self, tmp_path):
        path = tmp_path / "subdir" / "fresh.db"
        md = L2Metadata(path)
        assert path.exists(), "DB file should be created on init"
        assert md.db_path == path
        md.close()

    def test_schema_version_recorded(self, db):
        with db._lock:
            cur = db._conn.execute("SELECT MAX(version) FROM schema_version")
            assert cur.fetchone()[0] == SCHEMA_VERSION

    def test_migrate_is_idempotent(self, tmp_path):
        """Re-opening the same DB should not re-run migrations."""
        path = tmp_path / "idempotent.db"
        L2Metadata(path).close()
        L2Metadata(path).close()  # second open
        md = L2Metadata(path)
        with md._lock:
            cur = md._conn.execute("SELECT COUNT(*) FROM schema_version")
            assert cur.fetchone()[0] == 1  # only one row written
        md.close()


# ── Insert & roundtrip ───────────────────────────────────────────────────────


class TestInsertRoundtrip:
    def test_insert_then_get(self, db):
        mem = _make_memory("user prefers Italian", mid="m1")
        db.insert(mem, hnsw_label=42)
        loaded = db.get("m1")
        assert loaded is not None
        assert loaded.id == "m1"
        assert loaded.text == "user prefers Italian"
        assert loaded.tier == Tier.L2
        assert "food" in loaded.context.semantic_tags

    def test_insert_preserves_context_fully(self, db):
        ctx = ContextTags.now(
            time_bucket=TimeBucket.MORNING,
            location="office",
            foreground_app="VSCode",
            calendar_event="standup",
            semantic_tags=("code", "review"),
        )
        mem = Memory(
            id=MemoryId("m1"),
            text="t",
            embedding_ref="e",
            context=ctx,
            tier=Tier.L2,
            created_at=datetime.now(UTC),
            last_touch=datetime.now(UTC),
        )
        db.insert(mem, hnsw_label=0)
        loaded = db.get("m1")
        assert loaded.context.location == "office"
        assert loaded.context.foreground_app == "VSCode"
        assert loaded.context.calendar_event == "standup"
        assert loaded.context.semantic_tags == ("code", "review")

    def test_duplicate_insert_raises(self, db):
        import sqlite3

        db.insert(_make_memory(mid="m1"), hnsw_label=0)
        with pytest.raises(sqlite3.IntegrityError):
            db.insert(_make_memory(mid="m1"), hnsw_label=1)

    def test_duplicate_label_raises(self, db):
        import sqlite3

        db.insert(_make_memory(mid="m1"), hnsw_label=5)
        with pytest.raises(sqlite3.IntegrityError):
            db.insert(_make_memory(mid="m2"), hnsw_label=5)

    def test_upsert_replaces(self, db):
        db.insert(_make_memory("v1", mid="m1"), hnsw_label=0)
        new_mem = _make_memory("v2", mid="m1")
        db.upsert(new_mem, hnsw_label=0)
        assert db.get("m1").text == "v2"


# ── Lookups & counts ─────────────────────────────────────────────────────────


class TestLookups:
    def test_get_by_label(self, db):
        db.insert(_make_memory(mid="m1"), hnsw_label=42)
        loaded = db.get_by_label(42)
        assert loaded is not None and loaded.id == "m1"

    def test_label_for(self, db):
        db.insert(_make_memory(mid="m1"), hnsw_label=42)
        assert db.label_for("m1") == 42
        assert db.label_for("nope") is None

    def test_contains(self, db):
        db.insert(_make_memory(mid="m1"), hnsw_label=0)
        assert db.contains("m1")
        assert not db.contains("m2")

    def test_count_all(self, db):
        for i in range(5):
            db.insert(_make_memory(mid=f"m{i}"), hnsw_label=i)
        assert db.count() == 5

    def test_count_by_tier(self, db):
        for i in range(3):
            db.insert(_make_memory(mid=f"l2-{i}", tier=Tier.L2), hnsw_label=i)
        for i in range(2):
            db.insert(_make_memory(mid=f"l3-{i}", tier=Tier.L3), hnsw_label=100 + i)
        assert db.count(Tier.L2) == 3
        assert db.count(Tier.L3) == 2
        assert db.count(Tier.L1) == 0


# ── Mutations ────────────────────────────────────────────────────────────────


class TestMutations:
    def test_touch_bumps_access_count(self, db):
        db.insert(_make_memory(mid="m1"), hnsw_label=0)
        assert db.touch("m1") is True
        assert db.get("m1").access_count == 1
        db.touch("m1")
        assert db.get("m1").access_count == 2

    def test_touch_missing_returns_false(self, db):
        assert db.touch("nonexistent") is False

    def test_touch_updates_last_touch_timestamp(self, db):
        db.insert(_make_memory(mid="m1"), hnsw_label=0)
        before = db.get("m1").last_touch
        # Sleep a tiny bit so the timestamp moves
        import time

        time.sleep(0.01)
        db.touch("m1")
        after = db.get("m1").last_touch
        assert after > before

    def test_delete(self, db):
        db.insert(_make_memory(mid="m1"), hnsw_label=0)
        assert db.delete("m1") is True
        assert db.get("m1") is None
        assert db.delete("m1") is False

    def test_update_tier(self, db):
        db.insert(_make_memory(mid="m1", tier=Tier.L2), hnsw_label=0)
        assert db.update_tier("m1", Tier.L3) is True
        assert db.get("m1").tier == Tier.L3


# ── Cold-finder (the pruner's helper) ────────────────────────────────────────


class TestFindCold:
    def test_finds_old_memories(self, db):
        # Insert 3 memories; sleep + insert 2 more
        import time

        for i in range(3):
            db.insert(_make_memory(mid=f"old-{i}", tier=Tier.L2), hnsw_label=i)
        time.sleep(0.05)
        cutoff = datetime.now(UTC)
        time.sleep(0.05)
        for i in range(2):
            db.insert(_make_memory(mid=f"new-{i}", tier=Tier.L2), hnsw_label=100 + i)

        cold = db.find_cold(Tier.L2, older_than=cutoff)
        assert len(cold) == 3
        assert all(mid.startswith("old-") for mid in cold)

    def test_returns_empty_when_none_cold(self, db):
        db.insert(_make_memory(mid="m1", tier=Tier.L2), hnsw_label=0)
        # Cutoff far in the past — nothing is cold
        cutoff = datetime.now(UTC) - timedelta(days=365)
        assert db.find_cold(Tier.L2, older_than=cutoff) == []

    def test_returns_oldest_first(self, db):
        import time

        ids = ["c", "a", "b"]
        for mid in ids:
            db.insert(_make_memory(mid=mid, tier=Tier.L2), hnsw_label=hash(mid) % 10**6)
            time.sleep(0.01)
        cutoff = datetime.now(UTC) + timedelta(seconds=1)
        cold = db.find_cold(Tier.L2, older_than=cutoff)
        # Should be in insertion order since we slept between
        assert cold == ["c", "a", "b"]


# ── Streaming iteration ──────────────────────────────────────────────────────


class TestIterMemories:
    def test_iter_returns_all(self, db):
        for i in range(5):
            db.insert(_make_memory(mid=f"m{i}", tier=Tier.L2), hnsw_label=i)
        got = list(db.iter_memories())
        assert len(got) == 5

    def test_iter_filtered_by_tier(self, db):
        for i in range(3):
            db.insert(_make_memory(mid=f"l2-{i}", tier=Tier.L2), hnsw_label=i)
        for i in range(2):
            db.insert(_make_memory(mid=f"l3-{i}", tier=Tier.L3), hnsw_label=100 + i)
        l2 = list(db.iter_memories(Tier.L2))
        l3 = list(db.iter_memories(Tier.L3))
        assert len(l2) == 3 and len(l3) == 2
        assert all(m.tier == Tier.L2 for m in l2)
        assert all(m.tier == Tier.L3 for m in l3)


# ── Persistence ──────────────────────────────────────────────────────────────


class TestPersistence:
    def test_survives_reopen(self, tmp_path):
        path = tmp_path / "persist.db"
        md1 = L2Metadata(path)
        md1.insert(_make_memory("durable", mid="m1"), hnsw_label=0)
        md1.close()

        md2 = L2Metadata(path)
        loaded = md2.get("m1")
        assert loaded is not None
        assert loaded.text == "durable"
        md2.close()
