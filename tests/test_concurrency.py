"""Concurrency stress tests — multi-threaded torture of the substrate.

What we're testing
------------------
Every L1/L2/L3 method is supposed to be thread-safe. These tests
verify that by running the same operation from many threads at once
and asserting nothing breaks: no lost data, no corrupted counters,
no deadlocks, no stale reads.

These are STRESS tests — they intentionally try to break the system.
If they pass, your locking is genuinely correct, not just correct
in the happy-path tests. If they fail, you have a real concurrency
bug that would eventually surface in production.

Why these tests are rare (and valuable)
---------------------------------------
Most projects skip concurrency tests because they're harder to write
and the bugs they catch only surface under load. The result is
production code that works fine on a dev machine and crashes mysteriously
under real traffic. Catching these bugs in CI is the difference between
a hackathon project and a production-ready library.

Determinism note
----------------
Threading bugs are non-deterministic by nature — a test might pass
99 times and fail the 100th. We mitigate by:
  - Using high thread counts (8-16) and many ops per thread
  - Using barriers so threads all start at the same instant
  - Repeating critical assertions across multiple iterations where possible
  - Asserting INVARIANTS (sums, counts) not exact values

Every test runs under the global pytest timeout (see pyproject.toml:
timeout=120, timeout_method=thread) so a regression that reintroduces
a deadlock fails fast instead of hanging CI forever.
"""

from __future__ import annotations

import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import UTC, datetime

import numpy as np

from neuro_paging import ContextTags, TimeBucket
from neuro_paging.daemons.pruner import Pruner
from neuro_paging.daemons.types import PowerSnapshot
from neuro_paging.memory.l1_working import L1WorkingContext
from neuro_paging.memory.l2_cache import L2HotVectorCache
from neuro_paging.memory.l3_archive import L3ArchiveCache
from neuro_paging.memory.manager import MemoryManager
from neuro_paging.memory.types import Memory, MemoryId, Tier

# ── Helpers ──────────────────────────────────────────────────────────────────


def _make_memory(text: str, mid: str, tier: Tier = Tier.L2) -> Memory:
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


def _rand_vec(seed: int, dim: int = 384) -> np.ndarray:
    rng = np.random.default_rng(seed)
    v = rng.standard_normal(dim).astype(np.float32)
    v /= np.linalg.norm(v) + 1e-8
    return v


class _ScriptedPower:
    """Always says 'OK to run' — for pruner concurrency tests."""

    def snapshot(self) -> PowerSnapshot:
        return PowerSnapshot(
            battery_pct=100,
            is_charging=True,
            is_idle=True,
            is_foreground_app_active=False,
        )


# ── L1 concurrency ───────────────────────────────────────────────────────────


class TestL1Concurrency:
    """L1 uses a threading.RLock around its OrderedDict. Verify it holds."""

    def test_concurrent_inserts_no_lost_data(self):
        """16 threads × 100 inserts each. No insert should be lost."""
        l1 = L1WorkingContext(capacity_bytes=64 * 1024 * 1024)  # huge — no eviction
        n_threads = 16
        ops_per_thread = 100
        barrier = threading.Barrier(n_threads)

        def worker(thread_id: int) -> int:
            barrier.wait()  # all threads start simultaneously
            count = 0
            for i in range(ops_per_thread):
                mid = f"t{thread_id}-i{i}"
                mem = _make_memory(f"text {mid}", mid=mid, tier=Tier.L1)
                l1.insert(mem)
                count += 1
            return count

        with ThreadPoolExecutor(max_workers=n_threads) as pool:
            # Submit ALL workers first (materialise the list) so every thread
            # reaches the barrier. Passing a lazy generator to as_completed
            # could start blocking before all threads are submitted, and the
            # barrier would never fill — a deadlock.
            futures = [pool.submit(worker, t) for t in range(n_threads)]
            results = [f.result() for f in as_completed(futures)]

        # Every thread should report all ops done
        assert sum(results) == n_threads * ops_per_thread
        # And L1 should now contain all of them
        stats = l1.stats()
        assert stats.count == n_threads * ops_per_thread

    def test_concurrent_touches_dont_corrupt_counter(self):
        """100 threads touching the same memory. access_count should equal touch count."""
        l1 = L1WorkingContext(capacity_bytes=64 * 1024)
        mem = _make_memory("shared target", mid="shared", tier=Tier.L1)
        l1.insert(mem)

        n_threads = 100
        touches_per_thread = 10
        barrier = threading.Barrier(n_threads)

        def worker() -> None:
            barrier.wait()
            for _ in range(touches_per_thread):
                l1.touch("shared")

        with ThreadPoolExecutor(max_workers=n_threads) as pool:
            futures = [pool.submit(worker) for _ in range(n_threads)]
            list(as_completed(futures))

        # The access_count is incremented by touch(). If concurrent touches
        # raced, we'd see a count < n_threads * touches_per_thread.
        loaded = l1.get("shared")
        assert loaded.access_count == n_threads * touches_per_thread, (
            f"Lost touch updates: expected {n_threads * touches_per_thread}, "
            f"got {loaded.access_count}"
        )

    def test_concurrent_insert_with_eviction_byte_budget_holds(self):
        """Hammer L1 at a tight budget — byte budget must never be exceeded."""
        l1 = L1WorkingContext(capacity_bytes=8 * 1024)  # tight — sustained eviction
        n_threads = 8
        ops_per_thread = 200
        barrier = threading.Barrier(n_threads)

        def worker(thread_id: int) -> None:
            barrier.wait()
            for i in range(ops_per_thread):
                mid = f"t{thread_id}-i{i}"
                mem = _make_memory(f"memory text {mid} with filler content", mid=mid, tier=Tier.L1)
                l1.insert(mem)

        with ThreadPoolExecutor(max_workers=n_threads) as pool:
            futures = [pool.submit(worker, t) for t in range(n_threads)]
            list(as_completed(futures))

        # The invariant: bytes_used <= capacity_bytes, ALWAYS.
        stats = l1.stats()
        assert stats.bytes_used <= stats.capacity_bytes, (
            f"L1 byte budget violated under concurrent load: "
            f"{stats.bytes_used} > {stats.capacity_bytes}"
        )


# ── L2 concurrency ───────────────────────────────────────────────────────────


class TestL2Concurrency:
    """L2 has the hardest concurrency story — dual-store (HNSW + SQLite) atomicity."""

    def test_concurrent_inserts_metadata_and_index_stay_in_sync(self, tmp_path):
        """8 threads × 50 inserts. Every memory in metadata must be in HNSW and vice versa."""
        l2 = L2HotVectorCache(
            data_dir=tmp_path / "l2",
            capacity_bytes=64 * 1024 * 1024,
            max_elements=10_000,
            dim=384,
        )
        n_threads = 8
        ops_per_thread = 50
        barrier = threading.Barrier(n_threads)

        def worker(thread_id: int) -> int:
            barrier.wait()
            for i in range(ops_per_thread):
                mid = f"t{thread_id}-i{i}"
                mem = _make_memory(f"memory {mid}", mid=mid, tier=Tier.L2)
                vec = _rand_vec(seed=thread_id * 1000 + i)
                l2.insert(mem, vec)
            return ops_per_thread

        with ThreadPoolExecutor(max_workers=n_threads) as pool:
            # Materialise submissions before as_completed (see note above —
            # lazy generator + barrier = potential deadlock).
            futures = [pool.submit(worker, t) for t in range(n_threads)]
            results = [f.result() for f in as_completed(futures)]

        assert sum(results) == n_threads * ops_per_thread

        # Invariant 1: metadata.count == len(l2)
        assert len(l2) == n_threads * ops_per_thread

        # Invariant 2: every memory_id must be findable by both .get() and the
        # by-label lookup. A torn write would surface here as one half present
        # and the other half absent.
        all_metadata = list(l2._metadata.iter_memories(Tier.L2))
        assert len(all_metadata) == n_threads * ops_per_thread
        for mem in all_metadata:
            label = l2._metadata.label_for(mem.id)
            assert label is not None, f"Memory {mem.id} missing label mapping"
            roundtrip = l2._metadata.get_by_label(label)
            assert roundtrip is not None and roundtrip.id == mem.id, (
                f"Label {label} maps to wrong memory"
            )

        l2.close()

    def test_concurrent_inserts_no_duplicate_labels(self, tmp_path):
        """Labels are assigned monotonically. Under concurrency, no two memories share a label."""
        l2 = L2HotVectorCache(
            data_dir=tmp_path / "l2",
            capacity_bytes=64 * 1024 * 1024,
            max_elements=10_000,
            dim=384,
        )
        n_threads = 8
        ops_per_thread = 100
        barrier = threading.Barrier(n_threads)

        def worker(thread_id: int) -> None:
            barrier.wait()
            for i in range(ops_per_thread):
                mid = f"t{thread_id}-i{i}"
                mem = _make_memory(f"text {mid}", mid=mid, tier=Tier.L2)
                vec = _rand_vec(seed=thread_id * 1000 + i)
                l2.insert(mem, vec)

        with ThreadPoolExecutor(max_workers=n_threads) as pool:
            futures = [pool.submit(worker, t) for t in range(n_threads)]
            list(as_completed(futures))

        # Collect all labels — they must all be unique
        with l2._metadata._lock:
            cur = l2._metadata._conn.execute("SELECT hnsw_label FROM memories")
            labels = [r[0] for r in cur.fetchall()]
        assert len(labels) == len(set(labels)), (
            f"Duplicate labels found under concurrent insert! "
            f"{len(labels)} total, {len(set(labels))} unique"
        )
        l2.close()

    def test_concurrent_mixed_workload_no_corruption(self, tmp_path):
        """Mix inserts, queries, touches, removes across threads. No crash, no corruption."""
        l2 = L2HotVectorCache(
            data_dir=tmp_path / "l2",
            capacity_bytes=64 * 1024 * 1024,
            max_elements=10_000,
            dim=384,
        )

        # Pre-populate so queries + touches have targets
        for i in range(50):
            l2.insert(
                _make_memory(f"seed {i}", mid=f"seed-{i}", tier=Tier.L2),
                _rand_vec(seed=i),
            )

        n_threads = 12
        ops_per_thread = 50
        barrier = threading.Barrier(n_threads)
        errors: list[str] = []
        errors_lock = threading.Lock()

        def inserter(thread_id: int) -> None:
            barrier.wait()
            try:
                for i in range(ops_per_thread):
                    mid = f"i{thread_id}-{i}"
                    l2.insert(
                        _make_memory(f"inserted {mid}", mid=mid, tier=Tier.L2),
                        _rand_vec(seed=thread_id * 10000 + i),
                    )
            except Exception as e:  # noqa: BLE001
                with errors_lock:
                    errors.append(f"inserter t{thread_id}: {e}")

        def querier(thread_id: int) -> None:
            barrier.wait()
            try:
                for i in range(ops_per_thread):
                    l2.query(_rand_vec(seed=thread_id * 7 + i), k=5)
            except Exception as e:  # noqa: BLE001
                with errors_lock:
                    errors.append(f"querier t{thread_id}: {e}")

        def toucher(thread_id: int) -> None:
            barrier.wait()
            try:
                for i in range(ops_per_thread):
                    l2.touch(f"seed-{i % 50}")
            except Exception as e:  # noqa: BLE001
                with errors_lock:
                    errors.append(f"toucher t{thread_id}: {e}")

        # 4 inserters, 4 queriers, 4 touchers running concurrently
        with ThreadPoolExecutor(max_workers=n_threads) as pool:
            futures = []
            for t in range(4):
                futures.append(pool.submit(inserter, t))
            for t in range(4):
                futures.append(pool.submit(querier, t))
            for t in range(4):
                futures.append(pool.submit(toucher, t))
            for f in as_completed(futures):
                f.result()

        assert not errors, f"Concurrent mixed workload errored: {errors[:3]}"

        # Invariant: total live count = 50 (seeds) + 4 * 50 (inserts) = 250
        # (Assuming no eviction at this capacity)
        assert len(l2) == 250, f"Lost inserts under concurrent load: {len(l2)} != 250"

        l2.close()


# ── L3 concurrency ───────────────────────────────────────────────────────────


class TestL3Concurrency:
    """L3 shares L2's lock pattern. Verify it holds at archive scale."""

    def test_concurrent_inserts_no_corruption(self, tmp_path):
        l3 = L3ArchiveCache(
            data_dir=tmp_path / "l3",
            capacity_bytes=256 * 1024 * 1024,
            max_elements=50_000,
            dim=384,
        )
        n_threads = 8
        ops_per_thread = 100
        barrier = threading.Barrier(n_threads)

        def worker(thread_id: int) -> None:
            barrier.wait()
            for i in range(ops_per_thread):
                mid = f"t{thread_id}-i{i}"
                l3.insert(
                    _make_memory(f"archived {mid}", mid=mid, tier=Tier.L3),
                    _rand_vec(seed=thread_id * 1000 + i),
                )

        with ThreadPoolExecutor(max_workers=n_threads) as pool:
            futures = [pool.submit(worker, t) for t in range(n_threads)]
            list(as_completed(futures))

        assert len(l3) == n_threads * ops_per_thread
        l3.close()


# ── MemoryManager concurrency ────────────────────────────────────────────────


class TestManagerConcurrency:
    """The full manager: insert/query/promote/demote across threads."""

    def test_concurrent_inserts_across_tiers_no_corruption(self, tmp_path):
        """Insert directly into L1, L2, L3 simultaneously from different threads."""
        mgr = MemoryManager(
            data_dir=tmp_path / "np",
            l1_capacity_bytes=64 * 1024,  # generous enough not to spill
        )

        n_per_tier = 30
        barrier = threading.Barrier(3)

        def insert_l1() -> int:
            barrier.wait()
            for i in range(n_per_tier):
                mgr.insert(f"l1 {i}", ContextTags.now(), tier=Tier.L1)
            return n_per_tier

        def insert_l2() -> int:
            barrier.wait()
            for i in range(n_per_tier):
                mgr.insert(f"l2 {i}", ContextTags.now(), tier=Tier.L2)
            return n_per_tier

        def insert_l3() -> int:
            barrier.wait()
            for i in range(n_per_tier):
                mgr.insert(f"l3 {i}", ContextTags.now(), tier=Tier.L3)
            return n_per_tier

        # CRITICAL: submit ALL futures FIRST so all three threads start and
        # reach the barrier. Calling .result() inline would block the main
        # thread on the first future before the other two are submitted, and
        # the 3-party barrier would never be satisfied — a self-inflicted
        # deadlock.
        with ThreadPoolExecutor(max_workers=3) as pool:
            futures = [
                pool.submit(insert_l1),
                pool.submit(insert_l2),
                pool.submit(insert_l3),
            ]
            results = [f.result() for f in futures]

        assert sum(results) == 3 * n_per_tier

        stats = mgr.get_stats()
        # All 90 memories accounted for somewhere (allowing cascade)
        assert stats.total_count == 90, (
            f"Lost memories under concurrent insert: "
            f"L1={stats.l1_count} L2={stats.l2_count} L3={stats.l3_count}"
        )

        mgr.close()

    def test_pruner_runs_concurrently_with_inserts(self, tmp_path):
        """The money test: background pruner ticking while foreground inserts.

        Proves the daemon doesn't corrupt the substrate when it runs at the
        same time as normal app traffic. This is THE point of having locks.
        """
        from datetime import timedelta

        mgr = MemoryManager(data_dir=tmp_path / "np")

        # Pre-populate L2 with cold memories so the pruner has work to do.
        # Backdate their last_touch via direct metadata access.
        cold_ids = []
        for i in range(40):
            mid = mgr.insert(f"cold memory {i}", ContextTags.now(), tier=Tier.L2)
            cold_ids.append(mid)

        old_ts = (datetime.now(UTC) - timedelta(days=20)).isoformat()
        with mgr._l2._metadata._lock:
            mgr._l2._metadata._conn.executemany(
                "UPDATE memories SET last_touch = ? WHERE memory_id = ?",
                [(old_ts, mid) for mid in cold_ids],
            )

        pruner = Pruner(mgr, power=_ScriptedPower())

        n_insert_threads = 4
        ops_per_thread = 30
        barrier = threading.Barrier(n_insert_threads + 1)  # +1 for the pruner ticker
        errors: list[str] = []
        errors_lock = threading.Lock()

        def inserter(thread_id: int) -> None:
            barrier.wait()
            try:
                for i in range(ops_per_thread):
                    mgr.insert(f"fresh t{thread_id}-{i}", ContextTags.now(), tier=Tier.L2)
            except Exception as e:  # noqa: BLE001
                with errors_lock:
                    errors.append(f"inserter t{thread_id}: {e}")

        def pruner_ticker() -> int:
            barrier.wait()
            total = 0
            try:
                # Tick a few times while inserts are happening
                for _ in range(5):
                    total += pruner.tick_once()
            except Exception as e:  # noqa: BLE001
                with errors_lock:
                    errors.append(f"pruner: {e}")
            return total

        with ThreadPoolExecutor(max_workers=n_insert_threads + 1) as pool:
            futures = [pool.submit(inserter, t) for t in range(n_insert_threads)]
            pruner_future = pool.submit(pruner_ticker)
            for f in as_completed(futures + [pruner_future]):
                f.result()
            total_pruned = pruner_future.result()

        assert not errors, f"Pruner concurrency errored: {errors[:3]}"
        # Pruner moved at least some cold memories to L3
        assert total_pruned > 0, "Pruner found no work — sanity broken"

        stats = mgr.get_stats()
        # Total inserts (40 cold + 4 * 30 fresh = 160) should all still exist
        # somewhere (L2 + L3), nothing lost
        assert stats.total_count == 40 + n_insert_threads * ops_per_thread

        mgr.close()
