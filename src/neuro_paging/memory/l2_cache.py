"""L2 Hot Vector Cache — hnswlib + SQLite metadata sidecar.

The deck's L2 spec:
    size:       ~8 MB · ~10K float16 vectors
    structure:  HNSW + per-vector access counter + last-touch ts
    eviction:   cold for 14 days → demote to L3
    latency:    ~5 ms · top-K HNSW

This file makes that real. Two collaborating stores:

    HNSW index   : (hnsw_label: int) -> vector
    L2Metadata   : (memory_id: UUID) <-> (hnsw_label: int)
                    + text, context, access_count, last_touch, tier

The cache class enforces invariant: every memory in HNSW has a
metadata row, and vice versa. All mutations are "metadata-first,
then HNSW" so a metadata failure leaves HNSW untouched, and an
HNSW failure rolls metadata back.

Eviction model
--------------
hnswlib doesn't physically delete vectors — it has `mark_deleted()`
which tombstones the label. We:
  1. Track live count separately from the index's allocated size
  2. When live_count approaches max_elements, pre-evict BEFORE the
     next insert (hnswlib treats max_elements as a HARD ceiling
     and will raise RuntimeError if we try to exceed it)
  3. When the byte budget is exceeded, post-evict (soft ceiling,
     single over-by-one is fine)
  4. When deletion ratio exceeds REBUILD_THRESHOLD, rebuild the
     index to reclaim space (Sprint 2 — current impl skips this)

Capacity is measured in two ways:
  - max_elements: hard ceiling on HNSW's internal index size
  - byte budget: soft ceiling tracked via metadata.total_text_bytes()
    The byte budget mirrors the deck's "8 MB" claim; max_elements
    is a backstop and an absolute upper bound on live vectors.
"""

from __future__ import annotations

import threading
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

import hnswlib
import numpy as np
from loguru import logger

from neuro_paging.memory.l2_metadata import L2Metadata
from neuro_paging.memory.types import Memory, MemoryId, Tier

# ── Configuration ─────────────────────────────────────────────────────────────

# Vector dim — matches bge-small-en-v1.5. If we ever swap embedders,
# this changes (and the L2 needs to be rebuilt).
DEFAULT_DIM = 384

# HNSW hyperparams. Same values as bench/hnswlib_baseline.py for consistency.
DEFAULT_M = 16  # max neighbours per node
DEFAULT_EF_CONSTRUCTION = 200
DEFAULT_EF_SEARCH = 50

# Per-memory rough byte estimate for the byte-budget calculation.
# Counts: utf-8 text bytes + ~768 bytes for a float16[384] vector
# + ~512 bytes for metadata row + HNSW graph overhead per node.
# Tuned to be conservative (rounds up, so eviction triggers slightly early).
_PER_MEMORY_OVERHEAD_BYTES = 1536


@dataclass(frozen=True, slots=True)
class L2Stats:
    """Snapshot of L2 state. Reported up to MemoryManager.get_stats()."""

    count: int  # live memories (excludes tombstoned)
    bytes_estimate: int  # rough total — text + vectors + overhead
    capacity_bytes: int
    max_elements: int  # HNSW's hard cap

    # Lifetime counters
    inserts_total: int
    evictions_total: int
    tombstones_total: int  # HNSW labels marked deleted but space not reclaimed

    @property
    def utilization(self) -> float:
        return self.bytes_estimate / self.capacity_bytes if self.capacity_bytes else 0.0


# ── The cache ─────────────────────────────────────────────────────────────────


class L2HotVectorCache:
    """The L2 tier: HNSW vector index + SQLite metadata sidecar.

    Thread-safe. Mutations are atomic across the two stores.

    Typical usage from MemoryManager:

        l2 = L2HotVectorCache(
            data_dir=Path("/var/data/neuro-paging/l2"),
            capacity_bytes=8 * 1024 * 1024,
            max_elements=10_000,
            dim=384,
        )

        evicted = l2.insert(memory, embedding=np.ndarray(384,))
        for ev in evicted:
            l3.insert(ev)   # demote
    """

    def __init__(
        self,
        data_dir: Path | str,
        capacity_bytes: int = 8 * 1024 * 1024,
        max_elements: int = 10_000,
        dim: int = DEFAULT_DIM,
        hnsw_m: int = DEFAULT_M,
        ef_construction: int = DEFAULT_EF_CONSTRUCTION,
        ef_search: int = DEFAULT_EF_SEARCH,
    ) -> None:
        if capacity_bytes <= 0:
            raise ValueError(f"capacity_bytes must be > 0, got {capacity_bytes}")
        if max_elements <= 0:
            raise ValueError(f"max_elements must be > 0, got {max_elements}")

        self._data_dir = Path(data_dir)
        self._data_dir.mkdir(parents=True, exist_ok=True)

        self._capacity_bytes = capacity_bytes
        self._max_elements = max_elements
        self._dim = dim
        self._ef_search = ef_search

        # Sub-stores
        self._metadata = L2Metadata(self._data_dir / "metadata.sqlite")
        self._index = hnswlib.Index(space="cosine", dim=dim)
        index_path = self._data_dir / "hnsw.bin"
        if index_path.exists():
            self._index.load_index(str(index_path), max_elements=max_elements)
            logger.info(f"L2 loaded existing index from {index_path}")
        else:
            self._index.init_index(
                max_elements=max_elements,
                ef_construction=ef_construction,
                M=hnsw_m,
            )
            logger.debug(f"L2 created new index at {index_path}")
        self._index.set_ef(ef_search)
        self._index.set_num_threads(1)  # single-threaded → realistic mobile profile

        # Next HNSW label to assign. Bootstrap from metadata so labels stay
        # unique across process restarts.
        self._next_label = self._bootstrap_next_label()

        # Coarse counter: how many labels we've tombstoned via mark_deleted.
        # Sprint 2: trigger rebuild when this gets large relative to count.
        self._tombstones = 0

        # Lifetime counters (snapshot to L2Stats)
        self._inserts_total = 0
        self._evictions_total = 0

        # Single mutation lock. Reads under HNSW's own thread-safety; writes
        # serialise here because we need metadata + HNSW updates to be
        # atomic together.
        self._lock = threading.RLock()

        logger.debug(
            f"L2HotVectorCache initialised "
            f"capacity={capacity_bytes}B max_elements={max_elements} dim={dim} "
            f"loaded_count={self._metadata.count(Tier.L2)}"
        )

    # ── Bootstrap ────────────────────────────────────────────────────────────

    def _bootstrap_next_label(self) -> int:
        """On open, resume label assignment from one past the max we've used."""
        with self._metadata._lock:
            cur = self._metadata._conn.execute("SELECT MAX(hnsw_label) FROM memories")
            row = cur.fetchone()
        max_label = row[0]
        return (max_label + 1) if max_label is not None else 0

    # ── Sizing ───────────────────────────────────────────────────────────────

    @staticmethod
    def memory_size_estimate(memory: Memory) -> int:
        """Rough byte cost of holding this memory in L2 (text + vector + overhead)."""
        return len(memory.text.encode("utf-8")) + _PER_MEMORY_OVERHEAD_BYTES

    def _current_bytes_estimate(self) -> int:
        """Sum of text bytes + per-memory overhead × live count.

        Coarse but consistent. We never compare it to anything that needs
        exact byte accuracy.
        """
        live_count = self._metadata.count(Tier.L2)
        text_bytes = self._metadata.total_text_bytes(Tier.L2)
        return text_bytes + live_count * _PER_MEMORY_OVERHEAD_BYTES

    # ── Reads ────────────────────────────────────────────────────────────────

    def get(self, memory_id: MemoryId) -> Memory | None:
        return self._metadata.get(memory_id)

    def contains(self, memory_id: MemoryId) -> bool:
        return self._metadata.contains(memory_id)

    def __len__(self) -> int:
        return self._metadata.count(Tier.L2)

    def query(
        self,
        embedding: np.ndarray,
        k: int = 5,
    ) -> list[tuple[Memory, float]]:
        """Top-k ANN search. Returns [(memory, distance)] sorted by distance ASC.

        Distance is the cosine distance (0 = identical, 2 = opposite).
        Caller (manager's scorer) converts to similarity.

        Returns memories in the L2 tier only. Tombstoned (deleted) labels
        are skipped automatically by hnswlib.

        Raises:
            ValueError: if embedding dim mismatches L2 dim. Validated even
                on an empty cache to fail fast on bad inputs.
        """
        # Validate dim FIRST so we fail fast on bad inputs regardless of state.
        vec = np.asarray(embedding, dtype=np.float32).reshape(1, -1)
        if vec.shape[1] != self._dim:
            raise ValueError(f"Embedding dim {vec.shape[1]} != L2 dim {self._dim}")

        if k <= 0:
            return []
        live_count = self._metadata.count(Tier.L2)
        if live_count == 0:
            return []

        # Ask for at most live_count to avoid hnswlib raising on small indexes
        k_eff = min(k, live_count)
        try:
            labels, distances = self._index.knn_query(vec, k=k_eff)
        except RuntimeError as e:
            logger.warning(f"L2 knn_query failed: {e}")
            return []

        out: list[tuple[Memory, float]] = []
        for lbl, dist in zip(labels[0], distances[0], strict=True):
            mem = self._metadata.get_by_label(int(lbl))
            if mem is None:
                # Stale label — metadata says it's gone but HNSW didn't tombstone yet
                continue
            out.append((mem, float(dist)))
        return out

    # ── Writes ───────────────────────────────────────────────────────────────

    def insert(self, memory: Memory, embedding: np.ndarray) -> list[Memory]:
        """Insert a memory + its embedding into L2.

        Returns any memories evicted to make room (so caller can demote to L3).

        Atomicity: metadata row is written first, then HNSW. If HNSW fails,
        we roll back the metadata row. If both succeed and we exceed
        capacity, the eviction step inside the same lock removes the
        coldest entries and returns them.
        """
        vec = np.asarray(embedding, dtype=np.float32).reshape(-1)
        if vec.shape[0] != self._dim:
            raise ValueError(f"Embedding dim {vec.shape[0]} != L2 dim {self._dim}")

        evicted: list[Memory] = []

        with self._lock:
            # If the memory already lives here, treat as update.
            existing_label = self._metadata.label_for(memory.id)
            if existing_label is not None:
                # Update HNSW vector + metadata row in place. No eviction.
                self._index.mark_deleted(existing_label)
                self._tombstones += 1
                new_label = self._next_label
                self._next_label += 1
                self._metadata.upsert(memory, new_label)
                try:
                    self._index.add_items(vec.reshape(1, -1), np.array([new_label]))
                except RuntimeError:
                    # Roll back metadata
                    self._metadata.delete(memory.id)
                    raise
                self._inserts_total += 1
                return []

            # Pre-evict if we're at the HARD ceiling (max_elements). hnswlib
            # will refuse to add_items if the index is full, so eviction
            # must happen BEFORE the insert, not after.
            #
            # Byte-budget eviction can stay after insert (soft ceiling, single
            # over-by-one is fine), but max_elements is non-negotiable.
            evicted = self._ensure_capacity_for_insert()

            # Fresh insert
            label = self._next_label
            self._next_label += 1

            # Metadata first (so a HNSW failure leaves the row absent)
            self._metadata.insert(memory, label)

            try:
                self._index.add_items(vec.reshape(1, -1), np.array([label]))
            except RuntimeError as e:
                # Roll back metadata
                self._metadata.delete(memory.id)
                logger.error(f"L2 HNSW add_items failed, rolled back metadata: {e}")
                raise

            self._inserts_total += 1

            # Now that the insert succeeded, run byte-budget eviction.
            # Append to any pre-evictions we already did.
            evicted.extend(self._evict_to_fit())

        return evicted

    def _ensure_capacity_for_insert(self) -> list[Memory]:
        """Make room for one more vector.

        TWO INDEPENDENT CONCERNS — handle both:

        1. PHYSICAL slot exhaustion: hnswlib's index has a fixed allocated
           size (max_elements). mark_deleted() does NOT reclaim slots — it
           just tombstones the label. So even after evicting, the next
           add_items() can fail if the physical slots are full. Fix: check
           index.get_current_count() (physical slots used INCLUDING
           tombstones) and resize if needed.

        2. LOGICAL eviction: if live_count is at the desired ceiling
           (max_elements), evict the coldest entry. This keeps the live
           population bounded.

        Order matters: do physical grow FIRST (so add_items will succeed),
        then logical eviction (so live_count stays bounded).

        Returns evicted memories so caller can demote them to L3.
        Caller holds the lock.
        """
        evicted: list[Memory] = []

        # ── Concern 1: physical slot exhaustion ──
        # get_current_count() counts ALL labels ever added (including
        # tombstoned ones). If it's at the cap, the next add_items will
        # raise even if logically we have room.
        physical_count = self._index.get_current_count()
        if physical_count >= self._max_elements:
            # Grow by 50% headroom, minimum +16 slots
            new_max = max(self._max_elements * 3 // 2, self._max_elements + 16)
            try:
                self._index.resize_index(new_max)
                self._max_elements = new_max
                logger.debug(
                    f"L2 grew HNSW index: {physical_count} used "
                    f"(incl. tombstones), new max_elements={new_max}"
                )
            except RuntimeError as e:
                logger.error(f"L2 resize_index failed: {e}")
                raise

        # ── Concern 2: logical eviction ──
        # If live count is at the ceiling, evict the coldest entry to keep
        # the live population bounded. (We may have just grown, but we
        # still want eviction semantics for the byte budget and LRU policy.)
        live_count = self._metadata.count(Tier.L2)
        if live_count >= self._max_elements:
            far_future = datetime.now(UTC).replace(year=9999)
            candidates = self._metadata.find_cold(Tier.L2, older_than=far_future)
            if candidates:
                victim_id = candidates[0]
                victim = self._metadata.get(victim_id)
                if victim is not None:
                    label = self._metadata.label_for(victim_id)
                    if label is not None:
                        self._index.mark_deleted(label)
                        self._tombstones += 1
                    self._metadata.delete(victim_id)
                    self._evictions_total += 1
                    evicted.append(victim)

        return evicted

    def _evict_to_fit(self) -> list[Memory]:
        """Evict coldest entries until L2 is back within both budgets.

        Caller holds the lock. Returns evicted memories (caller demotes to L3).

        Cold = oldest last_touch. We pull up to 128 candidates per pass and
        evict in batches to avoid degenerate eviction storms.
        """
        evicted: list[Memory] = []

        # Loop because deletes change the byte estimate; one pass may not be enough.
        while True:
            current_bytes = self._current_bytes_estimate()
            current_count = self._metadata.count(Tier.L2)

            byte_ok = current_bytes <= self._capacity_bytes
            count_ok = current_count <= self._max_elements
            if byte_ok and count_ok:
                break

            # Find coldest ~128 candidates
            far_future = datetime.now(UTC).replace(year=9999)
            candidates = self._metadata.find_cold(Tier.L2, older_than=far_future)
            if not candidates:
                break  # nothing left to evict

            # Evict the single oldest, recheck. We don't batch because the
            # caller has just inserted ONE thing; evicting more than needed
            # is wasteful.
            victim_id = candidates[0]
            victim = self._metadata.get(victim_id)
            if victim is None:
                break

            label = self._metadata.label_for(victim_id)
            if label is not None:
                self._index.mark_deleted(label)
                self._tombstones += 1
            self._metadata.delete(victim_id)
            self._evictions_total += 1
            evicted.append(victim)

            # Safety: don't evict everything we just inserted in a runaway loop
            if len(evicted) >= self._max_elements:
                logger.warning("L2 eviction safety brake hit — stopping")
                break

        if evicted:
            logger.trace(f"L2 evicted {len(evicted)} memories (now {len(self)} live)")
        return evicted

    def touch(self, memory_id: MemoryId) -> bool:
        """Bump access_count + last_touch on this memory's metadata row.

        Called by the manager on cache hits to keep recently-used memories
        from being evicted first.
        """
        return self._metadata.touch(memory_id)

    def remove(self, memory_id: MemoryId) -> Memory | None:
        """Hard remove from L2 (mark_deleted in HNSW + DELETE metadata row).

        Returns the removed memory or None.
        """
        with self._lock:
            label = self._metadata.label_for(memory_id)
            if label is None:
                return None
            mem = self._metadata.get(memory_id)
            self._index.mark_deleted(label)
            self._tombstones += 1
            self._metadata.delete(memory_id)
            return mem

    # ── Persistence ──────────────────────────────────────────────────────────

    def save(self) -> None:
        """Persist HNSW index to disk. SQLite is already on-disk."""
        with self._lock:
            self._index.save_index(str(self._data_dir / "hnsw.bin"))

    def close(self) -> None:
        with self._lock:
            self.save()
            self._metadata.close()

    # ── Observability ────────────────────────────────────────────────────────

    def stats(self) -> L2Stats:
        with self._lock:
            return L2Stats(
                count=self._metadata.count(Tier.L2),
                bytes_estimate=self._current_bytes_estimate(),
                capacity_bytes=self._capacity_bytes,
                max_elements=self._max_elements,
                inserts_total=self._inserts_total,
                evictions_total=self._evictions_total,
                tombstones_total=self._tombstones,
            )

    @property
    def capacity_bytes(self) -> int:
        return self._capacity_bytes

    @property
    def dim(self) -> int:
        return self._dim
