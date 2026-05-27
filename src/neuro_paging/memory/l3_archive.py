"""L3 Archive Cache — long-term memory tier.

The deck's L3 spec:
    size:       100 MB+ · ~100K compressed vectors
    structure:  Product Quantization (int8) over HNSW
    eviction:   only on hard cap; otherwise persistent
    latency:    ~20 ms p95 · top-K
    backend:    on-disk SQLite + HNSW (full PQ in Sprint 2)

This v0 ships the **same backbone as L2** — HNSW (cosine) + SQLite
metadata sidecar — but with:
  - Larger default byte budget (128 MB)
  - Larger starting slot count (100K vs L2's 10K)
  - tier=Tier.L3 baked into all metadata writes
  - Slightly relaxed HNSW hyperparams (M=24 for better recall at scale)
  - On-disk persistence by default; reopen-safe

PQ-int8 compression is deferred to Sprint 2. The substitute for now
is float32 HNSW at lower fidelity (no compression). When we swap in
PQ later, this class is the only thing that changes — MemoryManager
and the cascade chain stay identical.

Why duplicate L2's shape instead of subclassing?
  - Clear ownership: L2 = HOT, L3 = ARCHIVE. They evolve independently.
  - L3 will diverge in Sprint 2 (PQ index, different file layout).
  - Inheritance would couple two tiers that should be peers.
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

DEFAULT_DIM = 384

# L3 uses M=24 (vs L2's M=16). Slightly denser graph → better recall at
# the cost of slightly more memory per node. Worth it for archive.
DEFAULT_M = 24
DEFAULT_EF_CONSTRUCTION = 200
DEFAULT_EF_SEARCH = 64

# Per-memory overhead. Same calculation as L2 since we use the same
# float32 vector representation today. Sprint 2 PQ will reduce this
# to ~96 bytes per memory (int8 × 384 / 4 subspaces).
_PER_MEMORY_OVERHEAD_BYTES = 1536


@dataclass(frozen=True, slots=True)
class L3Stats:
    """Snapshot of L3 state. Mirrors L2Stats."""

    count: int
    bytes_estimate: int
    capacity_bytes: int
    max_elements: int

    inserts_total: int
    evictions_total: int
    tombstones_total: int

    @property
    def utilization(self) -> float:
        return self.bytes_estimate / self.capacity_bytes if self.capacity_bytes else 0.0


# ── The archive ───────────────────────────────────────────────────────────────


class L3ArchiveCache:
    """The L3 tier: long-term archive.

    Same atomic dual-store contract as L2 (metadata-first, HNSW second,
    rollback on HNSW failure). Same grow-on-physical-exhaustion eviction.
    Same lifecycle (open / save / close).

    Differences from L2:
      - capacity_bytes default 128 MB (vs L2's 8 MB)
      - max_elements default 100K (vs L2's 10K)
      - HNSW M=24 (denser graph for archive-scale recall)
      - Inserts default tier=Tier.L3
      - Lives at <data_dir>/l3/ on disk

    Sprint 2 will add:
      - PQ-int8 quantization (reduces bytes/memory by ~16×)
      - Periodic index rebuild to reclaim tombstoned slots
      - Optional background compaction daemon

    Until then, this is honest float32 HNSW at archive scale. The
    cascade chain L1→L2→L3 is end-to-end real *today*.
    """

    def __init__(
        self,
        data_dir: Path | str,
        capacity_bytes: int = 128 * 1024 * 1024,
        max_elements: int = 100_000,
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

        # Reuse L2Metadata — it already handles the Tier column correctly
        self._metadata = L2Metadata(self._data_dir / "metadata.sqlite")

        self._index = hnswlib.Index(space="cosine", dim=dim)
        index_path = self._data_dir / "hnsw.bin"
        if index_path.exists():
            self._index.load_index(str(index_path), max_elements=max_elements)
            logger.info(f"L3 loaded existing index from {index_path}")
        else:
            self._index.init_index(
                max_elements=max_elements,
                ef_construction=ef_construction,
                M=hnsw_m,
            )
            logger.debug(f"L3 created new index at {index_path}")
        self._index.set_ef(ef_search)
        self._index.set_num_threads(1)

        self._next_label = self._bootstrap_next_label()
        self._tombstones = 0
        self._inserts_total = 0
        self._evictions_total = 0
        self._lock = threading.RLock()

        logger.debug(
            f"L3ArchiveCache initialised "
            f"capacity={capacity_bytes}B max_elements={max_elements} dim={dim} "
            f"loaded_count={self._metadata.count(Tier.L3)}"
        )

    # ── Bootstrap ────────────────────────────────────────────────────────────

    def _bootstrap_next_label(self) -> int:
        """Resume label assignment from max+1 across BOTH L2 and L3 metadata.

        L3 has its own metadata DB, so we only care about L3's labels here.
        """
        with self._metadata._lock:
            cur = self._metadata._conn.execute(
                "SELECT MAX(hnsw_label) FROM memories WHERE tier = ?",
                (Tier.L3.value,),
            )
            row = cur.fetchone()
        max_label = row[0]
        return (max_label + 1) if max_label is not None else 0

    # ── Sizing ───────────────────────────────────────────────────────────────

    @staticmethod
    def memory_size_estimate(memory: Memory) -> int:
        return len(memory.text.encode("utf-8")) + _PER_MEMORY_OVERHEAD_BYTES

    def _current_bytes_estimate(self) -> int:
        live_count = self._metadata.count(Tier.L3)
        text_bytes = self._metadata.total_text_bytes(Tier.L3)
        return text_bytes + live_count * _PER_MEMORY_OVERHEAD_BYTES

    # ── Reads ────────────────────────────────────────────────────────────────

    def get(self, memory_id: MemoryId) -> Memory | None:
        mem = self._metadata.get(memory_id)
        # Only return if it's actually in L3 (the metadata sidecar may
        # hold L2 entries if shared with another cache)
        if mem is not None and mem.tier != Tier.L3:
            return None
        return mem

    def contains(self, memory_id: MemoryId) -> bool:
        mem = self._metadata.get(memory_id)
        return mem is not None and mem.tier == Tier.L3

    def __len__(self) -> int:
        return self._metadata.count(Tier.L3)

    def query(
        self,
        embedding: np.ndarray,
        k: int = 5,
    ) -> list[tuple[Memory, float]]:
        """Top-k ANN search over the L3 archive.

        Same semantics as L2.query() but bounded to the archive tier.
        Raises ValueError on dim mismatch (validated before state checks).
        """
        vec = np.asarray(embedding, dtype=np.float32).reshape(1, -1)
        if vec.shape[1] != self._dim:
            raise ValueError(f"Embedding dim {vec.shape[1]} != L3 dim {self._dim}")

        if k <= 0:
            return []
        live_count = self._metadata.count(Tier.L3)
        if live_count == 0:
            return []

        # Over-fetch and filter by tier to avoid returning L2 hits when
        # L2 and L3 share a metadata DB (not the default, but defensive).
        k_eff = min(k * 4, live_count)
        try:
            labels, distances = self._index.knn_query(vec, k=k_eff)
        except RuntimeError as e:
            logger.warning(f"L3 knn_query failed: {e}")
            return []

        out: list[tuple[Memory, float]] = []
        for lbl, dist in zip(labels[0], distances[0], strict=True):
            mem = self._metadata.get_by_label(int(lbl))
            if mem is None or mem.tier != Tier.L3:
                continue
            out.append((mem, float(dist)))
            if len(out) >= k:
                break
        return out

    # ── Writes ───────────────────────────────────────────────────────────────

    def insert(self, memory: Memory, embedding: np.ndarray) -> list[Memory]:
        """Insert into L3. Returns memories evicted (for forget-style outflow).

        L3 is the bottom of the stack — evicted memories from L3 are *gone*
        (in Sprint 2 they'll go to "forgotten" status with a privacy timer).
        For now: return them so the caller can log / surface, but the
        manager treats them as terminally lost.
        """
        vec = np.asarray(embedding, dtype=np.float32).reshape(-1)
        if vec.shape[0] != self._dim:
            raise ValueError(f"Embedding dim {vec.shape[0]} != L3 dim {self._dim}")

        # Force tier=Tier.L3 — defensive against callers passing a Memory
        # with stale tier metadata (e.g., a memory cascading from L2)
        if memory.tier != Tier.L3:
            memory = Memory(
                id=memory.id,
                text=memory.text,
                embedding_ref=memory.embedding_ref,
                context=memory.context,
                tier=Tier.L3,
                created_at=memory.created_at,
                last_touch=memory.last_touch,
                access_count=memory.access_count,
                is_consolidated=memory.is_consolidated,
            )

        evicted: list[Memory] = []

        with self._lock:
            # Update path: same memory_id already in L3 → tombstone + re-add
            existing_label = self._metadata.label_for(memory.id)
            if existing_label is not None:
                self._index.mark_deleted(existing_label)
                self._tombstones += 1
                new_label = self._next_label
                self._next_label += 1
                self._metadata.upsert(memory, new_label)
                try:
                    self._index.add_items(vec.reshape(1, -1), np.array([new_label]))
                except RuntimeError:
                    self._metadata.delete(memory.id)
                    raise
                self._inserts_total += 1
                return []

            # Pre-evict if at the ceiling (physical OR logical)
            evicted = self._ensure_capacity_for_insert()

            # Fresh insert
            label = self._next_label
            self._next_label += 1
            self._metadata.insert(memory, label)

            try:
                self._index.add_items(vec.reshape(1, -1), np.array([label]))
            except RuntimeError as e:
                self._metadata.delete(memory.id)
                logger.error(f"L3 HNSW add_items failed, rolled back: {e}")
                raise

            self._inserts_total += 1

            # Byte-budget eviction after insert
            evicted.extend(self._evict_to_fit())

        return evicted

    def _ensure_capacity_for_insert(self) -> list[Memory]:
        """Same two-pronged approach as L2: grow physical if needed, then evict."""
        evicted: list[Memory] = []

        physical_count = self._index.get_current_count()
        if physical_count >= self._max_elements:
            new_max = max(self._max_elements * 3 // 2, self._max_elements + 16)
            try:
                self._index.resize_index(new_max)
                self._max_elements = new_max
                logger.debug(
                    f"L3 grew HNSW index: {physical_count} used "
                    f"(incl. tombstones), new max_elements={new_max}"
                )
            except RuntimeError as e:
                logger.error(f"L3 resize_index failed: {e}")
                raise

        live_count = self._metadata.count(Tier.L3)
        if live_count >= self._max_elements:
            far_future = datetime.now(UTC).replace(year=9999)
            candidates = self._metadata.find_cold(Tier.L3, older_than=far_future)
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
        """Byte-budget eviction. Same as L2."""
        evicted: list[Memory] = []

        while True:
            current_bytes = self._current_bytes_estimate()
            current_count = self._metadata.count(Tier.L3)
            byte_ok = current_bytes <= self._capacity_bytes
            count_ok = current_count <= self._max_elements
            if byte_ok and count_ok:
                break

            far_future = datetime.now(UTC).replace(year=9999)
            candidates = self._metadata.find_cold(Tier.L3, older_than=far_future)
            if not candidates:
                break

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

            if len(evicted) >= self._max_elements:
                logger.warning("L3 eviction safety brake hit")
                break

        if evicted:
            logger.trace(f"L3 evicted {len(evicted)} memories (now {len(self)} live)")
        return evicted

    def touch(self, memory_id: MemoryId) -> bool:
        """L3 memories are accessed rarely; touch is a no-op-like signal."""
        return self._metadata.touch(memory_id)

    def remove(self, memory_id: MemoryId) -> Memory | None:
        with self._lock:
            label = self._metadata.label_for(memory_id)
            if label is None:
                return None
            mem = self._metadata.get(memory_id)
            if mem is None or mem.tier != Tier.L3:
                return None
            self._index.mark_deleted(label)
            self._tombstones += 1
            self._metadata.delete(memory_id)
            return mem

    # ── Persistence ──────────────────────────────────────────────────────────

    def save(self) -> None:
        with self._lock:
            self._index.save_index(str(self._data_dir / "hnsw.bin"))

    def close(self) -> None:
        with self._lock:
            self.save()
            self._metadata.close()

    # ── Observability ────────────────────────────────────────────────────────

    def stats(self) -> L3Stats:
        with self._lock:
            return L3Stats(
                count=self._metadata.count(Tier.L3),
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
