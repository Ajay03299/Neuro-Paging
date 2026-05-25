"""L1 Working Context — the working set tier.

The deck's L1 spec:
    size:      ~32 KB
    contains:  current turn + recent context tags + freshest memories
    eviction:  FIFO at end-of-turn
    latency:   < 1 ms

This file makes that real. Replaces the in-memory dict stub in the
MemoryManager. Wired in via memory/manager.py in the next commit.

Design choices
--------------
- collections.OrderedDict for O(1) insert + O(1) FIFO eviction.
  Inserting a new item appends to the right; popitem(last=False) pops
  the leftmost (oldest) item. Both O(1).
- Strict byte budget enforced at insert time. New inserts that would
  exceed the budget trigger eviction. Eviction continues until the new
  item fits OR the buffer is empty. If a single item is larger than the
  budget, we reject the insert with a ValueError — the caller (manager)
  should route oversized memories straight to L2/L3.
- Every eviction returns the evicted Memory. The caller decides what
  to do with it (the manager demotes it to L2). L1 never silently drops.
- threading.Lock around mutations. Mobile agent = one main thread plus
  background daemons; readers (query) and writers (insert/evict) can race.
"""

from __future__ import annotations

import threading
from collections import OrderedDict
from collections.abc import Iterator
from dataclasses import dataclass
from datetime import UTC, datetime

from loguru import logger

from neuro_paging.memory.types import Memory, MemoryId

# Per-memory overhead: dataclass overhead + dict pointer + small bookkeeping.
# Conservative — we'd rather over-account and evict slightly earlier than
# blow past the budget. Measured empirically with sys.getsizeof on a few
# Memory instances; rounded up.
_PER_MEMORY_OVERHEAD_BYTES = 256


@dataclass(frozen=True, slots=True)
class L1Stats:
    """Snapshot of L1 state. Reported up to MemoryManager.get_stats()."""

    count: int
    bytes_used: int
    capacity_bytes: int
    inserts_total: int
    evictions_total: int
    rejects_total: int  # insert attempts rejected because item > capacity

    @property
    def utilization(self) -> float:
        return self.bytes_used / self.capacity_bytes if self.capacity_bytes else 0.0


class L1WorkingContext:
    """FIFO buffer with a strict byte budget. The working set tier (~32 KB).

    Thread-safe. O(1) insert, O(1) eviction.

    Typical usage from inside MemoryManager:

        l1 = L1WorkingContext(capacity_bytes=32 * 1024)
        evicted = l1.insert(memory)
        for ev in evicted:
            # demote each to L2
            l2.insert(ev)
    """

    def __init__(self, capacity_bytes: int = 32 * 1024) -> None:
        if capacity_bytes <= 0:
            raise ValueError(f"capacity_bytes must be > 0, got {capacity_bytes}")

        self._capacity_bytes = capacity_bytes
        self._buf: OrderedDict[MemoryId, Memory] = OrderedDict()
        self._bytes_used = 0
        self._lock = threading.Lock()

        # Lifetime counters (for stats / dashboard)
        self._inserts_total = 0
        self._evictions_total = 0
        self._rejects_total = 0

        logger.debug(f"L1WorkingContext initialised capacity={capacity_bytes}B")

    # ── Sizing ────────────────────────────────────────────────────────────────

    @staticmethod
    def memory_size_bytes(memory: Memory) -> int:
        """Estimate byte cost of a Memory. Used for budget accounting.

        Components:
          - utf-8 length of text
          - embedding reference string (short — the actual vector lives in L2/L3)
          - per-memory fixed overhead (dataclass + dict pointer + bookkeeping)

        We deliberately do NOT count the context dataclass — it's shared and
        small. If that ever changes, this is the place to fix.
        """
        return (
            len(memory.text.encode("utf-8"))
            + len(memory.embedding_ref.encode("utf-8"))
            + _PER_MEMORY_OVERHEAD_BYTES
        )

    # ── Read path (lock-free for the common case) ─────────────────────────────

    def get(self, memory_id: MemoryId) -> Memory | None:
        """Return the memory if present, else None. Does not affect order."""
        with self._lock:
            return self._buf.get(memory_id)

    def contains(self, memory_id: MemoryId) -> bool:
        with self._lock:
            return memory_id in self._buf

    def __len__(self) -> int:
        with self._lock:
            return len(self._buf)

    def __iter__(self) -> Iterator[Memory]:
        """Iterate memories oldest → newest. Snapshot, safe for concurrent writes."""
        with self._lock:
            snapshot = list(self._buf.values())
        return iter(snapshot)

    # ── Write path ────────────────────────────────────────────────────────────

    def insert(self, memory: Memory) -> list[Memory]:
        """Insert a memory into L1. Returns any memories evicted to make room.

        Eviction policy: FIFO. We pop oldest entries until the new memory fits.

        Raises:
            ValueError: if memory by itself is larger than capacity (caller
                should route oversized memories directly to L2/L3 instead).
        """
        cost = self.memory_size_bytes(memory)

        if cost > self._capacity_bytes:
            with self._lock:
                self._rejects_total += 1
            logger.warning(
                f"L1 reject: memory id={memory.id} cost={cost}B > "
                f"capacity={self._capacity_bytes}B — route directly to L2/L3"
            )
            raise ValueError(f"Memory size {cost}B exceeds L1 capacity {self._capacity_bytes}B")

        evicted: list[Memory] = []

        with self._lock:
            # If the memory_id already exists, treat as update: remove old,
            # then proceed with insert. Keeps FIFO order honest (re-inserts
            # move to the back, which is what we want for "freshest").
            if memory.id in self._buf:
                old = self._buf.pop(memory.id)
                self._bytes_used -= self.memory_size_bytes(old)

            # Evict until we have room. OrderedDict.popitem(last=False)
            # removes the oldest (leftmost) item in O(1).
            while self._bytes_used + cost > self._capacity_bytes and self._buf:
                _, victim = self._buf.popitem(last=False)
                self._bytes_used -= self.memory_size_bytes(victim)
                self._evictions_total += 1
                evicted.append(victim)

            # Insert the new memory at the back (newest position).
            self._buf[memory.id] = memory
            self._bytes_used += cost
            self._inserts_total += 1

        if evicted:
            logger.trace(
                f"L1 evicted {len(evicted)} memories to fit id={memory.id} "
                f"(now {self._bytes_used}/{self._capacity_bytes}B)"
            )

        return evicted

    def remove(self, memory_id: MemoryId) -> Memory | None:
        """Hard remove by id. Returns the removed memory or None.

        Used by manager.forget() and manager.demote() (when explicitly
        moving a memory out of L1).
        """
        with self._lock:
            mem = self._buf.pop(memory_id, None)
            if mem is not None:
                self._bytes_used -= self.memory_size_bytes(mem)
            return mem

    def touch(self, memory_id: MemoryId) -> bool:
        """Mark a memory as freshly accessed.

        Two effects:
          1. Move to the back of FIFO so it survives the next eviction round
          2. Bump access_count + refresh last_touch on the Memory record

        Called by the manager when a memory in L1 is returned by query().

        Returns True if the memory was in L1, False otherwise.
        """
        with self._lock:
            if memory_id not in self._buf:
                return False
            old = self._buf[memory_id]

            updated = Memory(
                id=old.id,
                text=old.text,
                embedding_ref=old.embedding_ref,
                context=old.context,
                tier=old.tier,
                created_at=old.created_at,
                last_touch=datetime.now(UTC),
                access_count=old.access_count + 1,
                is_consolidated=old.is_consolidated,
            )

            del self._buf[memory_id]
            self._buf[memory_id] = updated
            return True

    def clear(self) -> list[Memory]:
        """Drop everything. Returns the cleared memories (caller may demote).

        Used at end-of-turn if the deck's strict FIFO-at-end-of-turn policy
        is enforced. Currently we use natural eviction; this is here for
        tests and potential future use.
        """
        with self._lock:
            cleared = list(self._buf.values())
            self._buf.clear()
            self._bytes_used = 0
        return cleared

    # ── Observability ─────────────────────────────────────────────────────────

    def stats(self) -> L1Stats:
        with self._lock:
            return L1Stats(
                count=len(self._buf),
                bytes_used=self._bytes_used,
                capacity_bytes=self._capacity_bytes,
                inserts_total=self._inserts_total,
                evictions_total=self._evictions_total,
                rejects_total=self._rejects_total,
            )

    @property
    def capacity_bytes(self) -> int:
        return self._capacity_bytes
