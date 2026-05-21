"""MemoryManager — the public API between Ajay's substrate and Christine's
intelligence layer.

THIS IS A CONTRACT. After Sprint 0 ends (Fri May 22), the method
signatures here do not change. Behavior fills in across Sprints 1-4.

Christine's layer should NEVER reach behind this interface. If she
finds herself wanting to, that's a signal to widen the interface
together — not to bypass it.

Quick map of methods:
    insert(...)     → store a new memory. Returns MemoryId.
    query(...)      → top-k retrieval. Returns list[Hit].
    get(...)        → fetch a single memory by id. Returns Memory or None.
    promote(...)    → hint: warm this memory up (mostly daemons).
    demote(...)     → hint: cool this memory off (mostly daemons).
    forget(...)     → hard delete. Use sparingly.
    get_stats()     → live tier observability.

The stub implementation in this file uses an in-memory dict so Christine
can develop against it. Real L1/L2/L3 implementations land Sprint 1+.
"""

from __future__ import annotations

import time
import uuid
from datetime import UTC, datetime
from typing import Protocol

from loguru import logger

from neuro_paging.context.types import ContextTags
from neuro_paging.memory.types import (
    Hit,
    Memory,
    MemoryId,
    Provenance,
    Tier,
    TierStats,
)

# ── Scoring callback ──────────────────────────────────────────────────────────
# The manager doesn't know how to score. Christine provides a scorer at
# construction time. This is dependency inversion: the substrate calls into
# the intelligence layer for ranking decisions.


class Scorer(Protocol):
    """The contract the router must implement. Christine owns this."""

    def score(
        self,
        memory: Memory,
        query_text: str,
        query_context: ContextTags,
    ) -> tuple[float, Provenance]:
        """Return (final_score, provenance) for this memory given the query."""
        ...


class _DefaultStubScorer:
    """Trivial scorer used until Christine wires in the real one.

    Returns a stable but meaningless score based on text length. Lets
    end-to-end tests run without depending on her layer being ready.
    """

    def score(
        self,
        memory: Memory,
        query_text: str,
        query_context: ContextTags,
    ) -> tuple[float, Provenance]:
        start = time.perf_counter()
        # Pure stub — overlap of words. Replaced by Christine's real scorer.
        memory_words = set(memory.text.lower().split())
        query_words = set(query_text.lower().split())
        overlap = len(memory_words & query_words) / max(len(query_words), 1)

        score = overlap
        elapsed_ms = (time.perf_counter() - start) * 1000.0

        prov = Provenance(
            tier=memory.tier,
            raw_relevance=overlap,
            raw_context_sim=0.0,
            raw_frequency=0.0,
            weights=(1.0, 0.0, 0.0),
            final_score=score,
            elapsed_ms=elapsed_ms,
        )
        return score, prov


# ── Default tier capacities (overridable via constructor) ─────────────────────
# These mirror the deck's tier specs. Tune in Sprint 2 once we measure.
DEFAULT_L1_CAPACITY_BYTES = 32 * 1024  # 32 KB
DEFAULT_L2_CAPACITY_BYTES = 8 * 1024 * 1024  # 8 MB
DEFAULT_L3_CAPACITY_BYTES = 128 * 1024 * 1024  # 128 MB


# ── The contract ──────────────────────────────────────────────────────────────


class MemoryManager:
    """The public API for the tiered memory subsystem.

    Construct once at app startup, share across the pipeline.
    Thread-safety: methods are safe for concurrent reads but writes
    serialise on an internal lock (implementation detail of the real
    L2/L3 backends — the stub here is naïvely single-threaded).
    """

    def __init__(
        self,
        scorer: Scorer | None = None,
        l1_capacity_bytes: int = DEFAULT_L1_CAPACITY_BYTES,
        l2_capacity_bytes: int = DEFAULT_L2_CAPACITY_BYTES,
        l3_capacity_bytes: int = DEFAULT_L3_CAPACITY_BYTES,
    ) -> None:
        self._scorer = scorer or _DefaultStubScorer()
        self._l1_cap = l1_capacity_bytes
        self._l2_cap = l2_capacity_bytes
        self._l3_cap = l3_capacity_bytes

        # ── STUB STORAGE — replaced with real L1/L2/L3 backends in Sprint 1 ──
        self._memories: dict[MemoryId, Memory] = {}

        # Rolling counters for stats
        self._queries_24h = 0
        self._promotions_24h = 0
        self._demotions_24h = 0
        self._hits_24h = 0

        logger.debug(
            "MemoryManager initialised (stub backend) "
            f"L1={l1_capacity_bytes}B L2={l2_capacity_bytes}B L3={l3_capacity_bytes}B"
        )

    # ── Write path ────────────────────────────────────────────────────────────

    def insert(
        self,
        text: str,
        context: ContextTags,
        *,
        tier: Tier = Tier.L1,
    ) -> MemoryId:
        """Store a memory. Returns the assigned id.

        New memories default to L1. The daemons handle promotion to L2,
        and consolidation into L3. Direct insert to L2/L3 is supported
        for loading historical data (e.g. during a sync from another device
        in a future version, or for test fixtures).
        """
        if not text.strip():
            raise ValueError("Cannot insert empty memory text")

        mem_id = MemoryId(str(uuid.uuid4()))
        now = datetime.now(UTC)
        memory = Memory(
            id=mem_id,
            text=text,
            embedding_ref=f"stub-emb:{mem_id}",  # real impl owns the embedding
            context=context,
            tier=tier,
            created_at=now,
            last_touch=now,
            access_count=0,
            is_consolidated=False,
        )
        self._memories[mem_id] = memory
        logger.trace(f"insert id={mem_id} tier={tier.value} len={len(text)}")
        return mem_id

    # ── Read path ─────────────────────────────────────────────────────────────

    def query(
        self,
        text: str,
        context: ContextTags,
        *,
        k: int = 5,
        min_tier: Tier = Tier.L3,  # search down to this tier; default = all
    ) -> list[Hit]:
        """Retrieve top-k memories ranked by the scorer.

        `min_tier=L1` means L1-only (fastest, cheapest, lowest recall).
        `min_tier=L3` (default) means search all tiers.

        The deck's battery-aware behavior: callers can pass min_tier=L1
        when battery is critical, sidestepping disk reads.
        """
        if k <= 0:
            return []
        self._queries_24h += 1

        # Candidate set: everything in tiers at least as hot as min_tier.
        # `min_tier=L3` (default) → search all tiers (L3, L2, L1).
        # `min_tier=L1` (battery-critical mode) → L1 only, skip disk reads.
        # Real impl: ANN search per tier, merge, then re-rank.
        candidates = [m for m in self._memories.values() if m.tier.rank >= min_tier.rank]

        # Score each candidate
        scored: list[tuple[float, Memory, Provenance]] = []
        for mem in candidates:
            score, prov = self._scorer.score(mem, text, context)
            scored.append((score, mem, prov))

        # Top-k
        scored.sort(key=lambda triple: triple[0], reverse=True)
        top = scored[:k]

        # Convert to hits, update access counters
        hits: list[Hit] = []
        for score, mem, prov in top:
            # Update last_touch + access_count (frozen dataclass → replace)
            updated = Memory(
                id=mem.id,
                text=mem.text,
                embedding_ref=mem.embedding_ref,
                context=mem.context,
                tier=mem.tier,
                created_at=mem.created_at,
                last_touch=datetime.now(UTC),
                access_count=mem.access_count + 1,
                is_consolidated=mem.is_consolidated,
            )
            self._memories[mem.id] = updated

            hits.append(
                Hit(
                    memory_id=mem.id,
                    text=mem.text,
                    score=score,
                    context=mem.context,
                    provenance=prov,
                )
            )

        if hits:
            self._hits_24h += 1
        return hits

    def get(self, memory_id: MemoryId) -> Memory | None:
        """Fetch a memory by id. Returns None if not found."""
        return self._memories.get(memory_id)

    # ── Tier movement (daemons mostly) ────────────────────────────────────────

    def promote(self, memory_id: MemoryId) -> None:
        """Hint: warm this memory up. L3→L2, or L2→L1.

        Used by the prefetcher when it speculatively pulls memories in.
        No-op if the memory is already in L1.
        """
        mem = self._memories.get(memory_id)
        if mem is None:
            logger.warning(f"promote: unknown memory_id={memory_id}")
            return
        new_tier = {Tier.L3: Tier.L2, Tier.L2: Tier.L1, Tier.L1: Tier.L1}[mem.tier]
        if new_tier != mem.tier:
            self._update_tier(memory_id, new_tier)
            self._promotions_24h += 1

    def demote(self, memory_id: MemoryId) -> None:
        """Hint: cool this memory off. L1→L2, or L2→L3.

        Used by the pruner when an L2 memory has gone cold (14 days
        no access) or when L2 hits its capacity ceiling.
        """
        mem = self._memories.get(memory_id)
        if mem is None:
            logger.warning(f"demote: unknown memory_id={memory_id}")
            return
        new_tier = {Tier.L1: Tier.L2, Tier.L2: Tier.L3, Tier.L3: Tier.L3}[mem.tier]
        if new_tier != mem.tier:
            self._update_tier(memory_id, new_tier)
            self._demotions_24h += 1

    def forget(self, memory_id: MemoryId) -> bool:
        """Hard delete. Returns True if it existed.

        Use sparingly. The whole point of the system is to *keep* memory
        bounded but useful. Forgetting is a privacy lever, not a cleanup
        strategy.
        """
        return self._memories.pop(memory_id, None) is not None

    # ── Observability ─────────────────────────────────────────────────────────

    def get_stats(self) -> TierStats:
        """Return current tier occupancy + rolling 24h metrics.

        Powers the Streamlit dashboard.
        """
        by_tier = {Tier.L1: 0, Tier.L2: 0, Tier.L3: 0}
        bytes_by_tier = {Tier.L1: 0, Tier.L2: 0, Tier.L3: 0}
        for m in self._memories.values():
            by_tier[m.tier] += 1
            # Crude byte estimate: utf-8 text + ~1.5KB per float32[384] embedding ref
            bytes_by_tier[m.tier] += len(m.text.encode("utf-8")) + 1536

        hit_rate = self._hits_24h / max(self._queries_24h, 1)

        return TierStats(
            l1_count=by_tier[Tier.L1],
            l2_count=by_tier[Tier.L2],
            l3_count=by_tier[Tier.L3],
            l1_bytes=bytes_by_tier[Tier.L1],
            l2_bytes=bytes_by_tier[Tier.L2],
            l3_bytes=bytes_by_tier[Tier.L3],
            l1_capacity_bytes=self._l1_cap,
            l2_capacity_bytes=self._l2_cap,
            l3_capacity_bytes=self._l3_cap,
            queries_24h=self._queries_24h,
            hit_rate_24h=hit_rate,
            promotions_24h=self._promotions_24h,
            demotions_24h=self._demotions_24h,
        )

    # ── Internals ─────────────────────────────────────────────────────────────

    def _update_tier(self, memory_id: MemoryId, new_tier: Tier) -> None:
        mem = self._memories[memory_id]
        self._memories[memory_id] = Memory(
            id=mem.id,
            text=mem.text,
            embedding_ref=mem.embedding_ref,
            context=mem.context,
            tier=new_tier,
            created_at=mem.created_at,
            last_touch=datetime.now(UTC),
            access_count=mem.access_count,
            is_consolidated=mem.is_consolidated,
        )
        logger.trace(f"_update_tier {memory_id} {mem.tier.value}→{new_tier.value}")
