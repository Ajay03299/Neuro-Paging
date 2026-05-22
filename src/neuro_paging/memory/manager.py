"""MemoryManager — the public API between Ajay's substrate and Christine's
intelligence layer.

THIS IS A CONTRACT. Method signatures here are locked at git tag api-v0.1.0.
Behavior fills in across Sprints 1-4.

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

Backend status (May 22, 2026):
    L1: REAL — L1WorkingContext with FIFO + byte budget
    L2: stub — in-memory dict (real HNSW backend lands Sprint 1)
    L3: stub — in-memory dict (real PQ-int8 backend lands Sprint 2)

The L2/L3 stubs are intentionally minimal. The integration *pattern* —
how the manager talks to a tier backend — is fully exercised by the
real L1. When real L2/L3 land, they slot into the same call sites.
"""

from __future__ import annotations

import time
import uuid
from datetime import UTC, datetime
from typing import Protocol

from loguru import logger

from neuro_paging.context.types import ContextTags
from neuro_paging.memory.l1_working import L1WorkingContext
from neuro_paging.memory.types import (
    Hit,
    Memory,
    MemoryId,
    Provenance,
    Tier,
    TierStats,
)

# ── Scoring callback ──────────────────────────────────────────────────────────


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

    Returns a stable but meaningless score based on word overlap. Lets
    end-to-end tests run without depending on her layer being ready.
    """

    def score(
        self,
        memory: Memory,
        query_text: str,
        query_context: ContextTags,
    ) -> tuple[float, Provenance]:
        start = time.perf_counter()
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
DEFAULT_L1_CAPACITY_BYTES = 32 * 1024  # 32 KB
DEFAULT_L2_CAPACITY_BYTES = 8 * 1024 * 1024  # 8 MB
DEFAULT_L3_CAPACITY_BYTES = 128 * 1024 * 1024  # 128 MB


# ── The contract ──────────────────────────────────────────────────────────────


class MemoryManager:
    """The public API for the tiered memory subsystem.

    Construct once at app startup, share across the pipeline.
    Thread-safety: L1 is internally locked. The L2/L3 stubs are
    single-threaded today; real backends will be locked too.
    """

    def __init__(
        self,
        scorer: Scorer | None = None,
        l1_capacity_bytes: int = DEFAULT_L1_CAPACITY_BYTES,
        l2_capacity_bytes: int = DEFAULT_L2_CAPACITY_BYTES,
        l3_capacity_bytes: int = DEFAULT_L3_CAPACITY_BYTES,
    ) -> None:
        self._scorer = scorer or _DefaultStubScorer()
        self._l2_cap = l2_capacity_bytes
        self._l3_cap = l3_capacity_bytes

        # ── REAL L1 backend (FIFO + byte budget) ──
        self._l1 = L1WorkingContext(capacity_bytes=l1_capacity_bytes)

        # ── STUB L2/L3 backends — real implementations land Sprint 1/2 ──
        self._l2: dict[MemoryId, Memory] = {}
        self._l3: dict[MemoryId, Memory] = {}

        # Rolling counters for get_stats()
        self._queries_24h = 0
        self._promotions_24h = 0
        self._demotions_24h = 0
        self._hits_24h = 0

        logger.debug(
            "MemoryManager initialised — "
            f"L1=REAL({l1_capacity_bytes}B) L2=stub({l2_capacity_bytes}B) L3=stub({l3_capacity_bytes}B)"
        )

    # ── Internal: locate a memory across tiers ────────────────────────────────

    def _find(self, memory_id: MemoryId) -> Memory | None:
        """Look up a memory in any tier. Returns None if not found."""
        mem = self._l1.get(memory_id)
        if mem is not None:
            return mem
        mem = self._l2.get(memory_id)
        if mem is not None:
            return mem
        return self._l3.get(memory_id)

    def _remove_from_any_tier(self, memory_id: MemoryId) -> Memory | None:
        """Remove a memory from whichever tier it currently lives in."""
        mem = self._l1.remove(memory_id)
        if mem is not None:
            return mem
        if memory_id in self._l2:
            return self._l2.pop(memory_id)
        if memory_id in self._l3:
            return self._l3.pop(memory_id)
        return None

    def _put_in_tier(self, memory: Memory, tier: Tier) -> list[Memory]:
        """Place a memory into the target tier. Returns memories evicted
        from L1 (if any) so the caller can cascade them to L2.

        L2 and L3 are stubs today — no eviction logic. Real backends will
        return their own evicted memories from this method.
        """
        if tier == Tier.L1:
            return self._l1.insert(memory)
        elif tier == Tier.L2:
            self._l2[memory.id] = memory
            return []
        else:  # L3
            self._l3[memory.id] = memory
            return []

    def _with_tier(self, memory: Memory, new_tier: Tier) -> Memory:
        """Return a copy of memory with tier replaced (frozen dataclass)."""
        return Memory(
            id=memory.id,
            text=memory.text,
            embedding_ref=memory.embedding_ref,
            context=memory.context,
            tier=new_tier,
            created_at=memory.created_at,
            last_touch=datetime.now(UTC),
            access_count=memory.access_count,
            is_consolidated=memory.is_consolidated,
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

        New memories default to L1. If L1 is full, oldest entries cascade
        down to L2 automatically (mirrors the daemon-driven flow that
        Sprint 2 formalises).

        Direct insert to L2/L3 is supported for loading historical data
        and tests.
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

        try:
            evicted_from_l1 = self._put_in_tier(memory, tier)
        except ValueError:
            # Oversized for L1 — route directly to L2 instead.
            # In real backends, L2 will accept anything L1 can't.
            logger.info(f"insert id={mem_id} oversized for L1, routing to L2 directly")
            big_mem = self._with_tier(memory, Tier.L2)
            self._l2[mem_id] = big_mem
            return mem_id

        # Cascade L1 evictions down to L2. This is the natural-flow demotion
        # path. It complements (but does not replace) the explicit demote()
        # method that the pruner daemon uses.
        for ev in evicted_from_l1:
            demoted = self._with_tier(ev, Tier.L2)
            self._l2[ev.id] = demoted
            self._demotions_24h += 1
            logger.trace(f"cascade-demote L1->L2 id={ev.id}")

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
        """
        if k <= 0:
            return []
        self._queries_24h += 1

        # Build candidate set across tiers based on min_tier rank.
        # min_tier.rank: L3=0, L2=1, L1=2. We include tiers >= min_tier.rank.
        candidates: list[Memory] = []
        if Tier.L1.rank >= min_tier.rank:
            candidates.extend(iter(self._l1))
        if Tier.L2.rank >= min_tier.rank:
            candidates.extend(self._l2.values())
        if Tier.L3.rank >= min_tier.rank:
            candidates.extend(self._l3.values())

        # Score each candidate
        scored: list[tuple[float, Memory, Provenance]] = []
        for mem in candidates:
            score, prov = self._scorer.score(mem, text, context)
            scored.append((score, mem, prov))

        # Top-k
        scored.sort(key=lambda triple: triple[0], reverse=True)
        top = scored[:k]

        # Convert to hits + bump access counters
        hits: list[Hit] = []
        for score, mem, prov in top:
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

            # Write back to the tier the memory currently lives in.
            # Also touch() L1 so it survives the next eviction round.
            if mem.tier == Tier.L1:
                # Re-insert into L1 will update + move-to-back (handled by
                # L1WorkingContext). No L2 cascade expected here (we just
                # took it out of L1, so room exists).
                self._l1.insert(updated)
            elif mem.tier == Tier.L2:
                self._l2[mem.id] = updated
            else:
                self._l3[mem.id] = updated

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
        return self._find(memory_id)

    # ── Tier movement (daemons mostly) ────────────────────────────────────────

    def promote(self, memory_id: MemoryId) -> None:
        """Hint: warm this memory up. L3→L2, or L2→L1.

        Used by the prefetcher when it speculatively pulls memories in.
        No-op if the memory is already in L1.
        """
        mem = self._find(memory_id)
        if mem is None:
            logger.warning(f"promote: unknown memory_id={memory_id}")
            return

        new_tier = {Tier.L3: Tier.L2, Tier.L2: Tier.L1, Tier.L1: Tier.L1}[mem.tier]
        if new_tier == mem.tier:
            return

        # Remove from current tier, place into new tier.
        self._remove_from_any_tier(memory_id)
        promoted = self._with_tier(mem, new_tier)
        evicted_from_l1 = self._put_in_tier(promoted, new_tier)

        # Cascade any L1 evictions that resulted from the promotion.
        for ev in evicted_from_l1:
            demoted = self._with_tier(ev, Tier.L2)
            self._l2[ev.id] = demoted
            self._demotions_24h += 1

        self._promotions_24h += 1
        logger.trace(f"promote {memory_id} {mem.tier.value}->{new_tier.value}")

    def demote(self, memory_id: MemoryId) -> None:
        """Hint: cool this memory off. L1→L2, or L2→L3.

        Used by the pruner when an L2 memory has gone cold (14 days
        no access) or when L2 hits its capacity ceiling.
        """
        mem = self._find(memory_id)
        if mem is None:
            logger.warning(f"demote: unknown memory_id={memory_id}")
            return

        new_tier = {Tier.L1: Tier.L2, Tier.L2: Tier.L3, Tier.L3: Tier.L3}[mem.tier]
        if new_tier == mem.tier:
            return

        self._remove_from_any_tier(memory_id)
        demoted = self._with_tier(mem, new_tier)
        self._put_in_tier(demoted, new_tier)
        self._demotions_24h += 1
        logger.trace(f"demote {memory_id} {mem.tier.value}->{new_tier.value}")

    def forget(self, memory_id: MemoryId) -> bool:
        """Hard delete. Returns True if it existed.

        Use sparingly. The whole point of the system is to *keep* memory
        bounded but useful. Forgetting is a privacy lever, not a cleanup
        strategy.
        """
        return self._remove_from_any_tier(memory_id) is not None

    # ── Observability ─────────────────────────────────────────────────────────

    def get_stats(self) -> TierStats:
        """Return current tier occupancy + rolling 24h metrics.

        Powers the Streamlit dashboard.
        """
        l1_stats = self._l1.stats()

        # Stub L2/L3 byte estimates — same crude formula as before.
        l2_bytes = sum(len(m.text.encode("utf-8")) + 1536 for m in self._l2.values())
        l3_bytes = sum(len(m.text.encode("utf-8")) + 1536 for m in self._l3.values())

        hit_rate = self._hits_24h / max(self._queries_24h, 1)

        return TierStats(
            l1_count=l1_stats.count,
            l2_count=len(self._l2),
            l3_count=len(self._l3),
            l1_bytes=l1_stats.bytes_used,
            l2_bytes=l2_bytes,
            l3_bytes=l3_bytes,
            l1_capacity_bytes=l1_stats.capacity_bytes,
            l2_capacity_bytes=self._l2_cap,
            l3_capacity_bytes=self._l3_cap,
            queries_24h=self._queries_24h,
            hit_rate_24h=hit_rate,
            promotions_24h=self._promotions_24h,
            demotions_24h=self._demotions_24h,
        )
