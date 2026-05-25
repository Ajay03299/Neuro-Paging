"""MemoryManager — the public API between the substrate and
intelligence layer.

 Method signatures here are locked at git tag api-v0.1.0.
Behavior fills in across Sprints 1-4.


Quick map of methods:
    insert(...)     → store a new memory. Returns MemoryId.
    query(...)      → top-k retrieval. Returns list[Hit].
    get(...)        → fetch a single memory by id. Returns Memory or None.
    promote(...)    → hint: warm this memory up (mostly daemons).
    demote(...)     → hint: cool this memory off (mostly daemons).
    forget(...)     → hard delete. Use sparingly.
    get_stats()     → live tier observability.

Backend status (May 23, 2026):
    L1: REAL — L1WorkingContext with FIFO + byte budget
    L2: REAL — L2HotVectorCache with HNSW + SQLite metadata
    L3: stub — in-memory dict (real PQ-int8 backend lands Sprint 2)

Plug-in protocols:
    Scorer:   Christine's router. Called for every query candidate.
    Embedder: Christine's bge-small wrapper. Called once per insert
              when a memory crosses into L2.
"""

from __future__ import annotations

import time
import uuid
from datetime import UTC, datetime
from pathlib import Path
from typing import Protocol

import numpy as np
from loguru import logger

from neuro_paging.context.types import ContextTags
from neuro_paging.memory.l1_working import L1WorkingContext
from neuro_paging.memory.l2_cache import L2HotVectorCache
from neuro_paging.memory.types import (
    Hit,
    Memory,
    MemoryId,
    Provenance,
    Tier,
    TierStats,
)

# ── Plug-in protocols ─────────────────────────────────────────────────────────


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


class Embedder(Protocol):
    """The contract the embedder must implement. Christine owns this.

    Called by the manager when a memory needs an embedding (i.e. crosses
    into L2 or L3). For pure L1 use, the embedder is never invoked —
    manager works fine without one.
    """

    def embed(self, text: str) -> np.ndarray:
        """Return a (dim,) float32 ndarray for the given text."""
        ...


# ── Default stubs ─────────────────────────────────────────────────────────────


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


class _DefaultStubEmbedder:
    """Deterministic random embedder for tests + dev.

    Hashes the text to seed a numpy RNG, then samples a unit-norm
    vector. Same text always returns the same vector — so retrieval
    of "Italian food" twice in a row still finds the same neighbors.

    Replaced by Christine's bge-small wrapper at construction time.
    """

    def __init__(self, dim: int = 384) -> None:
        self._dim = dim

    def embed(self, text: str) -> np.ndarray:
        # Stable seed from text — same text → same vector
        seed = abs(hash(text)) % (2**32)
        rng = np.random.default_rng(seed)
        v = rng.standard_normal(self._dim).astype(np.float32)
        v /= np.linalg.norm(v) + 1e-8
        return v


# ── Default tier capacities (overridable via constructor) ─────────────────────
DEFAULT_L1_CAPACITY_BYTES = 32 * 1024  # 32 KB
DEFAULT_L2_CAPACITY_BYTES = 8 * 1024 * 1024  # 8 MB
DEFAULT_L3_CAPACITY_BYTES = 128 * 1024 * 1024  # 128 MB

# Default HNSW starting slot count for L2
DEFAULT_L2_MAX_ELEMENTS = 10_000

# Default embedding dimension (matches bge-small-en-v1.5)
DEFAULT_EMBEDDING_DIM = 384


# ── The contract ──────────────────────────────────────────────────────────────


class MemoryManager:
    """The public API for the tiered memory subsystem.

    Construct once at app startup, share across the pipeline.

    Args:
        data_dir: where L2 persists (HNSW index + SQLite metadata).
            Defaults to ".neuro-paging" under cwd. For tests, pass a
            tmp_path. For production, pass a per-user dir.
        scorer: Christine's routing function. Defaults to a stub.
        embedder: Christine's text→vector wrapper. Defaults to a
            deterministic stub.
        l1_capacity_bytes: L1 byte budget. Default 32 KB.
        l2_capacity_bytes: L2 byte budget. Default 8 MB.
        l2_max_elements: L2 HNSW starting slot count. Grows
            geometrically when exhausted.
        l3_capacity_bytes: L3 byte budget (stub for now). Default 128 MB.
        embedding_dim: L2 vector dim. Must match the embedder's output.

    Thread-safety: L1 and L2 have their own internal locks. The manager
    itself holds an RLock to keep multi-store mutations atomic.
    """

    def __init__(
        self,
        data_dir: Path | str | None = None,
        scorer: Scorer | None = None,
        embedder: Embedder | None = None,
        l1_capacity_bytes: int = DEFAULT_L1_CAPACITY_BYTES,
        l2_capacity_bytes: int = DEFAULT_L2_CAPACITY_BYTES,
        l2_max_elements: int = DEFAULT_L2_MAX_ELEMENTS,
        l3_capacity_bytes: int = DEFAULT_L3_CAPACITY_BYTES,
        embedding_dim: int = DEFAULT_EMBEDDING_DIM,
    ) -> None:
        self._scorer = scorer or _DefaultStubScorer()
        self._embedder = embedder or _DefaultStubEmbedder(dim=embedding_dim)
        self._l3_cap = l3_capacity_bytes
        self._embedding_dim = embedding_dim

        # Resolve data dir
        self._data_dir = Path(data_dir) if data_dir is not None else Path(".neuro-paging")
        self._data_dir.mkdir(parents=True, exist_ok=True)

        # ── REAL L1 backend ──
        self._l1 = L1WorkingContext(capacity_bytes=l1_capacity_bytes)

        # ── REAL L2 backend (HNSW + SQLite metadata sidecar) ──
        self._l2 = L2HotVectorCache(
            data_dir=self._data_dir / "l2",
            capacity_bytes=l2_capacity_bytes,
            max_elements=l2_max_elements,
            dim=embedding_dim,
        )

        # ── STUB L3 — real backend lands Sprint 2 ──
        self._l3: dict[MemoryId, Memory] = {}

        # Rolling counters for get_stats()
        self._queries_24h = 0
        self._promotions_24h = 0
        self._demotions_24h = 0
        self._hits_24h = 0

        logger.debug(
            "MemoryManager initialised — "
            f"L1=REAL({l1_capacity_bytes}B) "
            f"L2=REAL({l2_capacity_bytes}B/{l2_max_elements}slots) "
            f"L3=stub({l3_capacity_bytes}B) "
            f"data_dir={self._data_dir}"
        )

    # ── Lifecycle ─────────────────────────────────────────────────────────────

    def close(self) -> None:
        """Persist L2 to disk and close all backends.

        Call at app shutdown. Failing to call risks losing in-memory L2
        index changes (the SQLite sidecar is always persistent; only the
        HNSW index file is written on close()).
        """
        self._l2.close()

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

    def _which_tier(self, memory_id: MemoryId) -> Tier | None:
        """Which tier contains this memory? None if not found anywhere."""
        if self._l1.contains(memory_id):
            return Tier.L1
        if self._l2.contains(memory_id):
            return Tier.L2
        if memory_id in self._l3:
            return Tier.L3
        return None

    def _remove_from_any_tier(self, memory_id: MemoryId) -> Memory | None:
        """Remove a memory from whichever tier it currently lives in."""
        mem = self._l1.remove(memory_id)
        if mem is not None:
            return mem
        mem = self._l2.remove(memory_id)
        if mem is not None:
            return mem
        if memory_id in self._l3:
            return self._l3.pop(memory_id)
        return None

    def _put_in_l2(self, memory: Memory) -> list[Memory]:
        """Embed + insert into L2. Returns memories L2 evicted (for L3 demote).

        Helper for cascade-demote (L1 → L2) and direct L2 insert paths.
        """
        # Tier metadata may be wrong (e.g., a memory cascading from L1
        # still has tier=Tier.L1). Fix it before insertion.
        memory_for_l2 = self._with_tier(memory, Tier.L2)
        embedding = self._embedder.embed(memory_for_l2.text)
        return self._l2.insert(memory_for_l2, embedding)

    def _put_in_tier(self, memory: Memory, tier: Tier) -> list[Memory]:
        """Place a memory into the target tier. Returns memories evicted
        from the destination tier (so caller can cascade them).

        L1 → returns L1 evictions (which cascade to L2)
        L2 → returns L2 evictions (which cascade to L3)
        L3 → stub for now, no eviction logic
        """
        if tier == Tier.L1:
            return self._l1.insert(memory)
        elif tier == Tier.L2:
            return self._put_in_l2(memory)
        else:  # L3
            self._l3[memory.id] = self._with_tier(memory, Tier.L3)
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

    def _cascade_to_l3(self, evicted_from_l2: list[Memory]) -> None:
        """Land L2-evicted memories in the (stub) L3 store."""
        for ev in evicted_from_l2:
            self._l3[ev.id] = self._with_tier(ev, Tier.L3)
            self._demotions_24h += 1
            logger.trace(f"cascade-demote L2->L3 id={ev.id}")

    # ── Write path ────────────────────────────────────────────────────────────

    def insert(
        self,
        text: str,
        context: ContextTags,
        *,
        tier: Tier = Tier.L1,
    ) -> MemoryId:
        """Store a memory. Returns the assigned id.

        New memories default to L1. If L1 fills, evicted memories cascade
        to L2 automatically. If L2 fills, those memories cascade to L3.

        Direct insert to L2/L3 is supported for loading historical data,
        consolidator writes, and tests.
        """
        if not text.strip():
            raise ValueError("Cannot insert empty memory text")

        mem_id = MemoryId(str(uuid.uuid4()))
        now = datetime.now(UTC)
        memory = Memory(
            id=mem_id,
            text=text,
            embedding_ref=f"l2:{mem_id}",  # filled in by L2 if it lands there
            context=context,
            tier=tier,
            created_at=now,
            last_touch=now,
            access_count=0,
            is_consolidated=False,
        )

        try:
            evicted_from_target = self._put_in_tier(memory, tier)
        except ValueError:
            # Oversized for L1 — route directly to L2 instead.
            logger.info(f"insert id={mem_id} oversized for L1, routing to L2 directly")
            l2_evicted = self._put_in_l2(memory)
            self._cascade_to_l3(l2_evicted)
            return mem_id

        # Cascade chain:
        # - If we just inserted into L1, evictions are L1 spillover → L2
        # - If we just inserted into L2, evictions are L2 spillover → L3
        # - If we just inserted into L3, no further cascade
        if tier == Tier.L1 and evicted_from_target:
            for ev in evicted_from_target:
                l2_evicted = self._put_in_l2(ev)
                self._demotions_24h += 1
                logger.trace(f"cascade-demote L1->L2 id={ev.id}")
                # L2 might have evicted in turn — those land in L3
                self._cascade_to_l3(l2_evicted)
        elif tier == Tier.L2 and evicted_from_target:
            self._cascade_to_l3(evicted_from_target)
        # tier == Tier.L3: nothing to cascade

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

        Strategy: gather candidates from each tier (L1 via iter, L2 via
        HNSW ANN, L3 via dict iter), then re-rank with Christine's scorer.

        Note: L2 uses ANN over a single embedding of the query text. The
        scorer still sees every candidate and produces the final ranking,
        so the ANN's job is just to surface a useful candidate pool.
        """
        if k <= 0:
            return []
        self._queries_24h += 1

        # ── Gather candidates ──
        candidates: list[Memory] = []

        if Tier.L1.rank >= min_tier.rank:
            candidates.extend(iter(self._l1))

        if Tier.L2.rank >= min_tier.rank and len(self._l2) > 0:
            # Embed the query and ask L2 for top-(k * 4) — give the scorer
            # a wider pool than k since context-aware re-ranking may
            # promote a lower-distance candidate.
            try:
                query_embedding = self._embedder.embed(text)
                l2_hits = self._l2.query(query_embedding, k=max(k * 4, k))
                candidates.extend(mem for mem, _dist in l2_hits)
            except Exception as e:  # noqa: BLE001
                # Embedder failure shouldn't take down query — log and continue
                # with just L1 + L3 candidates
                logger.warning(f"L2 ANN query failed: {e}")

        if Tier.L3.rank >= min_tier.rank:
            candidates.extend(self._l3.values())

        if not candidates:
            return []

        # ── Score each candidate ──
        scored: list[tuple[float, Memory, Provenance]] = []
        for mem in candidates:
            score, prov = self._scorer.score(mem, text, context)
            scored.append((score, mem, prov))

        # ── Top-k ──
        scored.sort(key=lambda triple: triple[0], reverse=True)
        top = scored[:k]

        # ── Build hits + touch each tier ──
        hits: list[Hit] = []
        for score, mem, prov in top:
            # Touch in whichever tier the memory currently lives so it
            # survives the next eviction round.
            if mem.tier == Tier.L1:
                self._l1.touch(mem.id)
            elif mem.tier == Tier.L2:
                self._l2.touch(mem.id)
            # L3: no touch needed (stub has no eviction)

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
        """warm this memory up. L3→L2, or L2→L1.

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

        old_tier = mem.tier
        # Remove from current tier, update tier metadata, place into new tier.
        self._remove_from_any_tier(memory_id)
        promoted = self._with_tier(mem, new_tier)
        evicted = self._put_in_tier(promoted, new_tier)

        # Cascade any evictions resulting from the promotion.
        # L2 evictions go to L3. L1 evictions go through L2 (potentially
        # generating more L3 cascade).
        if new_tier == Tier.L1:
            for ev in evicted:
                l2_evicted = self._put_in_l2(ev)
                self._demotions_24h += 1
                self._cascade_to_l3(l2_evicted)
        elif new_tier == Tier.L2:
            self._cascade_to_l3(evicted)

        self._promotions_24h += 1
        logger.trace(f"promote {memory_id} {old_tier.value}->{new_tier.value}")

    def demote(self, memory_id: MemoryId) -> None:
        """cool this memory off. L1→L2, or L2→L3.

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

        old_tier = mem.tier
        self._remove_from_any_tier(memory_id)
        demoted = self._with_tier(mem, new_tier)
        evicted = self._put_in_tier(demoted, new_tier)
        if new_tier == Tier.L2:
            self._cascade_to_l3(evicted)
        self._demotions_24h += 1
        logger.trace(f"demote {memory_id} {old_tier.value}->{new_tier.value}")

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
        l2_stats = self._l2.stats()

        # Stub L3 byte estimate
        l3_bytes = sum(len(m.text.encode("utf-8")) + 1536 for m in self._l3.values())

        hit_rate = self._hits_24h / max(self._queries_24h, 1)

        return TierStats(
            l1_count=l1_stats.count,
            l2_count=l2_stats.count,
            l3_count=len(self._l3),
            l1_bytes=l1_stats.bytes_used,
            l2_bytes=l2_stats.bytes_estimate,
            l3_bytes=l3_bytes,
            l1_capacity_bytes=l1_stats.capacity_bytes,
            l2_capacity_bytes=l2_stats.capacity_bytes,
            l3_capacity_bytes=self._l3_cap,
            queries_24h=self._queries_24h,
            hit_rate_24h=hit_rate,
            promotions_24h=self._promotions_24h,
            demotions_24h=self._demotions_24h,
        )
