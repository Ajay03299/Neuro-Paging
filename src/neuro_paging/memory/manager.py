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

Backend status (May 25, 2026):
    L1: REAL — L1WorkingContext with FIFO + byte budget
    L2: REAL — L2HotVectorCache with HNSW + SQLite metadata
    L3: REAL — L3ArchiveCache (HNSW + SQLite; PQ-int8 compression in Sprint 2)

Plug-in protocols:
    Scorer:   Christine's router. Called for every query candidate.
    Embedder: Christine's bge-small wrapper. Called once per insert
              when a memory crosses into L2 or L3.
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
from neuro_paging.memory.l3_archive import L3ArchiveCache
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
    """The contract the embedder must implement. Christine owns this."""

    def embed(self, text: str) -> np.ndarray:
        """Return a (dim,) float32 ndarray for the given text."""
        ...


# ── Default stubs ─────────────────────────────────────────────────────────────


class _DefaultStubScorer:
    """Trivial scorer used until Christine wires in the real one."""

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
    """Deterministic random embedder for tests + dev."""

    def __init__(self, dim: int = 384) -> None:
        self._dim = dim

    def embed(self, text: str) -> np.ndarray:
        seed = abs(hash(text)) % (2**32)
        rng = np.random.default_rng(seed)
        v = rng.standard_normal(self._dim).astype(np.float32)
        v /= np.linalg.norm(v) + 1e-8
        return v


# ── Default tier capacities ───────────────────────────────────────────────────
DEFAULT_L1_CAPACITY_BYTES = 32 * 1024  # 32 KB
DEFAULT_L2_CAPACITY_BYTES = 8 * 1024 * 1024  # 8 MB
DEFAULT_L3_CAPACITY_BYTES = 128 * 1024 * 1024  # 128 MB

DEFAULT_L2_MAX_ELEMENTS = 10_000
DEFAULT_L3_MAX_ELEMENTS = 100_000

DEFAULT_EMBEDDING_DIM = 384


# ── The contract ──────────────────────────────────────────────────────────────


class MemoryManager:
    """The public API for the tiered memory subsystem.

    All three tiers are now real. The deck's L1/L2/L3 cascade is end-
    to-end live: L1 fills → cascade to L2 → if L2 fills cascade to L3.
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
        l3_max_elements: int = DEFAULT_L3_MAX_ELEMENTS,
        embedding_dim: int = DEFAULT_EMBEDDING_DIM,
    ) -> None:
        self._scorer = scorer or _DefaultStubScorer()
        self._embedder = embedder or _DefaultStubEmbedder(dim=embedding_dim)
        self._embedding_dim = embedding_dim

        self._data_dir = Path(data_dir) if data_dir is not None else Path(".neuro-paging")
        self._data_dir.mkdir(parents=True, exist_ok=True)

        # ── REAL L1 ──
        self._l1 = L1WorkingContext(capacity_bytes=l1_capacity_bytes)

        # ── REAL L2 ──
        self._l2 = L2HotVectorCache(
            data_dir=self._data_dir / "l2",
            capacity_bytes=l2_capacity_bytes,
            max_elements=l2_max_elements,
            dim=embedding_dim,
        )

        # ── REAL L3 ──
        self._l3 = L3ArchiveCache(
            data_dir=self._data_dir / "l3",
            capacity_bytes=l3_capacity_bytes,
            max_elements=l3_max_elements,
            dim=embedding_dim,
        )

        # Rolling counters for get_stats()
        self._queries_24h = 0
        self._promotions_24h = 0
        self._demotions_24h = 0
        self._hits_24h = 0
        # Memories evicted off the bottom of L3 — terminally forgotten.
        # Sprint 2 will add a "tombstoned" status with privacy timers.
        self._forgotten_24h = 0

        logger.debug(
            "MemoryManager initialised — "
            f"L1=REAL({l1_capacity_bytes}B) "
            f"L2=REAL({l2_capacity_bytes}B/{l2_max_elements}slots) "
            f"L3=REAL({l3_capacity_bytes}B/{l3_max_elements}slots) "
            f"data_dir={self._data_dir}"
        )

    # ── Lifecycle ─────────────────────────────────────────────────────────────

    def close(self) -> None:
        """Persist L2 + L3 to disk and close all backends."""
        self._l2.close()
        self._l3.close()

    # ── Internal: locate a memory across tiers ────────────────────────────────

    def _find(self, memory_id: MemoryId) -> Memory | None:
        mem = self._l1.get(memory_id)
        if mem is not None:
            return mem
        mem = self._l2.get(memory_id)
        if mem is not None:
            return mem
        return self._l3.get(memory_id)

    def _which_tier(self, memory_id: MemoryId) -> Tier | None:
        if self._l1.contains(memory_id):
            return Tier.L1
        if self._l2.contains(memory_id):
            return Tier.L2
        if self._l3.contains(memory_id):
            return Tier.L3
        return None

    def _remove_from_any_tier(self, memory_id: MemoryId) -> Memory | None:
        mem = self._l1.remove(memory_id)
        if mem is not None:
            return mem
        mem = self._l2.remove(memory_id)
        if mem is not None:
            return mem
        return self._l3.remove(memory_id)

    def _put_in_l2(self, memory: Memory) -> list[Memory]:
        """Embed + insert into L2. Returns memories L2 evicted (→ L3)."""
        memory_for_l2 = self._with_tier(memory, Tier.L2)
        embedding = self._embedder.embed(memory_for_l2.text)
        return self._l2.insert(memory_for_l2, embedding)

    def _put_in_l3(self, memory: Memory) -> list[Memory]:
        """Embed + insert into L3. Returns memories L3 evicted (forgotten)."""
        memory_for_l3 = self._with_tier(memory, Tier.L3)
        embedding = self._embedder.embed(memory_for_l3.text)
        return self._l3.insert(memory_for_l3, embedding)

    def _put_in_tier(self, memory: Memory, tier: Tier) -> list[Memory]:
        """Place a memory into the target tier. Returns evictions."""
        if tier == Tier.L1:
            return self._l1.insert(memory)
        elif tier == Tier.L2:
            return self._put_in_l2(memory)
        else:  # L3
            return self._put_in_l3(memory)

    def _with_tier(self, memory: Memory, new_tier: Tier) -> Memory:
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
        """Land L2-evicted memories in the real L3."""
        for ev in evicted_from_l2:
            l3_evicted = self._put_in_l3(ev)
            self._demotions_24h += 1
            logger.trace(f"cascade-demote L2->L3 id={ev.id}")
            # L3 evictions are terminal in v0
            for forgotten in l3_evicted:
                self._forgotten_24h += 1
                logger.info(f"L3 forgot id={forgotten.id} — bottom of stack")

    # ── Write path ────────────────────────────────────────────────────────────

    def insert(
        self,
        text: str,
        context: ContextTags,
        *,
        tier: Tier = Tier.L1,
    ) -> MemoryId:
        """Store a memory. Returns the assigned id."""
        if not text.strip():
            raise ValueError("Cannot insert empty memory text")

        mem_id = MemoryId(str(uuid.uuid4()))
        now = datetime.now(UTC)
        memory = Memory(
            id=mem_id,
            text=text,
            embedding_ref=f"emb:{mem_id}",
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
            # Oversized for L1 — route directly to L2
            logger.info(f"insert id={mem_id} oversized for L1, routing to L2 directly")
            l2_evicted = self._put_in_l2(memory)
            self._cascade_to_l3(l2_evicted)
            return mem_id

        # Cascade chain
        if tier == Tier.L1 and evicted_from_target:
            for ev in evicted_from_target:
                l2_evicted = self._put_in_l2(ev)
                self._demotions_24h += 1
                logger.trace(f"cascade-demote L1->L2 id={ev.id}")
                self._cascade_to_l3(l2_evicted)
        elif tier == Tier.L2 and evicted_from_target:
            self._cascade_to_l3(evicted_from_target)
        elif tier == Tier.L3 and evicted_from_target:
            # L3 evictions are terminal
            for _forgotten in evicted_from_target:
                self._forgotten_24h += 1

        logger.trace(f"insert id={mem_id} tier={tier.value} len={len(text)}")
        return mem_id

    # ── Read path ─────────────────────────────────────────────────────────────

    def query(
        self,
        text: str,
        context: ContextTags,
        *,
        k: int = 5,
        min_tier: Tier = Tier.L3,
    ) -> list[Hit]:
        """Retrieve top-k memories ranked by the scorer."""
        if k <= 0:
            return []
        self._queries_24h += 1

        candidates: list[Memory] = []

        if Tier.L1.rank >= min_tier.rank:
            candidates.extend(iter(self._l1))

        # L2 + L3 share the embedder call. Do it once.
        query_embedding: np.ndarray | None = None

        if Tier.L2.rank >= min_tier.rank and len(self._l2) > 0:
            try:
                query_embedding = self._embedder.embed(text)
                l2_hits = self._l2.query(query_embedding, k=max(k * 4, k))
                candidates.extend(mem for mem, _dist in l2_hits)
            except Exception as e:  # noqa: BLE001
                logger.warning(f"L2 ANN query failed: {e}")

        if Tier.L3.rank >= min_tier.rank and len(self._l3) > 0:
            try:
                if query_embedding is None:
                    query_embedding = self._embedder.embed(text)
                l3_hits = self._l3.query(query_embedding, k=max(k * 4, k))
                candidates.extend(mem for mem, _dist in l3_hits)
            except Exception as e:  # noqa: BLE001
                logger.warning(f"L3 ANN query failed: {e}")

        if not candidates:
            return []

        # Score
        scored: list[tuple[float, Memory, Provenance]] = []
        for mem in candidates:
            score, prov = self._scorer.score(mem, text, context)
            scored.append((score, mem, prov))

        # Top-k
        scored.sort(key=lambda triple: triple[0], reverse=True)
        top = scored[:k]

        # Touch + hits
        hits: list[Hit] = []
        for score, mem, prov in top:
            if mem.tier == Tier.L1:
                self._l1.touch(mem.id)
            elif mem.tier == Tier.L2:
                self._l2.touch(mem.id)
            elif mem.tier == Tier.L3:
                self._l3.touch(mem.id)

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
        return self._find(memory_id)

    # ── Tier movement (daemons mostly) ────────────────────────────────────────

    def promote(self, memory_id: MemoryId) -> None:
        """Warm a memory up: L3→L2, or L2→L1."""
        mem = self._find(memory_id)
        if mem is None:
            logger.warning(f"promote: unknown memory_id={memory_id}")
            return

        new_tier = {Tier.L3: Tier.L2, Tier.L2: Tier.L1, Tier.L1: Tier.L1}[mem.tier]
        if new_tier == mem.tier:
            return

        old_tier = mem.tier
        self._remove_from_any_tier(memory_id)
        promoted = self._with_tier(mem, new_tier)
        evicted = self._put_in_tier(promoted, new_tier)

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
        """Cool a memory off: L1→L2, or L2→L3."""
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
        elif new_tier == Tier.L3:
            for _ in evicted:
                self._forgotten_24h += 1
        self._demotions_24h += 1
        logger.trace(f"demote {memory_id} {old_tier.value}->{new_tier.value}")

    def forget(self, memory_id: MemoryId) -> bool:
        """Hard delete from whichever tier holds the memory."""
        return self._remove_from_any_tier(memory_id) is not None

    # ── Observability ─────────────────────────────────────────────────────────

    def get_stats(self) -> TierStats:
        l1_stats = self._l1.stats()
        l2_stats = self._l2.stats()
        l3_stats = self._l3.stats()

        hit_rate = self._hits_24h / max(self._queries_24h, 1)

        return TierStats(
            l1_count=l1_stats.count,
            l2_count=l2_stats.count,
            l3_count=l3_stats.count,
            l1_bytes=l1_stats.bytes_used,
            l2_bytes=l2_stats.bytes_estimate,
            l3_bytes=l3_stats.bytes_estimate,
            l1_capacity_bytes=l1_stats.capacity_bytes,
            l2_capacity_bytes=l2_stats.capacity_bytes,
            l3_capacity_bytes=l3_stats.capacity_bytes,
            queries_24h=self._queries_24h,
            hit_rate_24h=hit_rate,
            promotions_24h=self._promotions_24h,
            demotions_24h=self._demotions_24h,
        )
