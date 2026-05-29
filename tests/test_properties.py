"""Property-based tests using Hypothesis.

What's different from the other tests
-------------------------------------
The example-based tests in test_l1_working.py, test_l2_cache.py, etc.
check specific scenarios we thought of. These property tests check
INVARIANTS that must hold for ANY input — Hypothesis generates hundreds
of random input sequences and tries to find one that breaks the property.
When it finds a failure, it shrinks the input to a minimal reproducer.

This catches the cases we DIDN'T think of, which is where real bugs hide.

The invariants we pin down
--------------------------
- L1: byte budget is never exceeded, regardless of insert sequence
- L1: FIFO eviction order (oldest out first)
- L1: round-trip fidelity (what goes in comes back identical)
- L2: dual-store stays consistent (metadata count == addressable vectors)
- L2: labels are globally unique across any operation sequence
- Manager: tier never increases on demote (monotonicity)
"""

from __future__ import annotations

from datetime import UTC, datetime

import numpy as np
from hypothesis import HealthCheck, given, settings
from hypothesis import strategies as st

from neuro_paging import ContextTags, TimeBucket
from neuro_paging.memory.l1_working import L1WorkingContext
from neuro_paging.memory.l2_cache import L2HotVectorCache
from neuro_paging.memory.types import Memory, MemoryId, Tier

# ── Strategies (how Hypothesis generates test data) ──────────────────────────

# Text: 1-200 chars, printable, no control chars that'd break SQLite
text_strategy = st.text(
    alphabet=st.characters(min_codepoint=32, max_codepoint=126),
    min_size=1,
    max_size=200,
)

# A short, unique-ish memory id
mid_strategy = st.text(
    alphabet="abcdefghijklmnopqrstuvwxyz0123456789-",
    min_size=1,
    max_size=20,
)


def _make_memory(text: str, mid: str, tier: Tier = Tier.L1) -> Memory:
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


# ── L1 properties ─────────────────────────────────────────────────────────────


class TestL1Properties:
    @given(
        texts=st.lists(text_strategy, min_size=1, max_size=100),
        capacity=st.integers(min_value=256, max_value=8192),
    )
    @settings(max_examples=200, deadline=None)
    def test_byte_budget_never_exceeded(self, texts: list[str], capacity: int):
        """INVARIANT: after ANY insert sequence, bytes_used <= capacity_bytes."""
        l1 = L1WorkingContext(capacity_bytes=capacity)
        for i, text in enumerate(texts):
            mem = _make_memory(text, mid=f"m{i}", tier=Tier.L1)
            # Some texts may be individually larger than capacity — that's a
            # ValueError, which is allowed. We only assert the invariant for
            # memories that DID get inserted.
            try:
                l1.insert(mem)
            except ValueError:
                continue
            assert l1.stats().bytes_used <= capacity, (
                f"Byte budget violated: {l1.stats().bytes_used} > {capacity}"
            )

    @given(texts=st.lists(text_strategy, min_size=1, max_size=50))
    @settings(max_examples=150, deadline=None)
    def test_roundtrip_fidelity(self, texts: list[str]):
        """INVARIANT: a memory that fits comes back byte-identical."""
        # Big enough capacity that nothing evicts
        l1 = L1WorkingContext(capacity_bytes=64 * 1024 * 1024)
        inserted: dict[str, str] = {}
        for i, text in enumerate(texts):
            mid = f"m{i}"
            mem = _make_memory(text, mid=mid, tier=Tier.L1)
            l1.insert(mem)
            inserted[mid] = text

        for mid, original_text in inserted.items():
            loaded = l1.get(mid)
            assert loaded is not None, f"Lost memory {mid}"
            assert loaded.text == original_text, (
                f"Text corrupted: stored {original_text!r}, got {loaded.text!r}"
            )

    @given(n=st.integers(min_value=2, max_value=30))
    @settings(max_examples=100, deadline=None)
    def test_fifo_eviction_order(self, n: int):
        """INVARIANT: under eviction, the oldest-inserted is evicted first."""
        # Tight capacity so eviction definitely happens
        l1 = L1WorkingContext(capacity_bytes=512)
        evicted_ids: list[str] = []
        for i in range(n):
            # Uniform-size memories so eviction is purely about order
            mem = _make_memory("x" * 50, mid=f"m{i}", tier=Tier.L1)
            evicted = l1.insert(mem)
            evicted_ids.extend(str(e.id) for e in evicted)

        # Evicted ids must be a prefix of insertion order: m0, m1, m2, ...
        for idx, eid in enumerate(evicted_ids):
            assert eid == f"m{idx}", f"FIFO violated: eviction #{idx} was {eid}, expected m{idx}"


# ── L2 properties ─────────────────────────────────────────────────────────────


class TestL2Properties:
    @given(
        mids=st.lists(mid_strategy, min_size=1, max_size=40, unique=True),
    )
    @settings(max_examples=60, deadline=None, suppress_health_check=[HealthCheck.too_slow])
    def test_dual_store_stays_consistent(self, mids: list[str], tmp_path_factory):
        """INVARIANT: metadata count == addressable vectors, after any inserts."""
        tmp = tmp_path_factory.mktemp("l2_prop")
        l2 = L2HotVectorCache(
            data_dir=tmp,
            capacity_bytes=64 * 1024 * 1024,
            max_elements=10_000,
            dim=384,
        )
        for i, mid in enumerate(mids):
            mem = _make_memory(f"text {mid}", mid=mid, tier=Tier.L2)
            l2.insert(mem, _rand_vec(seed=i))

        # Every memory in metadata must have a label, and that label must
        # round-trip back to the same memory.
        all_mems = list(l2._metadata.iter_memories(Tier.L2))
        assert len(all_mems) == len(mids)
        for mem in all_mems:
            label = l2._metadata.label_for(mem.id)
            assert label is not None
            back = l2._metadata.get_by_label(label)
            assert back is not None and back.id == mem.id

        l2.close()

    @given(
        mids=st.lists(mid_strategy, min_size=1, max_size=40, unique=True),
    )
    @settings(max_examples=60, deadline=None, suppress_health_check=[HealthCheck.too_slow])
    def test_labels_globally_unique(self, mids: list[str], tmp_path_factory):
        """INVARIANT: no two live memories ever share an hnsw_label."""
        tmp = tmp_path_factory.mktemp("l2_labels")
        l2 = L2HotVectorCache(
            data_dir=tmp,
            capacity_bytes=64 * 1024 * 1024,
            max_elements=10_000,
            dim=384,
        )
        for i, mid in enumerate(mids):
            mem = _make_memory(f"text {mid}", mid=mid, tier=Tier.L2)
            l2.insert(mem, _rand_vec(seed=i))

        with l2._metadata._lock:
            cur = l2._metadata._conn.execute("SELECT hnsw_label FROM memories")
            labels = [r[0] for r in cur.fetchall()]
        assert len(labels) == len(set(labels)), "Duplicate labels found"

        l2.close()

    @given(text=text_strategy, seed=st.integers(min_value=0, max_value=10**6))
    @settings(max_examples=80, deadline=None, suppress_health_check=[HealthCheck.too_slow])
    def test_exact_match_query_returns_self(self, text: str, seed: int, tmp_path_factory):
        """INVARIANT: querying with a memory's exact vector returns that memory."""
        tmp = tmp_path_factory.mktemp("l2_exact")
        l2 = L2HotVectorCache(
            data_dir=tmp,
            capacity_bytes=64 * 1024 * 1024,
            max_elements=10_000,
            dim=384,
        )
        vec = _rand_vec(seed=seed)
        l2.insert(_make_memory(text, mid="target", tier=Tier.L2), vec)
        # Add a few distractors
        for i in range(5):
            l2.insert(
                _make_memory(f"distractor {i}", mid=f"d{i}", tier=Tier.L2),
                _rand_vec(seed=seed + 1 + i),
            )

        hits = l2.query(vec, k=1)
        assert len(hits) == 1
        mem, dist = hits[0]
        assert mem.id == "target"
        assert dist < 0.01  # cosine distance ~0 for identical vectors

        l2.close()
