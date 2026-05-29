# Benchmarks

All numbers are **measured**, not projected — produced by the scripts in
[`bench/`](./bench) and re-runnable on any machine. Each run writes a
timestamped JSON to `bench/results/` and a human-readable report to
[`notes/`](./notes).

**Hardware:** Apple Silicon M3 Pro · macOS · Python 3.11
**Last updated:** May 2026 (Sprint 1)

---

## Headline numbers

| Tier / component | Metric | Deck target | **Measured** | Margin |
| :--- | :--- | ---: | ---: | ---: |
| **L1** working context | insert p99 @ 32 KB | < 1 ms | **1.42 µs** | **704× under** |
| **L1** working context | insert throughput | — | **1.1M ops/sec** | flat across capacities |
| **L2** hot vector cache | query p50 @ 10K vectors | ~5 ms | **0.78 ms** | **6.4× under** |
| **L2** hot vector cache | insert throughput (steady) | — | **659 ops/sec** | dual-store atomic |
| hnswlib (raw reference) | query p95 @ 10K | ~5 ms | **0.356 ms** | 14× under |

> The hnswlib row is the bare library with no wrapper — it sets the floor.
> L2 adds SQLite metadata, atomic dual-store commits, and eviction on top,
> which is why L2's latency is higher than raw hnswlib. The +0.5 ms delta is
> the measured cost of durability and consistency.

---

## L1 — working context (in-memory FIFO + byte budget)

`bench/l1_baseline.py` · full report: [`notes/l1_baseline.md`](./notes/l1_baseline.md)

Per-op insert latency (microseconds), measured with the eviction path active:

| Capacity | p50 | p95 | p99 | Max |
| :--- | ---: | ---: | ---: | ---: |
| 32 KB (deck spec) | 0.96 | 1.29 | **1.42** | 10.83 |
| 128 KB (4× headroom) | 1.04 | 1.33 | 1.58 | 18.25 |
| 1 KB (eviction-heavy) | 1.00 | 1.25 | 1.38 | 13.04 |

Insert throughput is **~1.1M ops/sec and flat across all three capacities** —
eviction is genuinely O(1), no perf cliff under sustained pressure. The cost
is dominated by Python function-call overhead, not the OrderedDict operations,
which is why a C++ rewrite of L1 would not help (and is not planned).

---

## L2 — hot vector cache (HNSW + SQLite metadata)

`bench/l2_baseline.py` · full report: [`notes/l2_baseline.md`](./notes/l2_baseline.md)

### Query latency (top-K = 5)

| Corpus size | p50 | p95 | p99 | Max |
| ---: | ---: | ---: | ---: | ---: |
| 1,000 | 245 µs | 260 µs | 292 µs | 524 µs |
| **10,000** | **782 µs** | 861 µs | 938 µs | 1154 µs |
| 50,000 | 2626 µs | 2788 µs | 2875 µs | 3151 µs |

Latency scales sub-linearly: 10× corpus → ~3–4× latency, the expected
HNSW behavior plus a near-constant SQLite hydration cost per result.

### Insert performance

| Regime | Throughput | p50 | p99 | Notes |
| :--- | ---: | ---: | ---: | :--- |
| Steady (no eviction) | 659 ops/sec | 1.5 ms | 3.0 ms | 64 MB budget |
| Pressure (eviction) | 464 ops/sec | 2.5 ms | 4.3 ms | 2 MB budget, 3.7K evictions |

Insert cost is dominated by the SQLite metadata write (the atomic
"metadata-first, then HNSW" commit protocol). This is the price of
crash-safe, consistent dual-store mutations.

---

## hnswlib — raw reference

`bench/hnswlib_baseline.py` · full report: [`notes/hnswlib_baseline.md`](./notes/hnswlib_baseline.md)

Bare hnswlib with no metadata layer, measured so we can quantify exactly
what our L2 wrapper costs. p95 query at 10K vectors = **0.356 ms**. L2's
+142% overhead vs this floor buys durability, observability, and atomicity.

---

## Reproduce

```bash
uv venv && source .venv/bin/activate
uv pip install -e ".[dev]"

python bench/hnswlib_baseline.py   # raw reference
python bench/l1_baseline.py        # L1 working context
python bench/l2_baseline.py        # L2 hot vector cache
```

Each script prints a table, writes timestamped JSON to `bench/results/`,
and regenerates the matching report in `notes/`.

---

## Test coverage

The substrate ships with **169 tests**, including **9 concurrency stress
tests** that hammer L1/L2/L3 and the pruner daemon from up to 100 threads
simultaneously — verifying dual-store atomicity, race-free monotonic label
assignment, byte-budget invariants under contention, and that the background
pruner runs safely alongside foreground writes.

```bash
pytest tests/ -v
```