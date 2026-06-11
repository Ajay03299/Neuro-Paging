# Neuro-Paging — Architecture & Design Decisions

Tiered, context-aware, on-device AI memory. This document explains *why* the
system is built the way it is — the design reasoning, the measured tradeoffs,
and the bugs that shaped it. Companion docs: [BENCHMARKS.md](../BENCHMARKS.md)
(measured performance) and [eval/README.md](../eval/README.md) (retrieval
quality methodology).

---

## 1. The problem and the thesis

On-device agents need long-term memory, and the two common approaches both
fail at it:

- **Flat RAG**: one vector index over everything. Every query scans every
  memory; latency and energy grow with lifetime use; there is no notion of
  "hot" vs "cold" data. The index treats a memory used 50 times this week
  identically to one untouched for a year.
- **Sliding window**: keep the last N turns, forget the rest. Cheap, but it
  discards exactly the durable facts (preferences, routines, projects) that
  make a personal agent useful.

**The thesis: treat AI memory the way an OS treats RAM.** The mapping is
precise, not decorative:

| OS concept | Neuro-Paging |
| :-- | :-- |
| Cache hierarchy (L1/L2/L3) | Memory tiers with hard byte budgets |
| Page | A memory (text + embedding + context tags) |
| Page replacement / eviction | FIFO overflow cascade L1→L2→L3 |
| Background page daemon | Power-gated pruner demoting cold L2→L3 |
| Page table / metadata | SQLite sidecar mapping ids ↔ HNSW labels |

Two properties fall out of the design: **bounded resource use** (every tier
has a hard byte budget — nothing grows without limit) and **access-pattern
awareness** (hot memories stay fast, cold ones get archived, and retrieval
ranks by situation, not just similarity).

---

## 2. Tier design

### L1 — working context (in-memory FIFO, 32 KB)

An `OrderedDict` behind an `RLock`. Insert appends; overflow pops from the
front (FIFO); `touch()` bumps both the access count and the FIFO position so
recently-used memories survive longer. Everything is O(1).

*Why FIFO and not LRU/LFU?* Measured insert p99 is **1.42 µs** — the cost is
dominated by Python call overhead, not the data structure, so a smarter
policy buys nothing measurable at 32 KB scale while adding complexity. The
budget is **bytes, not item count**, because memory texts vary 10×+ in size;
a count budget would make the tier's real footprint unpredictable.

### L2 — hot vector cache (HNSW + SQLite dual-store, 8 MB)

Two stores that must never diverge:

- **hnswlib** (cosine) holds the vectors and answers ANN queries —
  measured p50 **0.78 ms at 10K vectors**.
- **SQLite** (WAL mode) holds everything else: text, context tags, access
  counts, timestamps, and the bidirectional memory-id ↔ HNSW-label mapping.

*Why a dual-store at all?* HNSW is the right structure for vector search but
has no durable metadata story; SQLite is the right structure for durable
structured data but can't do ANN. Each does what it's good at.

*How they stay consistent:* **metadata-first commit with rollback.** An
insert writes SQLite first (transactional), then adds to HNSW; if the HNSW
add fails, the SQLite row is rolled back. The invariant — every metadata row
has exactly one live label and vice versa — is enforced by 9 concurrency
stress tests (up to 100 threads) and property-based tests.

*The mark_deleted discovery:* hnswlib's `mark_deleted` tombstones a label but
**does not reclaim its physical slot**. So eviction alone cannot prevent slot
exhaustion — the physical count (including tombstones) can hit the cap while
the logical count is fine. The fix: check `get_current_count()` (physical)
before insert and grow the index ×1.5 when full, *then* apply logical
eviction. Getting this wrong produced real crashes at scale before the
two-concern split (physical growth vs logical eviction) was made explicit.

### L3 — archive (HNSW + SQLite, 128 MB)

Same backbone as L2 by **composition, not inheritance** — L3 wraps the same
dual-store machinery with its own budget and forces `tier=L3` on insert.
PQ-int8 compression is deliberately future work: float32 HNSW is correct and
simple, and compression is an optimization with no behavioral change, so it
can land later without touching the architecture.

### Movement between tiers

- **Cascade on overflow** (synchronous): L1 evicts → L2 inserts; L2 evicts →
  L3 inserts. A new memory always enters L1.
- **Pruner daemon** (asynchronous): APScheduler background job demoting cold
  L2 memories (>14 days untouched) to L3, capped per tick, and **power-gated**
  — it skips work on low battery (unless charging), foreground activity, or
  non-idle device. Memory maintenance must never cost the user battery.

---

## 3. Retrieval scoring, derived

Pure semantic search ranks by meaning alone. But a personal memory's
relevance also depends on *situation* (the same fact matters more in the
context where it was created) and *usage* (habits beat one-offs). Hence:
Score(m | q, c) = α·cos(e_m, e_q) + β·ctxSim(tags_m, c) + γ·log(1+freq_m)·decay(Δt)

- **Semantic, α = 0.60.** Cosine of bge-small embeddings. Dominant by
  design — a memory that isn't *about* the query shouldn't win on context.
  Because all embeddings are L2-normalized, cosine reduces to a dot product
  (the denominator ‖a‖‖b‖ = 1) — which is also what makes the native kernel
  possible (§4).
- **Context, β = 0.25.** Normalized match between the memory's stored
  situation and the current one: time-of-day bucket, foreground app,
  location, Jaccard overlap of tags. Missing fields are skipped, not
  penalized. This is the signal flat RAG throws away.
- **Frequency × recency, γ = 0.15.** `log(1+access_count)` gives
  diminishing returns (a memory used 100× isn't 100× a memory used once),
  normalized against a soft ceiling; multiplied by `exp(-Δt/half-life)` with
  a **7-day half-life** — a memory untouched for a week has half its
  frequency weight. Tie-breaker by design, hence the smallest weight.

Weights sum to 1.0 (validated at construction) and are **injected, not
hard-coded** — online per-user learning of α/β/γ from implicit feedback is a
clean future extension with no architectural change.

**The worked example** (real output, real bge embeddings): two memories with
*identical text* — "the CI deployment pipeline broke on the merge to main":

| Context at creation | sem | ctx | freq | **total** |
| :-- | --: | --: | --: | --: |
| morning · VSCode · used 12× · fresh | 0.830 | 1.000 | 0.734 | **0.858** |
| night · Netflix · never used · 20 days old | 0.830 | 0.000 | 0.000 | **0.498** |

Identical semantic score; the context and recency terms produce a decisive
rank gap. Flat RAG would tie them. This is the project's thesis in one table.

One performance note: the scorer exposes `set_query()` so the query is
embedded **once per query**, not once per candidate; the manager calls it
before the scoring loop. Queries use bge's query-instruction prefix
(`embed_query`), stored passages don't — the asymmetry bge was trained with.

---

## 4. The native NEON kernel

The retrieval hot path is "dot the query against N candidates."
`npaged-core/` implements it in C++17 with ARM NEON intrinsics, exposed via
pybind11 (zero-copy: C++ reads the NumPy buffers directly).

- **Vectorization:** 384 dims = 96 four-wide steps. `vld1q_f32` loads 4
  floats, `vfmaq_f32` fused-multiply-accumulates 4 lanes per instruction,
  `vaddvq_f32` sums the lanes at the end.
- **Threading:** candidate rows are partitioned across `std::thread`s, each
  writing a **disjoint output slice** — no locks, no contention. The GIL is
  released during compute (workers touch only raw float buffers).
- **Empirically tuned, not guessed** (1–11 thread sweep, median of 50 runs):
  the speedup peaks at **exactly 5 threads — the M3 Pro's P-core count**
  (3.67× over single-threaded) and *degrades* past it as efficiency-core
  chunks become stragglers. Threading is a net loss below ~10K candidates
  (launch overhead), so the kernel auto-falls-back to single-threaded under
  a measured 8192 threshold.

**Result:** at 50K candidates, **655 µs vs NumPy/Accelerate's 761 µs —
1.16× faster than BLAS.** Specialization (fixed dim, batched dot product,
hand-tuned threading) beats general GEMV at scale; BLAS wins at small N.
All variants verified against NumPy (max diff ~6e-8, float32 summation
order). Full sweep: [notes/native_kernel.md](../notes/native_kernel.md).

---

## 5. Evaluation methodology

Retrieval recall@k on **LongMemEval** (ICLR 2025): store each question's
session turns as memories (turn-level granularity, tagged with session id),
query with the question, score whether a gold evidence session appears in
the top-k.

Three methodology choices that make the number trustworthy:

1. **Oracle-first validation.** The oracle split (evidence-only, no
   distractors) must score ~1.0 by construction; it did (1.000 across all
   k), proving the harness before trusting the haystack number.
2. **Stride-sampling.** The dataset is grouped by question type; taking the
   first N samples only the easiest category. Stride-sampling covers all
   six ability types.
3. **Per-question isolation.** A fresh index per question (temp dir,
   cleaned up) — no cross-question contamination. (The first harness
   version got this wrong, accumulated 14K memories into one shared store,
   and crashed L2's resize — which exposed a real persistence bug, §6.)

**Result: recall@5 = 0.98** (50 questions, ~48 sessions each, bge-small
ONNX, on-device). Temporal-reasoning is the hard ceiling (0.92 even at
k=10) — time-relative queries need reasoning beyond embedding similarity.
These are retrieval-recall numbers, not official LongMemEval accuracy
scores. Full breakdown: [eval/README.md](../eval/README.md).

---

## 6. Decisions, tradeoffs, and bugs that shaped the system

**Bugs found by realistic load (each fixed with a regression test):**

- **HNSW resize-on-reload.** When L2 grew its index past the constructor's
  max_elements, the grown capacity wasn't persisted; reload reverted to the
  stale cap and the next insert computed a resize *below* the live count,
  which hnswlib rejects. Found by the LongMemEval scale test. Fix: on load,
  adopt max(requested, persisted capacity, live count); in resize, floor the
  grow base at the actual physical count. Defense in depth.
- **Duplicate query results.** Candidates were gathered from three tiers
  with no dedup; an over-fetched ANN pool could return the same memory
  twice. Caught by a months-old regression test the day real embeddings
  landed. Fix: dedup by memory id before scoring, keeping the hottest copy.
- **Concurrency-test deadlock.** The suite hung intermittently — the
  substrate's locking was correct; the *test harness* called `.result()`
  inline and passed lazy generators to `as_completed`, so a barrier never
  filled. Fix: materialize futures first; plus a permanent pytest-timeout
  (thread method) so any future hang fails fast with a thread dump.

**Deliberate scoping (documented, not built):** PQ-int8 for L3, the
consolidator, FP-Growth predictive prefetch, online weight learning, live
LLM generation (the `Generator` protocol ships with a transparent echo
default — a memory system's job is assembling the right context; generation
is pluggable). Each has a designed seam; none was rushed to 60%.

**Testing philosophy:** 216 tests in three styles — example-based
(correctness on known cases), concurrency stress (atomicity and budget
invariants under 100 threads), and property-based via Hypothesis
(invariants like "the byte budget is never exceeded for any insert
sequence" across generated inputs). Three layers because they catch
disjoint bug classes.
