<div align="center">

# 🧠 Neuro-Paging

**An operating system for AI memory.**

*Tiered. Context-aware. Quietly predictive. 100% on-device.*

[![License: Apache 2.0](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)
[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)
[![Samsung ennovateX](https://img.shields.io/badge/Samsung-ennovateX_2026-1428a0.svg)](https://ennovatex.io/ax-hackathon/)
[![Tests](https://img.shields.io/badge/tests-169_passing-brightgreen.svg)](#testing)
[![CI](https://github.com/Ajay03299/Neuro-Paging/actions/workflows/ci.yml/badge.svg)](https://github.com/Ajay03299/Neuro-Paging/actions/workflows/ci.yml)

</div>

---

## The pitch

> Treat AI memory the way an OS treats RAM. Page the right thoughts to the front of mind, archive the rest, and quietly learn the user's rhythms — so the pages worth keeping warm are already loaded before they ever hit send.

Built for **Samsung ennovateX AX Hackathon 2026 · Problem Statement #03 · Context-Aware Adaptive Memory for Mobile Agentic Systems** by team **ByteMe**.

---

## What's built right now

This is an active hackathon project. Here is an honest split of what is
**shipped and tested** versus what is **on the roadmap** — no vaporware.

| Component | Status | Evidence |
| :--- | :---: | :--- |
| **L1** working context (FIFO + byte budget) | ✅ **shipped** | measured 1.42 µs p99 insert · [benchmarks](./BENCHMARKS.md) |
| **L2** hot vector cache (HNSW + SQLite, atomic dual-store) | ✅ **shipped** | measured 0.78 ms p50 query @ 10K · [benchmarks](./BENCHMARKS.md) |
| **L3** archive tier (HNSW + SQLite) | ✅ **shipped** | cascade L1→L2→L3 end-to-end |
| **MemoryManager** public API (locked contract) | ✅ **shipped** | `api-v0.1.0` tag |
| **Pruner daemon** (cold L2→L3, power-gated) | ✅ **shipped** | APScheduler + power gating |
| **Concurrency-safe** under multi-threaded load | ✅ **shipped** | 9 stress tests, up to 100 threads |
| L3 PQ-int8 compression | 🚧 roadmap | Sprint 2 — L3 is float32 HNSW today |
| Context-aware routing (α·cos + β·ctx + γ·freq) | 🚧 roadmap | Sprint 3 — intelligence layer |
| Predictive prefetch (FP-Growth) | 🚧 roadmap | Sprint 3 |
| C++17 native core (ARM NEON SIMD) | 🚧 roadmap | Sprint 3, decision-gated |
| Consolidator (cluster → concept summary) | 🚧 roadmap | Sprint 2 |
| Live Streamlit dashboard | 🚧 roadmap | Sprint 1–2 |

**169 tests passing**, CI green, Apache-2.0. See [Roadmap](#roadmap) for the full plan.

---

## Measured performance

Every number below is **measured on Apple Silicon M3 Pro**, not projected.
Reproducible via `bench/`. Full detail: **[BENCHMARKS.md](./BENCHMARKS.md)**.

| Tier | Metric | Deck target | **Measured** | Margin |
| :--- | :--- | ---: | ---: | ---: |
| L1 | insert p99 @ 32 KB | < 1 ms | **1.42 µs** | **704× under** |
| L2 | query p50 @ 10K vectors | ~5 ms | **0.78 ms** | **6.4× under** |
| L2 | insert throughput | — | **659 ops/sec** | atomic dual-store |

```bash
python bench/l1_baseline.py    # L1 working context
python bench/l2_baseline.py    # L2 hot vector cache
python bench/hnswlib_baseline.py  # raw reference floor
```

---

## Why memory is the unglamorous bottleneck

On-device agents today usually pick one of two bad options:

| Trap A · Flat RAG | Trap B · Sliding Window |
| :--- | :--- |
| One giant vector blob. Every query scans every memory. The longer you use it, the slower it gets — and your phone heats up to remind you. | Forget aggressively, keep the last N turns. Throws away your routines, your projects, your preferences. Wakes up a stranger every morning. |

**Neuro-Paging is a third option.** A memory subsystem, not a RAG wrapper.

---

## The three prongs

### 1. Tiered Memory Architecture *(the filing system)* — ✅ shipped
Three tiers modeled on CPU caches:
- **L1** — working context, in-memory FIFO, 32 KB, measured **1.42 µs** insert
- **L2** — hot vector cache, HNSW + SQLite metadata, 8 MB, measured **0.78 ms** query @ 10K
- **L3** — archive vault, HNSW + SQLite, 128 MB (PQ-int8 compression in Sprint 2)

Each tier has a hard byte budget. Overflow cascades down automatically:
L1 → L2 → L3. Nothing unbounded. A background pruner demotes cold L2
memories to L3 on an idle, battery-aware schedule.

### 2. Context-Aware Routing *(the brain)* — 🚧 Sprint 3
Retrieval shouldn't just match words. The design scores every memory as:
Score(m | q, c) = α·cos(e_m, e_q) + β·ctxSim(tags_m, c) + γ·log(1 + freq_m)·decay(Δt)

Weights `α, β, γ` learned per user from implicit feedback. The substrate
already exposes a `Scorer` plug-in protocol; the learned scorer slots in
without touching the tiers.

### 3. Pruning & Consolidation *(the housekeeper)* — pruner ✅, consolidator 🚧
The **pruner is shipped**: a power-gated background daemon that demotes
stale L2 vectors to L3 (skips work on low battery / active foreground app /
non-idle device). The **consolidator** (cluster + summarize stale memories
into dense concepts) lands Sprint 2.

### Standout: Predictive Prefetching — 🚧 Sprint 3
FP-Growth pattern mining over `(time, app, topic)` tuples to detect routines
(Mon 9am IDE → "project status?"). When confidence > 0.7, warm the answer
into L1 before the user hits send. The substrate's `promote()` API is the
hook this will call.

---

## Architecture
┌─────────────────┐     ┌──────────────────────┐     ┌──────────────┐
│  User prompt    │────▶│  Context-Aware       │────▶│  On-device   │
│  + sensors      │     │  Router  (Sprint 3)  │     │  LLM         │
│  (time, app,    │     │                      │     │  Qwen2.5-1.5B│
│   loc, battery) │     │  α·R + β·C + γ·F      │     │  INT4        │
└─────────────────┘     └──────────────────────┘     └──────────────┘
│
┌────────────────────┼────────────────────┐
▼                    ▼                    ▼
┌──────────────┐     ┌──────────────┐     ┌──────────────┐
│  L1 Working  │     │  L2 Hot      │     │  L3 Archive  │
│  ~32 KB      │     │  ~8 MB       │     │  128 MB      │
│  1.42 µs ✅  │     │  0.78 ms ✅  │     │  HNSW ✅     │
│              │     │  HNSW+SQLite │     │  PQ-int8 🚧  │
└──────────────┘     └──────────────┘     └──────────────┘
▲  cascade L1→L2→L3 on overflow (shipped) ▲
└────────────────────┬────────────────────┘
│
┌────────────────────┴────────────────────┐
│  Background daemons                      │
│  · Pruner (L2→L3, power-gated)     ✅    │
│  · Consolidator (cluster+summary)   🚧   │
│  · Pattern learner (FP-Growth)      🚧   │
│  · Predictive prefetcher            🚧   │
└──────────────────────────────────────────┘

---

## What makes us different

The design goals vs prior art. ✅ = shipped, 🚧 = roadmap.

|  | Flat RAG | MemGPT | Letta | **Neuro-Paging** |
| :--- | :---: | :---: | :---: | :---: |
| Tiered hierarchy + promote/demote + budgets | ❌ flat | main/archival | block-based | ✅ **L1/L2/L3 shipped** |
| Atomic dual-store (vectors + metadata) | ❌ | ❌ | ❌ | ✅ **shipped + stress-tested** |
| Power-gated background pruning | ❌ | ❌ | ❌ | ✅ **shipped** |
| Retrieval weighted by sensor context | ❌ | ❌ | ❌ | 🚧 Sprint 3 |
| Predictive prefetch from routines | ❌ | ❌ | ❌ | 🚧 Sprint 3 |
| Native C++ retrieval engine | Python | Python | Python | 🚧 Sprint 3 |
| 100% on-device, zero cloud calls | depends | cloud LLM | supports local | ✅ **designed for it** |

---

## Testing

The substrate ships with **169 tests**, all passing, CI-enforced.

Notable: **9 concurrency stress tests** hammer L1/L2/L3 and the pruner
daemon from up to **100 threads simultaneously**, verifying:
- dual-store atomicity (HNSW index + SQLite metadata never diverge)
- race-free monotonic label assignment under contention
- byte-budget invariants hold under sustained concurrent eviction
- the background pruner runs safely alongside foreground writes

A global `pytest-timeout` (thread method) guards against any regression
that could reintroduce a deadlock — a hang fails fast with a thread dump
instead of wedging CI.

```bash
pytest tests/ -v        # all 169
pytest tests/test_concurrency.py -v   # the 9 stress tests
```

---

## Quick start

### Prerequisites
- Python 3.11+
- macOS (Apple Silicon recommended) or Linux ARM64
- 4 GB free disk for models (only needed for the ML runtime)

### Install

```bash
git clone https://github.com/Ajay03299/Neuro-Paging.git
cd Neuro-Paging

uv venv && source .venv/bin/activate
uv pip install -e ".[dev]"

# Verify
python -c "import neuro_paging; print('✅', neuro_paging.__version__)"
pytest tests/ -q
```

### 30-second taste

```python
from neuro_paging import MemoryManager, ContextTags

mgr = MemoryManager(data_dir="./my-memory")

# Store memories — they land in L1, cascade to L2/L3 as it fills
mgr.insert("User prefers Italian food for dinner", ContextTags.now())
mgr.insert("Standup is at 9am every weekday", ContextTags.now())

# Retrieve top-k across all tiers
hits = mgr.query("what does the user like to eat?", ContextTags.now(), k=3)
for h in hits:
    print(h.score, h.text)

mgr.close()  # persists L2 + L3 to disk
```

---

## Repository layout
Neuro-Paging/
├── src/neuro_paging/
│   ├── memory/            # L1/L2/L3 tiers + manager        (Ajay) ✅
│   ├── daemons/           # pruner (consolidator runner WIP) (Ajay) ✅
│   ├── context/           # context tags, sensor types      (shared)
│   ├── routing/           # scorer, online learner          (Christine) 🚧
│   ├── embed/             # bge-small ONNX runtime           (Christine) 🚧
│   ├── llm/               # Qwen llama.cpp runtime           (Christine) 🚧
│   ├── consolidator/      # clustering + summarisation       (Christine) 🚧
│   └── prefetch/          # FP-Growth + prefetcher           (Christine) 🚧
├── npaged-core/           # C++17 native retrieval core      (Ajay) 🚧
├── bench/                 # latency / throughput benchmarks  ✅
├── eval/                  # LongMemEval, LoCoMo, MSC         🚧
├── ui/                    # Streamlit live tier dashboard    🚧
├── tests/                 # 169 tests incl. concurrency      ✅
├── notes/                 # baselines, design docs, decisions ✅
├── BENCHMARKS.md          # consolidated measured results    ✅
└── agents.md              # agent contracts, tool schemas

---

## Roadmap

Phase 2 submission target: **June 22, 2026**.

| Sprint | Dates | Headline deliverable | Status |
| :--- | :--- | :--- | :---: |
| 0 | May 19–22 | Scaffold, baselines, locked API contract | ✅ |
| 1 | May 23–29 | Three tiers end-to-end, pruner, concurrency tests | ✅ |
| 2 | May 30–Jun 5 | PQ-int8 for L3, consolidator daemon, dashboard | 🚧 |
| 3 | Jun 6–12 | C++ native core, predictive prefetch, online learning | ⬜ |
| 4 | Jun 13–19 | Full benchmarks (LongMemEval/LoCoMo/MSC), MobileMem-Bench | ⬜ |
| 5 | Jun 20–22 | Code freeze, demo video, submission | ⬜ |

---

## Open datasets & models

**Benchmark datasets:**
- **[LongMemEval](https://huggingface.co/datasets/xiaowu0162/longmemeval)** (MIT) — 500 long-conversation tasks across 5 memory abilities
- **[LoCoMo](https://huggingface.co/datasets/snap-stanford/locomo)** (CC-BY) — multi-session conversations, ~9K turns each
- **[MSC · Multi-Session Chat](https://parl.ai/projects/msc/)** (CC-BY) — persona-grounded sessions across days
- **[PG-19](https://github.com/google-deepmind/pg19)** (Apache-2.0) — long-form English for PQ stress-testing

**Models (open-weight, locally executable):**
- **[Qwen2.5-1.5B-Instruct](https://huggingface.co/Qwen/Qwen2.5-1.5B-Instruct)** (Apache-2.0) — reasoning + summarisation
- **[bge-small-en-v1.5](https://huggingface.co/BAAI/bge-small-en-v1.5)** (MIT) — embeddings
- **[Llama-3.2-1B](https://huggingface.co/meta-llama/Llama-3.2-1B)** (LLAMA-3 Community) — fallback for low-end devices

**Planned releases:**
- **MobileMem-Bench** — ~5–10K synthetic mobile-agent dialogues with device context. CC-BY 4.0 on Hugging Face.
- **Qwen2.5-1.5B-Consolidator** — LoRA-tuned for cluster→concept distillation. Apache-2.0.

---

## Team

| | Name | Role | Owns |
| :---: | :--- | :--- | :--- |
| 👤 | **Ajay Javali** | Systems Layer | L1/L2/L3 tiers, C++ native core, daemons, benchmarks |
| 👤 | **Christine R** | Intelligence Layer | Routing, context, embed/LLM, prefetch, consolidator |

Both: IIIT-Bangalore, B.Tech CSE, 2nd Year. Team **ByteMe**.

---

## AI-assisted development disclosure

Per Samsung ennovateX rules, all AI-assisted development is disclosed.
Tools used during development: **Cursor**, **Claude Code** (paired-coding
agents), **aider** (repo-aware refactors), **Continue.dev** (IDE assistant).

No closed APIs are used at runtime. Every dependency is permissively
licensed and locally executable.

---

## License

Apache License 2.0 — see [LICENSE](./LICENSE).
Released datasets will be CC-BY 4.0; released models Apache-2.0.

---

<div align="center">

Built for **Samsung R&D Institute India–Bangalore** · ennovateX AX Hackathon 2026.
*Memory is the unglamorous bottleneck. We're making it fast.*

</div>