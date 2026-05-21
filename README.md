<div align="center">

# 🧠 Neuro-Paging

**An operating system for AI memory.**

*Tiered. Context-aware. Quietly predictive. 100% on-device.*

[![License: Apache 2.0](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)
[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)
[![Samsung ennovateX](https://img.shields.io/badge/Samsung-ennovateX_2026-1428a0.svg)](https://ennovatex.io/ax-hackathon/)
[![Status](https://img.shields.io/badge/status-Phase_2_in_progress-orange.svg)](#roadmap)
[![CI](https://github.com/Ajay03299/Neuro-Paging/actions/workflows/ci.yml/badge.svg)](https://github.com/Ajay03299/Neuro-Paging/actions/workflows/ci.yml)

</div>

---

## The pitch

> Treat AI memory the way an OS treats RAM. Page the right thoughts to the front of mind, archive the rest, and quietly learn the user's rhythms — so the pages worth keeping warm are already loaded before they ever hit send.

Built for **Samsung ennovateX AX Hackathon 2026 · Problem Statement #03 · Context-Aware Adaptive Memory for Mobile Agentic Systems** by team **ByteMe**.

---

## Why memory is the unglamorous bottleneck

On-device agents today usually pick one of two bad options:

| Trap A · Flat RAG | Trap B · Sliding Window |
| :--- | :--- |
| One giant vector blob. Every query scans every memory. The longer you use it, the slower it gets — and your phone heats up to remind you. | Forget aggressively, keep the last N turns. Throws away your routines, your projects, your preferences. Wakes up a stranger every morning. |

**Neuro-Paging is a third option.** A memory subsystem, not a RAG wrapper.

---

## The three prongs

### 1. Tiered Memory Architecture *(the filing system)*
Three tiers modeled on CPU caches: **L1** (working context, RAM, <1 ms), **L2** (hot vector cache, mmap, ~5 ms), **L3** (archive vault, disk, ~30 ms). Each tier has a hard size and lifetime budget. Nothing unbounded.

### 2. Context-Aware Routing *(the brain)*
Retrieval shouldn't just match words. Every memory scores against:
Score(m | q, c) = α·cos(e_m, e_q) + β·ctxSim(tags_m, c) + γ·log(1 + freq_m)·decay(Δt)

Weights `α, β, γ` are learned per user from implicit feedback. A "remind-me" user leans on `β`. A heavy researcher leans on `α`.

### 3. Pruning & Consolidation *(the housekeeper)*
A nightly background pass turns stale L2 vectors into dense L3 concepts — what your brain does while you sleep. Storage stays bounded; nothing worth keeping disappears.

### Standout: Predictive Prefetching
FP-Growth pattern mining over `(time, app, topic)` tuples detects routines (Mon 9am IDE → "project status?"). When confidence > 0.7, the answer is already in L1 **before the user hits send**.

---

## Architecture
┌─────────────────┐     ┌──────────────────────┐     ┌──────────────┐
│  User prompt    │────▶│  Context-Aware       │────▶│  On-device   │
│  + sensors      │     │  Router              │     │  LLM         │
│  (time, app,    │     │                      │     │  Qwen2.5-1.5B│
│   loc, battery) │     │  α·R + β·C + γ·F     │     │  INT4        │
└─────────────────┘     └──────────────────────┘     └──────────────┘
│
┌─────────────────┼─────────────────┐
▼                 ▼                 ▼
┌──────────────┐  ┌──────────────┐  ┌──────────────┐
│  L1 Working  │  │  L2 Hot      │  │  L3 Archive  │
│  ~32 KB      │  │  ~8 MB       │  │  100 MB+     │
│  <1 ms       │  │  ~5 ms HNSW  │  │  PQ-int8     │
└──────────────┘  └──────────────┘  └──────────────┘
▲
┌─────────────────┴─────────────────┐
│  Background daemons               │
│  · Pruner (L2→L3 nightly)         │
│  · Consolidator (cluster+summary) │
│  · Pattern learner (FP-Growth)    │
│  · Predictive prefetcher          │
└───────────────────────────────────┘

Full system diagram and tier specifications: see [`docs/architecture.md`](./docs/architecture.md) *(coming soon)*.

---

## What makes us different

|  | Flat RAG | MemGPT | Letta | **Neuro-Paging** |
| :--- | :---: | :---: | :---: | :---: |
| Tiered hierarchy with promote/demote | ❌ flat | main/archival only | block-based | ✅ **L1/L2/L3 + budgets** |
| Retrieval weighted by sensor context | ❌ | ❌ | ❌ | ✅ **time × loc × app × battery** |
| Predictive prefetch from routines | ❌ | ❌ | ❌ | ✅ **FP-Growth + confidence-gated** |
| Native C++ retrieval engine | Python | Python | Python | ✅ **C++17 + ARM NEON** |
| 100% on-device, zero cloud calls | depends | cloud LLM expected | supports local | ✅ **designed for it** |
| Storage stays bounded with usage | grows linearly | archival grows | unclear | ✅ **consolidation + PQ-int8** |

---

## Quick start

> **Status:** under active development. Phase 2 submission targets June 22, 2026. APIs and benchmarks land sprint-by-sprint — see [Roadmap](#roadmap).

### Prerequisites
- Python 3.11+
- macOS (Apple Silicon recommended) or Linux ARM64
- 4 GB free disk for models

### Install

```bash
# Clone
git clone https://github.com/Ajay03299/Neuro-Paging.git
cd Neuro-Paging

# Set up environment (uv-based)
uv venv && source .venv/bin/activate
uv pip install -e ".[dev,eval,build]"

# Optional: install ML runtime (for the LLM + embedder)
uv pip install -e ".[ml]"

# Verify
python -c "import hnswlib, numpy; print('✅ core deps OK')"
```

---

## Repository layout
Neuro-Paging/
├── src/neuro_paging/      # main package
│   ├── memory/            # L1/L2/L3 tier implementations  (Ajay)
│   ├── routing/           # scorer, online learner          (Christine)
│   ├── context/           # sensor providers                (Christine)
│   ├── embed/             # bge-small ONNX runtime          (Christine)
│   ├── llm/               # Qwen llama.cpp runtime          (Christine)
│   ├── consolidator/      # clustering + summarisation      (Christine)
│   ├── prefetch/          # FP-Growth + prefetcher          (Christine)
│   ├── daemons/           # pruner, consolidator runner     (Ajay)
│   └── pipeline/          # LangGraph orchestration         (joint)
├── npaged-core/           # C++17 native retrieval core     (Ajay)
├── bench/                 # latency, battery, throughput
├── eval/                  # LongMemEval, LoCoMo, MSC, MobileMem-Bench
├── ui/                    # Streamlit live tier dashboard
├── data/                  # fixtures, generated dialogues
├── tests/
├── notes/                 # design docs, baselines, decisions
├── agents.md              # agent contracts, tool schemas, refusal rules
└── pyproject.toml

---

## Roadmap

Phase 2 submission target: **June 22, 2026**. Sprint cadence below.

| Sprint | Dates | Headline deliverable |
| :--- | :--- | :--- |
| 0 | May 19–22 | Repo scaffold, baseline measurements, locked API contract |
| 1 | May 23–29 | Three tiers working end-to-end, live Streamlit dashboard |
| 2 | May 30–Jun 5 | Background daemons (pruner + consolidator), PQ-int8 |
| 3 | Jun 6–12 | C++ native core, predictive prefetch, online learning |
| 4 | Jun 13–19 | Full benchmarks (LongMemEval/LoCoMo/MSC), MobileMem-Bench release, LoRA consolidator |
| 5 | Jun 20–22 | Code freeze, demo video, submission |

---

## Open datasets

We benchmark on:
- **[LongMemEval](https://huggingface.co/datasets/xiaowu0162/longmemeval)** (MIT) — 500 long-conversation tasks across 5 memory abilities
- **[LoCoMo](https://huggingface.co/datasets/snap-stanford/locomo)** (CC-BY) — Multi-session conversations, ~9K turns each
- **[MSC · Multi-Session Chat](https://parl.ai/projects/msc/)** (CC-BY) — Persona-grounded sessions across days
- **[PG-19](https://github.com/google-deepmind/pg19)** (Apache-2.0) — Long-form English for stress-testing PQ compression

We release:
- **MobileMem-Bench** *(planned)* — ~5,000–10,000 synthetic mobile-agent dialogues annotated with device context. CC-BY 4.0 on Hugging Face.

## Open-weight models

- **[Qwen2.5-1.5B-Instruct](https://huggingface.co/Qwen/Qwen2.5-1.5B-Instruct)** (Apache-2.0) — reasoning + summarisation
- **[bge-small-en-v1.5](https://huggingface.co/BAAI/bge-small-en-v1.5)** (MIT) — embeddings
- **[Llama-3.2-1B](https://huggingface.co/meta-llama/Llama-3.2-1B)** (LLAMA-3 Community) — fallback for low-end devices

We release:
- **Qwen2.5-1.5B-Consolidator** *(planned)* — LoRA-tuned for cluster→concept-summary distillation. Apache-2.0.

---

## Team

| | Name | Role | Owns |
| :---: | :--- | :--- | :--- |
| 👤 | **Ajay Javali** | Systems Layer | L1/L2/L3 tiers, C++ native core, daemons, benchmarks, dashboard |
| 👤 | **Christine R** | Intelligence Layer | Routing, context, embed/LLM, prefetch, consolidator, MobileMem-Bench |

Both: IIIT-Bangalore, B.Tech CSE, 2nd Year. Team **ByteMe**.

---

## AI-assisted development disclosure

Per Samsung ennovateX rules, all AI-assisted development is disclosed. Tools used during development:
- **Cursor** & **Claude Code** — paired-coding agents
- **aider** — repo-aware refactors
- **Continue.dev** — on-device IDE assistant

No closed APIs are used at runtime. Every dependency is permissively licensed and locally executable.

---

## License

Apache License 2.0 — see [LICENSE](./LICENSE).
All released datasets are CC-BY 4.0; all released models are Apache-2.0.

---

<div align="center">

Built for **Samsung R&D Institute India–Bangalore** · ennovateX AX Hackathon 2026.
*Memory is the unglamorous bottleneck. We're making it fast.*

</div>