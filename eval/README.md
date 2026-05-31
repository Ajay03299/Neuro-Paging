# Evaluation — LongMemEval retrieval recall@k

This directory contains the retrieval evaluation of Neuro-Paging on
**LongMemEval** (Wu et al., ICLR 2025), the long-term interactive-memory
benchmark for chat assistants.

## Headline

**recall@5 = 0.98** across all five LongMemEval ability types — 50
questions, ~48 chat sessions per question, using bge-small-en-v1.5
embeddings, fully on-device (no cloud, no API).

| metric | value |
| :--- | ---: |
| recall@1  | 0.90 |
| recall@3  | 0.96 |
| **recall@5**  | **0.98** |
| recall@10 | 0.98 |

_50 stride-sampled questions, avg 47.5 sessions/question, bge-small (ONNX)._

## What is measured

Retrieval recall@k: for each question we store the question's chat-session
turns as memories, query with the question text, and check whether a **gold
evidence session** appears among the top-k retrieved memories.

    recall@k = (# questions where a gold session is in top-k) / (# questions)

This is **retrieval-only** — it isolates the job Neuro-Paging is responsible
for (surfacing the right memory) and does not involve answer generation or
LLM judging. These are **not** official LongMemEval accuracy scores; they are
retrieval recall on the LongMemEval haystack, the same protocol used by other
retrieval-only reports on this dataset.

## Methodology

- **Oracle-first validation.** Before trusting the haystack number, the
  harness was validated on the `oracle` split (evidence sessions only, no
  distractors), where recall must be ~1.0 by construction. Oracle
  recall@k = 1.000 across all k confirmed the store → tag → retrieve →
  map-to-session → score pipeline is correct, so any drop on the full
  haystack reflects real retrieval difficulty, not a harness bug.
- **Representative sampling.** The dataset is grouped by question type, so
  taking the first N questions would sample only the easiest category.
  `--limit` stride-samples evenly across the full set so all six ability
  types are represented.
- **Per-question isolation.** Each question builds a fresh index in its own
  temporary directory (cleaned up after), so there is no cross-question
  contamination.
- **Turn-level memory granularity.** Each conversation turn is stored as a
  separate memory tagged with its session id, matching how the system is
  actually used and how prior retrieval evals on this dataset operate. A
  retrieved turn maps back to its session for scoring.
- **Embedder.** bge-small-en-v1.5 via ONNX Runtime; queries use bge's
  query-instruction prefix, passages do not.

## Recall by question type

| ability type | @1 | @3 | @5 | @10 |
| :--- | ---: | ---: | ---: | ---: |
| knowledge-update          | 1.00 | 1.00 | 1.00 | 1.00 |
| single-session-assistant  | 1.00 | 1.00 | 1.00 | 1.00 |
| multi-session             | 0.93 | 1.00 | 1.00 | 1.00 |
| single-session-user       | 0.86 | 1.00 | 1.00 | 1.00 |
| temporal-reasoning        | 0.85 | 0.92 | 0.92 | 0.92 |
| single-session-preference | 0.67 | 0.67 | 1.00 | 1.00 |

> **Per-category n is small.** Stride-sampling 50 questions across 6 types
> leaves only ~3–13 questions per category, so per-type figures are
> directional, not precise (e.g. preference 0.67 = 2 of 3). The overall
> recall@5 = 0.98 over 50 questions is the stable headline.

Two observations the breakdown surfaces:

- **Temporal reasoning is the hard ceiling** — recall plateaus at 0.92 even
  at k=10. Time-relative queries ("the first issue *after* my service")
  require reasoning over *when* things happened, which pure embedding
  similarity can't fully capture. This is exactly where a re-ranking or
  generation layer on top of retrieval would help.
- **Preferences are retrieved but under-ranked at k=1** (0.67 → 1.00 by k=5).
  Preferences are often stated indirectly, so the right memory is found but
  not always ranked first — a known difficulty on this dataset.

## Reproduce

```bash
uv pip install -e ".[dev,ml]"

# 1. Validate the harness on oracle (should be ~1.0)
python eval/longmemeval.py --split oracle --limit 50

# 2. The headline number — stride-sampled across all ability types
python eval/longmemeval.py --split s --limit 50

# Full 500-question run (slow — ~48 sessions/question)
python eval/longmemeval.py --split s
```

The dataset (`xiaowu0162/longmemeval-cleaned`) downloads automatically on
first run to `eval/data/` (gitignored, ~264 MB). Results are written to
`eval/results/` (gitignored).

## Citation

```
@inproceedings{wu2025longmemeval,
  title={LongMemEval: Benchmarking Chat Assistants on Long-Term Interactive Memory},
  author={Wu, Di and others},
  booktitle={ICLR},
  year={2025}
}
```