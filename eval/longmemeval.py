"""LongMemEval recall@k harness for Neuro-Paging.

What this measures
------------------
Retrieval recall@k: for each LongMemEval question, we store the question's
chat sessions as memories, query with the question text, and check whether
a *gold evidence session* appears among the top-k retrieved memories.

    recall@k = (# questions where a gold session is in top-k) / (# questions)

This is a RETRIEVAL-ONLY evaluation (no answer generation / LLM judging) —
it isolates the thing Neuro-Paging is responsible for: surfacing the right
memory. We do not claim these as official "LongMemEval scores"; they are
retrieval recall on the LongMemEval haystack, the same protocol used by
other retrieval-only reports on this dataset.

Methodology note (why we run oracle first)
-------------------------------------------
The `oracle` split contains only evidence sessions (no distractors), so
recall should be ~1.0 — it's a CORRECTNESS CHECK ON THE HARNESS, not the
system. If oracle recall isn't near-perfect, the harness has a bug. Only
once oracle passes do we trust the number from the full `s` haystack
(~48 sessions/question), which is the real retrieval difficulty.

Memory granularity
------------------
We store each TURN as a memory (tagged with its session id), not whole
sessions. This matches how the system is actually used (individual
memories) and how prior retrieval evals on this dataset operate. A
retrieved turn maps back to its session id for scoring.

Usage
-----
    python eval/longmemeval.py --split oracle              # validate harness
    python eval/longmemeval.py --split s --limit 100       # real number, subset
    python eval/longmemeval.py --split s                   # full 500-question run
"""

from __future__ import annotations

import argparse
import json
import time
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path

from huggingface_hub import hf_hub_download

from neuro_paging.context.types import ContextTags
from neuro_paging.embed import BGESmallEmbedder
from neuro_paging.pipeline import MemoryAgent
from neuro_paging.routing import ContextAwareScorer

_REPO = "xiaowu0162/longmemeval-cleaned"
_SPLIT_FILES = {
    "oracle": "longmemeval_oracle.json",
    "s": "longmemeval_s_cleaned.json",
    "m": "longmemeval_m_cleaned.json",
}
_DATA_DIR = Path(__file__).parent / "data"
_RESULTS_DIR = Path(__file__).parent / "results"
_K_VALUES = (1, 3, 5, 10)


@dataclass
class QuestionResult:
    question_id: str
    question_type: str
    gold_session_ids: set[str]
    retrieved_session_ids_at_k: dict[int, list[str]]  # k -> ordered session ids

    def hit_at(self, k: int) -> bool:
        retrieved = set(self.retrieved_session_ids_at_k.get(k, []))
        return bool(retrieved & self.gold_session_ids)


@dataclass
class EvalReport:
    split: str
    n_questions: int
    recall_at_k: dict[int, float]
    recall_by_type: dict[str, dict[int, float]] = field(default_factory=dict)
    elapsed_seconds: float = 0.0
    avg_sessions_per_q: float = 0.0
    embedder: str = "bge-small-en-v1.5 (ONNX)"


def _download_split(split: str) -> Path:
    fname = _SPLIT_FILES[split]
    _DATA_DIR.mkdir(parents=True, exist_ok=True)
    path = _DATA_DIR / fname
    if path.exists():
        return path
    print(f"Downloading {fname} from {_REPO} …")
    dl = hf_hub_download(
        repo_id=_REPO, filename=fname, repo_type="dataset", local_dir=str(_DATA_DIR)
    )
    return Path(dl)


def _turn_text(turn: dict) -> str:
    """Extract storable text from a turn. Prefix with role for clarity."""
    role = turn.get("role", "user")
    content = turn.get("content", "")
    return f"{role}: {content}"


def _eval_question(q: dict, max_k: int) -> QuestionResult:
    """Run one question end-to-end: store its turns, query, record top-k sessions.

    A fresh in-memory agent per question (a clean index per question is the
    standard protocol — no cross-question contamination).
    """
    embedder = BGESmallEmbedder()
    scorer = ContextAwareScorer(embedder=embedder)
    # In-memory: use a unique temp dir per question via None data_dir behaviour.
    agent = MemoryAgent(data_dir=None, embedder=embedder, scorer=scorer)

    ctx = ContextTags.now()

    sessions = q["haystack_sessions"]
    session_ids = q["haystack_session_ids"]

    # Store every turn, tagged with its session id (so a retrieved turn maps
    # back to its session for scoring). We keep a local map memory_id->session.
    mem_to_session: dict[str, str] = {}
    for sess, sid in zip(sessions, session_ids, strict=True):
        for turn in sess:
            text = _turn_text(turn)
            if not text.strip():
                continue
            mid = agent.remember(text, ctx)
            mem_to_session[str(mid)] = sid

    # Query once; ask for the largest k we score at.
    result = agent.recall(q["question"], ctx, k=max_k)

    # For each k, the ordered session ids of the top-k retrieved memories.
    retrieved_at_k: dict[int, list[str]] = {}
    ordered_session_ids: list[str] = []
    for hit in result.hits:
        sid = mem_to_session.get(str(hit.memory_id))
        if sid is not None:
            ordered_session_ids.append(sid)
    for k in _K_VALUES:
        if k > max_k:
            continue
        # top-k memories -> their session ids (dedup preserving order)
        seen: set[str] = set()
        topk: list[str] = []
        for sid in ordered_session_ids[:k]:
            if sid not in seen:
                seen.add(sid)
                topk.append(sid)
        retrieved_at_k[k] = topk

    agent.close()

    return QuestionResult(
        question_id=q["question_id"],
        question_type=q.get("question_type", "unknown"),
        gold_session_ids=set(q["answer_session_ids"]),
        retrieved_session_ids_at_k=retrieved_at_k,
    )


def run_eval(split: str, limit: int | None = None) -> EvalReport:
    path = _download_split(split)
    with open(path) as f:
        data = json.load(f)
    if limit is not None:
        data = data[:limit]

    max_k = max(_K_VALUES)
    results: list[QuestionResult] = []
    total_sessions = 0
    start = time.perf_counter()

    for i, q in enumerate(data, start=1):
        total_sessions += len(q["haystack_sessions"])
        results.append(_eval_question(q, max_k))
        if i % 10 == 0 or i == len(data):
            print(f"  [{i}/{len(data)}] processed", flush=True)

    elapsed = time.perf_counter() - start

    # Aggregate recall@k overall
    recall_at_k: dict[int, float] = {}
    for k in _K_VALUES:
        hits = sum(1 for r in results if r.hit_at(k))
        recall_at_k[k] = hits / len(results) if results else 0.0

    # Recall@k by question type
    by_type: dict[str, list[QuestionResult]] = defaultdict(list)
    for r in results:
        by_type[r.question_type].append(r)
    recall_by_type: dict[str, dict[int, float]] = {}
    for qtype, rs in by_type.items():
        recall_by_type[qtype] = {k: sum(1 for r in rs if r.hit_at(k)) / len(rs) for k in _K_VALUES}

    return EvalReport(
        split=split,
        n_questions=len(results),
        recall_at_k=recall_at_k,
        recall_by_type=recall_by_type,
        elapsed_seconds=elapsed,
        avg_sessions_per_q=total_sessions / len(results) if results else 0.0,
    )


def print_report(report: EvalReport) -> None:
    print("\n" + "=" * 64)
    print(f"  LongMemEval retrieval recall@k — split={report.split}")
    print("=" * 64)
    print(f"  Questions:          {report.n_questions}")
    print(f"  Avg sessions/q:     {report.avg_sessions_per_q:.1f}")
    print(f"  Embedder:           {report.embedder}")
    print(f"  Elapsed:            {report.elapsed_seconds:.1f}s")
    print("-" * 64)
    print("  Overall recall@k:")
    for k in _K_VALUES:
        print(f"    recall@{k:<3} = {report.recall_at_k[k]:.3f}")
    print("-" * 64)
    print("  By question type (recall@k):")
    for qtype in sorted(report.recall_by_type):
        n = "  ".join(f"@{k}={report.recall_by_type[qtype][k]:.2f}" for k in _K_VALUES)
        print(f"    {qtype:<28} {n}")
    print("=" * 64 + "\n")


def save_report(report: EvalReport) -> Path:
    _RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    out = _RESULTS_DIR / f"recall_{report.split}.json"
    payload = {
        "split": report.split,
        "n_questions": report.n_questions,
        "avg_sessions_per_q": report.avg_sessions_per_q,
        "embedder": report.embedder,
        "elapsed_seconds": report.elapsed_seconds,
        "recall_at_k": report.recall_at_k,
        "recall_by_type": report.recall_by_type,
    }
    out.write_text(json.dumps(payload, indent=2))
    print(f"Saved → {out}")
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description="LongMemEval recall@k for Neuro-Paging")
    ap.add_argument("--split", choices=list(_SPLIT_FILES), default="oracle")
    ap.add_argument("--limit", type=int, default=None, help="cap # questions")
    args = ap.parse_args()

    print(f"Running LongMemEval recall@k — split={args.split} limit={args.limit}")
    report = run_eval(args.split, limit=args.limit)
    print_report(report)
    save_report(report)


if __name__ == "__main__":
    main()
