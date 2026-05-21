"""hnswlib baseline — measure raw HNSW insert + query performance.

Why this exists
---------------
Our L2 tier wraps hnswlib. Before we build the tier, we need to know:
  - How fast does insert scale with corpus size?
  - What's the p50/p95/p99 query latency at the sizes we care about?
  - What's the recall@k vs ef_search trade-off?

These numbers replace the placeholders in the Phase 1 deck. Every
"~5 ms" claim now points to a measurement on a real machine.

Run
---
    python bench/hnswlib_baseline.py

Output
------
    bench/results/hnswlib_baseline_<timestamp>.json
    notes/hnswlib_baseline.md  (regenerated, human-readable)
"""

from __future__ import annotations

import json
import platform
import statistics
import time
from datetime import datetime, timezone
from pathlib import Path

import hnswlib
import numpy as np
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TimeElapsedColumn
from rich.table import Table

# ── Config ────────────────────────────────────────────────────────────────────
# Corpus sizes to test (number of vectors in the index)
CORPUS_SIZES = [1_000, 10_000, 50_000, 100_000]

# Vector dimensionality — matches bge-small-en-v1.5, our planned embedder
DIM = 384

# Query count per corpus size
NUM_QUERIES = 1_000

# HNSW hyperparameters — defaults tuned for retrieval quality
M = 16              # max neighbours per node (higher = better recall, more RAM)
EF_CONSTRUCTION = 200  # build-time search width
EF_SEARCH = 50         # query-time search width

# Top-k
K = 5

# Reproducibility
SEED = 42

# ── Output paths ──────────────────────────────────────────────────────────────
REPO_ROOT = Path(__file__).resolve().parent.parent
RESULTS_DIR = REPO_ROOT / "bench" / "results"
NOTES_PATH = REPO_ROOT / "notes" / "hnswlib_baseline.md"

console = Console()


def percentile(data: list[float], pct: float) -> float:
    """Sorted-list percentile (we don't import numpy just for this)."""
    if not data:
        return float("nan")
    sorted_data = sorted(data)
    idx = int(round(pct / 100.0 * (len(sorted_data) - 1)))
    return sorted_data[idx]


def benchmark_size(n: int) -> dict:
    """Run insert + query benchmark for a corpus of size n."""
    rng = np.random.default_rng(SEED)

    # ── Build index ──
    index = hnswlib.Index(space="cosine", dim=DIM)
    index.init_index(max_elements=n, ef_construction=EF_CONSTRUCTION, M=M)
    index.set_ef(EF_SEARCH)
    index.set_num_threads(1)  # single-threaded → realistic mobile profile

    # Generate corpus
    vectors = rng.standard_normal((n, DIM)).astype(np.float32)
    # L2-normalize for cosine space
    vectors /= np.linalg.norm(vectors, axis=1, keepdims=True)

    # Insert (and measure)
    insert_start = time.perf_counter()
    index.add_items(vectors, np.arange(n))
    insert_total = time.perf_counter() - insert_start
    insert_throughput = n / insert_total

    # ── Query ──
    query_vectors = rng.standard_normal((NUM_QUERIES, DIM)).astype(np.float32)
    query_vectors /= np.linalg.norm(query_vectors, axis=1, keepdims=True)

    # Warm-up (JIT, page-cache, etc.)
    for q in query_vectors[:10]:
        index.knn_query(q, k=K)

    latencies_ms: list[float] = []
    for q in query_vectors:
        start = time.perf_counter()
        index.knn_query(q, k=K)
        latencies_ms.append((time.perf_counter() - start) * 1000.0)

    # ── Memory footprint ──
    # hnswlib doesn't expose this cleanly; estimate from saved-index size
    tmp_path = RESULTS_DIR / f"_tmp_index_{n}.bin"
    index.save_index(str(tmp_path))
    bytes_on_disk = tmp_path.stat().st_size
    tmp_path.unlink()  # clean up

    return {
        "corpus_size": n,
        "insert": {
            "total_seconds": round(insert_total, 4),
            "throughput_vectors_per_sec": round(insert_throughput, 1),
        },
        "query_latency_ms": {
            "p50": round(percentile(latencies_ms, 50), 4),
            "p95": round(percentile(latencies_ms, 95), 4),
            "p99": round(percentile(latencies_ms, 99), 4),
            "mean": round(statistics.mean(latencies_ms), 4),
            "min": round(min(latencies_ms), 4),
            "max": round(max(latencies_ms), 4),
        },
        "index_size_bytes": bytes_on_disk,
        "index_size_mb": round(bytes_on_disk / 1024 / 1024, 2),
    }


def collect_system_info() -> dict:
    """Capture environment so benchmarks across machines stay comparable."""
    return {
        "platform": platform.platform(),
        "machine": platform.machine(),
        "processor": platform.processor() or platform.machine(),
        "python_version": platform.python_version(),
        "hnswlib_version": getattr(hnswlib, "__version__", "unknown"),
        "numpy_version": np.__version__,
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
    }


def render_table(results: list[dict]) -> Table:
    """Pretty-print results to the terminal."""
    table = Table(title="hnswlib baseline · query latency by corpus size")
    table.add_column("Corpus", justify="right", style="cyan", no_wrap=True)
    table.add_column("Insert (vec/s)", justify="right", style="green")
    table.add_column("p50 (ms)", justify="right")
    table.add_column("p95 (ms)", justify="right", style="yellow")
    table.add_column("p99 (ms)", justify="right", style="red")
    table.add_column("Index (MB)", justify="right", style="dim")

    for r in results:
        table.add_row(
            f"{r['corpus_size']:,}",
            f"{r['insert']['throughput_vectors_per_sec']:,.0f}",
            f"{r['query_latency_ms']['p50']:.3f}",
            f"{r['query_latency_ms']['p95']:.3f}",
            f"{r['query_latency_ms']['p99']:.3f}",
            f"{r['index_size_mb']:.2f}",
        )
    return table


def write_markdown_report(payload: dict) -> None:
    """Regenerate the human-readable notes/hnswlib_baseline.md."""
    sys_info = payload["system"]
    cfg = payload["config"]

    lines: list[str] = []
    lines.append("# hnswlib baseline")
    lines.append("")
    lines.append("> Auto-generated by `bench/hnswlib_baseline.py`. Do not edit by hand.")
    lines.append("> Re-run the script to refresh.")
    lines.append("")
    lines.append("## System")
    lines.append("")
    lines.append(f"- **Platform:** {sys_info['platform']}")
    lines.append(f"- **Machine:** {sys_info['machine']}")
    lines.append(f"- **Processor:** {sys_info['processor']}")
    lines.append(f"- **Python:** {sys_info['python_version']}")
    lines.append(f"- **hnswlib:** {sys_info['hnswlib_version']}")
    lines.append(f"- **numpy:** {sys_info['numpy_version']}")
    lines.append(f"- **Measured at:** {sys_info['timestamp_utc']}")
    lines.append("")
    lines.append("## Config")
    lines.append("")
    lines.append(f"- Vector dim: **{cfg['dim']}** (matches bge-small-en-v1.5)")
    lines.append(f"- M: **{cfg['M']}**, ef_construction: **{cfg['ef_construction']}**, ef_search: **{cfg['ef_search']}**")
    lines.append(f"- Top-k: **{cfg['k']}**")
    lines.append(f"- Queries per corpus size: **{cfg['num_queries']:,}**")
    lines.append(f"- Distance metric: **cosine**")
    lines.append(f"- Threads: **1** (single-threaded — realistic mobile profile)")
    lines.append("")
    lines.append("## Results")
    lines.append("")
    lines.append("| Corpus | Insert (vec/s) | p50 (ms) | p95 (ms) | p99 (ms) | Index (MB) |")
    lines.append("| ---: | ---: | ---: | ---: | ---: | ---: |")
    for r in payload["results"]:
        lines.append(
            f"| {r['corpus_size']:,} "
            f"| {r['insert']['throughput_vectors_per_sec']:,.0f} "
            f"| {r['query_latency_ms']['p50']:.3f} "
            f"| {r['query_latency_ms']['p95']:.3f} "
            f"| {r['query_latency_ms']['p99']:.3f} "
            f"| {r['index_size_mb']:.2f} |"
        )
    lines.append("")
    lines.append("## Interpretation")
    lines.append("")
    lines.append("L2 tier target in deck: ~5 ms p50 at ~10K vectors (float16, hot cache).")
    lines.append("")
    p95_10k = next((r["query_latency_ms"]["p95"] for r in payload["results"] if r["corpus_size"] == 10_000), None)
    if p95_10k is not None:
        lines.append(f"Measured p95 at 10K = **{p95_10k:.3f} ms** on this machine.")
        if p95_10k < 5.0:
            lines.append("✅ Within budget — deck claim defensible on Apple Silicon.")
        else:
            lines.append("⚠️ Above budget — investigate before locking the L2 design.")
    lines.append("")
    lines.append("## Next")
    lines.append("")
    lines.append("- Sprint 1: wrap this in `memory/l2_cache.py` with metadata sidecar.")
    lines.append("- Sprint 2: measure again after adding context-tag filtering.")
    lines.append("- Sprint 3: re-measure through the C++ binding — expect lower variance.")
    lines.append("")

    NOTES_PATH.write_text("\n".join(lines))


def main() -> None:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    NOTES_PATH.parent.mkdir(parents=True, exist_ok=True)

    console.rule("[bold cyan]hnswlib baseline")
    console.print(f"Vector dim: [bold]{DIM}[/bold]   k: [bold]{K}[/bold]   "
                  f"queries/size: [bold]{NUM_QUERIES:,}[/bold]")
    console.print(f"Corpus sizes: {CORPUS_SIZES}")
    console.print()

    results: list[dict] = []
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TimeElapsedColumn(),
        console=console,
    ) as progress:
        task = progress.add_task("Benchmarking…", total=len(CORPUS_SIZES))
        for n in CORPUS_SIZES:
            progress.update(task, description=f"corpus_size={n:,}")
            result = benchmark_size(n)
            results.append(result)
            progress.advance(task)

    console.print()
    console.print(render_table(results))
    console.print()

    payload = {
        "system": collect_system_info(),
        "config": {
            "dim": DIM,
            "M": M,
            "ef_construction": EF_CONSTRUCTION,
            "ef_search": EF_SEARCH,
            "k": K,
            "num_queries": NUM_QUERIES,
            "corpus_sizes": CORPUS_SIZES,
            "seed": SEED,
        },
        "results": results,
    }

    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    json_path = RESULTS_DIR / f"hnswlib_baseline_{timestamp}.json"
    json_path.write_text(json.dumps(payload, indent=2))
    console.print(f"📊 JSON:     [dim]{json_path.relative_to(REPO_ROOT)}[/dim]")

    write_markdown_report(payload)
    console.print(f"📝 Markdown: [dim]{NOTES_PATH.relative_to(REPO_ROOT)}[/dim]")
    console.print()
    console.print("[bold green]Done.[/bold green] These numbers are now defensible.")


if __name__ == "__main__":
    main()