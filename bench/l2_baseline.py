"""L2 baseline — measure throughput + latency of L2HotVectorCache.

Why this exists
---------------
The deck claims L2 latency ~5ms p50 at ~10K hot vectors. The raw
hnswlib baseline already showed 0.356ms p95 on the same machine,
but that was the bare library. L2 adds:
  - SQLite metadata writes per insert
  - Atomic dual-store coordination (metadata first, HNSW second)
  - Eviction logic (cold-finder query, mark_deleted, delete row)
  - Tier metadata bookkeeping

This benchmark measures the FULL stack — wrapper overhead included.
Numbers replace the deck's projection with what L2 actually does.

Three regimes
-------------
  1. Steady-state inserts at low utilization (no eviction)
  2. Inserts at the byte-budget ceiling (sustained eviction)
  3. Query latency at corpus sizes: 1K, 10K, 50K

Run
---
    python bench/l2_baseline.py

Output
------
    bench/results/l2_baseline_<timestamp>.json
    notes/l2_baseline.md  (regenerated, human-readable)
"""

from __future__ import annotations

import json
import platform
import shutil
import statistics
import tempfile
import time
import uuid
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
from rich.console import Console
from rich.progress import BarColumn, Progress, SpinnerColumn, TextColumn, TimeElapsedColumn
from rich.table import Table

from neuro_paging import ContextTags, TimeBucket
from neuro_paging.memory.l2_cache import L2HotVectorCache
from neuro_paging.memory.types import Memory, MemoryId, Tier

# ── Config ────────────────────────────────────────────────────────────────────

# Vector dim — matches L2's default + bge-small-en-v1.5
DIM = 384

# Corpus sizes for the query-latency sweep
QUERY_CORPUS_SIZES = [1_000, 10_000, 50_000]

# How many queries to measure per corpus size
NUM_QUERIES = 500

# How many inserts to do in the steady-state and pressure regimes
INSERTS_STEADY = 5_000
INSERTS_PRESSURE = 5_000

# Top-k
K = 5

# L2 capacities
STEADY_CAPACITY_BYTES = 64 * 1024 * 1024  # 64 MB — no eviction expected
PRESSURE_CAPACITY_BYTES = 2 * 1024 * 1024  # 2 MB — sustained eviction
QUERY_CAPACITY_BYTES = 256 * 1024 * 1024  # 256 MB — fits all corpus sizes

# Reproducibility
SEED = 42

# ── Output paths ──────────────────────────────────────────────────────────────
REPO_ROOT = Path(__file__).resolve().parent.parent
RESULTS_DIR = REPO_ROOT / "bench" / "results"
NOTES_PATH = REPO_ROOT / "notes" / "l2_baseline.md"

console = Console()


# ── Helpers ───────────────────────────────────────────────────────────────────


def percentile(data: list[float], pct: float) -> float:
    if not data:
        return float("nan")
    s = sorted(data)
    idx = int(round(pct / 100.0 * (len(s) - 1)))
    return s[idx]


def make_memory(text: str, rng: np.random.Generator) -> tuple[Memory, np.ndarray]:
    """Build a memory + a unit-norm random embedding."""
    mid = MemoryId(str(uuid.uuid4()))
    now = datetime.now(UTC)
    mem = Memory(
        id=mid,
        text=text,
        embedding_ref=f"emb:{mid}",
        context=ContextTags.now(time_bucket=TimeBucket.EVENING),
        tier=Tier.L2,
        created_at=now,
        last_touch=now,
        access_count=0,
        is_consolidated=False,
    )
    vec = rng.standard_normal(DIM).astype(np.float32)
    vec /= np.linalg.norm(vec) + 1e-8
    return mem, vec


# ── Benchmark routines ────────────────────────────────────────────────────────


def bench_steady_insert(data_dir: Path, n: int) -> dict:
    """N inserts at low utilization. Measures the no-eviction insert path."""
    rng = np.random.default_rng(SEED)
    cache = L2HotVectorCache(
        data_dir=data_dir,
        capacity_bytes=STEADY_CAPACITY_BYTES,
        max_elements=max(n * 2, 100),
        dim=DIM,
    )

    latencies_us: list[float] = []
    total_evicted = 0
    start_total = time.perf_counter()
    for i in range(n):
        mem, vec = make_memory(f"steady memory {i}", rng)
        t0 = time.perf_counter()
        evicted = cache.insert(mem, vec)
        latencies_us.append((time.perf_counter() - t0) * 1_000_000)
        total_evicted += len(evicted)
    elapsed_total = time.perf_counter() - start_total

    stats = cache.stats()
    cache.close()

    return {
        "inserts": n,
        "elapsed_seconds": round(elapsed_total, 4),
        "throughput_ops_per_sec": round(n / elapsed_total, 1),
        "evictions": total_evicted,
        "final_count": stats.count,
        "p50_us": round(percentile(latencies_us, 50), 2),
        "p95_us": round(percentile(latencies_us, 95), 2),
        "p99_us": round(percentile(latencies_us, 99), 2),
        "mean_us": round(statistics.mean(latencies_us), 2),
    }


def bench_pressure_insert(data_dir: Path, n: int) -> dict:
    """N inserts at a tight byte budget. Measures the eviction-heavy path."""
    rng = np.random.default_rng(SEED + 1)
    cache = L2HotVectorCache(
        data_dir=data_dir,
        capacity_bytes=PRESSURE_CAPACITY_BYTES,
        max_elements=max(n // 4, 100),
        dim=DIM,
    )

    latencies_us: list[float] = []
    total_evicted = 0
    start_total = time.perf_counter()
    for i in range(n):
        mem, vec = make_memory(f"pressure memory text content {i}", rng)
        t0 = time.perf_counter()
        evicted = cache.insert(mem, vec)
        latencies_us.append((time.perf_counter() - t0) * 1_000_000)
        total_evicted += len(evicted)
    elapsed_total = time.perf_counter() - start_total

    stats = cache.stats()
    cache.close()

    return {
        "inserts": n,
        "elapsed_seconds": round(elapsed_total, 4),
        "throughput_ops_per_sec": round(n / elapsed_total, 1),
        "evictions": total_evicted,
        "final_count": stats.count,
        "bytes_used": stats.bytes_estimate,
        "capacity_bytes": stats.capacity_bytes,
        "p50_us": round(percentile(latencies_us, 50), 2),
        "p95_us": round(percentile(latencies_us, 95), 2),
        "p99_us": round(percentile(latencies_us, 99), 2),
        "mean_us": round(statistics.mean(latencies_us), 2),
    }


def bench_query_latency(data_dir: Path, corpus_size: int) -> dict:
    """Build an L2 of size N, then measure query latency over NUM_QUERIES."""
    rng = np.random.default_rng(SEED + 2)

    cache = L2HotVectorCache(
        data_dir=data_dir,
        capacity_bytes=QUERY_CAPACITY_BYTES,
        max_elements=corpus_size * 2,
        dim=DIM,
    )

    # Build corpus
    for i in range(corpus_size):
        mem, vec = make_memory(f"corpus memory {i}", rng)
        cache.insert(mem, vec)

    # Warm-up
    for _ in range(10):
        q = rng.standard_normal(DIM).astype(np.float32)
        q /= np.linalg.norm(q) + 1e-8
        cache.query(q, k=K)

    # Measure
    latencies_us: list[float] = []
    for _ in range(NUM_QUERIES):
        q = rng.standard_normal(DIM).astype(np.float32)
        q /= np.linalg.norm(q) + 1e-8
        t0 = time.perf_counter()
        cache.query(q, k=K)
        latencies_us.append((time.perf_counter() - t0) * 1_000_000)

    cache.close()

    return {
        "corpus_size": corpus_size,
        "queries": NUM_QUERIES,
        "p50_us": round(percentile(latencies_us, 50), 2),
        "p95_us": round(percentile(latencies_us, 95), 2),
        "p99_us": round(percentile(latencies_us, 99), 2),
        "mean_us": round(statistics.mean(latencies_us), 2),
        "max_us": round(max(latencies_us), 2),
    }


# ── System info ──────────────────────────────────────────────────────────────


def collect_system_info() -> dict:
    import neuro_paging

    return {
        "platform": platform.platform(),
        "machine": platform.machine(),
        "processor": platform.processor() or platform.machine(),
        "python_version": platform.python_version(),
        "neuro_paging_version": neuro_paging.__version__,
        "timestamp_utc": datetime.now(UTC).isoformat(),
    }


# ── Rendering ────────────────────────────────────────────────────────────────


def render_insert_table(steady: dict, pressure: dict) -> Table:
    table = Table(title="L2 baseline · insert performance")
    table.add_column("Regime", style="cyan")
    table.add_column("Inserts", justify="right")
    table.add_column("Throughput (ops/s)", justify="right", style="green")
    table.add_column("Evictions", justify="right", style="yellow")
    table.add_column("p50 (µs)", justify="right")
    table.add_column("p95 (µs)", justify="right", style="yellow")
    table.add_column("p99 (µs)", justify="right", style="red")

    for label, r in [("Steady (no eviction)", steady), ("Pressure (eviction)", pressure)]:
        table.add_row(
            label,
            f"{r['inserts']:,}",
            f"{r['throughput_ops_per_sec']:,.0f}",
            f"{r['evictions']:,}",
            f"{r['p50_us']:.1f}",
            f"{r['p95_us']:.1f}",
            f"{r['p99_us']:.1f}",
        )
    return table


def render_query_table(results: list[dict]) -> Table:
    table = Table(title="L2 baseline · query latency (top-K=5)")
    table.add_column("Corpus", justify="right", style="cyan")
    table.add_column("Queries", justify="right")
    table.add_column("p50 (µs)", justify="right")
    table.add_column("p95 (µs)", justify="right", style="yellow")
    table.add_column("p99 (µs)", justify="right", style="red")
    table.add_column("Max (µs)", justify="right", style="dim")

    for r in results:
        table.add_row(
            f"{r['corpus_size']:,}",
            f"{r['queries']:,}",
            f"{r['p50_us']:.1f}",
            f"{r['p95_us']:.1f}",
            f"{r['p99_us']:.1f}",
            f"{r['max_us']:.1f}",
        )
    return table


def write_markdown_report(payload: dict) -> None:
    sys_info = payload["system"]
    lines: list[str] = []
    lines.append("# L2 baseline")
    lines.append("")
    lines.append("> Auto-generated by `bench/l2_baseline.py`. Do not edit by hand.")
    lines.append("> Re-run the script to refresh.")
    lines.append("")
    lines.append("## System")
    lines.append("")
    lines.append(f"- **Platform:** {sys_info['platform']}")
    lines.append(f"- **Machine:** {sys_info['machine']}")
    lines.append(f"- **Python:** {sys_info['python_version']}")
    lines.append(f"- **neuro-paging:** {sys_info['neuro_paging_version']}")
    lines.append(f"- **Measured at:** {sys_info['timestamp_utc']}")
    lines.append("")
    lines.append("## Insert performance")
    lines.append("")
    lines.append("Two regimes: low-utilization steady-state, and tight-budget eviction-heavy.")
    lines.append("")
    lines.append(
        "| Regime | Inserts | Throughput (ops/s) | Evictions | p50 (µs) | p95 (µs) | p99 (µs) |"
    )
    lines.append("| :--- | ---: | ---: | ---: | ---: | ---: | ---: |")
    s = payload["steady_insert"]
    p = payload["pressure_insert"]
    lines.append(
        f"| Steady (no eviction) | {s['inserts']:,} "
        f"| {s['throughput_ops_per_sec']:,.0f} "
        f"| {s['evictions']:,} "
        f"| {s['p50_us']:.1f} | {s['p95_us']:.1f} | {s['p99_us']:.1f} |"
    )
    lines.append(
        f"| Pressure (eviction) | {p['inserts']:,} "
        f"| {p['throughput_ops_per_sec']:,.0f} "
        f"| {p['evictions']:,} "
        f"| {p['p50_us']:.1f} | {p['p95_us']:.1f} | {p['p99_us']:.1f} |"
    )
    lines.append("")
    lines.append("## Query latency (top-K = 5)")
    lines.append("")
    lines.append("| Corpus | Queries | p50 (µs) | p95 (µs) | p99 (µs) | Max (µs) |")
    lines.append("| ---: | ---: | ---: | ---: | ---: | ---: |")
    for r in payload["query_latency"]:
        lines.append(
            f"| {r['corpus_size']:,} | {r['queries']:,} "
            f"| {r['p50_us']:.1f} | {r['p95_us']:.1f} "
            f"| {r['p99_us']:.1f} | {r['max_us']:.1f} |"
        )
    lines.append("")
    lines.append("## Interpretation")
    lines.append("")
    lines.append("Deck claim: L2 latency **~5 ms p50** at ~10K hot vectors.")
    lines.append("")

    # Find 10K result for the headline
    target = next(
        (r for r in payload["query_latency"] if r["corpus_size"] == 10_000),
        None,
    )
    if target:
        p50_us = target["p50_us"]
        p50_ms = p50_us / 1000.0
        lines.append(
            f"Measured p50 query latency at 10K vectors = **{p50_us:.1f} µs** ({p50_ms:.3f} ms)."
        )
        if p50_ms < 5.0:
            ratio = 5.0 / p50_ms if p50_ms > 0 else float("inf")
            lines.append(f"✅ **{ratio:.0f}× under the deck's 5 ms budget.**")
        else:
            lines.append("⚠️ Above 5 ms — investigate wrapper overhead.")
    lines.append("")
    lines.append("Wrapper overhead vs raw hnswlib:")
    lines.append("")
    lines.append("Raw hnswlib at 10K queried p95 = 0.356 ms (from `notes/hnswlib_baseline.md`).")
    if target:
        wrapper_overhead_pct = (
            (target["p95_us"] / 1000.0 / 0.356 - 1) * 100 if target["p95_us"] > 0 else 0
        )
        lines.append(
            f"L2 p95 at 10K = {target['p95_us'] / 1000:.3f} ms. "
            f"Wrapper overhead = **+{wrapper_overhead_pct:.0f}%** vs raw hnswlib."
        )
    lines.append("")
    lines.append("## Next")
    lines.append("")
    lines.append("- Sprint 2: re-measure after PQ-int8 compression lands (L3)")
    lines.append("- Sprint 3: re-measure with C++ binding for the scoring path")
    lines.append("")

    NOTES_PATH.write_text("\n".join(lines))


# ── Main ──────────────────────────────────────────────────────────────────────


def main() -> None:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    NOTES_PATH.parent.mkdir(parents=True, exist_ok=True)

    console.rule("[bold cyan]L2 baseline")
    console.print(f"Vector dim:         [bold]{DIM}[/bold]")
    console.print(f"Inserts (steady):   [bold]{INSERTS_STEADY:,}[/bold]")
    console.print(f"Inserts (pressure): [bold]{INSERTS_PRESSURE:,}[/bold]")
    console.print(f"Query corpus sizes: {QUERY_CORPUS_SIZES}")
    console.print(f"Queries per corpus: [bold]{NUM_QUERIES:,}[/bold]")
    console.print()

    # Use a fresh temp directory per regime so on-disk state doesn't bleed
    tmp_root = Path(tempfile.mkdtemp(prefix="np_l2_bench_"))

    try:
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TimeElapsedColumn(),
            console=console,
        ) as progress:
            task = progress.add_task("Benchmarking…", total=2 + len(QUERY_CORPUS_SIZES))

            progress.update(task, description="Steady-state insert")
            steady = bench_steady_insert(tmp_root / "steady", INSERTS_STEADY)
            progress.advance(task)

            progress.update(task, description="Pressure insert (eviction)")
            pressure = bench_pressure_insert(tmp_root / "pressure", INSERTS_PRESSURE)
            progress.advance(task)

            query_results = []
            for n in QUERY_CORPUS_SIZES:
                progress.update(task, description=f"Query latency · corpus={n:,}")
                qr = bench_query_latency(tmp_root / f"query_{n}", n)
                query_results.append(qr)
                progress.advance(task)
    finally:
        shutil.rmtree(tmp_root, ignore_errors=True)

    console.print()
    console.print(render_insert_table(steady, pressure))
    console.print()
    console.print(render_query_table(query_results))
    console.print()

    payload = {
        "system": collect_system_info(),
        "config": {
            "dim": DIM,
            "inserts_steady": INSERTS_STEADY,
            "inserts_pressure": INSERTS_PRESSURE,
            "query_corpus_sizes": QUERY_CORPUS_SIZES,
            "num_queries": NUM_QUERIES,
            "k": K,
            "steady_capacity_bytes": STEADY_CAPACITY_BYTES,
            "pressure_capacity_bytes": PRESSURE_CAPACITY_BYTES,
            "query_capacity_bytes": QUERY_CAPACITY_BYTES,
            "seed": SEED,
        },
        "steady_insert": steady,
        "pressure_insert": pressure,
        "query_latency": query_results,
    }

    timestamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    json_path = RESULTS_DIR / f"l2_baseline_{timestamp}.json"
    json_path.write_text(json.dumps(payload, indent=2))
    console.print(f"📊 JSON:     [dim]{json_path.relative_to(REPO_ROOT)}[/dim]")

    write_markdown_report(payload)
    console.print(f"📝 Markdown: [dim]{NOTES_PATH.relative_to(REPO_ROOT)}[/dim]")
    console.print()
    console.print("[bold green]Done.[/bold green] L2 throughput + latency are now measured.")


if __name__ == "__main__":
    main()
