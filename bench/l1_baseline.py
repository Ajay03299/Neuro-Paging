"""L1 baseline — measure throughput + latency of L1WorkingContext.

Why this exists
---------------
The deck claims L1 latency <1ms. Before relying on that number in the
final write-up, we measure it on the dev machine. These numbers also
inform whether the FIFO + byte-budget design has any unexpected cost
(it shouldn't — OrderedDict ops are O(1) — but we verify).

Three regimes we benchmark:
  1. Inserts with no eviction (L1 below capacity)
  2. Inserts with sustained eviction (L1 at capacity)
  3. touch() throughput (cache-hit refresh path)

Run
---
    python bench/l1_baseline.py

Output
------
    bench/results/l1_baseline_<timestamp>.json
    notes/l1_baseline.md  (regenerated, human-readable)
"""

from __future__ import annotations

import json
import platform
import statistics
import time
import uuid
from datetime import UTC, datetime
from pathlib import Path

from rich.console import Console
from rich.progress import BarColumn, Progress, SpinnerColumn, TextColumn, TimeElapsedColumn
from rich.table import Table

from neuro_paging import ContextTags, TimeBucket
from neuro_paging.memory.l1_working import L1WorkingContext
from neuro_paging.memory.types import Memory, MemoryId, Tier

# ── Config ────────────────────────────────────────────────────────────────────

# Number of inserts per regime
N_INSERTS = 50_000

# Number of touch() ops to measure
N_TOUCHES = 50_000

# L1 capacities to test — small forces eviction; large is the "headroom" case
CAPACITIES = [
    ("32 KB (deck spec)", 32 * 1024),
    ("128 KB (4x headroom)", 128 * 1024),
    ("1 KB (eviction-heavy)", 1024),
]

# Latency samples — we measure per-op latency on a separate sub-run
LATENCY_SAMPLES = 5_000

# Reproducibility
SEED = 42

# ── Output paths ──────────────────────────────────────────────────────────────
REPO_ROOT = Path(__file__).resolve().parent.parent
RESULTS_DIR = REPO_ROOT / "bench" / "results"
NOTES_PATH = REPO_ROOT / "notes" / "l1_baseline.md"

console = Console()


# ── Helpers ───────────────────────────────────────────────────────────────────


def make_memory(text: str = "filler memory text content here") -> Memory:
    """Cheap synthetic memory with deterministic-ish shape."""
    mid = MemoryId(str(uuid.uuid4()))
    now = datetime.now(UTC)
    return Memory(
        id=mid,
        text=text,
        embedding_ref=f"stub-emb:{mid}",
        context=ContextTags.now(time_bucket=TimeBucket.EVENING),
        tier=Tier.L1,
        created_at=now,
        last_touch=now,
        access_count=0,
        is_consolidated=False,
    )


def percentile(data: list[float], pct: float) -> float:
    if not data:
        return float("nan")
    s = sorted(data)
    idx = int(round(pct / 100.0 * (len(s) - 1)))
    return s[idx]


# ── Benchmark routines ────────────────────────────────────────────────────────


def bench_insert_throughput(l1: L1WorkingContext, n: int) -> dict:
    """Insert n memories and report ops/sec + eviction count."""
    memories = [make_memory(f"memory {i} filler") for i in range(n)]

    start = time.perf_counter()
    total_evicted = 0
    for mem in memories:
        evicted = l1.insert(mem)
        total_evicted += len(evicted)
    elapsed = time.perf_counter() - start

    s = l1.stats()
    return {
        "inserts": n,
        "elapsed_seconds": round(elapsed, 4),
        "throughput_ops_per_sec": round(n / elapsed, 1),
        "evictions": total_evicted,
        "final_count": s.count,
        "final_bytes_used": s.bytes_used,
        "capacity_bytes": s.capacity_bytes,
        "utilization": round(s.utilization, 4),
    }


def bench_insert_latency(l1: L1WorkingContext, samples: int) -> dict:
    """Measure per-op insert latency. Warm-up + sample."""
    # Warm-up: fill L1 so we're in steady state (eviction path active)
    for i in range(min(samples, 1000)):
        l1.insert(make_memory(f"warmup {i}"))

    latencies_us: list[float] = []
    for i in range(samples):
        mem = make_memory(f"sample {i}")
        start = time.perf_counter()
        l1.insert(mem)
        latencies_us.append((time.perf_counter() - start) * 1_000_000)  # microseconds

    return {
        "samples": samples,
        "p50_us": round(percentile(latencies_us, 50), 3),
        "p95_us": round(percentile(latencies_us, 95), 3),
        "p99_us": round(percentile(latencies_us, 99), 3),
        "mean_us": round(statistics.mean(latencies_us), 3),
        "max_us": round(max(latencies_us), 3),
    }


def bench_touch_throughput(l1: L1WorkingContext, n: int) -> dict:
    """Pre-populate L1 to natural capacity, then measure touch() throughput.

    We fill L1 until eviction kicks in, then take whichever memories happen
    to be alive in the buffer as our touch targets. This is the realistic
    case — the touch() path is exercised on whatever's currently warm.
    """
    # Fill L1 until it starts evicting, plus a small safety margin
    target_inserts = 200
    for i in range(target_inserts):
        l1.insert(make_memory(f"preload {i}"))

    # Snapshot the currently-alive memory ids. These are guaranteed to be
    # in L1 *right now*. Cycling through them gives 100% hit rate.
    alive_ids = [m.id for m in l1]

    if not alive_ids:
        return {
            "touches": 0,
            "elapsed_seconds": 0.0,
            "throughput_ops_per_sec": 0.0,
            "hit_rate": 0.0,
            "note": "L1 empty after preload",
        }

    # Cycle through alive ids for n touches
    start = time.perf_counter()
    hits = 0
    for i in range(n):
        if l1.touch(alive_ids[i % len(alive_ids)]):
            hits += 1
    elapsed = time.perf_counter() - start

    return {
        "touches": n,
        "elapsed_seconds": round(elapsed, 4),
        "throughput_ops_per_sec": round(n / elapsed, 1),
        "hit_rate": round(hits / n, 4),
        "alive_ids_after_preload": len(alive_ids),
    }


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


def render_throughput_table(results: list[dict]) -> Table:
    table = Table(title="L1 baseline · insert throughput by capacity")
    table.add_column("Capacity", style="cyan")
    table.add_column("Inserts", justify="right")
    table.add_column("Throughput (ops/s)", justify="right", style="green")
    table.add_column("Evictions", justify="right", style="yellow")
    table.add_column("Utilization", justify="right", style="dim")

    for r in results:
        ins = r["insert_throughput"]
        table.add_row(
            r["capacity_label"],
            f"{ins['inserts']:,}",
            f"{ins['throughput_ops_per_sec']:,.0f}",
            f"{ins['evictions']:,}",
            f"{ins['utilization'] * 100:.1f}%",
        )
    return table


def render_latency_table(results: list[dict]) -> Table:
    table = Table(title="L1 baseline · per-op insert latency (microseconds)")
    table.add_column("Capacity", style="cyan")
    table.add_column("p50", justify="right")
    table.add_column("p95", justify="right", style="yellow")
    table.add_column("p99", justify="right", style="red")
    table.add_column("Max", justify="right", style="dim")

    for r in results:
        lat = r["insert_latency"]
        table.add_row(
            r["capacity_label"],
            f"{lat['p50_us']:.2f}",
            f"{lat['p95_us']:.2f}",
            f"{lat['p99_us']:.2f}",
            f"{lat['max_us']:.2f}",
        )
    return table


def write_markdown_report(payload: dict) -> None:
    sys_info = payload["system"]
    lines: list[str] = []
    lines.append("# L1 baseline")
    lines.append("")
    lines.append("> Auto-generated by `bench/l1_baseline.py`. Do not edit by hand.")
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
    lines.append("## Insert throughput")
    lines.append("")
    lines.append("| Capacity | Inserts | Throughput (ops/s) | Evictions | Utilization |")
    lines.append("| :--- | ---: | ---: | ---: | ---: |")
    for r in payload["results"]:
        ins = r["insert_throughput"]
        lines.append(
            f"| {r['capacity_label']} "
            f"| {ins['inserts']:,} "
            f"| {ins['throughput_ops_per_sec']:,.0f} "
            f"| {ins['evictions']:,} "
            f"| {ins['utilization'] * 100:.1f}% |"
        )
    lines.append("")
    lines.append("## Per-op insert latency (microseconds)")
    lines.append("")
    lines.append("| Capacity | p50 | p95 | p99 | Max |")
    lines.append("| :--- | ---: | ---: | ---: | ---: |")
    for r in payload["results"]:
        lat = r["insert_latency"]
        lines.append(
            f"| {r['capacity_label']} "
            f"| {lat['p50_us']:.2f} "
            f"| {lat['p95_us']:.2f} "
            f"| {lat['p99_us']:.2f} "
            f"| {lat['max_us']:.2f} |"
        )
    lines.append("")
    lines.append("## touch() throughput")
    lines.append("")
    lines.append("| Capacity | Touches | Throughput (ops/s) | Hit rate |")
    lines.append("| :--- | ---: | ---: | ---: |")
    for r in payload["results"]:
        t = r["touch_throughput"]
        lines.append(
            f"| {r['capacity_label']} "
            f"| {t['touches']:,} "
            f"| {t['throughput_ops_per_sec']:,.0f} "
            f"| {t.get('hit_rate', 0) * 100:.1f}% |"
        )
    lines.append("")
    lines.append("## Interpretation")
    lines.append("")
    lines.append("Deck claim: L1 latency **< 1 ms** per op.")
    lines.append("")

    # Find the 32 KB result for interpretation
    target = next((r for r in payload["results"] if "32 KB" in r["capacity_label"]), None)
    if target:
        p99_us = target["insert_latency"]["p99_us"]
        p99_ms = p99_us / 1000.0
        lines.append(
            f"Measured p99 insert latency at 32 KB = **{p99_us:.2f} µs** ({p99_ms:.4f} ms)."
        )
        if p99_ms < 1.0:
            lines.append(f"✅ {(1.0 / p99_ms):.0f}× under budget. L1 design is correct.")
        else:
            lines.append("⚠️ Above 1 ms budget — investigate.")
    lines.append("")
    lines.append("## Next")
    lines.append("")
    lines.append("- Sprint 1: re-measure after wiring L1 into the live MemoryManager")
    lines.append("- Sprint 3: re-measure under the C++ binding (router scoring)")
    lines.append("")

    NOTES_PATH.write_text("\n".join(lines))


# ── Main ──────────────────────────────────────────────────────────────────────


def main() -> None:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    NOTES_PATH.parent.mkdir(parents=True, exist_ok=True)

    console.rule("[bold cyan]L1 baseline")
    console.print(f"Inserts per regime: [bold]{N_INSERTS:,}[/bold]")
    console.print(f"Latency samples:    [bold]{LATENCY_SAMPLES:,}[/bold]")
    console.print(f"Touch ops:          [bold]{N_TOUCHES:,}[/bold]")
    console.print()

    results: list[dict] = []
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TimeElapsedColumn(),
        console=console,
    ) as progress:
        task = progress.add_task("Benchmarking…", total=len(CAPACITIES) * 3)

        for label, cap_bytes in CAPACITIES:
            # Fresh L1 per regime
            l1 = L1WorkingContext(capacity_bytes=cap_bytes)
            progress.update(task, description=f"{label}: insert throughput")
            ins = bench_insert_throughput(l1, N_INSERTS)
            progress.advance(task)

            l1_lat = L1WorkingContext(capacity_bytes=cap_bytes)
            progress.update(task, description=f"{label}: insert latency")
            lat = bench_insert_latency(l1_lat, LATENCY_SAMPLES)
            progress.advance(task)

            l1_touch = L1WorkingContext(capacity_bytes=cap_bytes)
            progress.update(task, description=f"{label}: touch throughput")
            touch = bench_touch_throughput(l1_touch, N_TOUCHES)
            progress.advance(task)

            results.append(
                {
                    "capacity_label": label,
                    "capacity_bytes": cap_bytes,
                    "insert_throughput": ins,
                    "insert_latency": lat,
                    "touch_throughput": touch,
                }
            )

    console.print()
    console.print(render_throughput_table(results))
    console.print()
    console.print(render_latency_table(results))
    console.print()

    payload = {
        "system": collect_system_info(),
        "config": {
            "n_inserts": N_INSERTS,
            "latency_samples": LATENCY_SAMPLES,
            "n_touches": N_TOUCHES,
            "capacities": [{"label": lbl, "bytes": b} for lbl, b in CAPACITIES],
            "seed": SEED,
        },
        "results": results,
    }

    timestamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    json_path = RESULTS_DIR / f"l1_baseline_{timestamp}.json"
    json_path.write_text(json.dumps(payload, indent=2))
    console.print(f"📊 JSON:     [dim]{json_path.relative_to(REPO_ROOT)}[/dim]")

    write_markdown_report(payload)
    console.print(f"📝 Markdown: [dim]{NOTES_PATH.relative_to(REPO_ROOT)}[/dim]")
    console.print()
    console.print("[bold green]Done.[/bold green] L1 latency is now measured, not projected.")


if __name__ == "__main__":
    main()
