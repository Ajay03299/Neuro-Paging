"""Tests for the benchmark scripts.

We don't run the full benchmark in CI (too slow). We just verify:
  1. The script is importable (no syntax errors)
  2. The percentile helper works correctly
  3. The benchmark runs on a tiny corpus without errors
"""

from __future__ import annotations

import sys
from pathlib import Path

# Add bench/ to path so we can import the script
sys.path.insert(0, str(Path(__file__).parent.parent / "bench"))


def test_hnswlib_baseline_importable():
    """Importing the benchmark must not have side effects."""
    import hnswlib_baseline  # noqa: F401


def test_percentile_helper():
    """percentile() must return correct values for known input.

    Note: our impl uses nearest-rank percentile with banker's rounding,
    which matches standard benchmark conventions but means the 'median'
    of a 10-element list is the 5th element (index 4), not the 6th.
    """
    from hnswlib_baseline import percentile

    data = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0]

    # Endpoints are well-defined
    assert percentile(data, 0) == 1.0
    assert percentile(data, 100) == 10.0

    # Monotonicity (the property we actually rely on for benchmarks)
    assert percentile(data, 50) <= percentile(data, 95)
    assert percentile(data, 95) <= percentile(data, 99)
    assert percentile(data, 99) <= percentile(data, 100)

    # Specific values — derived from the nearest-rank formula
    # idx = round(pct/100 * (n-1)) with banker's rounding
    assert percentile(data, 50) == 5.0  # idx = round(4.5) = 4 → data[4] = 5.0
    assert percentile(data, 95) == 10.0  # idx = round(8.55) = 9 → data[9] = 10.0
    assert percentile(data, 99) == 10.0  # idx = round(8.91) = 9 → data[9] = 10.0


def test_percentile_monotonic_on_realistic_data():
    """On realistic latency data, percentiles must be monotonically non-decreasing."""
    # Simulate latency data with a long tail (typical of HNSW queries)
    import random

    from hnswlib_baseline import percentile

    rng = random.Random(42)
    data = sorted([rng.gauss(0.3, 0.05) for _ in range(1000)])
    data.extend([rng.gauss(0.8, 0.1) for _ in range(50)])  # tail
    rng.shuffle(data)

    p50 = percentile(data, 50)
    p95 = percentile(data, 95)
    p99 = percentile(data, 99)

    assert p50 < p95 < p99, f"Percentiles not monotonic: p50={p50}, p95={p95}, p99={p99}"


def test_percentile_empty():
    """percentile() on empty input returns NaN."""
    import math

    from hnswlib_baseline import percentile

    assert math.isnan(percentile([], 50))


def test_benchmark_runs_on_tiny_corpus():
    """Smoke test: benchmark a 100-vector corpus and verify shape of output."""
    from hnswlib_baseline import benchmark_size

    result = benchmark_size(100)

    assert result["corpus_size"] == 100
    assert "insert" in result
    assert "query_latency_ms" in result
    assert result["query_latency_ms"]["p50"] > 0
    assert result["query_latency_ms"]["p95"] >= result["query_latency_ms"]["p50"]
    assert result["query_latency_ms"]["p99"] >= result["query_latency_ms"]["p95"]
    assert result["index_size_bytes"] > 0
