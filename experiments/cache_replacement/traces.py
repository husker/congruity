"""Workload trace generators for cache replacement benchmark.

Trace = sequence of integer object IDs (the keys being requested).
Generators are deterministic (fixed seeds) so all policies see identical input.
"""

from __future__ import annotations

import random


def zipf_trace(
    n_requests: int = 1_000_000,
    n_objects: int = 100_000,
    s: float = 0.99,
    seed: int = 1,
) -> list[int]:
    """Zipf-distributed access trace.

    s=0.99 is roughly the CDN/web workload regime (Breslau 1999, "Web caching
    and Zipf-like distributions").
    """
    rng = random.Random(seed)
    # Use a power-law weight table; sample by inverse CDF
    weights = [1.0 / ((i + 1) ** s) for i in range(n_objects)]
    total = sum(weights)
    cdf = []
    acc = 0.0
    for w in weights:
        acc += w / total
        cdf.append(acc)
    out: list[int] = []
    for _ in range(n_requests):
        u = rng.random()
        # Binary search for inverse CDF
        lo, hi = 0, n_objects - 1
        while lo < hi:
            m = (lo + hi) // 2
            if cdf[m] < u:
                lo = m + 1
            else:
                hi = m
        out.append(lo)
    return out


def loop_trace(n_objects: int = 50_000, loops: int = 4) -> list[int]:
    """Scan/loop workload: sweep through objects in order, repeat.

    This is the classic adversarial pattern for LRU when cache_size < n_objects:
    LRU evicts the object that's about to be requested next.
    """
    out: list[int] = []
    for _ in range(loops):
        out.extend(range(n_objects))
    return out


def scan_with_hot_trace(
    n_requests: int = 1_000_000,
    n_hot: int = 1_000,
    n_cold: int = 100_000,
    scan_fraction: float = 0.5,
    seed: int = 2,
) -> list[int]:
    """Mixed workload: hot working set + occasional scans through cold objects.

    This is where SIEVE/S3-FIFO win over LRU: scans pollute LRU but the
    one-bit-visited filters prevent scan objects from displacing hot ones.
    """
    rng = random.Random(seed)
    hot_ids = list(range(n_hot))
    cold_ids = list(range(n_hot, n_hot + n_cold))
    out: list[int] = []
    while len(out) < n_requests:
        if rng.random() < scan_fraction:
            out.append(rng.choice(cold_ids))
        else:
            out.append(rng.choice(hot_ids))
    return out


def paired_trace(
    n_requests: int = 200_000,
    n_pairs: int = 5_000,
    s: float = 0.99,
    seed: int = 3,
) -> list[int]:
    """Sanity-check workload with strong successor structure.

    Objects come in fixed pairs (2k, 2k+1). Each "draw" selects a pair via
    Zipf and emits both keys in order. A predictor that knows pairs co-occur
    can pre-warm the second member's visited bit. If Succ-SIEVE does not
    beat SIEVE here, the mechanism is broken; if it does, the mechanism
    works but Zipf/scan_hot traces simply lack this structure.
    """
    rng = random.Random(seed)
    weights = [1.0 / ((i + 1) ** s) for i in range(n_pairs)]
    total = sum(weights)
    cdf = []
    acc = 0.0
    for w in weights:
        acc += w / total
        cdf.append(acc)
    out: list[int] = []
    while len(out) < n_requests:
        u = rng.random()
        lo, hi = 0, n_pairs - 1
        while lo < hi:
            m = (lo + hi) // 2
            if cdf[m] < u:
                lo = m + 1
            else:
                hi = m
        pair = lo
        out.append(2 * pair)
        out.append(2 * pair + 1)
    return out[:n_requests]


TRACE_GENERATORS = {
    "zipf_s099": lambda: zipf_trace(n_requests=200_000, n_objects=20_000, s=0.99, seed=1),
    "zipf_s080": lambda: zipf_trace(n_requests=200_000, n_objects=20_000, s=0.80, seed=1),
    "loop_4x": lambda: loop_trace(n_objects=10_000, loops=20),
    "scan_hot": lambda: scan_with_hot_trace(
        n_requests=200_000, n_hot=500, n_cold=20_000, scan_fraction=0.4, seed=2
    ),
    "paired_zipf": lambda: paired_trace(n_requests=200_000, n_pairs=5_000, s=0.99, seed=3),
}


def trace_stats(trace: list[int]) -> dict:
    unique = set(trace)
    from collections import Counter

    c = Counter(trace)
    sorted_counts = sorted(c.values(), reverse=True)
    return {
        "n_requests": len(trace),
        "n_unique": len(unique),
        "top_10_share": sum(sorted_counts[:10]) / len(trace),
        "p50_count": sorted_counts[len(sorted_counts) // 2],
    }


if __name__ == "__main__":
    for name, gen in TRACE_GENERATORS.items():
        t = gen()
        s = trace_stats(t)
        print(
            f"{name:15s}  n_req={s['n_requests']:>7d}  n_unique={s['n_unique']:>6d}  "
            f"top10_share={s['top_10_share']:.3f}  p50_count={s['p50_count']}"
        )
