"""Delaunay candidate-recall verification for compose run 20260522-215133 C7.

C7's recipe asserts that Delaunay triangulation is a sound first stage for
Euclidean TSP: it claims P26's Delaunay-restricted candidate edge set
combined with spectral hypergraph partitioning + min-cost flow can beat
LKH-3. The recipe's own stop rule says:

    "Abandon as clean negative if Delaunay candidate recall is <99% on
     any small instance with known optimum"

This bench falsifies that floor before any of the heavier methods get
built. For each small Euclidean instance we:

  1. Generate uniform random points in [0,1]^2 (TSPLIB-style Euclidean).
  2. Solve TSP exactly via Held-Karp DP (n <= 18 keeps this fast).
  3. Compute the Delaunay triangulation via scipy.spatial.
  4. Count what fraction of the optimal tour's n edges appear as
     Delaunay edges. This is the "candidate recall" the recipe gates on.

A single instance below 99% recall falsifies C7 by its own stop rule
(the entire composition's value depends on Delaunay being a tight
filter, so even one miss is decisive).

Run:
    uv run --with scipy python -m experiments.tsp_delaunay.bench
"""

from __future__ import annotations

import argparse
import itertools
import math
import time

import numpy as np
from scipy.spatial import Delaunay


def euclidean_tour_length(coords: np.ndarray, tour: list[int]) -> float:
    total = 0.0
    n = len(tour)
    for i in range(n):
        a, b = tour[i], tour[(i + 1) % n]
        total += math.hypot(coords[a, 0] - coords[b, 0], coords[a, 1] - coords[b, 1])
    return total


def held_karp_optimum(coords: np.ndarray) -> tuple[float, list[int]]:
    """Exact optimal TSP via Held-Karp DP. O(2^n * n^2). Use up to n ~ 18.

    Returns (optimal tour length, optimal tour as a list of indices starting at 0).
    """
    n = len(coords)
    dist = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            if i != j:
                dist[i, j] = math.hypot(coords[i, 0] - coords[j, 0],
                                         coords[i, 1] - coords[j, 1])

    # dp[mask][i] = min cost to start at 0, visit cities in mask, end at i.
    # mask always includes bit 0 (the start city).
    full = (1 << n) - 1
    dp: dict[tuple[int, int], float] = {(1, 0): 0.0}
    parent: dict[tuple[int, int], int] = {}

    for mask in range(1, full + 1):
        if not (mask & 1):
            continue
        for last in range(n):
            if not (mask & (1 << last)):
                continue
            if (mask, last) not in dp:
                continue
            cost_here = dp[(mask, last)]
            for nxt in range(n):
                if mask & (1 << nxt):
                    continue
                new_mask = mask | (1 << nxt)
                new_cost = cost_here + dist[last, nxt]
                key = (new_mask, nxt)
                if key not in dp or new_cost < dp[key]:
                    dp[key] = new_cost
                    parent[key] = last

    # Close the cycle: pick the last city minimising dp[full][last] + dist[last][0].
    best_cost = math.inf
    best_last = -1
    for last in range(1, n):
        cost = dp.get((full, last), math.inf) + dist[last, 0]
        if cost < best_cost:
            best_cost = cost
            best_last = last

    # Recover the tour by walking parent pointers backwards.
    tour: list[int] = [best_last]
    mask = full
    last = best_last
    while (mask, last) in parent:
        prev = parent[(mask, last)]
        tour.append(prev)
        mask ^= (1 << last)
        last = prev
    tour.append(0)
    tour.reverse()
    return best_cost, tour


def delaunay_edge_set(coords: np.ndarray) -> set[tuple[int, int]]:
    """Edges of the Delaunay triangulation as a set of (min, max) index pairs."""
    tri = Delaunay(coords)
    edges: set[tuple[int, int]] = set()
    for simplex in tri.simplices:
        for a, b in itertools.combinations(simplex, 2):
            i, j = (int(a), int(b)) if a < b else (int(b), int(a))
            edges.add((i, j))
    return edges


def tour_edge_set(tour: list[int]) -> set[tuple[int, int]]:
    """Edges of the tour as a set of (min, max) index pairs. Held-Karp
    returns the tour with a duplicate start vertex ([0, 0, v1, ..., vn-1])
    after reverse, so we skip self-loops to avoid a spurious (0,0) edge
    inflating the denominator by 1."""
    edges: set[tuple[int, int]] = set()
    n_tour = len(tour)
    for i in range(n_tour):
        a, b = tour[i], tour[(i + 1) % n_tour]
        if a == b:
            continue
        e = (a, b) if a < b else (b, a)
        edges.add(e)
    return edges


def recall_one_instance(n: int, seed: int) -> dict:
    rng = np.random.default_rng(seed)
    coords = rng.random((n, 2))
    t0 = time.monotonic()
    opt_cost, opt_tour = held_karp_optimum(coords)
    held_karp_secs = time.monotonic() - t0
    delaunay_edges = delaunay_edge_set(coords)
    optimal_edges = tour_edge_set(opt_tour)
    contained = optimal_edges & delaunay_edges
    missing = optimal_edges - delaunay_edges
    recall = len(contained) / len(optimal_edges)
    return {
        "n": n,
        "seed": seed,
        "optimum": opt_cost,
        "n_optimal_edges": len(optimal_edges),
        "n_delaunay_edges": len(delaunay_edges),
        "n_contained": len(contained),
        "n_missing": len(missing),
        "recall": recall,
        "held_karp_secs": round(held_karp_secs, 3),
        "missing_edges": sorted(missing),
    }


def run(sizes: list[int], seeds_per_size: int) -> dict:
    """Run all (n, seed) combinations and return a summary."""
    rows = []
    print("  n |  seed | held-karp |  edges | recall | missing edges")
    print("----+-------+-----------+--------+--------+----------------")
    for n in sizes:
        for s in range(seeds_per_size):
            r = recall_one_instance(n, s)
            rows.append(r)
            print(f" {n:2d} | {s:5d} | {r['held_karp_secs']:7.2f}s  | "
                  f"{r['n_optimal_edges']:4d}/{r['n_delaunay_edges']:4d} | "
                  f"{r['recall']*100:5.1f}% | "
                  f"{r['n_missing']} missing")

    # Aggregate
    recalls = [r["recall"] for r in rows]
    by_n: dict[int, list[float]] = {}
    for r in rows:
        by_n.setdefault(r["n"], []).append(r["recall"])

    summary = {
        "n_instances": len(rows),
        "mean_recall": float(np.mean(recalls)),
        "min_recall": float(np.min(recalls)),
        "max_recall": float(np.max(recalls)),
        "n_below_99pct": sum(1 for r in recalls if r < 0.99),
        "n_below_95pct": sum(1 for r in recalls if r < 0.95),
        "n_perfect": sum(1 for r in recalls if r == 1.0),
        "per_size_mean_recall": {n: float(np.mean(rs)) for n, rs in by_n.items()},
    }
    return {"rows": rows, "summary": summary}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--sizes", type=int, nargs="+", default=[10, 12, 14, 16, 18],
                    help="Instance sizes (must be <= ~18 for held-karp to stay fast)")
    ap.add_argument("--seeds", type=int, default=10,
                    help="Number of random instances per size")
    args = ap.parse_args()

    print("\n=== Delaunay candidate-recall verification (compose 20260522-215133 C7) ===")
    print(f"sizes: {args.sizes}, seeds per size: {args.seeds}")
    print("stop rule from recipe: 'Abandon as clean negative if Delaunay")
    print("  candidate recall is <99% on any small instance with known optimum'\n")

    result = run(args.sizes, args.seeds)
    s = result["summary"]

    print("\n=== summary ===")
    print(f"instances run:       {s['n_instances']}")
    print(f"mean recall:         {s['mean_recall']*100:.2f}%")
    print(f"min recall:          {s['min_recall']*100:.2f}%")
    print(f"max recall:          {s['max_recall']*100:.2f}%")
    print(f"instances at 100%:   {s['n_perfect']}/{s['n_instances']}")
    print(f"instances < 99%:     {s['n_below_99pct']}/{s['n_instances']}")
    print(f"instances < 95%:     {s['n_below_95pct']}/{s['n_instances']}")
    print("per-size mean recall:")
    for n in sorted(s["per_size_mean_recall"]):
        print(f"  n={n}: {s['per_size_mean_recall'][n]*100:.2f}%")

    print("\n=== verdict ===")
    if s["n_below_99pct"] > 0:
        print(f"FALSIFIED (clean-negative): {s['n_below_99pct']} instance(s) below 99% recall floor.")
        print("  C7's recipe stop rule fires: 'Abandon as clean negative if recall < 99%'.")
        print("  The composition's foundational premise (Delaunay as a tight TSP candidate")
        print("  filter) does not hold uniformly even on small uniform-random instances,")
        print("  so the heavier hypergraph-partition + min-cost-flow stages would be")
        print("  building on an unreliable foundation.")
    else:
        print(f"PASSES first-pass floor: all {s['n_instances']} instances >=99% recall.")
        print("  C7 is NOT falsified by its own stop rule. The candidate filter is")
        print("  reliable enough on small uniform-random instances to warrant the next")
        print("  experimental step: implement the spectral hypergraph partition + flow")
        print("  stages and benchmark against LKH-3 on TSPLIB95.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
