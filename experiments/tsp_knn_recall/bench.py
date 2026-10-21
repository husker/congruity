"""C4 candidate-edge recall verification — compose run 20260523-003811.

C4's recipe is a generate-filter-solve pipeline: a learned (diffusion) sampler
proposes candidate TSP/HC edges, a formal safety filter prunes them, and a
SAT solver completes the partial solution. Its stop rule:

    "Abandon early if candidate edge recall for known optimum tours is
     below 99% on 20 pilot TSPLIB instances at the chosen pruning budget,
     because exact completion cannot recover missing optimum edges."

The recipe doesn't specify the diffusion architecture or budget. We can
still bound the composition by testing the cheapest possible learned
proposer surrogate: K-nearest-neighbors. K-NN edges are the trivial
distance-only filter; any worthwhile diffusion model has to do at least
as well as K-NN at a given budget, and ideally better.

For each (n, seed) we:

  1. Generate n uniform random Euclidean points in [0,1]^2.
  2. Solve TSP exactly via Held-Karp DP (n <= 18 keeps this fast).
  3. For each K, build the symmetric K-NN candidate set
     (edges where at least one endpoint sees the other in its K nearest).
  4. Count what fraction of optimal-tour edges are in the K-NN set.

Per-K mean recall + the minimum K hitting 99% mean recall determines the
lower bound any diffusion-based proposer must beat. If K=n/2 still can't
hit 99% even on toy uniform points, candidate-pruning approaches like C4
are bounded.

Run:
    uv run --with numpy python -m experiments.tsp_knn_recall.bench
"""

from __future__ import annotations

import argparse
import math

import numpy as np


def held_karp_optimum(coords: np.ndarray) -> tuple[float, list[int]]:
    """Exact optimal TSP via Held-Karp DP. O(2^n n^2)."""
    n = len(coords)
    dist = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            if i != j:
                dist[i, j] = math.hypot(
                    coords[i, 0] - coords[j, 0], coords[i, 1] - coords[j, 1]
                )

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

    best_cost = math.inf
    best_last = -1
    for last in range(1, n):
        cost = dp.get((full, last), math.inf) + dist[last, 0]
        if cost < best_cost:
            best_cost = cost
            best_last = last

    tour: list[int] = [best_last]
    mask = full
    last = best_last
    while (mask, last) in parent:
        prev = parent[(mask, last)]
        tour.append(prev)
        mask ^= 1 << last
        last = prev
    tour.append(0)
    tour.reverse()
    return best_cost, tour


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


def knn_edge_set(coords: np.ndarray, k: int) -> set[tuple[int, int]]:
    """Symmetric K-NN edge set: edge (i,j) exists if j is among i's K
    nearest OR i is among j's K nearest (the standard symmetrized form
    used as a candidate set in TSP heuristics like LKH)."""
    n = len(coords)
    edges: set[tuple[int, int]] = set()
    # Pairwise distances
    diff = coords[:, None, :] - coords[None, :, :]
    dist = np.sqrt((diff * diff).sum(-1))
    np.fill_diagonal(dist, np.inf)
    # For each i, the K closest indices
    nearest = np.argsort(dist, axis=1)[:, :k]
    for i in range(n):
        for j in nearest[i]:
            a, b = (int(i), int(j)) if i < j else (int(j), int(i))
            edges.add((a, b))
    return edges


def recall_one_instance(n: int, seed: int, ks: list[int]) -> dict:
    rng = np.random.default_rng(seed)
    coords = rng.random((n, 2))
    _opt_cost, opt_tour = held_karp_optimum(coords)
    optimal_edges = tour_edge_set(opt_tour)
    per_k = {}
    for k in ks:
        if k >= n:
            per_k[k] = 1.0
            continue
        knn_edges = knn_edge_set(coords, k)
        contained = optimal_edges & knn_edges
        per_k[k] = len(contained) / len(optimal_edges)
    return {
        "n": n,
        "seed": seed,
        "n_optimal_edges": len(optimal_edges),
        "recall_by_k": per_k,
    }


def run(sizes: list[int], seeds_per_size: int, ks: list[int]) -> dict:
    rows = []
    header = "  n |  seed | " + " | ".join(f"K={k:>2d}" for k in ks)
    print(header)
    print("-" * len(header))
    for n in sizes:
        for s in range(seeds_per_size):
            r = recall_one_instance(n, s, ks)
            rows.append(r)
            cells = " | ".join(f"{r['recall_by_k'][k] * 100:5.1f}%" for k in ks)
            print(f" {n:2d} | {s:5d} | {cells}")

    # Aggregate: mean recall per K, per size
    summary: dict = {"per_k": {}, "per_n": {}, "per_n_per_k": {}}
    for k in ks:
        all_r = [r["recall_by_k"][k] for r in rows]
        summary["per_k"][k] = {
            "mean": float(np.mean(all_r)),
            "min": float(np.min(all_r)),
            "n_below_99pct": sum(1 for x in all_r if x < 0.99),
            "n_perfect": sum(1 for x in all_r if x == 1.0),
        }
    for n in sizes:
        for k in ks:
            sub = [r["recall_by_k"][k] for r in rows if r["n"] == n]
            if sub:
                summary["per_n_per_k"].setdefault(n, {})[k] = float(np.mean(sub))

    # What's the smallest K hitting 99% mean recall across all instances?
    min_k_for_99 = None
    for k in sorted(ks):
        if summary["per_k"][k]["mean"] >= 0.99 and summary["per_k"][k]["n_below_99pct"] == 0:
            min_k_for_99 = k
            break
    summary["min_k_for_99pct_universal"] = min_k_for_99

    # Smallest K with mean >= 99% (weaker bar: average not min)
    min_k_for_99_mean = None
    for k in sorted(ks):
        if summary["per_k"][k]["mean"] >= 0.99:
            min_k_for_99_mean = k
            break
    summary["min_k_for_99pct_mean"] = min_k_for_99_mean

    return {"rows": rows, "summary": summary}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--sizes", type=int, nargs="+", default=[10, 12, 14, 16, 18],
        help="Instance sizes (n <= ~18 keeps held-karp fast)",
    )
    ap.add_argument("--seeds", type=int, default=10)
    ap.add_argument(
        "--ks", type=int, nargs="+", default=[3, 5, 8, 10, 15],
        help="K-NN budgets to test",
    )
    args = ap.parse_args()

    print("\n=== C4 candidate-edge recall (compose 20260523-003811, K-NN baseline) ===")
    print(f"sizes: {args.sizes}, seeds per size: {args.seeds}, K values: {args.ks}")
    print("stop rule: 'candidate edge recall >= 99% at the chosen pruning budget'\n")

    result = run(args.sizes, args.seeds, args.ks)
    s = result["summary"]

    print(f"\n=== per-K aggregate (across all {sum(1 for _ in result['rows'])} instances) ===")
    for k in sorted(s["per_k"]):
        d = s["per_k"][k]
        print(
            f"  K={k:>2d}: mean recall {d['mean'] * 100:5.2f}%, "
            f"min {d['min'] * 100:5.2f}%, "
            f"{d['n_below_99pct']} instances below 99%, "
            f"{d['n_perfect']} perfect"
        )

    print("\n=== per-size per-K mean recall ===")
    sizes_sorted = sorted(s["per_n_per_k"])
    ks_sorted = sorted(args.ks)
    header = "  n    | " + " | ".join(f"K={k:>2d}" for k in ks_sorted)
    print(header)
    print("-" * len(header))
    for n in sizes_sorted:
        cells = " | ".join(f"{s['per_n_per_k'][n][k] * 100:5.1f}%" for k in ks_sorted)
        print(f" {n:>4d} | {cells}")

    print("\n=== verdict ===")
    universal = s["min_k_for_99pct_universal"]
    avg = s["min_k_for_99pct_mean"]
    if universal is not None:
        print(
            f"K-NN at K={universal} achieves >=99% recall on EVERY instance tested."
        )
        print(
            "  Any diffusion proposer must (a) beat this budget while keeping >=99% recall,"
        )
        print(
            "  or (b) exploit something K-NN can't see. C4 is NOT falsified at its stop rule;"
        )
        print(
            "  the open question is whether learned proposers can do better than trivial K-NN."
        )
    elif avg is not None:
        print(
            f"K-NN at K={avg} achieves >=99% MEAN recall, but worst-case instances still"
        )
        print(
            "  fall below 99%. C4's stop rule (which gates on per-instance recall) would"
        )
        print(
            "  fire on those instances unless the proposer covers what K-NN misses."
        )
    else:
        print(
            f"No K in {args.ks} achieves >=99% mean recall. C4's premise is challenged: even"
        )
        print(
            "  generous K-NN budgets miss optimal tour edges. A diffusion proposer must learn"
        )
        print(
            "  something more discriminating than nearest-neighbor distance to clear the bar."
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
