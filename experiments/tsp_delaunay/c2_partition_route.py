"""C2 verification: Delaunay candidate edges + k-way partition + local solve + stitch.

Rescued composition from run 20260522-215133 C2 (gate-rejected L3=invalid; flipped
to purely_testable under the baseline-relative L4 in the 2026-05-29 rescue sweep):

  "P26's Delaunay triangulation gives a sparse candidate edge set for Euclidean
   TSP, P88's GPU k-way graph partitioner partitions that sparse graph quickly,
   and P86's global-local framework solves the clustered routing subproblems —
   a scaled solver that avoids the full complete graph."

L3's strongest objection: the missing boundary/reconnection layer — partitioning
and solving clusters independently can produce bad tours at cluster seams. So the
decisive test holds the candidate set fixed (Delaunay neighbor lists, whose recall
the C7 bench already validated) and asks:

    Does Delaunay + PARTITION + per-cluster 2-opt + stitch match the tour quality
    of Delaunay + GLOBAL 2-opt (no partition), and is it faster?

If partition+stitch matches quality at a speedup -> C2 holds (the reconnection
layer is adequate). If it loses material tour quality -> L3's objection is upheld.

Pure numpy + scipy.spatial.Delaunay. Tour quality measured as gap vs the best
tour found by any method (and vs Held-Karp optimum on small n as a sanity check).

Run:
    uv run --with scipy python -m experiments.tsp_delaunay.c2_partition_route
"""

from __future__ import annotations

import time

import numpy as np
from scipy.spatial import Delaunay

from experiments.tsp_delaunay.bench import euclidean_tour_length, held_karp_optimum


def _dist(coords: np.ndarray, a: int, b: int) -> float:
    return float(np.hypot(coords[a, 0] - coords[b, 0], coords[a, 1] - coords[b, 1]))


def delaunay_neighbors(coords: np.ndarray) -> list[list[int]]:
    """Adjacency lists from the Delaunay triangulation (the C2 candidate set)."""
    n = len(coords)
    nbrs: list[set[int]] = [set() for _ in range(n)]
    tri = Delaunay(coords)
    for s in tri.simplices:
        for i in range(3):
            for j in range(i + 1, 3):
                a, b = int(s[i]), int(s[j])
                nbrs[a].add(b)
                nbrs[b].add(a)
    # Sort each neighbor list by distance (2-opt tries closest candidates first).
    out = []
    for a in range(n):
        out.append(sorted(nbrs[a], key=lambda b: _dist(coords, a, b)))
    return out


def nearest_neighbor_tour(coords: np.ndarray, idx: np.ndarray | None = None) -> list[int]:
    """Greedy NN tour over the points indexed by `idx` (default: all)."""
    if idx is None:
        idx = np.arange(len(coords))
    idx = list(idx)
    if len(idx) <= 1:
        return idx
    remaining = set(idx)
    start = idx[0]
    tour = [start]
    remaining.discard(start)
    cur = start
    while remaining:
        nxt = min(remaining, key=lambda b: _dist(coords, cur, b))
        tour.append(nxt)
        remaining.discard(nxt)
        cur = nxt
    return tour


def two_opt_neighbor_list(
    coords: np.ndarray, tour: list[int], neighbors: list[list[int]],
    max_passes: int = 30,
) -> list[int]:
    """Classic neighbor-list 2-opt with don't-look bits (Bentley).

    Only considers adding candidate edges (a, c) where c is a Delaunay neighbor of
    a — so the candidate edge set is exactly C2's Delaunay restriction.
    """
    n = len(tour)
    if n < 4:
        return tour
    pos = {city: i for i, city in enumerate(tour)}
    dont_look = {c: False for c in tour}
    tourset = set(tour)
    queue = list(tour)

    def d(a, b):
        return _dist(coords, a, b)

    improved_any = True
    passes = 0
    while improved_any and passes < max_passes:
        improved_any = False
        passes += 1
        for a in queue:
            if dont_look[a]:
                continue
            improved = False
            i = pos[a]
            a_next = tour[(i + 1) % n]
            a_prev = tour[(i - 1) % n]
            for c in neighbors[a]:
                if c not in tourset:
                    continue
                d_ac = d(a, c)
                # Direction 1: replace (a, a_next) + (c, c_next) with (a, c) + (a_next, c_next)
                if d_ac < d(a, a_next):
                    j = pos[c]
                    c_next = tour[(j + 1) % n]
                    if c != a_next and c_next != a:
                        delta = (d_ac + d(a_next, c_next)) - (d(a, a_next) + d(c, c_next))
                        if delta < -1e-12:
                            _reverse_segment(tour, pos, (i + 1) % n, j, n)
                            dont_look[a] = dont_look[a_next] = False
                            dont_look[c] = dont_look[c_next] = False
                            improved = improved_any = True
                            break
                # Direction 2: use predecessor edge
                if d_ac < d(a_prev, a):
                    j = pos[c]
                    c_prev = tour[(j - 1) % n]
                    if c != a_prev and c_prev != a:
                        delta = (d_ac + d(a_prev, c_prev)) - (d(a_prev, a) + d(c_prev, c))
                        if delta < -1e-12:
                            _reverse_segment(tour, pos, j, (i - 1) % n, n)
                            dont_look[a] = dont_look[a_prev] = False
                            dont_look[c] = dont_look[c_prev] = False
                            improved = improved_any = True
                            break
            if not improved:
                dont_look[a] = True
    return tour


def _reverse_segment(tour: list[int], pos: dict, i: int, j: int, n: int) -> None:
    """Reverse tour[i..j] inclusive (cyclic), updating pos. Reverses the shorter arc."""
    seg_len = (j - i) % n + 1
    if seg_len > n // 2:
        # Reverse the complementary arc instead (equivalent tour, cheaper).
        i, j = (j + 1) % n, (i - 1) % n
        seg_len = n - seg_len
    a, b = i, j
    for _ in range(seg_len // 2):
        tour[a], tour[b] = tour[b], tour[a]
        pos[tour[a]] = a
        pos[tour[b]] = b
        a = (a + 1) % n
        b = (b - 1) % n
    if seg_len % 2 == 1:
        pos[tour[a]] = a


def global_2opt(coords: np.ndarray, neighbors: list[list[int]]) -> list[int]:
    """Baseline: NN start + Delaunay-restricted global 2-opt to convergence."""
    tour = nearest_neighbor_tour(coords)
    return two_opt_neighbor_list(coords, tour, neighbors)


def _rcb_partition(coords: np.ndarray, k: int) -> list[np.ndarray]:
    """Recursive coordinate bisection into ~k balanced clusters (P88 stand-in)."""
    clusters = [np.arange(len(coords))]
    while len(clusters) < k:
        clusters.sort(key=len, reverse=True)
        big = clusters.pop(0)
        if len(big) <= 1:
            clusters.append(big)
            break
        pts = coords[big]
        axis = 0 if np.ptp(pts[:, 0]) >= np.ptp(pts[:, 1]) else 1
        order = big[np.argsort(coords[big, axis])]
        mid = len(order) // 2
        clusters.append(order[:mid])
        clusters.append(order[mid:])
    return clusters


def c2_partition_route(coords: np.ndarray, neighbors: list[list[int]], k: int) -> list[int]:
    """C2: Delaunay candidate edges + k-way partition + per-cluster 2-opt + stitch.

    Stitch (the reconnection layer L3 said was missing): order clusters by a tour
    over their centroids, then merge each cluster's sub-tour into the global tour
    by the cheapest endpoint join; finally run a global Delaunay-2opt cleanup pass
    (bounded) to repair seams.
    """
    clusters = _rcb_partition(coords, k)
    # Solve each cluster locally with NN + restricted 2-opt on the FULL Delaunay
    # neighbor lists (intra-cluster neighbors only contribute within the subtour).
    sub_tours = []
    centroids = []
    for cl in clusters:
        if len(cl) == 0:
            continue
        sub = nearest_neighbor_tour(coords, cl)
        sub = two_opt_neighbor_list(coords, sub, _restrict_neighbors(neighbors, set(cl)))
        sub_tours.append(sub)
        centroids.append(coords[cl].mean(axis=0))
    # Order clusters by a tour over centroids (NN over centroids).
    centroids = np.array(centroids)
    if len(sub_tours) > 1:
        corder = nearest_neighbor_tour(centroids)
    else:
        corder = [0]
    # Concatenate sub-tours in centroid order, joining at nearest endpoints.
    full: list[int] = list(sub_tours[corder[0]])
    for ci in corder[1:]:
        nxt = sub_tours[ci]
        # rotate `nxt` so its start is closest to the current tail
        tail = full[-1]
        best_start = min(range(len(nxt)), key=lambda s: _dist(coords, tail, nxt[s]))
        nxt = nxt[best_start:] + nxt[:best_start]
        full.extend(nxt)
    # Seam-repair: one bounded global 2-opt cleanup on Delaunay neighbors.
    full = two_opt_neighbor_list(coords, full, neighbors, max_passes=8)
    return full


def _restrict_neighbors(neighbors: list[list[int]], keep: set[int]) -> list[list[int]]:
    return [[b for b in neighbors[a] if b in keep] if a in keep else []
            for a in range(len(neighbors))]


def run(sizes_seeds: list[tuple[int, int]], k_frac: float = 0.05) -> None:
    print(f"\n{'n':>6} {'seed':>5} {'method':>14} {'len':>10} {'gap%':>7} {'secs':>7}")
    print("-" * 56)
    for n, seed in sizes_seeds:
        rng = np.random.default_rng(seed)
        coords = rng.random((n, 2))
        nbrs = delaunay_neighbors(coords)
        results = {}

        t0 = time.monotonic()
        g = global_2opt(coords, nbrs)
        results["delaunay-global2opt"] = (euclidean_tour_length(coords, g), time.monotonic() - t0)

        k = max(2, int(n * k_frac))
        t0 = time.monotonic()
        c = c2_partition_route(coords, nbrs, k)
        results[f"C2-partition(k={k})"] = (euclidean_tour_length(coords, c), time.monotonic() - t0)

        t0 = time.monotonic()
        nn = nearest_neighbor_tour(coords)
        results["NN-only"] = (euclidean_tour_length(coords, nn), time.monotonic() - t0)

        ref = None
        if n <= 12:
            ref = held_karp_optimum(coords)[0]
        best = ref if ref is not None else min(v[0] for v in results.values())
        for name, (length, secs) in results.items():
            gap = 100 * (length - best) / best
            print(f"{n:>6} {seed:>5} {name:>14} {length:>10.4f} {gap:>7.2f} {secs:>7.3f}")
        if ref is not None:
            print(f"{n:>6} {seed:>5} {'held-karp-opt':>14} {ref:>10.4f} {0.0:>7.2f}")
        print()


if __name__ == "__main__":
    cases = [(10, 0), (10, 1),
             (200, 0), (200, 1), (200, 2),
             (500, 0), (500, 1), (500, 2),
             (1000, 0), (1000, 1),
             (2000, 0)]
    run(cases)
