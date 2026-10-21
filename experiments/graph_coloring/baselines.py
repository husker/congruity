"""Baseline graph-coloring solvers.

All solvers return a color assignment ``list[int]`` (one color per vertex,
colors are 0-based ints). Callers must re-validate with
:func:`validate_coloring`; never trust a solver's own report.
"""

from __future__ import annotations

import random


def validate_coloring(adj: list[set[int]], colors: list[int]) -> tuple[bool, int, int]:
    """Recompute validity from the emitted assignment.

    Returns ``(is_proper, n_colors, conflict_edges)``.
    """
    if len(colors) != len(adj) or any(c < 0 for c in colors):
        return False, 0, -1
    conflicts = 0
    for u, nbrs in enumerate(adj):
        for v in nbrs:
            if v > u and colors[v] == colors[u]:
                conflicts += 1
    return conflicts == 0, len(set(colors)), conflicts


def greedy(adj: list[set[int]], order: list[int]) -> list[int]:
    """Color vertices in the given order with the smallest feasible color."""
    n = len(adj)
    colors = [-1] * n
    for v in order:
        used = {colors[u] for u in adj[v] if colors[u] >= 0}
        c = 0
        while c in used:
            c += 1
        colors[v] = c
    return colors


def greedy_random(adj: list[set[int]], seed: int, restarts: int = 50) -> list[int]:
    """Best greedy coloring over `restarts` random vertex orders."""
    rng = random.Random(seed)
    n = len(adj)
    best: list[int] | None = None
    best_k = n + 1
    order = list(range(n))
    for _ in range(restarts):
        rng.shuffle(order)
        colors = greedy(adj, order)
        k = max(colors) + 1
        if k < best_k:
            best_k = k
            best = colors[:]
    assert best is not None
    return best


def dsatur(adj: list[set[int]]) -> list[int]:
    """DSATUR: color the vertex with highest saturation degree first."""
    n = len(adj)
    colors = [-1] * n
    sat: list[set[int]] = [set() for _ in range(n)]
    degrees = [len(adj[v]) for v in range(n)]
    uncolored = set(range(n))
    for _ in range(n):
        v = max(uncolored, key=lambda u: (len(sat[u]), degrees[u]))
        c = 0
        while c in sat[v]:
            c += 1
        colors[v] = c
        uncolored.discard(v)
        for u in adj[v]:
            sat[u].add(c)
    return colors


def _tabucol_fixed_k(
    adj: list[set[int]],
    k: int,
    rng: random.Random,
    max_iters: int,
    init: list[int] | None = None,
) -> tuple[list[int], int]:
    """TabuCol (Hertz & de Werra): minimize conflict edges at fixed k.

    Returns ``(assignment, conflict_edges)``; a proper k-coloring has 0.
    """
    n = len(adj)
    colors = init[:] if init is not None else [rng.randrange(k) for _ in range(n)]
    # gamma[v][c]: how many neighbors of v currently have color c.
    gamma = [[0] * k for _ in range(n)]
    for v in range(n):
        for u in adj[v]:
            gamma[v][colors[u]] += 1
    f = sum(gamma[v][colors[v]] for v in range(n)) // 2
    best_f = f
    best = colors[:]
    tabu = [[0] * k for _ in range(n)]

    for it in range(1, max_iters + 1):
        if f == 0:
            break
        conflicted = [v for v in range(n) if gamma[v][colors[v]] > 0]
        best_delta = None
        moves: list[tuple[int, int]] = []
        for v in conflicted:
            cur = colors[v]
            g_cur = gamma[v][cur]
            for c in range(k):
                if c == cur:
                    continue
                delta = gamma[v][c] - g_cur
                if tabu[v][c] >= it and f + delta >= best_f:
                    continue  # tabu and fails aspiration
                if best_delta is None or delta < best_delta:
                    best_delta = delta
                    moves = [(v, c)]
                elif delta == best_delta:
                    moves.append((v, c))
        if best_delta is None:
            continue  # everything tabu this iteration
        v, c = moves[rng.randrange(len(moves))]
        old = colors[v]
        colors[v] = c
        for u in adj[v]:
            gamma[u][old] -= 1
            gamma[u][c] += 1
        f += best_delta
        tabu[v][old] = it + int(0.6 * len(conflicted)) + rng.randrange(10)
        if f < best_f:
            best_f = f
            best = colors[:]
    return best, best_f


def tabucol(adj: list[set[int]], seed: int, iters_per_k: int = 100_000) -> list[int]:
    """Descending-k TabuCol: start from DSATUR, lower k until tabu search fails."""
    rng = random.Random(seed)
    legal = dsatur(adj)
    k = max(legal) + 1
    while k > 1:
        target = k - 1
        # Re-seed from the legal coloring, squashing the largest color class.
        init = [c if c < target else rng.randrange(target) for c in legal]
        candidate, conflicts = _tabucol_fixed_k(adj, target, rng, iters_per_k, init)
        if conflicts != 0:
            break
        legal = candidate
        k = target
    return legal


SOLVERS = {
    "greedy": lambda adj, seed, _iters: greedy_random(adj, seed),
    "dsatur": lambda adj, _seed, _iters: dsatur(adj),
    "tabucol": lambda adj, seed, iters: tabucol(adj, seed, iters_per_k=iters),
}
