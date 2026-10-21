"""Deterministic graph-coloring benchmark instances.

A graph is an adjacency list: ``list[set[int]]`` over vertices ``0..n-1``.
Every generator is seeded so the same name always yields the same graph.
"""

from __future__ import annotations

import random
from collections.abc import Callable


def gnp(n: int, p: float, seed: int) -> list[set[int]]:
    """Erdos-Renyi G(n, p) random graph (DSJC-style when p=0.5)."""
    rng = random.Random(seed)
    adj: list[set[int]] = [set() for _ in range(n)]
    for u in range(n):
        for v in range(u + 1, n):
            if rng.random() < p:
                adj[u].add(v)
                adj[v].add(u)
    return adj


def queen_graph(n: int) -> list[set[int]]:
    """Queen n^2 graph: squares of an n x n board; edges where queens attack."""
    adj: list[set[int]] = [set() for _ in range(n * n)]
    for r1 in range(n):
        for c1 in range(n):
            u = r1 * n + c1
            for r2 in range(n):
                for c2 in range(n):
                    v = r2 * n + c2
                    if v <= u:
                        continue
                    if r1 == r2 or c1 == c2 or abs(r1 - r2) == abs(c1 - c2):
                        adj[u].add(v)
                        adj[v].add(u)
    return adj


def planted_k(n: int, k: int, p: float, seed: int) -> list[set[int]]:
    """k-partite random graph: chromatic number <= k by construction.

    Vertices are split round-robin into k groups; edges are added only
    between groups with probability p, so the planted k-coloring is valid.
    """
    rng = random.Random(seed)
    group = [i % k for i in range(n)]
    adj: list[set[int]] = [set() for _ in range(n)]
    for u in range(n):
        for v in range(u + 1, n):
            if group[u] != group[v] and rng.random() < p:
                adj[u].add(v)
                adj[v].add(u)
    return adj


# Name -> zero-arg constructor. Fixed seeds keep every instance reproducible.
INSTANCE_GENERATORS: dict[str, Callable[[], list[set[int]]]] = {
    "gnp60_05": lambda: gnp(60, 0.5, seed=1),
    "gnp70_05": lambda: gnp(70, 0.5, seed=11),
    "gnp80_05": lambda: gnp(80, 0.5, seed=2),
    "gnp125_05": lambda: gnp(125, 0.5, seed=3),
    "gnp100_01": lambda: gnp(100, 0.1, seed=4),
    "queen8": lambda: queen_graph(8),
    "queen10": lambda: queen_graph(10),
    "planted80_k8": lambda: planted_k(80, 8, 0.5, seed=13),
    "planted90_k9": lambda: planted_k(90, 9, 0.5, seed=5),
}

SMOKE_INSTANCES = ["gnp60_05", "queen8"]

# Search/confirm split for autosearch: candidates are iterated against TRAIN
# and the recorded label comes from one confirmation run on HOLDOUT (disjoint
# instances, same families, unseen solver seeds via --seed-base). Never tune
# against HOLDOUT.
TRAIN_INSTANCES = ["gnp60_05", "gnp100_01", "queen8", "planted90_k9"]
HOLDOUT_INSTANCES = ["gnp70_05", "gnp80_05", "gnp125_05", "queen10", "planted80_k8"]
