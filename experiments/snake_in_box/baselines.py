"""Snake-in-the-box baselines + validator.

A *snake* in the hypercube Q_n is an induced (chord-free) open path:
consecutive vertices differ in exactly one bit, and no two non-consecutive
vertices are adjacent. Score = number of edges. All searches start from the
canonical edge 0 -> 1 (Q_n is vertex- and edge-transitive, so this loses no
generality and collapses the symmetry group).

Budget unit: candidate-extension evaluations (each generated successor
counts 1). Both baselines consume the full budget via restarts.
"""

from __future__ import annotations

import random


def neighbors(v: int, n: int) -> list[int]:
    return [v ^ (1 << i) for i in range(n)]


def validate_snake(n: int, seq: list[int]) -> tuple[bool, float]:
    """Re-derive validity + length (edges) from the raw sequence.

    Never trusts the solver: checks range, distinctness, unit-Hamming steps,
    and chord-freeness (the "skin" condition).
    """
    if not isinstance(seq, (list, tuple)) or len(seq) < 2:
        return False, 0.0
    seq = list(seq)
    if len(set(seq)) != len(seq):
        return False, 0.0
    if any(not isinstance(v, int) or not (0 <= v < (1 << n)) for v in seq):
        return False, 0.0
    for a, b in zip(seq, seq[1:]):
        if bin(a ^ b).count("1") != 1:
            return False, 0.0
    for i in range(len(seq)):
        for j in range(i + 2, len(seq)):
            if bin(seq[i] ^ seq[j]).count("1") == 1:
                return False, 0.0
    return True, float(len(seq) - 1)


def _extensions(tail: int, used: set[int], forbidden: set[int],
                n: int) -> list[int]:
    return [u for u in neighbors(tail, n) if u not in used and u not in forbidden]


def greedy_rollouts(n: int, seed: int, budget: int) -> list[int]:
    """Naive baseline: repeated randomized rollouts until the budget is spent.

    Each rollout extends the snake with a uniformly random legal move until
    stuck. `forbidden` tracks vertices adjacent to interior vertices — the
    incremental form of the chord-free condition.
    """
    rng = random.Random(seed)
    best = [0, 1]
    spent = 0
    while spent < budget:
        path = [0, 1]
        used = {0, 1}
        forbidden = set(neighbors(0, n))
        while True:
            cands = _extensions(path[-1], used, forbidden, n)
            spent += max(len(cands), 1)
            if not cands or spent >= budget:
                break
            u = rng.choice(cands)
            forbidden.update(neighbors(path[-1], n))
            path.append(u)
            used.add(u)
        if len(path) > len(best):
            best = path[:]
    return best


def stochastic_beam(n: int, seed: int, budget: int, width: int = 48) -> list[int]:
    """Strong reproducible baseline: restarted stochastic beam search.

    Keeps the `width` longest partial snakes each depth, with random
    tie-break jitter so restarts explore different corridors. This is a
    standard heuristic family for snake-in-the-box search — a fair
    reproducible bar, NOT the published record machinery (that bar lives in
    records.json once human-transcribed).
    """
    rng = random.Random(seed ^ 0x9E3779B9)
    best = [0, 1]
    spent = 0
    while spent < budget:
        beam: list[tuple[list[int], set[int], set[int]]] = [
            ([0, 1], {0, 1}, set(neighbors(0, n)))]
        while beam and spent < budget:
            successors: list[tuple[float, list[int], set[int], set[int]]] = []
            for path, used, forbidden in beam:
                cands = _extensions(path[-1], used, forbidden, n)
                spent += max(len(cands), 1)
                for u in cands:
                    new_forbidden = forbidden | set(neighbors(path[-1], n))
                    successors.append((
                        len(path) + rng.random(),
                        path + [u], used | {u}, new_forbidden))
                if len(path) > len(best):
                    best = path[:]
            if not successors:
                break
            successors.sort(key=lambda t: -t[0])
            beam = [(p, u, f) for _, p, u, f in successors[:width]]
        for path, _, _ in beam:
            if len(path) > len(best):
                best = path[:]
    return best


SOLVERS = {
    "greedy": greedy_rollouts,
    "stochastic_beam": stochastic_beam,
}
