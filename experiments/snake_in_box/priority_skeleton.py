"""FunSearch-style skeleton for snake-in-the-box: a fixed rollout driver
whose ONLY evolvable part is the priority function between the markers.

The FunSearch result (Nature 2023) — and its stated design lesson — is that
evolving a small priority function inside a frozen greedy skeleton beats
evolving whole solvers: the search space is narrow, every sample is
runnable, and the skeleton guarantees validity/budget accounting. This is
the snake-in-the-box instance; autosearch --function-skeleton splices
codegen output into the marked region and never touches the driver.

The seed priority is Kinny's record-breaking NMCS playout heuristic:
favor moves that lead to states with the fewest legal continuations
(forcing moves), which drives the snake along walls instead of leaving
usable space behind.

Standard candidate interface: solve(n, seed, budget).
"""

from __future__ import annotations

import random


# <PRIORITY-FUNCTION>
def priority(n: int, cur: int, nxt: int, succ_after: int, depth: int,
             dims_used: list[int], noise: float) -> float:
    """Score a candidate extension; the driver plays the argmax.

    Args:
      n: hypercube dimension.
      cur: current head vertex.
      nxt: the vertex this move would extend to.
      succ_after: how many legal continuations nxt would have.
      depth: current snake length (edges so far).
      dims_used: count of times each dimension has been flipped so far.
      noise: uniform [0,1) sample for stochastic tie-breaking.
    Higher is better. Kinny's heuristic: prefer forcing moves (fewest
    continuations, but never zero unless nothing else exists).
    """
    if succ_after == 0:
        return -100.0 + noise
    return -float(succ_after) + 0.1 * noise


# Marks the unmodified Kinny priority so the driver may use the C fast core.
# Lives INSIDE the markers, so a FunSearch splice removes it — an evolved
# priority function then correctly runs through the pure-Python driver.
priority._kinny_default = True
# </PRIORITY-FUNCTION>


def _legal_with_counts(n: int, cur: int, neighbor_count: list[int],
                       on_snake: list[bool]) -> list[tuple[int, int]]:
    """(next_vertex, continuations_after) for each legal extension of cur."""
    out = []
    for b in range(n):
        v = cur ^ (1 << b)
        if on_snake[v] or neighbor_count[v] != 1:
            continue
        succ = 0
        for bb in range(n):
            w = v ^ (1 << bb)
            # after moving to v, w is legal iff unvisited and its only
            # snake neighbor would be v itself (count becomes 1)
            if not on_snake[w] and w != cur and neighbor_count[w] == 0:
                succ += 1
        out.append((v, succ))
    return out


def solve(n: int, seed: int, budget: int) -> list[int]:
    if getattr(priority, "_kinny_default", False):
        from experiments.snake_in_box import fastcore
        if fastcore.CORE_AVAILABLE:
            return fastcore.rollout_kinny(n, seed, budget)
    rng = random.Random(seed)
    size = 1 << n
    best: list[int] = [0]
    evals = 0
    while evals < budget:
        neighbor_count = [0] * size
        on_snake = [False] * size
        path = [0]
        on_snake[0] = True
        for b in range(n):
            neighbor_count[1 << b] += 1
        dims_used = [0] * n
        while True:
            cur = path[-1]
            options = _legal_with_counts(n, cur, neighbor_count, on_snake)
            evals += 1 + len(options)
            if not options or evals >= budget:
                break
            scored = [(priority(n, cur, v, succ, len(path) - 1, dims_used,
                                rng.random()), v) for v, succ in options]
            scored.sort(reverse=True)
            v = scored[0][1]
            dims_used[(v ^ cur).bit_length() - 1] += 1
            on_snake[v] = True
            path.append(v)
            for b in range(n):
                neighbor_count[v ^ (1 << b)] += 1
        if len(path) > len(best):
            best = list(path)
    return best
