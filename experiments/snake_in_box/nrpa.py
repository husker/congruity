"""NRPA solver for snake-in-the-box — the Meta-NRPA lineage, as a candidate.

Nested Rollout Policy Adaptation (Rosin 2011) with the additions that set
the d10-13 records (Dang et al., AAAI-23): primer warm-start (replay a
stored certificate to pre-adapt the policy before searching) and forced
exploration (mutate one choice of the incumbent best). The policy is a
softmax weight table over (vertex, dimension) moves, adapted toward the
best sequence at each nesting level — the *adaptive* half that plain
replay/DFS primer playouts lack.

Exposes the pack's standard candidate interface, so it runs under bench,
record_run, autoverify, and --polish unchanged:

    def solve(n: int, seed: int, budget: int) -> list[int]

Budget accounting: one candidate-extension evaluation per legal-successor
scan during a playout step, comparable with other candidates. Primer
warm-start is opt-in via the CONGRUITY_NRPA_PRIMER env var (path to a
certificate JSON with "seq"); the candidate interface stays frozen.
"""

from __future__ import annotations

import json
import math
import os
import random


def _legal_moves(n: int, cur: int, neighbor_count: list[int],
                 on_snake: list[bool]) -> list[int]:
    out = []
    for b in range(n):
        v = cur ^ (1 << b)
        if not on_snake[v] and neighbor_count[v] == 1:
            out.append(b)
    return out


class _State:
    __slots__ = ("n", "path", "neighbor_count", "on_snake")

    def __init__(self, n: int):
        self.n = n
        size = 1 << n
        self.neighbor_count = [0] * size
        self.on_snake = [False] * size
        self.path = [0]
        self.on_snake[0] = True
        for b in range(n):
            self.neighbor_count[1 << b] += 1

    def push(self, bit: int) -> None:
        v = self.path[-1] ^ (1 << bit)
        self.on_snake[v] = True
        self.path.append(v)
        for b in range(self.n):
            self.neighbor_count[v ^ (1 << b)] += 1

    def pop_to(self, length: int) -> None:
        while len(self.path) > length:
            v = self.path.pop()
            self.on_snake[v] = False
            for b in range(self.n):
                self.neighbor_count[v ^ (1 << b)] -= 1


def _playout(state: _State, policy: dict, rng: random.Random,
             evals: list[int], budget: int) -> list[int]:
    """One rollout to a dead end, sampling moves by softmax policy weight."""
    moves: list[int] = []
    while evals[0] < budget:
        cur = state.path[-1]
        legal = _legal_moves(state.n, cur, state.neighbor_count, state.on_snake)
        evals[0] += 1
        if not legal:
            break
        weights = [math.exp(min(20.0, policy.get((cur, b), 0.0))) for b in legal]
        total = sum(weights)
        r = rng.random() * total
        acc = 0.0
        pick = legal[-1]
        for b, w in zip(legal, weights):
            acc += w
            if r <= acc:
                pick = b
                break
        moves.append(pick)
        state.push(pick)
    return moves


def _adapt(policy: dict, n: int, seq_moves: list[int], alpha: float = 1.0) -> None:
    """Shift policy weight toward each move of the target sequence (NRPA rule)."""
    state = _State(n)
    for bit in seq_moves:
        cur = state.path[-1]
        legal = _legal_moves(n, cur, state.neighbor_count, state.on_snake)
        if bit not in legal:
            break  # target diverges from what is now legal; stop adapting
        weights = [math.exp(min(20.0, policy.get((cur, b), 0.0))) for b in legal]
        total = sum(weights)
        for b, w in zip(legal, weights):
            policy[(cur, b)] = policy.get((cur, b), 0.0) - alpha * w / total
        policy[(cur, bit)] = policy.get((cur, bit), 0.0) + alpha
        state.push(bit)


def _moves_from_seq(seq: list[int]) -> list[int]:
    return [(seq[i] ^ seq[i + 1]).bit_length() - 1 for i in range(len(seq) - 1)]


def _nrpa(level: int, n: int, policy: dict, rng: random.Random,
          evals: list[int], budget: int,
          incumbent: list[int] | None) -> list[int]:
    if level == 0:
        state = _State(n)
        return _playout(state, policy, rng, evals, budget)
    best: list[int] = list(incumbent) if incumbent else []
    child = dict(policy)
    iterations = 0
    while evals[0] < budget:
        iterations += 1
        moves = _nrpa(level - 1, n, child, rng, evals, budget, None)
        if len(moves) > len(best):
            best = moves
        if best:
            # forced exploration (Meta-NRPA): occasionally mutate one choice
            if iterations % 7 == 0 and len(best) > 2:
                i = rng.randrange(len(best))
                mutated = best[:i] + [rng.randrange(n)]
                _adapt(child, n, mutated)
            else:
                _adapt(child, n, best)
        if level >= 2 and iterations >= 10:
            break
    return best


def solve(n: int, seed: int, budget: int) -> list[int]:
    rng = random.Random(seed)
    policy: dict = {}
    evals = [0]
    incumbent: list[int] | None = None

    primer_path = os.environ.get("CONGRUITY_NRPA_PRIMER", "")
    if primer_path and os.path.exists(primer_path):
        try:
            data = json.load(open(primer_path))
            seq = data.get("seq") or data["best"][str(n)]["seq"]
            incumbent = _moves_from_seq(seq)
            for _ in range(3):  # pre-adapt the policy toward the primer
                _adapt(policy, n, incumbent)
        except Exception:
            incumbent = None  # bad primer degrades to plain NRPA

    best = _nrpa(2, n, policy, rng, evals, budget, incumbent)
    # replay best moves into a vertex sequence
    state = _State(n)
    for bit in best:
        cur = state.path[-1]
        if bit not in _legal_moves(n, cur, state.neighbor_count, state.on_snake):
            break
        state.push(bit)
    return list(state.path)
