"""Primer playouts (Meta-NRPA harness design) in complete_certificate.py."""

import random

from experiments.snake_in_box.baselines import validate_snake
from experiments.snake_in_box.complete_certificate import complete, primer_attempt


def test_randomized_dfs_with_forbidden_first_child():
    # from [0,1] in Q3, forbid vertex 3 (the 0->1->3 continuation); the DFS
    # must diverge yet still find a valid snake
    rng = random.Random(7)
    best, nodes, exhausted = complete(3, [0, 1], node_cap=10_000, rng=rng,
                                      forbid_first={3})
    assert exhausted and nodes > 0
    assert best[2] != 3  # divergence enforced
    ok, _ = validate_snake(3, best)
    assert ok


def test_primer_attempt_returns_valid_snake_at_least_prefix_deep():
    # a suboptimal Q4 snake as primer; primer attempts must always return a
    # valid snake, and over a handful of seeds should find length >= primer
    primer = [0, 1, 3, 7, 6, 14]  # length-5 Q4 snake
    ok, plen = validate_snake(4, primer)
    assert ok and plen == 5
    best = 0
    for seed in range(6):
        out = primer_attempt(4, primer, seed, budget=20_000)
        ok, length = validate_snake(4, out)
        assert ok
        best = max(best, int(length))
    assert best >= 5  # never worse than replaying the primer
    # Q4 optimum is 7; randomized divergent DFS should reach it quickly
    assert best == 7


def test_primer_attempt_deterministic_per_seed():
    primer = [0, 1, 3, 7, 6, 14]
    a = primer_attempt(4, primer, 3, budget=5_000)
    b = primer_attempt(4, primer, 3, budget=5_000)
    assert a == b
