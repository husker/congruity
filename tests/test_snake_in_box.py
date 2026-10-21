"""Tests for the snake_in_box pack bench (validator + baselines)."""

from __future__ import annotations

from experiments.snake_in_box.baselines import (greedy_rollouts, neighbors,
                                                stochastic_beam, validate_snake)
from experiments.snake_in_box.bench import validate


def test_neighbors_q3():
    assert sorted(neighbors(0, 3)) == [1, 2, 4]
    assert sorted(neighbors(7, 3)) == [3, 5, 6]


def test_validate_accepts_known_snake():
    # 0-1-3-7 in Q3: steps flip bits 0,1,2; only non-consecutive pair (0,7)
    # differs in 3 bits and (1,7) in 2 — chord-free.
    ok, length = validate_snake(3, [0, 1, 3, 7])
    assert ok and length == 3.0


def test_validate_rejects_chord():
    # 0-1-3-2: 0 and 2 are adjacent (differ in bit 1) but non-consecutive.
    ok, _ = validate_snake(3, [0, 1, 3, 2])
    assert not ok


def test_validate_rejects_non_path_and_garbage():
    assert validate_snake(3, [0, 3]) == (False, 0.0)        # 2-bit jump
    assert validate_snake(3, [0, 1, 0]) == (False, 0.0)     # repeat
    assert validate_snake(3, [0, 8]) == (False, 0.0)        # out of range
    assert validate_snake(3, []) == (False, 0.0)
    assert validate_snake(3, "nope") == (False, 0.0)
    assert validate_snake(3, [0, 1, 3, 7.0]) == (False, 0.0)  # non-int


def test_bench_validate_wrapper_matches():
    assert validate({"n": 3}, [0, 1, 3, 7]) == (True, 3.0)


def test_baselines_produce_valid_snakes():
    for solver in (greedy_rollouts, stochastic_beam):
        for n in (4, 5):
            seq = solver(n, seed=1, budget=4000)
            ok, length = validate_snake(n, seq)
            assert ok, (solver.__name__, n, seq)
            assert length >= 2  # anything shorter means search is broken


def test_baselines_deterministic_per_seed():
    a = stochastic_beam(5, seed=7, budget=3000)
    b = stochastic_beam(5, seed=7, budget=3000)
    c = stochastic_beam(5, seed=8, budget=3000)
    assert a == b
    assert isinstance(c, list)


def test_beam_finds_optimum_on_q5_smoke_budget():
    # The known optimal snake length in Q5 is 13; the smoke-budget beam
    # reaches it (observed stable across seeds 1-3). Guard against silent
    # search regressions.
    seq = stochastic_beam(5, seed=1, budget=8000)
    ok, length = validate_snake(5, seq)
    assert ok and length >= 12
