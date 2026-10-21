"""C inner-loop core (experiments/snake_in_box/fastcore.py + snake_core.c).

The load-bearing test is EXACT equality between the C deterministic completion
and the pure-Python reference — that is what lets complete() dispatch to C
without changing any downstream result. Skipped cleanly where no compiler is
available (the fallback path is then what runs in production)."""

import json

import pytest

from experiments.snake_in_box import fastcore
from experiments.snake_in_box.baselines import validate_snake
from experiments.snake_in_box.complete_certificate import _complete_py, complete

pytestmark = pytest.mark.skipif(
    not fastcore.CORE_AVAILABLE, reason=f"no C core: {fastcore.LOAD_ERROR}")


def _seq():
    return json.load(open("data/snake_in_box/certificates/q7_len49.json"))["seq"]


def test_c_complete_exactly_matches_python_reference():
    seq = _seq()
    for k in (12, 20, 28):
        prefix = seq[: len(seq) - k]
        py = _complete_py(7, prefix, 20_000_000)
        c = fastcore.complete(7, prefix, 20_000_000, rng_seed=-1)
        assert py == c, f"mismatch at unwind {k}"


def test_complete_dispatches_to_c_and_stays_correct():
    # complete() (the public entry) must give the same answer as the reference
    seq = _seq()
    prefix = seq[: len(seq) - 24]
    assert complete(7, prefix, 20_000_000) == _complete_py(7, prefix, 20_000_000)


def test_c_complete_node_cap_reports_unexhausted():
    seq = _seq()
    prefix = seq[: len(seq) - 32]
    _best, nodes, exhausted = fastcore.complete(7, prefix, 1000, rng_seed=-1)
    assert not exhausted and nodes <= 1001


def test_c_rollout_produces_valid_snakes():
    for n, budget in ((7, 2_000_000), (9, 5_000_000)):
        for s in range(2):
            seq = fastcore.rollout_kinny(n, s, budget)
            ok, _ = validate_snake(n, seq)
            assert ok


def test_c_rollout_reaches_kinny_level():
    best = max(int(validate_snake(9, fastcore.rollout_kinny(9, s, 10_000_000))[1])
               for s in range(3))
    assert best >= 150  # Kinny prior level at Q9


def test_skeleton_solve_uses_c_and_stays_valid():
    from experiments.snake_in_box.priority_skeleton import solve
    ok, length = validate_snake(7, solve(7, 3, 2_000_000))
    assert ok and int(length) >= 44
