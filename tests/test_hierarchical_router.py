"""Hierarchical router v2 (experiments/snake_in_box/hierarchical_router.py).

Two properties matter: every output survives the pack validator (routers
have produced chord-carrying "snakes" before — the crossing-vs-blocked
bug), and probe_cap=0 reproduces the pre-probe greedy behavior exactly,
so the bilevel scoring is a strict opt-in and the 152 baseline remains
reproducible for A/B.
"""

import pytest

from experiments.snake_in_box import fastcore
from experiments.snake_in_box.baselines import validate_snake
from experiments.snake_in_box.hierarchical_router import hroute

pytestmark = pytest.mark.skipif(
    not fastcore.CORE_AVAILABLE, reason=f"no C core: {fastcore.LOAD_ERROR}")

KS7 = {7: 1, 8: 0, 9: 0}


def test_greedy_q7_valid():
    snake = hroute(7, 0, set(), set(), KS7, 200_000, beam_width=3)
    ok, length = validate_snake(7, snake)
    assert ok and length >= 30


def test_probe_q7_valid():
    snake = hroute(7, 0, set(), set(), KS7, 200_000, beam_width=3,
                   probe_cap=50_000, probe_top=8)
    ok, length = validate_snake(7, snake)
    assert ok and length >= 30


def test_probe_zero_reproduces_greedy():
    a = hroute(7, 0, set(), set(), KS7, 200_000, beam_width=3)
    b = hroute(7, 0, set(), set(), KS7, 200_000, beam_width=3, probe_cap=0)
    assert a == b


def test_probe_q8_valid_multicrossing():
    ks = {7: 2, 8: 1, 9: 0}
    snake = hroute(8, 0, set(), set(), ks, 200_000, beam_width=3,
                   probe_cap=50_000, probe_top=8)
    ok, length = validate_snake(8, snake)
    assert ok and length >= 60
