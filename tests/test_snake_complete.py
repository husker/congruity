"""Exhaustive certificate completion (experiments/snake_in_box/complete_certificate.py)."""

from experiments.snake_in_box.baselines import validate_snake
from experiments.snake_in_box.complete_certificate import complete, polish_sequence

# Q3: 0-1-3-7 is a valid length-3 snake; the Q3 optimum is length 4
# (e.g. 0-1-3-7-6). Unwinding back to the 0-1 prefix must find a 4.
_SUBOPTIMAL_Q3 = [0, 1, 3, 7]


def test_complete_finds_q3_optimum_from_prefix():
    best, nodes, exhausted = complete(3, [0, 1], node_cap=10_000)
    assert exhausted
    assert nodes > 0
    assert len(best) - 1 == 4
    ok, length = validate_snake(3, best)
    assert ok and length == 4


def test_complete_respects_node_cap():
    best, nodes, exhausted = complete(7, [0, 1], node_cap=50)
    assert not exhausted
    assert nodes <= 51
    ok, _ = validate_snake(7, best)
    assert ok


def test_polish_improves_suboptimal_snake():
    # unwind 4 removes all of 1,3,7 back past the improvable point
    out = polish_sequence(3, [0, 1, 3, 7, 6], max_unwind=4)
    ok, length = validate_snake(3, out)
    assert ok and length >= 4


def test_polish_passes_through_invalid_or_tiny_input():
    assert polish_sequence(3, [0, 5], max_unwind=8) == [0, 5]  # not a snake edge
    assert polish_sequence(3, [0, 1, 3], max_unwind=8) == [0, 1, 3]  # too short
