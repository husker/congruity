"""Mid-body window repair (experiments/snake_in_box/window_repair.py)."""

import json

from experiments.snake_in_box.baselines import validate_snake
from experiments.snake_in_box.window_repair import window_repair


def _cert(name):
    return json.load(open(f"data/snake_in_box/certificates/{name}.json"))["seq"]


def test_extra0_recovers_a_valid_snake():
    seq = _cert("q7_len50_completed")
    repaired, nodes, exhausted = window_repair(7, seq, 10, 25, extra=0)
    assert repaired is not None and nodes > 0
    ok, length = validate_snake(7, repaired)
    assert ok and int(length) == 50


def test_positive_control_end_window_repairs_49_to_50():
    # the one improvement we KNOW exists: the 49 -> 50 completion (found
    # 2026-07-04 at unwind 24). As an end-window (empty suffix) the repair
    # must rediscover it — proving the machinery FINDS real improvements.
    seq49 = _cert("q7_len49")
    i = 25  # keep 26 vertices, the prefix shared with the optimum
    repaired, _n, _e = window_repair(7, seq49, i, len(seq49), extra=1,
                                     node_cap=200_000_000)
    assert repaired is not None
    ok, length = validate_snake(7, repaired)
    assert ok and int(length) == 50


def test_not_found_reports_exhausted_or_capped_honestly():
    seq = _cert("q7_len50_completed")  # proven optimum: +1 must not exist
    repaired, _nodes, exhausted = window_repair(7, seq, 12, 30, extra=1)
    assert repaired is None
    assert exhausted  # small window in Q7: cap not hit, so this is a proof
