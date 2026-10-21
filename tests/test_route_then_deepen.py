"""Route-then-deepen (experiments/snake_in_box/route_then_deepen.py).

The pipeline must (a) never emit anything that fails the pack validator,
(b) never return worse than its input, and (c) actually recover known
headroom: a truncated Q7 optimum polishes back to 50.
"""

import json

import pytest

from experiments.snake_in_box import fastcore
from experiments.snake_in_box.baselines import validate_snake
from experiments.snake_in_box.route_then_deepen import deepen

pytestmark = pytest.mark.skipif(
    not fastcore.CORE_AVAILABLE, reason=f"no C core: {fastcore.LOAD_ERROR}")


def _q7_50():
    return json.load(open(
        "data/snake_in_box/certificates/q7_len50_completed.json"))["seq"]


def test_recovers_truncated_optimum(tmp_path):
    seq = _q7_50()[:-4]                       # 46e, headroom 4
    out = deepen(7, seq, widths=[8], extras=[1], unwind=16,
                 polish_cap=5_000_000, window_cap=2_000_000, stride=4,
                 checkpoint=tmp_path / "ck.json", max_rounds=2)
    ok, length = validate_snake(7, out)
    assert ok and int(length) == 50


def test_fixed_point_on_optimum(tmp_path):
    seq = _q7_50()
    out = deepen(7, seq, widths=[8], extras=[1], unwind=8,
                 polish_cap=2_000_000, window_cap=1_000_000, stride=8,
                 checkpoint=tmp_path / "ck.json", max_rounds=1)
    ok, length = validate_snake(7, out)
    assert ok and int(length) == 50
