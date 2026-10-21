"""FunSearch mode: skeleton splicing (pipeline/autosearch.py) and the
snake priority skeleton (experiments/snake_in_box/priority_skeleton.py)."""

import pytest

from experiments.snake_in_box.baselines import validate_snake
from experiments.snake_in_box.priority_skeleton import solve
from pipeline import autosearch
from pipeline.autosearch import extract_priority, splice_skeleton

_SKELETON = """# header
# <PRIORITY-FUNCTION>
def priority(x):
    return 0.0
# </PRIORITY-FUNCTION>
def solve(n, seed, budget):
    return [0, 1]
"""


def test_splice_replaces_marked_region_only():
    out = splice_skeleton(_SKELETON, "def priority(x):\n    return x * 2\n")
    assert "return x * 2" in out
    assert "return 0.0" not in out
    assert out.startswith("# header")
    assert "def solve(n, seed, budget):" in out


def test_splice_rejects_output_without_priority():
    with pytest.raises(ValueError):
        splice_skeleton(_SKELETON, "def other(): pass")


def test_extract_priority_roundtrip():
    spliced = splice_skeleton(_SKELETON, "def priority(x):\n    return 7\n")
    assert extract_priority(spliced).startswith("def priority(x):")
    assert "return 7" in extract_priority(spliced)


def test_generate_initial_skeleton_mode_splices(monkeypatch, tmp_path):
    def fake_codegen(prompt, dest, stage):
        assert "ONLY the replacement" in prompt
        dest.write_text("def priority(x):\n    return 42.0\n")
        return dest
    monkeypatch.setattr(autosearch, "_codegen", fake_codegen)
    dest = tmp_path / "cand.py"
    autosearch.generate_initial({"synthesis_claim": "c"}, {}, None, "p",
                                dest, "e", stage="s", skeleton=_SKELETON)
    text = dest.read_text()
    assert "return 42.0" in text and "def solve(n, seed, budget):" in text


def test_snake_skeleton_is_a_working_solver():
    seq = solve(4, 3, budget=40_000)
    ok, length = validate_snake(4, seq)
    assert ok and int(length) >= 6  # Kinny seed should be near Q4's 7


def test_snake_skeleton_deterministic():
    assert solve(5, 9, 20_000) == solve(5, 9, 20_000)
