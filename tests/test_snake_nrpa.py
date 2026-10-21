"""NRPA candidate (experiments/snake_in_box/nrpa.py)."""

import json

from experiments.snake_in_box.baselines import validate_snake
from experiments.snake_in_box.nrpa import solve


def test_nrpa_returns_valid_snake_and_finds_q4_optimum():
    best = 0
    for seed in range(4):
        seq = solve(4, seed, budget=30_000)
        ok, length = validate_snake(4, seq)
        assert ok
        best = max(best, int(length))
    assert best == 7  # Q4 proven optimum


def test_nrpa_deterministic_per_seed():
    assert solve(5, 11, 20_000) == solve(5, 11, 20_000)


def test_nrpa_q5_reaches_near_optimum():
    best = max(int(validate_snake(5, solve(5, s, 60_000))[1]) for s in range(4))
    assert best >= 11  # proven optimum 13; near-optimal at tiny budget


def test_nrpa_primer_warm_start(tmp_path, monkeypatch):
    # prime with a known-good Q4 snake; warm-started solve must never
    # return something worse than a valid snake and should reach 7 fast
    primer = {"n": 4, "length": 5, "seq": [0, 1, 3, 7, 6, 14]}
    p = tmp_path / "primer.json"
    p.write_text(json.dumps(primer))
    monkeypatch.setenv("CONGRUITY_NRPA_PRIMER", str(p))
    seq = solve(4, 1, budget=30_000)
    ok, length = validate_snake(4, seq)
    assert ok and int(length) >= 5


def test_nrpa_bad_primer_degrades_gracefully(tmp_path, monkeypatch):
    p = tmp_path / "junk.json"
    p.write_text("{not json")
    monkeypatch.setenv("CONGRUITY_NRPA_PRIMER", str(p))
    seq = solve(4, 2, budget=10_000)
    ok, _ = validate_snake(4, seq)
    assert ok
