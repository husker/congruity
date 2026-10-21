"""Tests for the snake-in-the-box record miner."""

from __future__ import annotations

import json

from experiments.snake_in_box.baselines import validate_snake
from experiments.snake_in_box.record_run import main


def _run(tmp_path, extra):
    state = tmp_path / "state.json"
    rc = main(["--method", "greedy", "--dims", "4",
               "--attempt-budget", "2000", "--workers", "1",
               "--state", str(state)] + extra)
    assert rc == 0
    return json.loads(state.read_text())


def test_miner_finds_valid_best_and_saves_certificate(tmp_path):
    state = _run(tmp_path, ["--attempts", "3"])
    assert state["total_attempts"] == 3
    best = state["best"]["4"]
    ok, length = validate_snake(4, best["seq"])
    assert ok and int(length) == best["length"]
    certs = list((tmp_path / "certificates").glob("q4_len*.json"))
    assert certs, "certificate file written on improvement"
    cert = json.loads(certs[-1].read_text())
    assert validate_snake(4, cert["seq"])[0]


def test_miner_resume_never_reuses_seeds_and_keeps_best(tmp_path):
    s1 = _run(tmp_path, ["--attempts", "3"])
    s2 = _run(tmp_path, ["--attempts", "2"])
    assert s2["total_attempts"] == 5
    assert s2["next_seed"] == s1["next_seed"] + 2
    assert s2["best"]["4"]["length"] >= s1["best"]["4"]["length"]


def test_miner_merges_new_dims_into_state(tmp_path):
    _run(tmp_path, ["--attempts", "2"])
    state = tmp_path / "state.json"
    rc = main(["--method", "greedy", "--dims", "3,4",
               "--attempt-budget", "1000", "--workers", "1",
               "--attempts", "2", "--state", str(state)])
    assert rc == 0
    s = json.loads(state.read_text())
    assert s["dims"] == [3, 4]
    assert "3" in s["best"]
