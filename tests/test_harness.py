"""Tests for the shared experiment harness (experiments/_harness.py)."""

from __future__ import annotations

import json

import pytest

from experiments._harness import (_THREAD_ENV_VARS, budget_loop,
                                  build_result_contract, cap_threads,
                                  emit_result_contract, make_rng)


def test_cap_threads_sets_all_vars(monkeypatch):
    for var in _THREAD_ENV_VARS:
        monkeypatch.delenv(var, raising=False)
    cap_threads()
    import os
    assert all(os.environ[v] == "1" for v in _THREAD_ENV_VARS)


def test_cap_threads_respects_explicit_exports(monkeypatch):
    monkeypatch.setenv("OMP_NUM_THREADS", "8")
    cap_threads()
    import os
    assert os.environ["OMP_NUM_THREADS"] == "8"  # shell export wins


def test_make_rng_deterministic_and_decorrelated():
    a1 = make_rng(42, 13, 0).random(4)
    a2 = make_rng(42, 13, 0).random(4)
    b = make_rng(42, 13, 1).random(4)
    assert (a1 == a2).all()
    assert not (a1 == b).all()


def test_budget_loop_min_iterations():
    assert list(budget_loop(0.0, min_iterations=3)) == [0, 1, 2]


def test_budget_loop_stops_on_budget():
    import time
    start = time.perf_counter()
    n = 0
    for _ in budget_loop(0.05):
        time.sleep(0.02)
        n += 1
    assert n >= 1
    assert time.perf_counter() - start < 1.0


def test_build_result_contract_validates_method_keys():
    with pytest.raises(ValueError, match="composition_method"):
        build_result_contract(
            metric="m", higher_is_better=True, scores={"sota": 1.0},
            composition_method="comp", home_field_sota_method="sota",
            n_trials=1)
    with pytest.raises(ValueError, match="home_field_sota_method"):
        build_result_contract(
            metric="m", higher_is_better=True, scores={"comp": 1.0},
            composition_method="comp", home_field_sota_method="sota",
            n_trials=1)


def test_emit_result_contract_last_line_parses(capsys):
    out = emit_result_contract(
        metric="min_area", higher_is_better=True,
        scores={"comp": 2.0, "sota": 1.0},
        composition_method="comp", home_field_sota_method="sota",
        n_trials=5, notes="ok", record_claims=[{"n": 13, "value": 2.0}])
    last_line = capsys.readouterr().out.strip().splitlines()[-1]
    parsed = json.loads(last_line)
    assert parsed == out
    assert parsed["record_claims"] == [{"n": 13, "value": 2.0}]


def test_contract_shape_matches_autoverify_expectations():
    from pipeline.autoverify import RESULT_CONTRACT
    contract = build_result_contract(
        metric="m", higher_is_better=False, scores={"c": 1.0, "s": 2.0},
        composition_method="c", home_field_sota_method="s", n_trials=3,
        notes="n")
    assert set(RESULT_CONTRACT) <= set(contract)
