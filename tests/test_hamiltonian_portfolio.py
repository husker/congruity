"""Tests for the Hamiltonian portfolio metric logic (offline, synthetic)."""

from __future__ import annotations

import numpy as np

from experiments.hamiltonian_regime_classifier.portfolio import (
    Data,
    eval_cost_aware,
    eval_speculative,
    mutuality,
    totals,
)


def _toy() -> Data:
    # 10 instances, 1 feature. CaDiCaL steady at 0.10; DFS usually faster
    # but a disaster on instance 2 (the fat tail). Hand-computed totals:
    #   sat_total = 1.00, dfs_total = 5.35, oracle = 0.45, best-single = CaDiCaL.
    sat = np.full(10, 0.10)
    dfs = np.array([0.01, 0.02, 5.00, 0.09, 0.03, 0.04, 0.02, 0.08, 0.01, 0.05])
    X = np.arange(10, dtype=float).reshape(-1, 1)
    fl = np.zeros(10, dtype=bool)
    return Data(X=X, sat=sat, dfs=dfs, dfs_timeout=fl.copy(), sat_timeout=fl.copy(), features=["f"])


def test_totals_oracle_and_best_single():
    d = _toy()
    t = totals(d)
    assert abs(t["oracle_total"] - 0.45) < 1e-9
    assert abs(t["best_single"] - 1.00) < 1e-9
    assert t["best_single_name"] == "CaDiCaL"
    # DFS wins all but instance 2 (9), CaDiCaL wins instance 2 (1)
    assert t["dfs_wins"] == 9
    assert t["sat_wins"] == 1
    assert abs(t["oracle_gain"] - (1.00 - 0.45) / 1.00) < 1e-12


def test_mutuality_absolute_catastrophe_one_sided():
    d = _toy()
    m = mutuality(d, catastrophe_s=0.5)
    # DFS has the absolute disaster (5.0s); CaDiCaL steady at 0.10s.
    assert m["sat_catastrophes"] == 0  # CaDiCaL never hits absolute catastrophe
    assert m["dfs_catastrophes"] == 1
    assert m["dfs_catastrophes_rescued_by_sat"] == 1  # CaDiCaL solves it fast
    assert m["sat_catastrophes_rescued_by_dfs"] == 0
    assert m["dfs_max"] == 5.0
    assert m["sat_max"] == 0.10


def test_policy_bounds_hold():
    d = _toy()
    t = totals(d)
    margins = [0.0, 0.5, 1.0]
    cost_total, _ = eval_cost_aware(d, seed=0, margins=margins)
    # any selection policy lies between oracle and the worse single total
    assert t["oracle_total"] - 1e-9 <= cost_total <= max(t["sat_total"], t["dfs_total"]) + 1e-9


def test_speculative_latency_le_cpu_and_safe():
    d = _toy()
    t = totals(d)
    spec = eval_speculative(d, seed=0, margins=[0.0, 0.5, 1.0])
    # 2-core latency never exceeds always-CaDiCaL, and CPU is >= latency
    assert spec["latency"] <= t["sat_total"] + 1e-9
    assert spec["cpu"] >= spec["latency"] - 1e-9
    # latency is bounded below by the oracle (can't beat picking the min)
    assert spec["latency"] >= t["oracle_total"] - 1e-9
