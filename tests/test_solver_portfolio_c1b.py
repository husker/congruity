from __future__ import annotations

import numpy as np

from experiments.solver_portfolio_c1 import bench as c1
from experiments.solver_portfolio_c1b import bench as c1b


def test_hybrid_instances_include_unseen_families():
    instances = c1b.generate_hybrid_instances(sizes=[20], seeds_per_family=2, base_seed=7)

    assert {inst.family for inst in instances} == set(c1b.HYBRID_FAMILIES)
    assert len(instances) == len(c1b.HYBRID_FAMILIES) * 2
    for inst in instances:
        assert inst.coords.shape == (20, 2)
        assert np.all(inst.coords >= 0.0)
        assert np.all(inst.coords <= 1.0)


def test_leave_one_family_out_splits_hold_out_each_family():
    families = np.array(["a", "a", "b", "b", "c", "c"])
    splits = c1b.leave_one_family_out_splits(families)

    assert [name for name, _, _ in splits] == ["a", "b", "c"]
    for held_out, train_idx, test_idx in splits:
        assert set(families[test_idx]) == {held_out}
        assert held_out not in set(families[train_idx])
        assert len(train_idx) == 4
        assert len(test_idx) == 2


def test_cost_sensitive_selector_learns_solver_loss_surface():
    features = np.array(
        [
            [0.00],
            [0.10],
            [0.20],
            [0.80],
            [0.90],
            [1.00],
        ],
        dtype=float,
    )
    losses = np.array(
        [
            [0.01, 0.35],
            [0.02, 0.30],
            [0.03, 0.28],
            [0.25, 0.03],
            [0.30, 0.02],
            [0.35, 0.01],
        ],
        dtype=float,
    )
    train_idx = np.array([0, 1, 3, 4])
    test_idx = np.array([2, 5])
    single = c1.evaluate_single_best(losses, train_idx, test_idx, solver_names=["left", "right"])

    result, importances = c1b.evaluate_cost_sensitive_regressor(
        features=features,
        losses=losses,
        train_idx=train_idx,
        test_idx=test_idx,
        baseline_regret=single.mean_regret,
        seed=3,
        feature_names=["shape_signal"],
        solver_names=["left", "right"],
    )

    assert result.mean_regret < single.mean_regret
    assert result.chosen_solver_names == ["left", "right"]
    assert importances[0][0] == "shape_signal"
    assert importances[0][1] > 0.0
