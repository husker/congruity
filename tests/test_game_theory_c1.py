from __future__ import annotations

import numpy as np

from experiments.game_theory_c1 import bench


def test_project_simplex_returns_probability_vector():
    projected = bench.project_simplex(np.array([0.6, -0.2, 1.6]))

    assert np.all(projected >= 0)
    assert np.isclose(projected.sum(), 1.0)
    assert np.allclose(projected, np.array([0.0, 0.0, 1.0]))


def test_nashconv_is_zero_for_matching_pennies_equilibrium():
    game = bench.matching_pennies_game()
    uniform = np.array([0.5, 0.5])

    metrics = bench.profile_metrics(game, uniform, uniform)
    pure_metrics = bench.profile_metrics(game, np.array([1.0, 0.0]), np.array([1.0, 0.0]))

    assert metrics.nashconv == 0.0
    assert pure_metrics.nashconv > 1.0


def test_no_u_turn_trigger_detects_simple_reversal_not_monotone_motion():
    reversal = [
        np.array([0.0, 0.0]),
        np.array([1.0, 0.0]),
        np.array([0.35, 0.0]),
    ]
    monotone = [
        np.array([0.0, 0.0]),
        np.array([0.4, 0.0]),
        np.array([0.8, 0.0]),
    ]

    assert bench.no_u_turn_trigger(reversal, window=2)
    assert not bench.no_u_turn_trigger(monotone, window=2)


def test_solver_returns_valid_strategy_and_history():
    game = bench.random_general_sum_game(8, np.random.default_rng(7))

    result = bench.solve_game(
        game,
        variant="c1_no_u_turn",
        iterations=80,
        seed=11,
        step_size=0.08,
    )

    assert result.variant == "c1_no_u_turn"
    assert result.x.shape == (8,)
    assert result.y.shape == (8,)
    assert np.isclose(result.x.sum(), 1.0)
    assert np.isclose(result.y.sum(), 1.0)
    assert np.all(result.x >= 0)
    assert np.all(result.y >= 0)
    assert result.best_nashconv <= result.history[0].nashconv
    assert result.elapsed_s >= 0.0


def test_c1_smoke_benchmark_compares_baselines_on_same_instances():
    report = bench.run_benchmark(
        families=["matching_pennies", "cyclic"],
        sizes=[6],
        seeds=[0, 1],
        iterations=100,
        step_sizes=[0.03, 0.08],
    )

    variants = {row.variant for row in report.rows}
    assert variants == {
        "multiplicative_weights",
        "regret_matching",
        "projected_gradient",
        "extragradient",
        "generic_restart",
        "c1_no_u_turn",
    }

    instance_ids = {row.instance_id for row in report.rows}
    for instance_id in instance_ids:
        rows = [row for row in report.rows if row.instance_id == instance_id]
        assert {row.variant for row in rows} == variants
        assert len({row.family for row in rows}) == 1

    for row in report.rows:
        assert row.best_nashconv >= 0.0
        assert row.final_nashconv >= 0.0
        assert row.x.shape == row.y.shape
        assert np.isclose(row.x.sum(), 1.0)
        assert np.isclose(row.y.sum(), 1.0)

    findings = bench.format_findings(report)
    assert "no-u-turn" in findings
    assert "multiplicative weights" in findings
    assert report.verdict in {"clean-positive", "partial-positive", "partial-negative", "clean-negative"}
