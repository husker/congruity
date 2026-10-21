from __future__ import annotations

from experiments.cerny_c3 import bench as c3
from experiments.cerny_c3b import bench as c3b
from experiments.cerny_c3c import bench


def test_distance_gate_rejects_c3b_best_candidate():
    c3b_best = c3.DFA(
        transitions=(
            (1, 0),
            (2, 4),
            (3, 2),
            (4, 3),
            (5, 4),
            (6, 5),
            (7, 6),
            (8, 7),
            (9, 8),
            (0, 9),
        )
    )

    score = c3b.score_strict(c3b_best, label="c3b_best")

    assert score.distance_from_cerny == 2
    assert not bench.passes_distance_gate(score, min_distance=4)


def test_seeded_far_cycle_candidate_preserves_threshold_distance():
    dfa = bench.far_cycle_seed(n_states=8, min_distance=4)
    score = c3b.score_strict(dfa, label="far_seed")

    assert c3b.is_strongly_connected(dfa)
    assert score.distance_from_cerny >= 4


def test_c3c_smoke_reports_distance_thresholds_separately():
    report = bench.run_benchmark(
        n_values=[7],
        distance_thresholds=[4, 6],
        trials_per_family=1,
        budget=50,
        seed=789,
    )

    assert {row.min_distance for row in report.rows} == {4, 6}
    assert {row.family for row in report.rows} == {
        "far_random",
        "far_hill_climb",
        "far_seeded_hill_climb",
        "far_cycle_evolution",
    }

    for row in report.rows:
        assert row.best.synchronizing
        assert row.best.strongly_connected
        assert row.best.closed_core_size is None
        assert row.best.distance_from_cerny >= row.min_distance
        assert row.best.strictly_novel

    findings = bench.format_findings(report)
    assert "distance >= 4" in findings
    assert "distance >= 6" in findings
    assert "near-Černý advantage survives" in findings or "advantage collapses" in findings
