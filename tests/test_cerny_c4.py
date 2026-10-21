from __future__ import annotations

from experiments.cerny_c3 import bench as c3
from experiments.cerny_c4 import bench


def test_pair_landmarks_are_real_reset_obligations():
    dfa = c3.cerny_automaton(4)
    distances = bench.pair_merge_distances(dfa)

    assert distances[(0, 1)] == 1
    assert distances[(0, 2)] > 1
    assert bench.unresolved_pair_count((1 << dfa.n_states) - 1) == 6
    assert bench.landmark_distance_sum(dfa, 1, distances) == 0


def test_landmark_greedy_finds_valid_reset_word_on_cerny():
    dfa = c3.cerny_automaton(5)
    result = bench.landmark_greedy_solver(dfa)

    assert result.reset_word is not None
    assert c3.is_reset_word(dfa, result.reset_word)
    assert result.found_length == len(result.reset_word)


def test_bounded_certificate_reports_optimality_gap():
    dfa = c3.cerny_automaton(5)
    optimal = c3.shortest_reset_word(dfa)[0]
    greedy = bench.landmark_greedy_solver(dfa)

    certified = bench.certify_result(dfa, greedy)

    assert certified.optimal_length == optimal
    assert certified.optimality_gap == greedy.found_length - optimal
    assert certified.certified_optimal == (greedy.found_length == optimal)


def test_bounded_certificate_penalizes_missing_incumbent():
    dfa = c3.cerny_automaton(5)
    missing = bench.SolverResult(
        variant="plain_mcts",
        dfa=dfa,
        instance_id="missing",
        reset_word=None,
        found_length=30,
        iterations=1,
        expanded_nodes=1,
        elapsed_s=0.0,
    )

    certified = bench.certify_result(dfa, missing)

    assert certified.optimal_length == 16
    assert certified.optimality_gap == 14
    assert not certified.certified_optimal


def test_c4_smoke_compares_mcts_variants_on_same_instances():
    report = bench.run_benchmark(
        n_values=[6],
        instances_per_n=2,
        iterations=80,
        rollout_depth_factor=3,
        seed=321,
    )

    variants = {row.variant for row in report.rows}
    assert variants == {"greedy_pair", "plain_mcts", "landmark_mcts"}

    instance_ids = {row.instance_id for row in report.rows}
    for instance_id in instance_ids:
        rows = [row for row in report.rows if row.instance_id == instance_id]
        assert {row.variant for row in rows} == variants
        assert len({row.optimal_length for row in rows}) == 1

    for row in report.rows:
        if row.reset_word is not None:
            assert c3.is_reset_word(row.dfa, row.reset_word)
        assert row.found_length >= row.optimal_length

    findings = bench.format_findings(report)
    assert "landmark MCTS" in findings
    assert "bounded exact-BFS certificate" in findings
