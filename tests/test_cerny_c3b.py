from __future__ import annotations

from experiments.cerny_c3 import bench as c3
from experiments.cerny_c3b import bench


def test_previous_c3_candidate_is_rejected_as_attached_core():
    previous = c3.DFA(
        transitions=(
            (1, 1),
            (2, 1),
            (3, 2),
            (4, 3),
            (5, 4),
            (6, 5),
            (0, 6),
            (2, 7),
        )
    )

    strict = bench.score_strict(previous, label="previous_c3")

    assert strict.reset_length == 36
    assert not strict.strongly_connected
    assert strict.closed_core_size == 7
    assert strict.closed_core_reset_length == 36
    assert not strict.strictly_novel


def test_cerny_reference_is_strict_but_not_novel():
    dfa = c3.cerny_automaton(6)
    strict = bench.score_strict(dfa, label="cerny")

    assert strict.synchronizing
    assert strict.strongly_connected
    assert strict.closed_core_size is None
    assert strict.reset_length == 25
    assert not strict.strictly_novel


def test_letter_mutation_preserves_strong_connectivity():
    rng = bench.random.Random(7)
    mutated = bench.mutate_cycle_letter(c3.cerny_automaton(7), rng)
    strict = bench.score_strict(mutated, label="mutated")

    assert strict.strongly_connected


def test_c3b_smoke_reports_only_strict_novel_candidates_for_search_families():
    report = bench.run_benchmark(
        n_values=[6],
        trials_per_family=2,
        budget=40,
        seed=456,
    )

    families = {row.family for row in report.rows}
    assert families == {
        "cerny_reference",
        "strict_random",
        "strict_hill_climb",
        "strict_cycle_evolution",
    }

    for row in report.rows:
        assert row.best.synchronizing
        assert row.best.strongly_connected
        assert row.best.closed_core_size is None
        if row.family != "cerny_reference":
            assert row.best.strictly_novel

    findings = bench.format_findings(report)
    assert "## Best Strict Novel Candidate" in findings
    assert "closed-core attachments" in findings
    assert "near-Černý" in findings
