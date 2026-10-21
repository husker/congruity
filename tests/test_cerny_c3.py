from __future__ import annotations

from experiments.cerny_c3 import bench


def test_cerny_family_exact_threshold_matches_known_small_values():
    for n in (3, 4, 5):
        dfa = bench.cerny_automaton(n)
        score = bench.score_dfa(dfa)

        assert score.synchronizing
        assert score.reset_length == (n - 1) ** 2
        assert score.reset_word is not None
        assert bench.is_reset_word(dfa, score.reset_word)


def test_exact_oracle_identifies_nonsynchronizing_permutation_automaton():
    # Both letters are permutations, so no word can merge two states.
    dfa = bench.DFA(
        transitions=(
            (1, 2),
            (2, 0),
            (0, 1),
        )
    )

    score = bench.score_dfa(dfa)

    assert not score.synchronizing
    assert score.reset_length is None
    assert score.reset_word is None


def test_pareto_front_removes_dominated_and_duplicate_candidates():
    dfa = bench.cerny_automaton(3)
    rows = [
        bench.CandidateScore("a", dfa, True, 4, 10, (0, 1, 0, 1), "sig-a", 0),
        bench.CandidateScore("b", dfa, True, 3, 8, (0, 1, 0), "sig-b", 2),
        bench.CandidateScore("c", dfa, True, 4, 9, (0, 1, 0, 1), "sig-a", 1),
        bench.CandidateScore("d", dfa, True, 4, 11, (0, 1, 0, 1), "sig-d", 3),
    ]

    front = bench.pareto_front(rows)

    assert [row.label for row in front] == ["d"]


def test_novel_detection_excludes_state_relabelled_cerny_family():
    reference = bench.cerny_automaton(4)
    permutation = (2, 0, 3, 1)
    inverse = {old: new for new, old in enumerate(permutation)}
    relabelled = bench.DFA(
        transitions=tuple(
            tuple(inverse[reference.transitions[old_state][letter]] for letter in range(2))
            for old_state in permutation
        )
    )

    score = bench.score_dfa(relabelled, label="relabelled", reference=reference)

    assert score.signature == bench.score_dfa(reference, label="reference").signature
    assert score.distance_from_cerny == 0
    assert bench._best_novel([score]) is None


def test_benchmark_smoke_scores_all_generators_with_exact_validation():
    report = bench.run_benchmark(
        n_values=[5],
        alphabet_size=2,
        trials_per_family=2,
        budget=20,
        seed=123,
    )

    families = {row.family for row in report.rows}
    assert families == {"cerny_seed", "random_sampling", "hill_climb", "pareto_evolution"}

    for row in report.rows:
        assert row.best.synchronizing
        assert row.best.reset_length is not None
        assert row.best.reset_word is not None
        assert bench.is_reset_word(row.best.dfa, row.best.reset_word)

    cerny_rows = [row for row in report.rows if row.family == "cerny_seed"]
    assert cerny_rows[0].best.reset_length == 16
    assert report.summary["best_family_vs_random_delta"] >= 0

    findings = bench.format_findings(report)
    assert "## Best Novel Candidate" in findings
    assert "transition table" in findings
