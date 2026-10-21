from __future__ import annotations

from experiments.snake_mcts_c2 import bench


def test_induced_path_validator_catches_shortcuts_and_bad_steps():
    assert bench.is_induced_path([0, 1, 3, 7], dimension=3)

    # 0 -> 1 -> 3 -> 2 uses valid hypercube edges, but 2 is adjacent to the
    # nonconsecutive start vertex 0, so the path is not induced.
    assert not bench.is_induced_path([0, 1, 3, 2], dimension=3)

    # 0 -> 3 flips two bits, so it is not even a hypercube path.
    assert not bench.is_induced_path([0, 3], dimension=3)


def test_legal_extensions_exclude_shortcut_neighbors():
    path = [0, 1, 3]

    assert bench.legal_extensions(path, dimension=3) == [7]


def test_benchmark_smoke_returns_valid_paths_with_matched_budgets():
    report = bench.run_benchmark(
        dimensions=[5],
        seeds_per_dimension=2,
        iterations=120,
        rollout_limit=40,
        greedy_restarts=120,
        seed=11,
    )

    variants = {row.variant for row in report.rows}
    assert variants == {"greedy", "plain_mcts", "novelty_metadynamic_mcts"}
    assert {row.dimension for row in report.rows} == {5}

    for row in report.rows:
        assert bench.is_induced_path(row.best_path, row.dimension)
        assert row.best_vertices == len(row.best_path)
        assert row.best_vertices >= 3

    plain = [row for row in report.rows if row.variant == "plain_mcts"]
    novel = [row for row in report.rows if row.variant == "novelty_metadynamic_mcts"]
    assert {row.seed for row in plain} == {row.seed for row in novel}
    assert all(row.iterations == 120 for row in plain + novel)
