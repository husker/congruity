"""Tests for the graph-coloring bench: instances, validator, and baselines."""

from __future__ import annotations

from experiments.graph_coloring.baselines import (
    dsatur,
    greedy_random,
    tabucol,
    validate_coloring,
)
from experiments.graph_coloring.instances import (
    INSTANCE_GENERATORS,
    gnp,
    planted_k,
    queen_graph,
)


def test_instances_are_deterministic():
    a = gnp(40, 0.5, seed=7)
    b = gnp(40, 0.5, seed=7)
    assert a == b
    assert gnp(40, 0.5, seed=8) != a


def test_named_instances_load():
    for name, gen in INSTANCE_GENERATORS.items():
        adj = gen()
        assert len(adj) > 0, name
        # Adjacency must be symmetric with no self-loops.
        for u, nbrs in enumerate(adj):
            assert u not in nbrs
            for v in nbrs:
                assert u in adj[v]


def test_validator_rejects_conflicts_and_counts_colors():
    adj = gnp(30, 0.5, seed=1)
    colors = dsatur(adj)
    ok, n_colors, conflicts = validate_coloring(adj, colors)
    assert ok and conflicts == 0
    assert n_colors == max(colors) + 1

    # Force a conflict: give a vertex the same color as one of its neighbors.
    u = next(v for v in range(len(adj)) if adj[v])
    bad = colors[:]
    bad[u] = colors[next(iter(adj[u]))]
    ok, _, conflicts = validate_coloring(adj, bad)
    assert not ok and conflicts > 0


def test_baselines_produce_proper_colorings():
    adj = gnp(40, 0.5, seed=2)
    for colors in (greedy_random(adj, seed=1, restarts=10), dsatur(adj),
                   tabucol(adj, seed=1, iters_per_k=2000)):
        ok, _, conflicts = validate_coloring(adj, colors)
        assert ok and conflicts == 0


def test_tabucol_beats_dsatur_on_queen8():
    adj = queen_graph(8)
    k_dsatur = max(dsatur(adj)) + 1
    colors = tabucol(adj, seed=1, iters_per_k=5000)
    ok, k_tabu, _ = validate_coloring(adj, colors)
    assert ok
    # queen8 has chromatic number 9; TabuCol should find it, DSATUR does not.
    assert k_tabu == 9
    assert k_tabu < k_dsatur


def test_tabucol_recovers_planted_coloring():
    adj = planted_k(60, 6, 0.5, seed=3)
    colors = tabucol(adj, seed=1, iters_per_k=5000)
    ok, k, _ = validate_coloring(adj, colors)
    assert ok
    assert k <= 6


def _run_bench_json(capsys, extra_args):
    import json

    from experiments.graph_coloring import bench

    rc = bench.main(["--instances", "gnp60_05", "--seeds", "1",
                     "--tabu-iters", "2000", "--json", *extra_args])
    out = capsys.readouterr().out.strip().splitlines()
    return rc, json.loads(out[-1])


def test_bench_json_contract(capsys):
    rc, result = _run_bench_json(capsys, [])
    assert rc == 0
    assert result["metric"] == "colors_used"
    assert result["higher_is_better"] is False
    assert result["home_field_sota_method"] == "tabucol"
    assert result["scores"]["tabucol"] == result["scores"]["tabucol"]  # finite
    assert result["scores"]["tabucol"] <= result["scores"]["dsatur"]
    assert result["n_trials"] == 1
    assert result["per_instance"]["tabucol"]["gnp60_05"] == result["scores"]["tabucol"]


def test_bench_seed_base_changes_stochastic_runs(capsys):
    # greedy with a different seed base shuffles differently; the run must
    # succeed and report the same contract either way.
    rc, base1 = _run_bench_json(capsys, ["--seed-base", "1", "--methods", "greedy"])
    assert rc == 0
    rc, base101 = _run_bench_json(capsys, ["--seed-base", "101", "--methods", "greedy"])
    assert rc == 0
    assert base1["metric"] == base101["metric"] == "colors_used"
    assert base101["scores"]["greedy"] == base101["scores"]["greedy"]  # finite


def test_bench_candidate_plugs_in(capsys, tmp_path):
    cand = tmp_path / "cand.py"
    cand.write_text(
        "from experiments.graph_coloring.baselines import dsatur\n"
        "def solve(adj, seed, iters):\n"
        "    return dsatur(adj)\n")
    rc, result = _run_bench_json(capsys, ["--candidate", str(cand)])
    assert rc == 0
    assert result["composition_method"] == "candidate"
    assert result["scores"]["candidate"] == result["scores"]["dsatur"]


def test_bench_per_trial_and_ablation(capsys, tmp_path):
    cand = tmp_path / "cand.py"
    cand.write_text(
        "from experiments.graph_coloring.baselines import dsatur, greedy_random\n"
        "def solve(adj, seed, iters):\n"
        "    return dsatur(adj)\n"
        "def solve_ablated(adj, seed, iters):\n"
        "    return greedy_random(adj, seed, restarts=5)\n")
    rc, result = _run_bench_json(capsys, ["--candidate", str(cand)])
    assert rc == 0
    assert result["ablated_method"] == "candidate_ablated"
    assert result["scores"]["candidate_ablated"] == result["scores"]["candidate_ablated"]
    # per_trial: one instance, one seed, every method traced
    assert result["per_trial"]["candidate"]["gnp60_05"][0] == \
        result["scores"]["candidate"]
    assert len(result["per_trial"]["tabucol"]["gnp60_05"]) == 1


def test_bench_invalid_candidate_scores_nan(capsys, tmp_path):
    cand = tmp_path / "cheat.py"
    cand.write_text(
        "def solve(adj, seed, iters):\n"
        "    return [0] * len(adj)\n")  # one color: massively conflicted
    rc, result = _run_bench_json(capsys, ["--candidate", str(cand)])
    assert rc == 0  # json mode reports via NaN, not exit code
    assert result["scores"]["candidate"] != result["scores"]["candidate"]  # NaN
