"""Tests for pipeline/pack_tools.py (pack doctor + pack new)."""

from __future__ import annotations

import json

import pytest

from pipeline.pack_tools import doctor, scaffold


def _write_minimal_pack(root, pack_id="toy", smoke="python3 -c 'print(1)'"):
    d = root / pack_id
    d.mkdir(parents=True)
    (d / "pack.json").write_text(json.dumps({
        "id": pack_id, "display_name": "Toy", "description": "toy pack",
        "problem_shape": {
            "objective": "minimize x", "input_representation": "n",
            "output_representation": "x", "constraints": ["x >= 0"],
            "failure_modes": ["cheating"], "verification_kind": "exact_oracle",
        },
        "recommended_problem_template": "solve toy",
        "compose_query_hints": ["toy"],
    }))
    (d / "evaluator.json").write_text(json.dumps({
        "metric_primary": "x", "higher_is_better": False,
        "validator": "recheck", "smoke_command": smoke,
        "full_command": smoke, "pass_criteria": "beat sota",
    }))
    (d / "baselines.json").write_text(json.dumps({
        "baselines": [{"id": "sota", "name": "SOTA",
                       "level": "strong_reproducible",
                       "command": smoke, "notes": ""}],
    }))
    (d / "datasets.json").write_text(json.dumps({
        "datasets": [{"id": "syn", "name": "Synthetic", "kind": "synthetic",
                      "availability": "generated", "command": "", "notes": ""}],
    }))
    (d / "sota.json").write_text(json.dumps({
        "bar_levels": [{"level": "strong_reproducible", "available": True,
                        "comparator": "sota", "source": "", "notes": ""}],
    }))
    (d / "gotchas.md").write_text("# gotchas\n")
    return d


def test_doctor_ok_on_minimal_pack(tmp_path):
    _write_minimal_pack(tmp_path)
    report = doctor("toy", root=tmp_path, run_smoke=True, smoke_timeout_s=30)
    assert report.ok, report.render()
    assert any(name == "smoke run (exit 0)" and passed
               for name, passed, _ in report.checks)


def test_doctor_fails_on_broken_smoke_command(tmp_path):
    _write_minimal_pack(tmp_path, smoke="python3 -c 'import sys; sys.exit(3)'")
    report = doctor("toy", root=tmp_path, run_smoke=True, smoke_timeout_s=30)
    assert not report.ok
    assert "exit 3" in report.render()


def test_doctor_fails_on_missing_bench_module(tmp_path):
    _write_minimal_pack(
        tmp_path, smoke="uv run python -m experiments.does_not_exist.bench")
    report = doctor("toy", root=tmp_path, run_smoke=False)
    assert not report.ok
    assert "experiments.does_not_exist.bench" in report.render()


def test_doctor_incomputable_pack_reports_missing(tmp_path):
    (tmp_path / "empty").mkdir()
    report = doctor("empty", root=tmp_path, run_smoke=False)
    assert not report.ok
    assert not report.preflight["computable"]


def test_scaffold_writes_valid_loadable_pack(tmp_path):
    from pipeline.computable import load_pack, preflight_pack

    written = scaffold("my_domain", root=tmp_path / "packs",
                       experiments_root=tmp_path / "experiments")
    names = {p.name for p in written}
    assert {"pack.json", "evaluator.json", "baselines.json", "datasets.json",
            "sota.json", "gotchas.md", "bench.py"} <= names

    # The scaffold must load through the strict pydantic schemas as-is.
    pack = load_pack("my_domain", root=tmp_path / "packs")
    assert pack.pack.id == "my_domain"
    assert "{candidate}" in pack.evaluator.candidate_command
    assert "experiments.my_domain.bench" in pack.evaluator.smoke_command

    pre = preflight_pack("my_domain", root=tmp_path / "packs")
    assert pre.candidate_ready and pre.search_ready

    # The bench skeleton must at least be valid python.
    import ast
    bench = next(p for p in written if p.name == "bench.py")
    ast.parse(bench.read_text())


def test_scaffold_refuses_overwrite_and_bad_ids(tmp_path):
    scaffold("dup", root=tmp_path / "packs",
             experiments_root=tmp_path / "experiments")
    with pytest.raises(FileExistsError):
        scaffold("dup", root=tmp_path / "packs",
                 experiments_root=tmp_path / "experiments")
    with pytest.raises(ValueError):
        scaffold("Bad-Id", root=tmp_path / "packs")
