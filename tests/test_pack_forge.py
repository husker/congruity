"""Tests for pipeline/pack_forge.py — the deterministic forge spine with
stubbed codex, a real local git repo for pin, and stubbed record fetches."""

from __future__ import annotations

import json
import subprocess

import pytest

import pipeline.llm_io as llm_io
from pipeline.pack_forge import forge, promote, workspace_for

BENCH_PY = '''\
"""Toy bench for forge tests."""


def validate(instance, solution):
    ok = solution == instance["target"]
    return ok, float(solution)
'''

BROKEN_BENCH_PY = '''\
def validate(instance, solution):
    return True, 0.0  # accepts everything — battery must catch this
'''


def _minimal_pack_dicts(pack_id: str, smoke: str) -> dict:
    return {
        "pack": {
            "id": pack_id, "display_name": "Toy", "description": "toy pack",
            "problem_shape": {
                "objective": "minimize x", "input_representation": "n",
                "output_representation": "x", "constraints": ["x >= 0"],
                "failure_modes": ["cheating"],
                "verification_kind": "exact_oracle",
            },
            "recommended_problem_template": "solve toy",
            "compose_query_hints": ["toy"],
        },
        "evaluator": {
            "metric_primary": "x", "higher_is_better": False,
            "validator": "recheck", "smoke_command": smoke,
            "full_command": smoke, "pass_criteria": "beat sota",
        },
        "baselines": {"baselines": [
            {"id": "sota", "name": "SOTA", "level": "strong_reproducible",
             "command": smoke, "notes": ""}]},
        "datasets": {"datasets": [
            {"id": "syn", "name": "Synthetic", "kind": "synthetic",
             "availability": "generated", "command": "", "notes": ""}]},
        "sota": {"bar_levels": [
            {"level": "strong_reproducible", "available": True,
             "comparator": "sota", "source": "", "notes": ""}]},
    }


def _make_sota_repo(tmp_path, value: float = 1.23):
    repo = tmp_path / "sota_repo"
    repo.mkdir()
    (repo / "repro.py").write_text(
        f'import json; print("warmup"); print(json.dumps({{"value": {value}}}))\n')
    for cmd in (["git", "init", "-q"],
                ["git", "add", "-A"],
                ["git", "-c", "user.email=t@t", "-c", "user.name=t",
                 "commit", "-qm", "init"]):
        subprocess.run(cmd, cwd=repo, check=True, capture_output=True)
    return repo


def _draft_json(pack_id: str, repo_url: str, *, bench: str = BENCH_PY,
                published: float = 1.25, tol: float = 0.05,
                record_sources: list | None = None,
                smoke: str = "python3 -c 'print(1)'") -> str:
    draft = _minimal_pack_dicts(pack_id, smoke)
    draft.update({
        "gotchas_md": "# gotchas\n",
        "bench_py": bench,
        "sota_provenance": {
            "method_name": "ToySOTA",
            "paper_title": "A Toy Paper",
            "paper_url": "https://example.org/paper",
            "repo_url": repo_url,
            "repo_ref": "",
            "reproduce_command": "python3 repro.py",
            "published_result": {
                "benchmark": "toy-bench", "metric": "x",
                "value": published, "rel_tolerance": tol,
                "source": "Table 1 of A Toy Paper",
            },
        },
        "validator_cases": [
            {"name": "basic", "instance": {"target": 3},
             "good_solution": 3, "bad_solution": 4, "good_value": 3.0},
            {"name": "zero", "instance": {"target": 0},
             "good_solution": 0, "bad_solution": 1},
        ],
        "record_sources": record_sources or [],
    })
    return json.dumps(draft)


def _stub_codex(monkeypatch, response: str, *, fail_if_called: bool = False):
    calls = []

    def fake(prompt, reasoning, timeout_s=None):
        if fail_if_called:
            raise AssertionError("codex was called but should not have been")
        calls.append(prompt)
        return True, response, ""

    monkeypatch.setattr(llm_io, "_invoke_codex", fake)
    return calls


def test_forge_all_stages_pass(tmp_path, monkeypatch):
    repo = _make_sota_repo(tmp_path, value=1.23)
    _stub_codex(monkeypatch, _draft_json("toy", str(repo),
                                         record_sources=["https://records.example/t"]))
    fetched = []

    def fake_fetch(url):
        fetched.append(url)
        return "<html>records</html>"

    state = forge("toy", "toy domain", root=tmp_path / "forge", fetch=fake_fetch)
    assert all(state["stages"][s]["ok"] for s in
               ("draft", "pin", "reproduce", "battery", "records", "evidence")), state
    # reproduce: |1.23 - 1.25| / 1.25 = 0.016 <= 0.05
    assert state["stages"]["reproduce"]["got"] == 1.23
    assert state["stages"]["pin"]["sha"]
    assert fetched == ["https://records.example/t"]

    ws = workspace_for("toy", tmp_path / "forge")
    evidence = (ws / "evidence.md").read_text()
    assert "ToySOTA" in evidence and "PASS" in evidence
    assert (ws / "records" / "source_00.html").exists()
    assert (ws / "records" / "source_00.meta.json").exists()


def test_reproduce_fails_outside_tolerance(tmp_path, monkeypatch):
    repo = _make_sota_repo(tmp_path, value=2.0)  # far from published 1.25
    _stub_codex(monkeypatch, _draft_json("toy", str(repo)))
    state = forge("toy", "d", root=tmp_path / "forge")
    assert state["stages"]["reproduce"]["ok"] is False
    assert state["stages"]["reproduce"]["rel_err"] > 0.05
    # chain stops after the failure; battery never ran, evidence still renders
    assert "battery" not in state["stages"]
    ws = workspace_for("toy", tmp_path / "forge")
    assert "FAIL" in (ws / "evidence.md").read_text()


def test_battery_catches_accept_everything_validator(tmp_path, monkeypatch):
    repo = _make_sota_repo(tmp_path)
    _stub_codex(monkeypatch, _draft_json("toy", str(repo), bench=BROKEN_BENCH_PY))
    state = forge("toy", "d", root=tmp_path / "forge")
    assert state["stages"]["battery"]["ok"] is False
    cases = state["stages"]["battery"]["cases"]
    assert any(c.get("bad_rejected") is False for c in cases)


def test_forge_resume_skips_passed_stages(tmp_path, monkeypatch):
    repo = _make_sota_repo(tmp_path)
    _stub_codex(monkeypatch, _draft_json("toy", str(repo)))
    forge("toy", "d", root=tmp_path / "forge")
    # Second run: codex must not be called — every stage is cached as passed.
    _stub_codex(monkeypatch, "", fail_if_called=True)
    state = forge("toy", "d", root=tmp_path / "forge")
    assert all(s["ok"] for s in state["stages"].values())


def test_promote_requires_passed_stages_and_signature(tmp_path):
    ws_root = tmp_path / "forge"
    workspace_for("toy", ws_root).mkdir(parents=True)
    with pytest.raises(ValueError, match="not passed"):
        promote("toy", "James", root=ws_root,
                packs_root=tmp_path / "packs",
                experiments_root=tmp_path / "experiments")
    with pytest.raises(ValueError, match="approve"):
        promote("toy", "  ", root=ws_root,
                packs_root=tmp_path / "packs",
                experiments_root=tmp_path / "experiments")


def test_promote_copies_pack_and_records_provenance(tmp_path, monkeypatch):
    repo = _make_sota_repo(tmp_path)
    _stub_codex(monkeypatch, _draft_json("toy", str(repo)))
    forge("toy", "d", root=tmp_path / "forge")

    prov = promote("toy", "James", root=tmp_path / "forge",
                   packs_root=tmp_path / "packs",
                   experiments_root=tmp_path / "experiments")
    assert prov["approved_by"] == "James"
    assert (tmp_path / "packs" / "toy" / "pack.json").exists()
    assert (tmp_path / "packs" / "toy" / "provenance.json").exists()
    assert (tmp_path / "experiments" / "toy" / "bench.py").exists()
    # promoted pack loads through the strict schemas
    from pipeline.computable import load_pack
    load_pack("toy", root=tmp_path / "packs")

    with pytest.raises(FileExistsError):
        promote("toy", "James", root=tmp_path / "forge",
                packs_root=tmp_path / "packs",
                experiments_root=tmp_path / "experiments")


def test_promote_doctor_gate_rolls_back(tmp_path, monkeypatch):
    repo = _make_sota_repo(tmp_path)
    _stub_codex(monkeypatch, _draft_json(
        "toy", str(repo), smoke="python3 -c 'import sys; sys.exit(3)'"))
    forge("toy", "d", root=tmp_path / "forge")

    with pytest.raises(RuntimeError, match="doctor gate"):
        promote("toy", "James", root=tmp_path / "forge",
                packs_root=tmp_path / "packs",
                experiments_root=tmp_path / "experiments",
                run_doctor=True)
    assert not (tmp_path / "packs" / "toy").exists()
    assert not (tmp_path / "experiments" / "toy").exists()


def test_promote_waive_is_recorded(tmp_path, monkeypatch):
    repo = _make_sota_repo(tmp_path, value=99.0)  # reproduce will fail
    _stub_codex(monkeypatch, _draft_json("toy", str(repo)))
    state = forge("toy", "d", root=tmp_path / "forge")
    assert state["stages"]["reproduce"]["ok"] is False
    # battery/records/evidence never ran (chain stopped); waive them too is
    # not allowed — only explicitly named stages are waived.
    with pytest.raises(ValueError):
        promote("toy", "James", root=tmp_path / "forge",
                packs_root=tmp_path / "packs",
                experiments_root=tmp_path / "experiments",
                waive=["reproduce"])
