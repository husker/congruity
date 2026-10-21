"""Tests for the automated-verification spine (grade / parse / ledger / pack mode).

Candidate codegen is not unit-tested (it calls codex); everything else — the
deterministic core that decides labels, plus the pack-bench execution path —
IS, because that core is what protects the rigor of the flywheel.
"""

from __future__ import annotations

import json
import shutil
import sys
from datetime import datetime, timezone
from pathlib import Path

import pytest

from pipeline import autoverify
from pipeline.autoverify import (
    build_outcome,
    confidence_from_audit,
    execute_pack_bench,
    grade,
    orchestrate,
    parse_min_effect,
    record,
    structural_check,
)

REPO = Path(__file__).resolve().parent.parent


def test_confidence_from_audit():
    # invalid result -> quarantined regardless of audit
    assert confidence_from_audit(False, {"fair": True, "sota_faithful": True}) == "quarantined"
    # unfair harness -> quarantined even if valid
    assert confidence_from_audit(True, {"fair": False, "sota_faithful": True}) == "quarantined"
    # fair + faithful SOTA -> medium (auto ceiling)
    assert confidence_from_audit(True, {"fair": True, "sota_faithful": True}) == "medium"
    # fair but weak/strawman SOTA -> low
    assert confidence_from_audit(True, {"fair": True, "sota_faithful": False}) == "low"
    # no audit -> low
    assert confidence_from_audit(True, None) == "low"
    # audit unparseable (fair=None) -> low (not quarantined; only explicit False quarantines)
    assert confidence_from_audit(True, {"fair": None, "sota_faithful": None}) == "low"


def _result(scores, comp, sota, hib=True, n=10):
    return {"metric": "m", "higher_is_better": hib, "scores": scores,
            "composition_method": comp, "home_field_sota_method": sota,
            "n_trials": n, "notes": "x"}


def test_parse_min_effect():
    assert parse_min_effect("+0.02 recall@10 over SOTA") == pytest.approx(0.02)
    assert parse_min_effect(0.5) == 0.5
    assert parse_min_effect("") == 0.0
    assert parse_min_effect(None) == 0.0
    assert parse_min_effect("-3 dB") == pytest.approx(3.0)


def test_grade_transfer_win_higher_better():
    r = _result({"comp": 0.9, "sota": 0.8, "weak": 0.5}, "comp", "sota")
    g, valid, _ = grade(r, {"min_meaningful_effect_size": "+0.05"})
    assert (g, valid) == ("transfer-win", True)


def test_grade_partial_beats_other_not_sota():
    # beats the weak baseline but not the home-field SOTA
    r = _result({"comp": 0.7, "sota": 0.8, "weak": 0.5}, "comp", "sota")
    g, valid, _ = grade(r, {"min_meaningful_effect_size": "+0.05"})
    assert (g, valid) == ("partial", True)


def test_grade_negative():
    r = _result({"comp": 0.4, "sota": 0.8, "weak": 0.5}, "comp", "sota")
    g, _, _ = grade(r, {"min_meaningful_effect_size": "+0.05"})
    assert g == "negative"


def test_grade_lower_is_better():
    # compression ratio: lower beats. comp 0.55 < sota 0.61 -> transfer win
    r = _result({"comp": 0.55, "sota": 0.61}, "comp", "sota", hib=False)
    g, valid, _ = grade(r, {"min_meaningful_effect_size": "0.02"})
    assert (g, valid) == ("transfer-win", True)


def test_grade_invalid_missing_sota():
    r = _result({"comp": 0.9, "other": 0.8}, "comp", "sota")
    g, valid, reason = grade(r, {})
    assert g == "invalid" and not valid and "home-field" in reason


def test_grade_invalid_too_few_scores():
    r = _result({"comp": 0.9}, "comp", "sota")
    g, valid, _ = grade(r, {})
    assert (g, valid) == ("invalid", False)


def test_grade_invalid_non_finite():
    r = _result({"comp": float("inf"), "sota": 0.8}, "comp", "sota")
    g, valid, _ = grade(r, {})
    assert (g, valid) == ("invalid", False)


def test_grade_invalid_zero_trials():
    r = _result({"comp": 0.9, "sota": 0.8}, "comp", "sota", n=0)
    g, valid, _ = grade(r, {})
    assert (g, valid) == ("invalid", False)


def test_grade_degenerate_when_per_trial_identical():
    # candidate is behaviorally identical to the SOTA on every instance+seed:
    # a wrapper, not a contribution.
    r = _result({"comp": 10.0, "sota": 10.0}, "comp", "sota", hib=False)
    r["per_trial"] = {"comp": {"i1": [10, 10], "i2": [10, 11]},
                     "sota": {"i1": [10, 10], "i2": [10, 11]}}
    g, valid, _ = grade(r, {})
    assert (g, valid) == ("degenerate", True)


def test_grade_statistical_significant_win():
    # consistent +1 color improvement over 8 paired trials (lower better)
    r = _result({"comp": 9.0, "sota": 10.0}, "comp", "sota", hib=False)
    r["per_trial"] = {"comp": {"i1": [9, 9, 9, 9], "i2": [9, 9, 9, 9]},
                     "sota": {"i1": [10, 10, 10, 10], "i2": [10, 10, 10, 10]}}
    g, _, _ = grade(r, {"min_meaningful_effect_size": "1 color"})
    assert g == "transfer-win"


def test_grade_statistical_insignificant_mean_is_not_a_win():
    # mean delta +1 but wildly mixed signs: CI includes 0 -> partial, not win
    r = _result({"comp": 9.0, "sota": 10.0}, "comp", "sota", hib=False)
    r["per_trial"] = {
        "comp": {"i1": [4, 16, 4, 16], "i2": [4, 16, 4, 16]},
        "sota": {"i1": [10, 11, 10, 11], "i2": [10, 11, 10, 11]},
    }
    g, _, _ = grade(r, {"min_meaningful_effect_size": "1 color"})
    assert g == "partial"


def test_grade_paired_pairs_with_failures_are_dropped():
    from pipeline.autoverify import paired_deltas

    r = _result({"comp": 9.0, "sota": 10.0}, "comp", "sota", hib=False)
    r["per_trial"] = {"comp": {"i1": [9, None, 9]},
                     "sota": {"i1": [10, 10, None]}}
    assert paired_deltas(r) == [1.0]


def test_build_outcome_records_ablation_delta(tmp_path, monkeypatch):
    monkeypatch.setattr(autoverify, "OUTCOMES", tmp_path / "outcomes.jsonl")
    comp = {"id": "C1", "synthesis_claim": "x", "papers": []}
    r = _result({"comp": 9.0, "sota": 10.0, "comp_abl": 10.5},
                "comp", "sota", hib=False)
    r["ablated_method"] = "comp_abl"
    now = datetime(2026, 6, 11, tzinfo=timezone.utc)
    out = build_outcome("r", "C1", comp, {}, r, now)
    # lower-better: mechanism delta = ablated - candidate = +1.5 (helps)
    assert "mechanism_delta=+1.5" in out.notes


def test_build_outcome_and_record_roundtrip(tmp_path, monkeypatch):
    monkeypatch.setattr(autoverify, "OUTCOMES", tmp_path / "outcomes.jsonl")
    comp = {"id": "C1", "synthesis_claim": "x", "papers": []}
    recipe = {"min_meaningful_effect_size": "+0.05", "home_field_sota": "SOTA"}
    r = _result({"comp": 0.9, "sota": 0.8}, "comp", "sota")
    now = datetime(2026, 5, 30, tzinfo=timezone.utc)
    out = build_outcome("20260530-000000", "C1", comp, recipe, r, now)
    assert out.grade == "transfer-win" and out.beats_home_field is True
    assert out.provenance == "auto" and out.confidence == "low"
    record(out)
    line = json.loads((tmp_path / "outcomes.jsonl").read_text().strip())
    assert line["run_id"] == "20260530-000000"
    assert line["beats_home_field"] is True
    assert line["delta_vs_home"] == pytest.approx(0.1)


# --------------------------------------------------------------------------- #
# Pack-bench mode
# --------------------------------------------------------------------------- #
def test_structural_check_against_real_pack():
    from pipeline.computable import load_pack

    pack = load_pack("graph_coloring")
    ok = {"metric": "colors_used", "higher_is_better": False}
    assert structural_check(ok, pack) == ""
    assert "metric" in structural_check({"metric": "hit_rate", "higher_is_better": False}, pack)
    assert "higher_is_better" in structural_check(
        {"metric": "colors_used", "higher_is_better": True}, pack)


def test_execute_pack_bench_parses_last_line(tmp_path):
    script = tmp_path / "ok.py"
    script.write_text(
        "print('table noise')\n"
        "print('{\"metric\": \"m\", \"scores\": {\"a\": 1.0}}')\n")
    result = execute_pack_bench(f"{sys.executable} {script}")
    assert result == {"metric": "m", "scores": {"a": 1.0}}

    bad = tmp_path / "bad.py"
    bad.write_text("import sys; sys.exit(3)\n")
    assert "_error" in execute_pack_bench(f"{sys.executable} {bad}")

    noise = tmp_path / "noise.py"
    noise.write_text("print('not json')\n")
    assert "unparseable" in execute_pack_bench(f"{sys.executable} {noise}")["_error"]


def _pack_mode_fixture(tmp_path, monkeypatch, bench_body: str):
    """Stub pack (cloned from graph_coloring) + composition + ledger in tmp."""
    packs = tmp_path / "packs"
    packs.mkdir()
    shutil.copytree(REPO / "computable_packs" / "graph_coloring", packs / "graph_coloring")
    # a pack with no candidate interface, for the rejection test
    shutil.copytree(REPO / "computable_packs" / "cache_replacement", packs / "cache_replacement")
    stub_bench = tmp_path / "stub_bench.py"
    stub_bench.write_text(bench_body)
    ev_path = packs / "graph_coloring" / "evaluator.json"
    ev = json.loads(ev_path.read_text())
    ev["candidate_command"] = f"{sys.executable} {stub_bench} {{candidate}}"
    ev_path.write_text(json.dumps(ev))

    import pipeline.computable as computable
    real_load, real_pre = computable.load_pack, computable.preflight_pack
    monkeypatch.setattr(computable, "load_pack",
                        lambda name, root=None: real_load(name, root=packs))
    monkeypatch.setattr(computable, "preflight_pack",
                        lambda name, root=None: real_pre(name, root=packs))

    comps = tmp_path / "compositions"
    (comps / "r1").mkdir(parents=True)
    (comps / "r1" / "state.json").write_text(json.dumps({
        "problem": "color graphs with fewer colors than tabucol",
        "compositions": [{
            "id": "C1", "synthesis_claim": "x", "papers": [],
            "verification": {"recipe": {
                "home_field_sota": "descending-k TabuCol",
                "min_meaningful_effect_size": "1 color",
            }},
        }],
    }))
    monkeypatch.setattr(autoverify, "COMPOSITIONS", comps)
    monkeypatch.setattr(autoverify, "OUTCOMES", tmp_path / "outcomes.jsonl")

    candidate = tmp_path / "candidate.py"
    candidate.write_text("def solve(adj, seed, iters):\n    return []\n")
    return candidate


def test_orchestrate_pack_mode_transfer_win(tmp_path, monkeypatch):
    bench = (
        "import json, sys\n"
        "assert len(sys.argv) > 1  # candidate path is passed through\n"
        "print('header noise')\n"
        "print(json.dumps({'metric': 'colors_used', 'higher_is_better': False,\n"
        "                  'scores': {'candidate': 9.0, 'tabucol': 11.0},\n"
        "                  'composition_method': 'candidate',\n"
        "                  'home_field_sota_method': 'tabucol',\n"
        "                  'n_trials': 6, 'notes': 'stub'}))\n"
    )
    candidate = _pack_mode_fixture(tmp_path, monkeypatch, bench)
    now = datetime(2026, 6, 11, tzinfo=timezone.utc)
    summary = orchestrate("r1", "C1", now, pack_id="graph_coloring",
                          candidate_path=str(candidate))
    assert summary["grade"] == "transfer-win"
    assert summary["confidence"] == "medium"
    assert summary["delta_vs_home"] == pytest.approx(2.0)
    line = json.loads((tmp_path / "outcomes.jsonl").read_text().strip())
    assert line["audit"]["fair"] is True and line["audit"]["sota_faithful"] is True
    assert line["provenance"] == "auto"


def test_orchestrate_pack_mode_crash_is_quarantined(tmp_path, monkeypatch):
    candidate = _pack_mode_fixture(tmp_path, monkeypatch,
                                   "import sys; sys.exit(2)\n")
    now = datetime(2026, 6, 11, tzinfo=timezone.utc)
    summary = orchestrate("r1", "C1", now, pack_id="graph_coloring",
                          candidate_path=str(candidate))
    assert summary["grade"] == "invalid"
    line = json.loads((tmp_path / "outcomes.jsonl").read_text().strip())
    assert line["valid"] is False and line["confidence"] == "quarantined"


def test_orchestrate_rejects_unknown_or_unready_pack(tmp_path, monkeypatch):
    candidate = _pack_mode_fixture(tmp_path, monkeypatch, "print('x')\n")
    now = datetime(2026, 6, 11, tzinfo=timezone.utc)
    assert "error" in orchestrate("r1", "C1", now, pack_id="nope",
                                  candidate_path=str(candidate))
    # cache_replacement has no candidate interface -> refused
    summary = orchestrate("r1", "C1", now, pack_id="cache_replacement",
                          candidate_path=str(candidate))
    assert "candidate interface" in summary.get("error", "")
