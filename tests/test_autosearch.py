"""Tests for the autosearch loop — deterministic, no codex, stub benches.

Codegen is stubbed (it calls codex); the loop mechanics ARE tested: selection
by train fitness, early stop on a train win, single holdout confirmation as
the recorded label, and quarantine when nothing runs.
"""

from __future__ import annotations

import json
import shutil
import sys
from datetime import datetime, timezone
from pathlib import Path

import pytest

from pipeline import autosearch, autoverify
from pipeline.autosearch import fitness, orchestrate_search, score_table

REPO = Path(__file__).resolve().parent.parent

# Stub bench: candidate module content is a float; that's the candidate's
# score. SOTA is fixed at 11; metric colors_used (lower better).
STUB_BENCH = """\
import json, sys
value = float(open(sys.argv[1]).read().strip())
print("table noise")
print(json.dumps({
    "metric": "colors_used", "higher_is_better": False,
    "scores": {"candidate": value, "tabucol": 11.0},
    "per_instance": {"candidate": {"i1": value}, "tabucol": {"i1": 11.0}},
    "composition_method": "candidate", "home_field_sota_method": "tabucol",
    "n_trials": 4, "notes": "stub"}))
"""


def test_fitness_sign_and_failures():
    base = {"composition_method": "c", "home_field_sota_method": "s"}
    # lower-better: candidate 9 vs sota 11 -> +2
    assert fitness({**base, "higher_is_better": False,
                    "scores": {"c": 9.0, "s": 11.0}}) == pytest.approx(2.0)
    # higher-better: candidate 9 vs sota 11 -> -2
    assert fitness({**base, "higher_is_better": True,
                    "scores": {"c": 9.0, "s": 11.0}}) == pytest.approx(-2.0)
    assert fitness({"_error": "boom"}) is None
    assert fitness({**base, "scores": {"c": float("nan"), "s": 11.0}}) is None
    assert fitness({**base, "scores": {"s": 11.0}}) is None


def test_score_table_renders_per_instance():
    result = {"composition_method": "candidate", "home_field_sota_method": "tabucol",
              "per_instance": {"candidate": {"a": 9.0}, "tabucol": {"a": 11.0}},
              "scores": {"candidate": 9.0, "tabucol": 11.0}}
    table = score_table(result)
    assert "a" in table and "9.00" in table and "11.00" in table


def _search_fixture(tmp_path, monkeypatch, search_body=STUB_BENCH,
                    confirm_body=STUB_BENCH):
    packs = tmp_path / "packs"
    packs.mkdir()
    shutil.copytree(REPO / "computable_packs" / "graph_coloring",
                    packs / "graph_coloring")
    search_script = tmp_path / "search_bench.py"
    search_script.write_text(search_body)
    confirm_script = tmp_path / "confirm_bench.py"
    confirm_script.write_text(confirm_body)
    ev_path = packs / "graph_coloring" / "evaluator.json"
    ev = json.loads(ev_path.read_text())
    ev["search_command"] = f"{sys.executable} {search_script} {{candidate}}"
    ev["confirm_command"] = f"{sys.executable} {confirm_script} {{candidate}}"
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
        "problem": "fewer colors than tabucol",
        "compositions": [{
            "id": "C1", "synthesis_claim": "the idea", "papers": [],
            "verification": {"recipe": {
                "home_field_sota": "descending-k TabuCol",
                "min_meaningful_effect_size": "1 color",
            }},
        }],
    }))
    monkeypatch.setattr(autoverify, "COMPOSITIONS", comps)
    monkeypatch.setattr(autosearch, "COMPOSITIONS", comps)
    monkeypatch.setattr(autoverify, "OUTCOMES", tmp_path / "outcomes.jsonl")
    monkeypatch.setattr(autosearch, "AUTO_EXPERIMENTS", tmp_path / "auto")
    return tmp_path


def _stub_codegen(monkeypatch, initial_values, refined_values):
    """Codegen stubs write scripted 'scores' instead of calling codex."""
    initial = iter(initial_values)
    refined = iter(refined_values)

    def fake_initial(comp, recipe, pack, problem, dest, emphasis, stage, **_kw):
        dest.write_text(f"{next(initial)}\n")
        return dest

    def fake_refine(comp, pack, parent_code, parent_result, dest, stage, **_kw):
        # refinement sees the parent's train result; assert plumbing intact
        assert parent_result["scores"]["tabucol"] == 11.0
        dest.write_text(f"{next(refined)}\n")
        return dest

    monkeypatch.setattr(autosearch, "generate_initial", fake_initial)
    monkeypatch.setattr(autosearch, "refine", fake_refine)
    return initial, refined


def test_search_selects_refines_and_confirms_on_holdout(tmp_path, monkeypatch):
    _search_fixture(tmp_path, monkeypatch)
    # initial: 12.0 (delta -1), 10.5 (delta +0.5, best but < min effect 1)
    # gen 1 refines the best -> 9.0 (delta +2 -> early stop before gen 2)
    _stub_codegen(monkeypatch, initial_values=[12.0, 10.5], refined_values=[9.0])

    now = datetime(2026, 6, 11, tzinfo=timezone.utc)
    summary = orchestrate_search("r1", "C1", now, pack_id="graph_coloring",
                                 population=2, generations=3, elites=1)

    assert summary["grade"] == "transfer-win"
    assert summary["candidates_evaluated"] == 3  # 2 initial + 1 refined
    assert summary["train_delta_best"] == pytest.approx(2.0)
    assert summary["holdout_delta"] == pytest.approx(2.0)
    assert summary["best_candidate"].endswith("g1_e0.py")

    # exactly ONE ledger line, labeled from the holdout result
    lines = (tmp_path / "outcomes.jsonl").read_text().strip().splitlines()
    assert len(lines) == 1
    out = json.loads(lines[0])
    assert out["grade"] == "transfer-win" and out["confidence"] == "medium"
    assert "held-out" in out["audit"]["notes"]
    assert "autosearch" in out["notes"]

    # search history persisted
    hist = json.loads((tmp_path / "auto" / "r1-C1" / "search" / "history.json")
                      .read_text())
    assert len(hist) == 3


def test_search_quarantines_when_nothing_runs(tmp_path, monkeypatch):
    _search_fixture(tmp_path, monkeypatch,
                    search_body="import sys; sys.exit(2)\n")
    _stub_codegen(monkeypatch, initial_values=[10.0, 10.0], refined_values=[])

    now = datetime(2026, 6, 11, tzinfo=timezone.utc)
    summary = orchestrate_search("r1", "C1", now, pack_id="graph_coloring",
                                 population=2, generations=2, elites=1)
    assert summary["grade"] == "invalid"
    out = json.loads((tmp_path / "outcomes.jsonl").read_text().strip())
    assert out["valid"] is False and out["confidence"] == "quarantined"


def test_control_arm_hides_composition_and_skips_ledger(tmp_path, monkeypatch):
    _search_fixture(tmp_path, monkeypatch)
    seen_claims: list[str] = []

    def fake_initial(comp, recipe, pack, problem, dest, emphasis, stage, **_kw):
        seen_claims.append(comp.get("synthesis_claim", ""))
        dest.write_text("10.5\n")
        return dest

    monkeypatch.setattr(autosearch, "generate_initial", fake_initial)
    monkeypatch.setattr(autosearch, "refine",
                        lambda *a, **kw: (_ for _ in ()).throw(AssertionError))

    now = datetime(2026, 6, 11, tzinfo=timezone.utc)
    summary = orchestrate_search("r1", "C1", now, pack_id="graph_coloring",
                                 population=1, generations=0, elites=1,
                                 arm="unseeded")
    assert summary["arm"] == "unseeded"
    # the control arm never sees the composition's claim
    assert "the idea" not in seen_claims[0]
    assert "first principles" in seen_claims[0]
    # control arms do not write the ledger
    assert not (tmp_path / "outcomes.jsonl").exists()
    # but the summary is persisted in an arm-specific dir
    assert (tmp_path / "auto" / "r1-C1" / "search_unseeded" /
            "summary.json").exists()


def test_unknown_arm_rejected(tmp_path, monkeypatch):
    _search_fixture(tmp_path, monkeypatch)
    now = datetime(2026, 6, 11, tzinfo=timezone.utc)
    assert "error" in orchestrate_search("r1", "C1", now,
                                         pack_id="graph_coloring", arm="bogus")


def test_search_rejects_non_search_ready_pack(tmp_path, monkeypatch):
    fixture_root = _search_fixture(tmp_path, monkeypatch)
    # strip the search commands -> not search-ready
    ev_path = fixture_root / "packs" / "graph_coloring" / "evaluator.json"
    ev = json.loads(ev_path.read_text())
    ev["search_command"] = ""
    ev_path.write_text(json.dumps(ev))
    now = datetime(2026, 6, 11, tzinfo=timezone.utc)
    summary = orchestrate_search("r1", "C1", now, pack_id="graph_coloring")
    assert "search-ready" in summary.get("error", "")

def test_resume_reuses_candidates_and_cached_evals(tmp_path, monkeypatch):
    """A restarted search re-pays neither codegen nor the search bench."""
    _search_fixture(tmp_path, monkeypatch)
    _stub_codegen(monkeypatch, initial_values=[12.0, 10.5], refined_values=[9.0])
    now = datetime(2026, 6, 11, tzinfo=timezone.utc)
    first = orchestrate_search("r1", "C1", now, pack_id="graph_coloring",
                               population=2, generations=3, elites=1)
    assert first["grade"] == "transfer-win"

    # Second run: codegen stubs are exhausted (StopIteration if called) and
    # the search bench now explodes (bench error if executed). Only the
    # holdout confirm — which re-runs by design — still works.
    search_script = tmp_path / "search_bench.py"
    search_script.write_text("import sys; sys.exit(3)\n")
    (tmp_path / "outcomes.jsonl").unlink()

    second = orchestrate_search("r1", "C1", now, pack_id="graph_coloring",
                                population=2, generations=3, elites=1)
    assert second["grade"] == "transfer-win"
    assert second["train_delta_best"] == first["train_delta_best"]
    assert second["candidates_evaluated"] == 3

    hist = json.loads((tmp_path / "auto" / "r1-C1" / "search" / "history.json")
                      .read_text())
    resumed = [h for h in hist if h.get("cached")]
    assert len(resumed) == 3  # every evaluation in run 2 came from the cache


# ---------------------------------------------------------------------------
# Record mode (--record-dim): tail fitness, no ledger, certificate hand-off
# ---------------------------------------------------------------------------

RECORD_STUB_BENCH = """\
import json, sys
value = float(open(sys.argv[1]).read().strip())
# args: candidate path, then --dims N appears in argv when record_command runs
n = int(sys.argv[sys.argv.index("--dims") + 1]) if "--dims" in sys.argv else 0
print(json.dumps({
    "metric": "snake_length", "higher_is_better": True,
    "scores": {"candidate": value, "tabucol": 11.0},
    "composition_method": "candidate", "home_field_sota_method": "tabucol",
    "n_trials": 2, "notes": "stub",
    "record_claims": [{"params": {"n": n}, "value": value}],
    "best_sequences": {"q%d" % n: [0, 1, 3]}}))
"""


def _record_fixture(tmp_path, monkeypatch, record_value=50.0):
    root = _search_fixture(tmp_path, monkeypatch)
    packs = root / "packs"
    record_script = tmp_path / "record_bench.py"
    record_script.write_text(RECORD_STUB_BENCH)
    ev_path = packs / "graph_coloring" / "evaluator.json"
    ev = json.loads(ev_path.read_text())
    ev["record_command"] = (f"{sys.executable} {record_script} "
                            "{candidate} --dims {n}")
    ev_path.write_text(json.dumps(ev))
    (packs / "graph_coloring" / "records.json").write_text(json.dumps({
        "domain": "graph_coloring", "metric": "snake_length",
        "higher_is_better": True, "tolerance": 1e-9,
        "records": [{"id": "q7", "params": {"n": 7}, "value": record_value,
                     "status": "best_known", "source": "test"}],
    }))
    return root


def test_record_fitness_extracts_target_dim():
    from pipeline.autosearch import record_fitness
    r = {"record_claims": [{"params": {"n": 7}, "value": 44.0},
                           {"params": {"n": 8}, "value": 90.0}]}
    assert record_fitness(r, 7) == 44.0
    assert record_fitness(r, 9) is None
    assert record_fitness({"_error": "x"}, 7) is None


def test_record_mode_selects_by_tail_and_skips_ledger(tmp_path, monkeypatch):
    _record_fixture(tmp_path, monkeypatch)
    _stub_codegen(monkeypatch, initial_values=[44.0, 46.0], refined_values=[47.0])
    now = datetime(2026, 7, 2, tzinfo=timezone.utc)
    summary = orchestrate_search("r1", "C1", now, pack_id="graph_coloring",
                                 population=2, generations=1, elites=1,
                                 record_dim=7)
    assert summary["mode"] == "record"
    assert summary["best_fitness"] == 47.0
    assert summary["record_value"] == 50.0
    assert summary["gap"] == -3.0
    assert summary["verdict"] == "below-record"
    # no ledger row in record mode
    assert not (tmp_path / "outcomes.jsonl").exists()
    # certificate persisted from the bench's best_sequences
    cert = json.loads(Path(summary["certificate"]).read_text())
    assert cert["n"] == 7 and cert["length"] == 47


def test_record_mode_early_stops_on_new_record(tmp_path, monkeypatch):
    _record_fixture(tmp_path, monkeypatch, record_value=45.0)
    # gen0 candidate already exceeds the 45 record -> early stop, no refine
    _stub_codegen(monkeypatch, initial_values=[46.0, 40.0], refined_values=[])
    now = datetime(2026, 7, 2, tzinfo=timezone.utc)
    summary = orchestrate_search("r1", "C1", now, pack_id="graph_coloring",
                                 population=2, generations=3, elites=1,
                                 record_dim=7)
    assert summary["verdict"] == "new-record"
    assert summary["best_fitness"] == 46.0


def test_record_mode_requires_record_command(tmp_path, monkeypatch):
    _search_fixture(tmp_path, monkeypatch)  # no record_command in evaluator
    _stub_codegen(monkeypatch, initial_values=[1.0], refined_values=[])
    now = datetime(2026, 7, 2, tzinfo=timezone.utc)
    summary = orchestrate_search("r1", "C1", now, pack_id="graph_coloring",
                                 population=1, generations=1, elites=1,
                                 record_dim=7)
    assert "record_command" in summary["error"]
