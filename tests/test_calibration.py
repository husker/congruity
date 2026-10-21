"""Tests for the calibration markdown report."""

from __future__ import annotations

import json
from pathlib import Path


def _make_run(root: Path, run_id: str, status: str, verdicts: list[str],
              relevance_skipped: bool = False) -> None:
    d = root / run_id
    d.mkdir(parents=True, exist_ok=True)
    state = {
        "run_id": run_id,
        "problem": f"problem for {run_id}",
        "corpus_size": 10,
        "dimensions": [
            {"name": "x", "description": "y", "example_question": "z"},
        ],
        "compositions": [
            {
                "id": f"C{i + 1}",
                "verification": {
                    "L3": {"verdict": v},
                    "L4": {"summary": {"net_testability_score": "PURELY_TESTABLE"}},
                },
            }
            for i, v in enumerate(verdicts)
        ],
        "status": status,
        "relevance_skipped": relevance_skipped,
    }
    (d / "state.json").write_text(json.dumps(state))


def test_calibration_report_contains_required_sections(tmp_path):
    from pipeline import calibration
    _make_run(tmp_path, "20260101-000000", "complete", ["sound", "weak"])
    _make_run(tmp_path, "20260102-000000", "complete", ["invalid", "weak", "weak"])
    out = calibration.generate_report(tmp_path)
    assert "Run summary" in out
    assert "L3 verdict mix" in out
    assert "L4 testability mix" in out
    assert "Skipped gates" in out
    assert "sound" in out
    assert "weak" in out
    assert "invalid" in out
    # 5 compositions total → 3 weak, 1 sound, 1 invalid
    assert "**weak**: 3" in out
    assert "**sound**: 1" in out
    assert "**invalid**: 1" in out


def test_calibration_handles_missing_root(tmp_path):
    from pipeline import calibration
    out = calibration.generate_report(tmp_path / "does-not-exist")
    assert "no compose runs found" in out.lower()


def test_calibration_handles_empty_dir(tmp_path):
    from pipeline import calibration
    out = calibration.generate_report(tmp_path)
    assert "no compose runs found" in out.lower()


def test_calibration_handles_run_with_no_compositions(tmp_path):
    from pipeline import calibration
    d = tmp_path / "20260103-000000"
    d.mkdir()
    (d / "state.json").write_text(json.dumps({
        "run_id": "20260103-000000",
        "problem": "x",
        "status": "no_dimensions_extracted",
        "compositions": [],
    }))
    out = calibration.generate_report(tmp_path)
    assert "20260103-000000" in out
    assert "no_dimensions_extracted" in out


def test_calibration_counts_skipped_relevance(tmp_path):
    from pipeline import calibration
    _make_run(tmp_path, "20260104-000000", "complete", ["weak"], relevance_skipped=True)
    _make_run(tmp_path, "20260105-000000", "complete", ["weak"], relevance_skipped=False)
    out = calibration.generate_report(tmp_path)
    assert "relevance check skipped: **1** / 2" in out


# ---------------------------------------------------------------------------
# HTML report tests
# ---------------------------------------------------------------------------


def test_html_report_is_self_contained(tmp_path):
    from pipeline import calibration
    _make_run(tmp_path, "20260106-000000", "complete", ["weak", "sound"])
    out = calibration.generate_html_report(tmp_path)
    # Single document, embedded CSS + JS, no external <link>/<script src=>
    assert out.startswith("<!doctype html>") or out.startswith("<!DOCTYPE html>")
    assert "<style>" in out
    assert "<script>" in out
    assert 'src="' not in out  # no external scripts/images
    assert 'href="http' not in out  # no external stylesheets


def test_html_report_includes_verdict_counts(tmp_path):
    from pipeline import calibration
    _make_run(tmp_path, "20260107-000000", "complete", ["weak", "weak", "sound", "invalid"])
    out = calibration.generate_html_report(tmp_path)
    # Each verdict label appears in the SVG bars
    assert ">weak<" in out
    assert ">sound<" in out
    assert ">invalid<" in out
    # And the counts appear nearby
    assert ">2<" in out  # weak count


def test_html_report_table_is_sortable(tmp_path):
    from pipeline import calibration
    _make_run(tmp_path, "20260108-000000", "complete", ["weak"])
    out = calibration.generate_html_report(tmp_path)
    # The sort handler attaches to tables with class="sortable"
    assert 'class="sortable"' in out
    # Headers include the expected columns
    for col in ["run_id", "problem", "corpus", "dims", "comps",
                "status", "llm_s", "rel.skip"]:
        assert f">{col}<" in out


def test_html_report_embeds_raw_json_for_export(tmp_path):
    from pipeline import calibration
    _make_run(tmp_path, "20260109-000000", "complete", ["sound"])
    out = calibration.generate_html_report(tmp_path)
    assert "<details>" in out
    assert "raw JSON" in out
    # Look for at least one key from the payload structure
    assert "l3_counts" in out


def test_html_report_handles_empty(tmp_path):
    from pipeline import calibration
    out = calibration.generate_html_report(tmp_path)
    assert "No compose runs found" in out
    assert out.startswith("<!doctype html>") or out.startswith("<!DOCTYPE html>")


def test_html_report_metric_tooltips_present(tmp_path):
    from pipeline import calibration
    _make_run(tmp_path, "20260110-000000", "complete", ["weak"])
    out = calibration.generate_html_report(tmp_path)
    # Metrics get a title= explaining what they count.
    assert calibration.TIPS_METRIC["runs"] in out
    assert calibration.TIPS_METRIC["compositions"] in out
    # The "runs" key label is rendered.
    assert ">runs<" in out


def test_html_report_column_header_tooltips_present(tmp_path):
    from pipeline import calibration
    _make_run(tmp_path, "20260111-000000", "complete", ["weak"])
    out = calibration.generate_html_report(tmp_path)
    # Each known column has its tooltip text in a title= attribute.
    for col, tip in calibration.TIPS_COL.items():
        # Header text appears
        assert f">{col}<" in out
        # And its tooltip is somewhere in the document
        assert tip in out


def test_html_report_l3_chart_has_svg_title_tooltips(tmp_path):
    from pipeline import calibration
    _make_run(tmp_path, "20260112-000000", "complete", ["weak", "sound", "invalid"])
    out = calibration.generate_html_report(tmp_path)
    # SVG <title> elements provide native tooltips on hover.
    for verdict in ("weak", "sound", "invalid"):
        assert f"<title>{calibration.TIPS_L3[verdict]}</title>" in out


def test_html_report_l4_chart_has_svg_title_tooltips(tmp_path):
    from pipeline import calibration
    _make_run(tmp_path, "20260113-000000", "complete", ["weak", "weak"])
    out = calibration.generate_html_report(tmp_path)
    # PURELY_TESTABLE is the only L4 verdict the helper emits for the fixture.
    assert f"<title>{calibration.TIPS_L4['PURELY_TESTABLE']}</title>" in out


def test_html_report_outcome_badges_get_tooltips_via_index(tmp_path, monkeypatch):
    """If findings/INDEX.md is present, outcome words inside it (clean-positive,
    clean-negative, etc.) gain a tooltip-decorated span."""
    from pipeline import calibration
    fake_index = tmp_path / "INDEX.md"
    fake_index.write_text(
        "# x\n\n| date | outcome |\n|---|---|\n| 2026-05-21 | **clean-positive** |\n"
    )
    monkeypatch.setattr(calibration, "INDEX_PATH", fake_index)
    _make_run(tmp_path, "20260114-000000", "complete", ["weak"])
    out = calibration.generate_html_report(tmp_path)
    assert 'class="outcome"' in out
    assert calibration.TIPS_OUTCOME["clean-positive"] in out


# ---------------------------------------------------------------------------
# Outcome column join
# ---------------------------------------------------------------------------


def _write_index(index_path: Path, rows: list[tuple[str, str, str]]) -> None:
    """rows: (run_id, experiment, outcome)."""
    lines = [
        "# Track record",
        "",
        "| run_id | experiment | composition | outcome | doc |",
        "|---|---|---|---|---|",
    ]
    for run_id, exp, outcome in rows:
        lines.append(f"| {run_id} | {exp} | x | **{outcome}** | `findings/x.md` |")
    index_path.write_text("\n".join(lines) + "\n")


def test_parse_index_outcomes_maps_run_id_to_outcome(tmp_path, monkeypatch):
    from pipeline import calibration
    idx = tmp_path / "INDEX.md"
    _write_index(idx, [
        ("20260101-000000", "Heilbronn", "clean-positive"),
        ("20260102-000000", "Cache", "clean-negative"),
    ])
    monkeypatch.setattr(calibration, "INDEX_PATH", idx)
    out = calibration._parse_index_outcomes()
    assert out["20260101-000000"]["outcome"] == "clean-positive"
    assert out["20260102-000000"]["outcome"] == "clean-negative"
    assert out["20260101-000000"]["experiment"] == "Heilbronn"


def test_parse_index_outcomes_ignores_unknown_outcome_labels(tmp_path, monkeypatch):
    from pipeline import calibration
    idx = tmp_path / "INDEX.md"
    _write_index(idx, [("20260101-000000", "x", "unrecognized-label")])
    monkeypatch.setattr(calibration, "INDEX_PATH", idx)
    out = calibration._parse_index_outcomes()
    assert out == {}


def test_runs_table_shows_outcome_column_and_badges(tmp_path, monkeypatch):
    from pipeline import calibration
    idx = tmp_path / "INDEX.md"
    _write_index(idx, [
        ("20260201-000000", "Heilbronn", "clean-positive"),
        ("20260202-000000", "Embed quant", "clean-negative"),
    ])
    monkeypatch.setattr(calibration, "INDEX_PATH", idx)
    _make_run(tmp_path, "20260201-000000", "complete", ["weak"])
    _make_run(tmp_path, "20260202-000000", "complete", ["invalid"])
    _make_run(tmp_path, "20260203-000000", "complete", ["weak"])  # unverified
    out = calibration.generate_html_report(tmp_path)

    # Outcome column header is present
    assert ">outcome<" in out
    # Both badge classes rendered exactly once each
    assert out.count('class="badge positive"') >= 1
    assert out.count('class="badge negative"') >= 1
    # Row-level marker classes
    assert 'class="row-positive"' in out
    assert 'class="row-negative"' in out
    assert 'class="row-unverified"' in out
    # Outcome summary chips at the top of the section
    assert "× clean-positive" in out
    assert "× clean-negative" in out
    # Verified-only filter UI
    assert 'id="only-verified"' in out


def test_runs_table_unverified_runs_have_dash_placeholder(tmp_path, monkeypatch):
    from pipeline import calibration
    idx = tmp_path / "INDEX.md"
    _write_index(idx, [])
    monkeypatch.setattr(calibration, "INDEX_PATH", idx)
    _make_run(tmp_path, "20260301-000000", "complete", ["weak"])
    out = calibration.generate_html_report(tmp_path)
    # Unverified summary chip
    assert "no verified runs in this set" in out
    # All rows marked unverified
    assert 'class="row-unverified"' in out


def _make_run_with_reconciled(root: Path, run_id: str,
                              reconciled_statuses: list[str]) -> None:
    """Create a run whose compositions carry explicit reconciled blocks."""
    d = root / run_id
    d.mkdir(parents=True, exist_ok=True)
    compositions = []
    for i, status in enumerate(reconciled_statuses):
        inconsistent = status == "invalid_disagreement"
        actual_status = "invalid" if inconsistent else status
        compositions.append({
            "id": f"C{i + 1}",
            "verification": {
                "L3": {"verdict": "invalid" if inconsistent else "weak"},
                "L4": {"summary": {"net_testability_score": "PURELY_TESTABLE"}},
                "reconciled": {
                    "status": actual_status,
                    "l3_l4_consistent": not inconsistent,
                    "reason": "test",
                },
            },
        })
    state = {
        "run_id": run_id,
        "problem": "x",
        "corpus_size": 10,
        "dimensions": [{"name": "d", "description": "x", "example_question": "y"}],
        "compositions": compositions,
        "status": "complete",
    }
    (d / "state.json").write_text(json.dumps(state))


def test_html_dashboard_renders_reconciled_chart(tmp_path):
    from pipeline import calibration
    _make_run_with_reconciled(tmp_path, "20260601-000000",
                              ["actionable", "invalid", "weak_untestable"])
    out = calibration.generate_html_report(tmp_path)
    # The new section appears
    assert "Reconciled L3+L4 verdict" in out
    # Each reconciled status the helper saw is rendered (text in SVG label)
    for status in ("actionable", "invalid", "weak_untestable"):
        assert f">{status}<" in out
    # Each status has its tooltip from TIPS_RECONCILED
    for status in ("actionable", "invalid", "weak_untestable"):
        assert calibration.TIPS_RECONCILED[status] in out


def test_html_dashboard_flags_l3_l4_inconsistencies(tmp_path):
    from pipeline import calibration
    # 2 of 3 compositions are inconsistent (L3=invalid but L4=PURELY_TESTABLE).
    _make_run_with_reconciled(
        tmp_path, "20260602-000000",
        ["actionable", "invalid_disagreement", "invalid_disagreement"],
    )
    out = calibration.generate_html_report(tmp_path)
    assert "L3 vs L4 disagreements" in out
    assert ">2<" in out  # the disagreement count
    assert "different stories" in out


def test_html_dashboard_no_disagreement_callout_when_consistent(tmp_path):
    from pipeline import calibration
    _make_run_with_reconciled(tmp_path, "20260603-000000",
                              ["actionable", "weak_untestable", "sound"])
    out = calibration.generate_html_report(tmp_path)
    # The card still renders but with the "all consistent" copy
    assert "L3 vs L4 disagreements" in out
    assert "all 3 compositions had consistent" in out


def test_aggregate_falls_back_for_runs_without_reconciled_field(tmp_path, monkeypatch):
    """Older runs (pre-reconciler) don't have `verification.reconciled`.
    The dashboard recomputes it on the fly."""
    from pipeline import calibration
    d = tmp_path / "20260101-000000"
    d.mkdir()
    state = {
        "run_id": "20260101-000000",
        "problem": "x", "corpus_size": 1, "dimensions": [{"name": "d", "description": "x", "example_question": "y"}],
        "compositions": [
            {
                "id": "C1",
                "verification": {
                    "L3": {"verdict": "invalid"},
                    "L4": {"summary": {"net_testability_score": "PURELY_TESTABLE",
                                       "fundamental_count": 0, "empirical_count": 7}},
                    # NOTE: no "reconciled" field — older run
                },
            },
        ],
        "status": "complete",
    }
    (d / "state.json").write_text(json.dumps(state))
    out = calibration.generate_html_report(tmp_path)
    # Reconciled chart still rendered using on-the-fly derivation
    assert ">invalid<" in out
    # And the inconsistency surfaces
    assert ">1<" in out  # one inconsistent comp
    assert "L3 vs L4 disagreements" in out


def test_outcome_count_appears_in_aggregate(tmp_path, monkeypatch):
    from pipeline import calibration
    idx = tmp_path / "INDEX.md"
    _write_index(idx, [
        ("20260301-000000", "x", "clean-positive"),
        ("20260302-000000", "y", "clean-positive"),
        ("20260303-000000", "z", "clean-negative"),
    ])
    monkeypatch.setattr(calibration, "INDEX_PATH", idx)
    for rid in [
        "20260301-000000", "20260302-000000", "20260303-000000", "20260304-000000",
    ]:
        _make_run(tmp_path, rid, "complete", ["weak"])
    out = calibration.generate_html_report(tmp_path)
    assert "2 × clean-positive" in out
    assert "1 × clean-negative" in out
