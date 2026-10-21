"""Tests for the product-facing static web UI."""

from __future__ import annotations

import json
from pathlib import Path


def _write_index(index_path: Path, run_id: str, outcome: str = "clean-positive") -> None:
    index_path.write_text(
        "\n".join(
            [
                "# Verification track record",
                "",
                "| run_id | experiment | composition | outcome | finding doc |",
                "|---|---|---|---|---|",
                f"| {run_id} | Heilbronn small-points | C1 | **{outcome}** | `findings/x.md` |",
            ]
        )
        + "\n",
        encoding="utf-8",
    )


def _make_run(root: Path, run_id: str, *, status: str = "complete",
              compositions: list[dict] | None = None) -> None:
    run_dir = root / run_id
    run_dir.mkdir(parents=True)
    state = {
        "run_id": run_id,
        "status": status,
        "problem": (
            "Find compositions that improve TSP solving via spectral graph methods, "
            "cross-disciplinary partitioning, and Markov-chain mixing-time analysis."
        ),
        "corpus_size": 42,
        "dimensions": [{"name": "routing", "description": "x"}],
        "topics_covered": 4,
        "scoring_coverage": 0.91,
        "compositions": compositions or [],
    }
    (run_dir / "state.json").write_text(json.dumps(state), encoding="utf-8")
    (run_dir / "corpus_gate.json").write_text(
        json.dumps({"total": 42, "passed": 40, "failed": 2, "skipped_checks": []}),
        encoding="utf-8",
    )


def _actionable_composition() -> dict:
    return {
        "id": "C1",
        "synthesis_claim": (
            "A spectral partitioner plus local search diagnostics could improve a "
            "partitioned TSP solver."
        ),
        "confidence": 8,
        "open_questions": ["Does the partition objective preserve tour quality?"],
        "verification": {
            "L2": {"passed": True},
            "L3": {"verdict": "weak", "verdict_reason": "Plausible but unproven."},
            "L4": {
                "summary": {
                    "net_testability_score": "PURELY_TESTABLE",
                    "empirical_count": 2,
                    "fundamental_count": 0,
                }
            },
            "reconciled": {
                "status": "actionable",
                "l3_l4_consistent": True,
                "reason": "L3=weak and L4=PURELY_TESTABLE; recipe-eligible",
            },
            "recipe": {
                "named_baselines": ["LKH-3", "Concorde"],
                "benchmark_dataset": "TSPLIB95",
                "metric_primary": "mean percent optimality gap",
                "stop_rule": "Stop if the method trails LKH-3 on first five instances.",
                "likely_failure_mode": "Partition boundaries make stitching worse.",
            },
        },
    }


def test_build_site_creates_dashboard_and_run_pages(tmp_path):
    from pipeline import web_ui

    root = tmp_path / "runs"
    out = tmp_path / "site"
    index = tmp_path / "INDEX.md"
    _write_index(index, "20260101-000000")
    _make_run(root, "20260101-000000", compositions=[_actionable_composition()])

    manifest = web_ui.build_site(root=root, out_dir=out, index_path=index)

    assert manifest["index"].exists()
    assert (out / "runs" / "20260101-000000.html").exists()
    index_html = manifest["index"].read_text(encoding="utf-8")
    assert "Congruity Evidence Board" in index_html
    assert "Can an AI system turn papers into discoveries you can actually check?" in index_html
    assert "Point-placement geometry test" in index_html
    assert "how to place points so the smallest triangle" in index_html
    assert "Clear win" in index_html


def test_run_page_explains_layperson_summary_and_advanced_details(tmp_path):
    from pipeline import web_ui

    root = tmp_path / "runs"
    out = tmp_path / "site"
    index = tmp_path / "INDEX.md"
    _write_index(index, "20260102-000000", outcome="clean-negative")
    _make_run(root, "20260102-000000", compositions=[_actionable_composition()])

    web_ui.build_site(root=root, out_dir=out, index_path=index)
    page = (out / "runs" / "20260102-000000.html").read_text(encoding="utf-8")

    assert "What was being tested?" in page
    assert "Did this prove Congruity works?" in page
    assert "Plain-English translation" in page
    assert "Technical label:" in page
    assert "Baselines, metrics, and stop rule" in page
    assert "LKH-3" in page
    assert "Raw candidate JSON" in page
    assert "falsify ideas" in page


def test_dashboard_filters_include_actionable_and_stopped_runs(tmp_path):
    from pipeline import web_ui

    root = tmp_path / "runs"
    out = tmp_path / "site"
    index = tmp_path / "INDEX.md"
    index.write_text("# Verification track record\n", encoding="utf-8")
    _make_run(root, "20260103-000000", compositions=[_actionable_composition()])
    _make_run(root, "20260104-000000", status="aborted_corpus_gate")

    web_ui.build_site(root=root, out_dir=out, index_path=index)
    index_html = (out / "index.html").read_text(encoding="utf-8")

    assert 'value="actionable"' in index_html
    assert 'value="stopped"' in index_html
    assert 'data-actionable="true"' in index_html
    assert 'data-stopped="true"' in index_html


def test_cerny_problem_gets_automata_plain_language_before_compression():
    from pipeline.web_ui import Outcome, _infer_run_language

    title, goal, context, _ = _infer_run_language(
        (
            "Find compositional research directions for the Cerny conjecture on "
            "synchronizing deterministic finite automata: keep ideas that share the "
            "shape of compressing a large finite state set to one state with a reset word."
        ),
        Outcome(),
    )

    assert title == "Reset-word automata discovery test"
    assert "reset" in goal.lower()
    assert "synchronizing automaton" in context
