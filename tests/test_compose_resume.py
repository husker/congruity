"""Resume tests for compose_one: a crashed run re-pays only unfinished stages.

Builds on the stubbed-codex integration fixtures: run once to completion,
then resume and assert zero codex calls; then delete a mid-pipeline
artifact and assert only that stage re-runs.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from tests.test_compose_integration import _fake_codex_responses, _make_fake_codex


@pytest.fixture
def fixture_corpus() -> list[dict]:
    return [
        {
            "openalex_id": f"https://openalex.org/W{i:04d}",
            "title": f"Paper {i} on embedding quantization",
            "abstract": "We propose a novel embedding compression scheme. " * 30,
            "year": 2024,
            "doi": None,
            "source_topic": "embedding compression",
            "source_topic_id": "T1",
        }
        for i in range(1, 6)
    ]


def _counting_codex(monkeypatch):
    """Install the canned-response stub, returning a per-call log."""
    from pipeline import llm_io

    inner = _make_fake_codex(_fake_codex_responses())
    calls: list[str] = []

    def fake(prompt: str, reasoning: str) -> tuple[bool, str, str]:
        calls.append(prompt[:80])
        return inner(prompt, reasoning)

    monkeypatch.setattr(llm_io, "_invoke_codex", fake)
    return calls


def _run_fresh(tmp_path: Path, monkeypatch, fixture_corpus) -> str:
    from pipeline import compose

    monkeypatch.setattr(compose, "sample_topic", lambda *a, **kw: list(fixture_corpus))
    monkeypatch.setattr(compose, "COMPOSE_DIR", tmp_path / "runs")
    topic_list = tmp_path / "topics.json"
    topic_list.write_text(json.dumps([
        {"id": "T1", "display_name": "embedding compression", "works_count": 100},
    ]))
    state = compose.compose_one(
        topic_list_path=topic_list,
        problem="compress sentence embeddings to under 64 bits per vector",
        papers_per_topic=5, max_papers=5,
        skip_gate=True, strict_gate=False, dry_run=False,
    )
    assert state["status"] == "complete"
    return state["run_id"]


def test_resume_completed_run_makes_no_codex_calls(tmp_path, monkeypatch,
                                                   fixture_corpus):
    from pipeline import compose

    _counting_codex(monkeypatch)
    run_id = _run_fresh(tmp_path, monkeypatch, fixture_corpus)

    calls = _counting_codex(monkeypatch)  # fresh counter for the resume
    state = compose.compose_one(
        topic_list_path=None, problem="compress sentence embeddings",
        skip_gate=True, strict_gate=False,
        resume_run_id=run_id,
    )
    assert state["status"] == "complete"
    assert len(state["compositions"]) == 1
    assert calls == []  # every stage reused a persisted artifact


def test_resume_reruns_only_missing_stage(tmp_path, monkeypatch, fixture_corpus):
    from pipeline import compose

    _counting_codex(monkeypatch)
    run_id = _run_fresh(tmp_path, monkeypatch, fixture_corpus)
    run_dir = tmp_path / "runs" / run_id

    # Simulate a crash after dimensions: scoring and everything downstream
    # of it never persisted — but verification progress DID (it happened
    # before our simulated deletion in real crash order; here we only care
    # that untouched stages stay cached).
    (run_dir / "scores.json").unlink()
    (run_dir / "state.json").unlink()

    calls = _counting_codex(monkeypatch)
    state = compose.compose_one(
        topic_list_path=None, problem="compress sentence embeddings",
        skip_gate=True, strict_gate=False,
        resume_run_id=run_id,
    )
    assert state["status"] == "complete"
    # Only the scoring stage re-ran (5 papers = 1 batch = 1 codex call);
    # dimensions, compositions, and all L1/L3/L4 verification were reused.
    assert len(calls) == 1
    assert "scoring" in calls[0] or "scor" in calls[0].lower()


def test_resume_unknown_run_id_raises(tmp_path, monkeypatch):
    from pipeline import compose

    monkeypatch.setattr(compose, "COMPOSE_DIR", tmp_path / "runs")
    with pytest.raises(ValueError, match="cannot resume"):
        compose.compose_one(topic_list_path=None, problem="x",
                            resume_run_id="19700101-000000")


def test_resume_and_dry_run_are_exclusive(tmp_path, monkeypatch):
    from pipeline import compose

    monkeypatch.setattr(compose, "COMPOSE_DIR", tmp_path / "runs")
    (tmp_path / "runs" / "r1").mkdir(parents=True)
    with pytest.raises(ValueError, match="mutually exclusive"):
        compose.compose_one(topic_list_path=None, problem="x",
                            dry_run=True, resume_run_id="r1")
