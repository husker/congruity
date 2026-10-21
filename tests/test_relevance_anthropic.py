"""Tests for the Anthropic LLM-judge relevance backend."""

from __future__ import annotations

import json
import re

import pytest


def _fixture_papers() -> list[dict]:
    return [
        {
            "paper_id": "P1",
            "openalex_id": "https://openalex.org/W1",
            "title": "Spectral graph TSP",
            "abstract": "We use the graph Laplacian eigenstructure to derive a TSP heuristic.",
        },
        {
            "paper_id": "P2",
            "openalex_id": "https://openalex.org/W2",
            "title": "Cardiac valve replacement outcomes",
            "abstract": "A retrospective cohort study of mitral valve replacement.",
        },
        {
            "paper_id": "P3",
            "openalex_id": "https://openalex.org/W3",
            "title": "Graph Laplacian spectral sparsification",
            "abstract": "Spectral sparsification preserves the graph Laplacian under cuts.",
        },
    ]


def test_paper_relevance_score_clipped():
    from pydantic import ValidationError
    from pipeline.relevance import PaperRelevance
    PaperRelevance(paper_id="P1", score=0.5)
    with pytest.raises(ValidationError):
        PaperRelevance(paper_id="P1", score=1.5)
    with pytest.raises(ValidationError):
        PaperRelevance(paper_id="P1", score=-0.1)


def test_resolve_mode_explicit_passthrough():
    from pipeline.relevance import resolve_mode
    assert resolve_mode("openai") == "openai"
    assert resolve_mode("anthropic") == "anthropic"
    assert resolve_mode("codex") == "codex"
    assert resolve_mode("none") == "none"


def test_resolve_mode_auto_prefers_codex_over_env_keys(monkeypatch):
    from pipeline import relevance
    import shutil
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-x")
    monkeypatch.setenv("OPENAI_API_KEY", "sk-x")
    monkeypatch.setattr(shutil, "which", lambda name: "/usr/local/bin/codex"
                        if name == "codex" else None)
    assert relevance.resolve_mode("auto") == "codex"


def test_resolve_mode_auto_prefers_anthropic_over_openai_when_no_codex(monkeypatch):
    from pipeline import relevance
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-x")
    monkeypatch.setenv("OPENAI_API_KEY", "sk-x")
    assert relevance.resolve_mode("auto") == "anthropic"


def test_resolve_mode_auto_falls_through_to_openai(monkeypatch):
    from pipeline import relevance
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    monkeypatch.setenv("OPENAI_API_KEY", "sk-x")
    assert relevance.resolve_mode("auto") == "openai"


def test_resolve_mode_auto_returns_none_when_no_keys(monkeypatch):
    from pipeline import relevance
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    # Pretend codex isn't on PATH either.
    import shutil
    monkeypatch.setattr(shutil, "which", lambda name: None)
    assert relevance.resolve_mode("auto") == "none"


def test_resolve_mode_auto_falls_through_to_codex_when_no_env_keys(monkeypatch):
    from pipeline import relevance
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    import shutil
    monkeypatch.setattr(shutil, "which", lambda name: "/usr/local/bin/codex"
                        if name == "codex" else None)
    assert relevance.resolve_mode("auto") == "codex"


def test_check_relevance_codex_skips_when_codex_binary_missing(monkeypatch):
    from pipeline import relevance
    import shutil
    monkeypatch.setattr(shutil, "which", lambda name: None)
    issues, skipped = relevance.check_relevance_codex(
        _fixture_papers(), "spectral TSP", threshold=0.5,
    )
    assert skipped
    assert issues == []


def test_check_relevance_codex_filters_below_threshold(monkeypatch):
    from pipeline import relevance, llm_io
    import shutil
    monkeypatch.setattr(shutil, "which", lambda name: "/usr/local/bin/codex"
                        if name == "codex" else None)

    canned = json.dumps({
        "scores": [
            {"paper_id": "P1", "score": 0.92, "reason": "directly on topic"},
            {"paper_id": "P2", "score": 0.05, "reason": "unrelated"},
            {"paper_id": "P3", "score": 0.78, "reason": "adjacent"},
        ]
    })

    def fake_invoke(prompt, reasoning):
        return True, canned, ""

    monkeypatch.setattr(llm_io, "_invoke_codex", fake_invoke)

    issues, skipped = relevance.check_relevance_codex(
        _fixture_papers(), "spectral TSP", threshold=0.5,
    )
    assert not skipped
    assert len(issues) == 1
    pid, score, _ = issues[0]
    assert pid == "P2"
    assert score == pytest.approx(0.05)


def test_score_papers_codex_logs_batch_progress(monkeypatch, capsys):
    from pipeline import llm_io, relevance

    def fake_invoke(prompt, reasoning):
        ids = re.findall(r"\[(P\d+)\]", prompt)
        return True, json.dumps({
            "scores": [
                {"paper_id": pid, "score": 0.8, "reason": "adjacent"}
                for pid in ids
            ]
        }), ""

    monkeypatch.setattr(llm_io, "_invoke_codex", fake_invoke)

    scores = relevance.score_papers_codex(
        _fixture_papers(),
        "spectral TSP",
        batch_size=2,
    )

    assert scores == {"P1": 0.8, "P2": 0.8, "P3": 0.8}
    out = capsys.readouterr().out
    assert "codex relevance batch 1/2: scored 2/3 papers" in out
    assert "codex relevance batch 2/2: scored 3/3 papers" in out


def test_validate_corpus_dispatches_to_codex_when_mode_set(monkeypatch):
    from pipeline import corpus_gate, llm_io
    import shutil
    monkeypatch.setattr(shutil, "which", lambda name: "/usr/local/bin/codex"
                        if name == "codex" else None)
    monkeypatch.setattr(
        llm_io, "_invoke_codex",
        lambda prompt, reasoning: (True, json.dumps({
            "scores": [
                {"paper_id": "P1", "score": 0.9, "reason": "x"},
                {"paper_id": "P2", "score": 0.1, "reason": "unrelated"},
                {"paper_id": "P3", "score": 0.8, "reason": "x"},
            ]
        }), ""),
    )
    papers = _fixture_papers()
    for p in papers:
        p["abstract"] = (p["abstract"] + " ") * 8
        p["year"] = 2024

    report = corpus_gate.validate_corpus(
        papers, problem="spectral TSP",
        relevance_mode="codex",
    )
    assert "relevance" not in report.skipped_checks
    assert report.by_check["relevance"] == 1
    assert "P2" in report.drop_ids


def test_validate_corpus_codex_honors_configured_threshold(monkeypatch):
    from pipeline import corpus_gate, llm_io
    import shutil
    monkeypatch.setattr(shutil, "which", lambda name: "/usr/local/bin/codex"
                        if name == "codex" else None)
    monkeypatch.setattr(
        llm_io, "_invoke_codex",
        lambda prompt, reasoning: (True, json.dumps({
            "scores": [
                {"paper_id": "P1", "score": 0.9, "reason": "x"},
                {"paper_id": "P2", "score": 0.4, "reason": "adjacent"},
                {"paper_id": "P3", "score": 0.8, "reason": "x"},
            ]
        }), ""),
    )
    papers = _fixture_papers()
    for p in papers:
        p["abstract"] = (p["abstract"] + " ") * 8
        p["year"] = 2024

    report = corpus_gate.validate_corpus(
        papers,
        problem="spectral TSP",
        relevance_mode="codex",
        relevance_threshold=0.30,
    )

    assert "relevance" not in report.skipped_checks
    assert report.by_check["relevance"] == 0
    assert "P2" not in report.drop_ids


def test_check_relevance_anthropic_skips_without_api_key():
    from pipeline.relevance import check_relevance_anthropic
    issues, skipped = check_relevance_anthropic(
        _fixture_papers(), "spectral TSP", threshold=0.5, api_key=None,
    )
    assert skipped
    assert issues == []


def test_check_relevance_anthropic_filters_below_threshold(monkeypatch):
    from pipeline import relevance

    canned = json.dumps({
        "scores": [
            {"paper_id": "P1", "score": 0.92, "reason": "directly on topic"},
            {"paper_id": "P2", "score": 0.05, "reason": "unrelated"},
            {"paper_id": "P3", "score": 0.78, "reason": "adjacent"},
        ]
    })

    def fake_post(api_key, prompt, model=relevance.DEFAULT_ANTHROPIC_MODEL,
                  max_tokens=4096, timeout=90):
        return canned

    monkeypatch.setattr(relevance, "_post_anthropic", fake_post)

    issues, skipped = relevance.check_relevance_anthropic(
        _fixture_papers(), "spectral TSP", threshold=0.5, api_key="sk-ant-x",
    )
    assert not skipped
    # Only P2 (score 0.05) is below threshold 0.5
    assert len(issues) == 1
    pid, score, detail = issues[0]
    assert pid == "P2"
    assert score == pytest.approx(0.05)
    assert "0.50" in detail


def test_check_relevance_anthropic_handles_missing_scores_gracefully(monkeypatch):
    """If the judge forgets a paper_id, we backfill 1.0 so it isn't penalized."""
    from pipeline import relevance

    canned = json.dumps({
        "scores": [
            {"paper_id": "P1", "score": 0.9, "reason": "y"},
            # P2 and P3 missing
        ]
    })
    monkeypatch.setattr(
        relevance, "_post_anthropic",
        lambda *a, **kw: canned,
    )
    issues, skipped = relevance.check_relevance_anthropic(
        _fixture_papers(), "x", threshold=0.5, api_key="sk-ant-x",
    )
    # P2/P3 backfilled to 1.0 so no issues raised
    assert issues == []
    assert not skipped


def test_check_relevance_anthropic_returns_skipped_on_persistent_failure(monkeypatch):
    from pipeline import relevance

    def fake_post(*a, **kw):
        raise RuntimeError("HTTP 500: server error")

    monkeypatch.setattr(relevance, "_post_anthropic", fake_post)
    issues, skipped = relevance.check_relevance_anthropic(
        _fixture_papers(), "x", threshold=0.5, api_key="sk-ant-x",
    )
    assert skipped
    assert issues == []


def test_check_relevance_anthropic_repairs_invalid_json_once(monkeypatch):
    from pipeline import relevance

    calls: list[str] = []

    def fake_post(api_key, prompt, model=relevance.DEFAULT_ANTHROPIC_MODEL,
                  max_tokens=4096, timeout=90):
        calls.append(prompt)
        if len(calls) == 1:
            return "not json at all"
        return json.dumps({
            "scores": [
                {"paper_id": "P1", "score": 0.9, "reason": "x"},
                {"paper_id": "P2", "score": 0.1, "reason": "x"},
                {"paper_id": "P3", "score": 0.8, "reason": "x"},
            ]
        })

    monkeypatch.setattr(relevance, "_post_anthropic", fake_post)
    issues, skipped = relevance.check_relevance_anthropic(
        _fixture_papers(), "x", threshold=0.5, api_key="sk-ant-x",
    )
    assert not skipped
    assert len(calls) == 2  # initial + one repair
    # P2 is filtered
    assert any(pid == "P2" for pid, _s, _d in issues)


def test_validate_corpus_dispatches_to_anthropic_when_mode_set(monkeypatch, tmp_path):
    from pipeline import corpus_gate, relevance

    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-x")
    monkeypatch.setattr(
        relevance, "_post_anthropic",
        lambda *a, **kw: json.dumps({
            "scores": [
                {"paper_id": "P1", "score": 0.9, "reason": "x"},
                {"paper_id": "P2", "score": 0.1, "reason": "unrelated"},
                {"paper_id": "P3", "score": 0.8, "reason": "x"},
            ]
        }),
    )
    papers = _fixture_papers()
    # Pad abstracts to meet structure-check minimum (30 words).
    for p in papers:
        p["abstract"] = (p["abstract"] + " ") * 8
        p["year"] = 2024

    report = corpus_gate.validate_corpus(
        papers, problem="spectral TSP",
        relevance_mode="anthropic",
    )
    # Relevance ran and flagged P2
    assert "relevance" not in report.skipped_checks
    assert report.by_check["relevance"] == 1
    assert "P2" in report.drop_ids
