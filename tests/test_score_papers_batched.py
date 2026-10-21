"""Tests for the batched score_papers stage.

The old single-call scoring stage silently degraded to ``{"scores": []}`` on
long corpora because the model couldn't fit ~760 structured entries in one
response. Three guards introduced together:
  1. ScoresOutput.scores min_length=1 (schema)
  2. Batched calls (no single call asks for >batch_size×|dims| entries)
  3. Coverage tracking + warning in meta.json when realized coverage dips
     below SCORING_MIN_COVERAGE
"""

from __future__ import annotations

import json
from pathlib import Path


def _make_papers(n: int) -> list[dict]:
    return [
        {
            "paper_id": f"P{i + 1}",
            "openalex_id": f"W{i + 1:04d}",
            "title": f"Paper {i + 1}",
            "abstract": "abstract " * 30,
            "year": 2024,
            "source_topic": "test",
        }
        for i in range(n)
    ]


def _make_dimensions() -> list[dict]:
    return [
        {"name": "d1", "description": "x", "example_question": "y"},
        {"name": "d2", "description": "x", "example_question": "y"},
    ]


def _scores_json_for(papers, dims, all_score=3) -> str:
    return json.dumps({
        "scores": [
            {
                "paper_id": p["paper_id"],
                "title_short": p["title"],
                "dimensions": {
                    d["name"]: {"score": all_score, "contribution": "x"}
                    for d in dims
                },
            }
            for p in papers
        ]
    })


def test_score_papers_uses_batches_for_long_corpora(tmp_path: Path, monkeypatch):
    from pipeline import compose, llm_io

    papers = _make_papers(45)
    dims = _make_dimensions()
    calls: list[str] = []

    def fake_invoke(prompt: str, reasoning: str) -> tuple[bool, str, str]:
        calls.append(prompt)
        # Identify which papers are in THIS batch's prompt and return scores for them.
        batch_papers = [p for p in papers if f"[{p['paper_id']}]" in prompt
                        or f"{p['paper_id']}: " in prompt]
        return True, _scores_json_for(batch_papers, dims), ""

    monkeypatch.setattr(llm_io, "_invoke_codex", fake_invoke)

    result = compose.score_papers(papers, dims, "test", run_dir=tmp_path,
                                  batch_size=20)
    # 45 papers / 20 = 3 batches → 3 codex calls
    assert len(calls) == 3
    # All 45 papers covered exactly once
    pids = {s["paper_id"] for s in result["scores"]}
    assert len(pids) == 45
    assert pids == {p["paper_id"] for p in papers}


def test_score_papers_empty_response_triggers_id_aware_repair(tmp_path: Path,
                                                               monkeypatch):
    """If the model returns {"scores": []} (the original bug), the new
    score_papers fires a SECOND codex call with explicit paper IDs and
    dimension names. If the repair also returns empty, the batch is
    dropped (no scores recovered) but the run continues."""
    from pipeline import compose, llm_io

    papers = _make_papers(10)
    dims = _make_dimensions()

    counter = {"n": 0}

    def always_empty(prompt: str, reasoning: str) -> tuple[bool, str, str]:
        counter["n"] += 1
        return True, json.dumps({"scores": []}), ""

    monkeypatch.setattr(llm_io, "_invoke_codex", always_empty)
    result = compose.score_papers(papers, dims, "test", run_dir=tmp_path,
                                  batch_size=20)
    # 1 primary call + 1 id-aware repair = 2 codex calls.
    assert counter["n"] == 2
    assert result["scores"] == []
    # Both attempts produced an artifact dir.
    assert (tmp_path / "llm_calls" / "scoring").is_dir()
    assert (tmp_path / "llm_calls" / "scoring_repair").is_dir()


def test_score_papers_id_aware_repair_recovers_missing_paper(tmp_path: Path,
                                                              monkeypatch):
    """The id-aware repair only accepts scores for the requested paper IDs,
    guarding against the observed pattern of the model returning a bogus
    score for an unrelated paper."""
    from pipeline import compose, llm_io

    papers = _make_papers(2)
    dims = _make_dimensions()
    state = {"n": 0}

    def stub(prompt: str, reasoning: str) -> tuple[bool, str, str]:
        state["n"] += 1
        if state["n"] == 1:
            # Primary returns score for P1 only (P2 missing).
            return True, _scores_json_for([papers[0]], dims), ""
        # Repair must include P2 in the prompt and ONLY ask for P2.
        assert "P2" in prompt
        # Model bogusly returns scores for both P1 and P2; only P2 should
        # be accepted (P1 was already scored).
        return True, _scores_json_for(papers, dims), ""

    monkeypatch.setattr(llm_io, "_invoke_codex", stub)
    result = compose.score_papers(papers, dims, "test", run_dir=tmp_path,
                                  batch_size=20)
    assert state["n"] == 2
    pids = {s["paper_id"] for s in result["scores"]}
    assert pids == {"P1", "P2"}
    # No duplicates — repair-supplied P1 should be deduped against primary
    assert len(result["scores"]) == 2


def test_score_papers_writes_coverage_warning_when_a_batch_returns_empty(
    tmp_path: Path, monkeypatch,
):
    """Per-batch tracking: even at 50% overall coverage, a wholly empty
    batch must surface a warning. The original failure mode (one batch
    full, one batch empty) used to slip through when only overall
    coverage was checked."""
    from pipeline import compose, llm_io

    papers = _make_papers(40)  # 2 batches at batch_size=20
    dims = _make_dimensions()
    state = {"call_count": 0}

    def half_scoring(prompt: str, reasoning: str) -> tuple[bool, str, str]:
        state["call_count"] += 1
        # First batch: return scores for the 20 papers in the prompt.
        # Second batch + its repair: empty.
        if state["call_count"] == 1:
            batch_papers = [p for p in papers if f"{p['paper_id']}: " in prompt]
            return True, _scores_json_for(batch_papers, dims), ""
        return True, json.dumps({"scores": []}), ""

    monkeypatch.setattr(llm_io, "_invoke_codex", half_scoring)
    result = compose.score_papers(papers, dims, "test", run_dir=tmp_path,
                                  batch_size=20)
    # Only the first batch's 20 scores landed.
    assert len(result["scores"]) == 20

    metas = sorted((tmp_path / "llm_calls" / "scoring").glob("*/meta.json"))
    assert metas, "expected scoring artifact dirs"
    last_meta = json.loads(metas[-1].read_text())
    assert last_meta["scoring_total_papers"] == 40
    assert last_meta["scoring_n_batches"] == 2
    assert last_meta["scoring_coverage"] == 0.5
    assert last_meta["scoring_empty_batches"] == 1
    # One batch returned no scores -> warning fires even though overall
    # coverage is exactly at the floor.
    assert last_meta["scoring_coverage_warning"] is not None
    assert "1/2 batches" in last_meta["scoring_coverage_warning"]


def test_score_papers_warning_fires_below_threshold(tmp_path: Path, monkeypatch):
    from pipeline import compose, llm_io

    papers = _make_papers(60)  # 3 batches at batch_size=20
    dims = _make_dimensions()

    def first_only(prompt: str, reasoning: str) -> tuple[bool, str, str]:
        # Content-keyed: see half_empty above. Only batch 1 (P1..P20) scores.
        if "P1: " in prompt and "P21: " not in prompt and "P41: " not in prompt:
            batch_papers = [p for p in papers if f"{p['paper_id']}: " in prompt]
            return True, _scores_json_for(batch_papers, dims), ""
        return True, json.dumps({"scores": []}), ""

    monkeypatch.setattr(llm_io, "_invoke_codex", first_only)
    result = compose.score_papers(papers, dims, "test", run_dir=tmp_path,
                                  batch_size=20)
    # 20 of 60 papers scored = 0.33 < 0.5 threshold
    assert len(result["scores"]) == 20
    metas = sorted((tmp_path / "llm_calls" / "scoring").glob("*/meta.json"))
    last_meta = json.loads(metas[-1].read_text())
    assert last_meta["scoring_coverage"] < 0.5
    assert last_meta["scoring_coverage_warning"] is not None
    assert "33%" in last_meta["scoring_coverage_warning"]


def test_score_papers_empty_corpus_returns_empty(tmp_path: Path):
    from pipeline import compose
    result = compose.score_papers([], _make_dimensions(), "test", run_dir=tmp_path)
    assert result["scores"] == []
    assert result["coverage"] == 0.0
    assert result["n_papers"] == 0
    assert result["n_batches"] == 0


def test_score_papers_returns_coverage_metadata(tmp_path: Path, monkeypatch):
    """score_papers returns coverage/n_batches/empty_batches so compose_one
    can gate Stage 4 on scoring health without re-parsing artifact meta."""
    from pipeline import compose, llm_io

    papers = _make_papers(40)
    dims = _make_dimensions()

    def half_empty(prompt: str, reasoning: str) -> tuple[bool, str, str]:
        # Content-keyed (not call-order-keyed): batches run concurrently, so
        # "first call" is nondeterministic. Only the batch containing P1
        # gets scores; every other call (batch 2, repairs) returns empty.
        if "P1: " in prompt and "P21: " not in prompt:
            batch_papers = [p for p in papers if f"{p['paper_id']}: " in prompt]
            return True, _scores_json_for(batch_papers, dims), ""
        return True, json.dumps({"scores": []}), ""

    monkeypatch.setattr(llm_io, "_invoke_codex", half_empty)
    result = compose.score_papers(papers, dims, "test", run_dir=tmp_path,
                                  batch_size=20)
    assert result["n_papers"] == 40
    assert result["n_batches"] == 2
    assert result["empty_batches"] == 1
    assert result["coverage"] == 0.5
    assert result["per_batch_yield"] == [20, 0]


def test_find_compositions_refuses_empty_scores(tmp_path: Path):
    """Stage 4 must not run when Stage 3 produced no scores — pre-fix runs
    surfaced 'compositions' from random clusters when scoring returned []."""
    from pipeline import compose
    result = compose.find_compositions(
        papers=[{"paper_id": "P1", "title": "x", "abstract": "x", "year": 2024}],
        dimensions=[{"name": "d1", "description": "x", "example_question": "y"}],
        scores=[],
        problem="test",
        run_dir=tmp_path,
    )
    assert result["error"] == "empty_scores"
    assert result["compositions"] == []


def test_find_compositions_prompt_uses_top_cluster_candidates(
    tmp_path: Path, monkeypatch,
):
    """Large trawls should be scored broadly, then compose from the bounded
    top candidates instead of pasting every surviving paper into Stage 4.
    """
    from pipeline import compose

    papers = _make_papers(30)
    dims = [{"name": "d1", "description": "x", "example_question": "y"}]
    scores = [
        {
            "paper_id": p["paper_id"],
            "title_short": p["title"],
            "dimensions": {
                "d1": {
                    "score": 5 if i < 20 else 2,
                    "contribution": f"contribution {i}",
                }
            },
        }
        for i, p in enumerate(papers)
    ]
    captured: dict[str, str] = {}

    def fake_run_codex_with_artifacts(
        stage: str,
        prompt: str,
        run_dir: Path,
        reasoning: str,
    ) -> tuple[bool, str, str, Path]:
        captured["prompt"] = prompt
        artifact_dir = run_dir / "llm_calls" / stage / "001"
        artifact_dir.mkdir(parents=True, exist_ok=True)
        return True, json.dumps({
            "compositions": [
                {
                    "id": "C1",
                    "synthesis_claim": "P1 and P2 compose",
                    "papers": [
                        {"paper_id": "P1", "role": "x", "quote": "x"},
                        {"paper_id": "P2", "role": "y", "quote": "y"},
                    ],
                }
            ]
        }), "", artifact_dir

    monkeypatch.setattr(compose, "run_codex_with_artifacts", fake_run_codex_with_artifacts)

    result = compose.find_compositions(
        papers=papers,
        dimensions=dims,
        scores=scores,
        problem="test",
        run_dir=tmp_path,
    )

    assert len(result["compositions"]) == 1
    assert "P15:" in captured["prompt"]
    assert "P16:" not in captured["prompt"]
    assert "P30:" not in captured["prompt"]


def test_compose_one_aborts_before_stage_4_on_zero_coverage(
    tmp_path: Path, monkeypatch,
):
    """compose_one must not run Stage 4 if Stage 3 produced no scores. Run
    20260522-062501 was a regression where 0/95 scored papers still
    surfaced 7 compositions; this test pins the gate behavior."""
    import json as _json

    from pipeline import compose, llm_io

    # Stub the corpus assembly so we don't hit OpenAlex.
    monkeypatch.setattr(compose, "assemble_corpus", lambda *_a, **_kw: [
        {"paper_id": "P1", "openalex_id": "W1", "title": "T1",
         "abstract": "abstract here", "year": 2024, "source_topic": "x"},
        {"paper_id": "P2", "openalex_id": "W2", "title": "T2",
         "abstract": "abstract here", "year": 2024, "source_topic": "x"},
    ])
    # Use a tmp compose dir so we don't pollute the real one.
    monkeypatch.setattr(compose, "COMPOSE_DIR", tmp_path)

    state = {"n": 0}

    def fake_codex(prompt: str, reasoning: str) -> tuple[bool, str, str]:
        state["n"] += 1
        if "dimensions" in prompt and "synthesis" not in prompt:
            return True, _json.dumps({
                "dimensions": [
                    {"name": "d1", "description": "x", "example_question": "y"}
                ]
            }), ""
        # Stage 3 (scoring) and any other stage: always return empty,
        # so Stage 4 must be gated off.
        return True, _json.dumps({"scores": []}), ""

    monkeypatch.setattr(llm_io, "_invoke_codex", fake_codex)

    result = compose.compose_one(
        topic_list_path=Path("/dev/null"),
        problem="test problem",
        skip_gate=True,  # bypass relevance gate; we're testing the scoring gate
        with_recipes=False,
        with_novelty=False,
        strict_scoring=True,
    )
    assert result["status"] == "insufficient_scoring_coverage"
    assert "scoring_abort_reason" in result
    # Stage 4 (compositions) should NOT have been called.
    assert not any(
        "synthesis" in str(p)
        for p in (tmp_path / result["run_id"] / "llm_calls").rglob("prompt.txt")
    ) or not (tmp_path / result["run_id"] / "llm_calls" / "compositions").exists()


def test_compose_one_no_strict_scoring_bypasses_gate(
    tmp_path: Path, monkeypatch,
):
    """--no-strict-scoring (strict_scoring=False) lets the run continue past
    Stage 3 even with low coverage. Documented as a debugging-only escape
    hatch."""
    import json as _json

    from pipeline import compose, llm_io

    monkeypatch.setattr(compose, "assemble_corpus", lambda *_a, **_kw: [
        {"paper_id": "P1", "openalex_id": "W1", "title": "T1",
         "abstract": "abstract here", "year": 2024, "source_topic": "x"},
        {"paper_id": "P2", "openalex_id": "W2", "title": "T2",
         "abstract": "abstract here", "year": 2024, "source_topic": "x"},
    ])
    monkeypatch.setattr(compose, "COMPOSE_DIR", tmp_path)

    def fake_codex(prompt: str, reasoning: str) -> tuple[bool, str, str]:
        if "dimensions" in prompt and "synthesis" not in prompt:
            return True, _json.dumps({
                "dimensions": [
                    {"name": "d1", "description": "x", "example_question": "y"}
                ]
            }), ""
        # Scoring returns empty; with strict_scoring=False the run still
        # tries Stage 4, where find_compositions refuses to run on empty
        # scores and returns its own error.
        return True, _json.dumps({"scores": []}), ""

    monkeypatch.setattr(llm_io, "_invoke_codex", fake_codex)

    result = compose.compose_one(
        topic_list_path=Path("/dev/null"),
        problem="test problem",
        skip_gate=True,
        with_recipes=False,
        with_novelty=False,
        strict_scoring=False,
    )
    # Not aborted at scoring; the run proceeded to Stage 4 which itself
    # refused to operate on empty scores.
    assert result["status"] != "insufficient_scoring_coverage"
