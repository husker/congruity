"""Tests for the performance/cost layer: tiered effort, race-safe artifact
dirs, usage aggregation, and the embedding cache."""

from __future__ import annotations

import json

import pipeline.corpus_gate as corpus_gate
from pipeline.llm_io import (_next_seq_dir, aggregate_llm_usage,
                             run_codex_with_artifacts, stage_reasoning)


# --------------------------------------------------------------------------
# stage_reasoning
# --------------------------------------------------------------------------

def test_stage_reasoning_default_passthrough(monkeypatch):
    monkeypatch.delenv("CONGRUITY_EFFORT_SCORING", raising=False)
    assert stage_reasoning("scoring", "high") == "high"


def test_stage_reasoning_env_override(monkeypatch):
    monkeypatch.setenv("CONGRUITY_EFFORT_SCORING", "medium")
    assert stage_reasoning("scoring", "high") == "medium"


def test_stage_reasoning_strips_composition_suffix(monkeypatch):
    monkeypatch.setenv("CONGRUITY_EFFORT_VERIFY_L1", "low")
    assert stage_reasoning("verify_l1_C3", "high") == "low"
    assert stage_reasoning("verify_l1_C12", "high") == "low"


def test_stage_reasoning_rejects_invalid_value(monkeypatch):
    monkeypatch.setenv("CONGRUITY_EFFORT_SCORING", "turbo")
    assert stage_reasoning("scoring", "high") == "high"


def test_reasoning_override_reaches_codex(tmp_path, monkeypatch):
    import pipeline.llm_io as llm_io

    seen = {}

    def fake(prompt, reasoning, timeout_s=None):
        seen["reasoning"] = reasoning
        return True, "{}", ""

    monkeypatch.setattr(llm_io, "_invoke_codex", fake)
    monkeypatch.setenv("CONGRUITY_EFFORT_DIMENSIONS", "medium")
    run_codex_with_artifacts("dimensions", "p", tmp_path, reasoning="high")
    assert seen["reasoning"] == "medium"


# --------------------------------------------------------------------------
# _next_seq_dir
# --------------------------------------------------------------------------

def test_next_seq_dir_sequential(tmp_path):
    a = _next_seq_dir(tmp_path / "stage")
    b = _next_seq_dir(tmp_path / "stage")
    assert a.name == "01" and b.name == "02"
    assert a.is_dir() and b.is_dir()  # claimed at allocation time


def test_next_seq_dir_skips_collision(tmp_path):
    stage = tmp_path / "stage"
    _next_seq_dir(stage)  # 01
    (stage / "02").mkdir()  # simulate a racing claimant
    c = _next_seq_dir(stage)
    assert c.name == "03"


# --------------------------------------------------------------------------
# meta.json cost fields + aggregate_llm_usage
# --------------------------------------------------------------------------

def test_meta_records_cost_proxies(tmp_path, monkeypatch):
    import pipeline.llm_io as llm_io

    monkeypatch.setattr(llm_io, "_invoke_codex",
                        lambda p, r, t=None: (True, "x" * 400, ""))
    _, _, _, artifact_dir = run_codex_with_artifacts(
        "dimensions", "y" * 800, tmp_path)
    meta = json.loads((artifact_dir / "meta.json").read_text())
    assert meta["prompt_chars"] == 800
    assert meta["stdout_chars"] == 400
    assert meta["est_prompt_tokens"] == 200
    assert meta["est_output_tokens"] == 100


def test_aggregate_llm_usage_groups_by_base_stage(tmp_path):
    for stage, dur in (("dimensions", 10.0), ("verify_l1_C1", 5.0),
                       ("verify_l1_C2", 7.0)):
        d = tmp_path / "llm_calls" / stage / "01"
        d.mkdir(parents=True)
        (d / "meta.json").write_text(json.dumps({
            "stage": stage, "duration_s": dur,
            "est_prompt_tokens": 100, "est_output_tokens": 50,
        }))
    usage = aggregate_llm_usage(tmp_path)
    assert usage["calls"] == 3
    assert usage["duration_s"] == 22.0
    assert usage["est_prompt_tokens"] == 300
    assert set(usage["by_stage"]) == {"dimensions", "verify_l1"}
    assert usage["by_stage"]["verify_l1"]["calls"] == 2
    assert usage["by_stage"]["verify_l1"]["duration_s"] == 12.0


def test_aggregate_llm_usage_empty_run(tmp_path):
    usage = aggregate_llm_usage(tmp_path)
    assert usage["calls"] == 0
    assert usage["by_stage"] == {}


# --------------------------------------------------------------------------
# embedding cache
# --------------------------------------------------------------------------

def _fake_embedder(counter):
    def fake(inputs, api_key):
        counter.append(len(inputs))
        return [[1.0, 0.0] for _ in inputs]
    return fake


def test_embed_cache_second_call_hits_no_api(tmp_path, monkeypatch):
    monkeypatch.setattr(corpus_gate, "EMBED_CACHE_DIR", tmp_path / "cache")
    calls: list[int] = []
    monkeypatch.setattr(corpus_gate, "_openai_embed", _fake_embedder(calls))
    papers = [{"paper_id": "P1", "title": "t", "abstract": "a"}]

    issues1, skipped1 = corpus_gate.check_relevance(papers, "problem", 0.1, "sk-x")
    assert not skipped1
    assert sum(calls) == 2  # problem + 1 paper

    issues2, skipped2 = corpus_gate.check_relevance(papers, "problem", 0.1, "sk-x")
    assert not skipped2
    assert sum(calls) == 2  # fully served from cache
    assert issues1 == issues2


def test_embed_cache_partial_miss_only_embeds_new(tmp_path, monkeypatch):
    monkeypatch.setattr(corpus_gate, "EMBED_CACHE_DIR", tmp_path / "cache")
    calls: list[int] = []
    monkeypatch.setattr(corpus_gate, "_openai_embed", _fake_embedder(calls))
    p1 = {"paper_id": "P1", "title": "t1", "abstract": "a1"}
    p2 = {"paper_id": "P2", "title": "t2", "abstract": "a2"}

    corpus_gate.check_relevance([p1], "problem", 0.1, "sk-x")
    assert sum(calls) == 2
    corpus_gate.check_relevance([p1, p2], "problem", 0.1, "sk-x")
    assert sum(calls) == 3  # only p2 was new


def test_embed_cache_corrupt_file_is_ignored(tmp_path, monkeypatch):
    monkeypatch.setattr(corpus_gate, "EMBED_CACHE_DIR", tmp_path / "cache")
    (tmp_path / "cache").mkdir()
    corpus_gate._embed_cache_path().write_text("{not json")
    calls: list[int] = []
    monkeypatch.setattr(corpus_gate, "_openai_embed", _fake_embedder(calls))
    _, skipped = corpus_gate.check_relevance(
        [{"paper_id": "P1", "title": "t", "abstract": "a"}], "q", 0.1, "sk-x")
    assert not skipped
    assert sum(calls) == 2
