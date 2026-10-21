"""End-to-end smoke tests for compose_one with stubbed LLM + corpus.

These tests exercise the full pipeline orchestrator (corpus assembly →
quality gate → dimensions → scoring → compositions → L1/L2/L3/L4
verification) without spawning real codex or hitting OpenAlex. They
catch regressions in:

  - state.json shape (top-level keys, status values)
  - corpus_raw.json / corpus_validated.json dual-write
  - llm_calls/<stage>/NN/ artifact persistence
  - strict-gate abort path

The fake codex outputs are deliberately minimal but match the actual
prompt contracts so parse_or_repair validates them against the
Pydantic schemas without needing the repair retry.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest


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


def _fake_codex_responses() -> dict[str, str]:
    """Canned LLM outputs per stage. Field names and casing match the
    actual prompt contracts (e.g. L3 verdicts are lowercase)."""
    return {
        "dimensions": json.dumps({
            "dimensions": [
                {"name": "compression_ratio",
                 "description": "bits per vector",
                 "example_question": "how many bits?"},
                {"name": "recall",
                 "description": "fraction of true neighbors recovered",
                 "example_question": "recall@10?"},
            ]
        }),
        "scoring": json.dumps({
            "scores": [
                {"paper_id": f"P{i}",
                 "title_short": f"Paper {i}",
                 "dimensions": {
                     "compression_ratio": {"score": 3, "contribution": "x"},
                     "recall": {"score": 2, "contribution": "y"},
                 }}
                for i in range(1, 6)
            ]
        }),
        "compositions": json.dumps({
            "compositions": [
                {
                    "id": "C1",
                    "synthesis_claim": "P1 + P2 together beat either alone",
                    "papers": [
                        {"paper_id": "P1", "role": "provides X", "quote": "..."},
                        {"paper_id": "P2", "role": "provides Y", "quote": "..."},
                    ],
                }
            ]
        }),
        "verify_l1_C1": json.dumps({
            "consistent": True,
            "issues": [],
        }),
        "verify_l3_C1": json.dumps({
            "verdict": "weak",
            "verdict_reason": "claim is plausible but the cited papers do not jointly establish it",
            "falsification_attempts": [
                {"angle": "novelty",
                 "argument": "P1 alone may already deliver the claimed effect",
                 "would_invalidate": False},
                {"angle": "rate-distortion",
                 "argument": "the proposed budget may be too tight to preserve recall",
                 "would_invalidate": True},
            ],
        }),
        "verify_l4_C1": json.dumps({
            "objection_classifications": [],
            "summary": {
                "total_objections": 0,
                "fundamental_count": 0,
                "empirical_count": 0,
                "net_testability_score": "PURELY_TESTABLE",
                "verdict_for_unsolved_problem": "worth a small computational test",
            },
        }),
    }


def _make_fake_codex(responses: dict[str, str]):
    """Build a stub for llm_io._invoke_codex that inspects the prompt and
    picks the right canned response based on which stage's prompt template
    is recognizable in the prompt text."""

    def fake(prompt: str, reasoning: str) -> tuple[bool, str, str]:
        # Match the most-specific stage markers first; the prompts share
        # boilerplate phrases like "compositional synthesis" so the order
        # of checks matters. The phrases used here are pulled from the
        # actual prompt templates in pipeline/compose.py.
        if "FUNDAMENTAL or EMPIRICAL" in prompt:
            return True, responses["verify_l4_C1"], ""
        if "FALSIFY a" in prompt or "Attempt to falsify" in prompt:
            return True, responses["verify_l3_C1"], ""
        if "auditing a proposed compositional synthesis" in prompt:
            return True, responses["verify_l1_C1"], ""
        if "identifying compositional syntheses" in prompt:
            return True, responses["compositions"], ""
        if "scoring papers in a corpus" in prompt:
            return True, responses["scoring"], ""
        if "Identify 5-8 *problem dimensions*" in prompt:
            return True, responses["dimensions"], ""
        # Fallback: surface an obvious-to-debug error rather than silently
        # returning a wrong-stage response.
        return False, "", f"fake_codex: no matcher hit; prompt[:200]={prompt[:200]!r}"

    return fake


def test_compose_one_end_to_end_with_stubs(tmp_path: Path, monkeypatch, fixture_corpus):
    from pipeline import compose, llm_io

    # sample_topic is imported into compose.py's namespace, so patch it there.
    monkeypatch.setattr(compose, "sample_topic", lambda *a, **kw: list(fixture_corpus))
    monkeypatch.setattr(llm_io, "_invoke_codex", _make_fake_codex(_fake_codex_responses()))
    # Redirect runs into tmp_path so test artifacts don't pile up under data/.
    monkeypatch.setattr(compose, "COMPOSE_DIR", tmp_path / "runs")

    topic_list = tmp_path / "topics.json"
    topic_list.write_text(json.dumps([
        {"id": "T1", "display_name": "embedding compression", "works_count": 100},
    ]))

    state = compose.compose_one(
        topic_list_path=topic_list,
        problem="compress sentence embeddings to under 64 bits per vector",
        papers_per_topic=5,
        max_papers=5,
        skip_gate=True,        # don't hit the OpenAI embedding API
        strict_gate=False,     # not needed when skip_gate=True
        dry_run=False,
    )

    # Top-level state shape.
    assert state["status"] == "complete"
    assert state["corpus_size"] == 5
    assert "dimensions" in state
    assert "compositions" in state
    assert len(state["compositions"]) == 1

    # Run dir on disk
    run_dirs = list((tmp_path / "runs").glob("*"))
    assert len(run_dirs) == 1
    run_dir = run_dirs[0]

    # Corpus dual-write
    assert (run_dir / "corpus_raw.json").exists()
    assert (run_dir / "corpus_validated.json").exists()
    raw = json.loads((run_dir / "corpus_raw.json").read_text())
    val = json.loads((run_dir / "corpus_validated.json").read_text())
    assert len(raw) == 5
    assert len(val) == 5
    assert all("paper_uid" in p for p in raw)

    # Per-stage artifact persistence
    dims_dir = run_dir / "llm_calls" / "dimensions"
    assert dims_dir.is_dir()
    nn = sorted(d for d in dims_dir.iterdir() if d.is_dir())
    assert nn, "expected at least one dimensions/NN/ directory"
    meta = json.loads((nn[0] / "meta.json").read_text())
    assert meta["parse_status"] == "ok"
    assert meta["stage"] == "dimensions"
    assert (nn[0] / "prompt.txt").exists()
    assert (nn[0] / "stdout.txt").exists()

    # Per-composition verification artifacts
    assert (run_dir / "llm_calls" / "verify_l1_C1").is_dir()
    assert (run_dir / "llm_calls" / "verify_l3_C1").is_dir()
    assert (run_dir / "llm_calls" / "verify_l4_C1").is_dir()


def test_compose_one_aborts_on_strict_gate_no_api_key(tmp_path: Path, monkeypatch,
                                                      fixture_corpus):
    from pipeline import compose

    monkeypatch.setattr(compose, "sample_topic", lambda *a, **kw: list(fixture_corpus))
    monkeypatch.setattr(compose, "COMPOSE_DIR", tmp_path / "runs")
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)

    topic_list = tmp_path / "topics.json"
    topic_list.write_text(json.dumps([
        {"id": "T1", "display_name": "x", "works_count": 100},
    ]))

    state = compose.compose_one(
        topic_list_path=topic_list,
        problem="x",
        papers_per_topic=5,
        max_papers=5,
        skip_gate=False,
        strict_gate=True,
        dry_run=False,
    )

    assert state["status"] == "aborted_corpus_gate"
    assert "relevance" in state["corpus_gate"]["abort_reason"].lower()
