"""Tests for the verification-recipe stage."""

from __future__ import annotations

import json

import pytest


def test_verification_recipe_happy_path():
    from pipeline.schemas import VerificationRecipe
    r = VerificationRecipe(
        named_baselines=["PQ", "OPQ"],
        benchmark_dataset="BEIR-fiqa",
        metric_primary="recall@10",
        metric_secondary=["wall_time_s"],
        min_meaningful_effect_size="+0.02 recall@10",
        expected_runtime="30 min on a single CPU",
        likely_failure_mode="codebook size dominates the bit budget at 32b",
        stop_rule="abort if any seed's recall@10 < 0.5 at 64b",
    )
    assert r.named_baselines == ["PQ", "OPQ"]


def test_verification_recipe_rejects_empty_baselines():
    from pydantic import ValidationError
    from pipeline.schemas import VerificationRecipe
    with pytest.raises(ValidationError):
        VerificationRecipe(
            named_baselines=[],
            benchmark_dataset="x",
            metric_primary="x",
            min_meaningful_effect_size="x",
            expected_runtime="x",
            likely_failure_mode="x",
            stop_rule="x",
        )


def test_generate_recipe_with_stubbed_codex(tmp_path, monkeypatch):
    from pipeline import llm_io, recipe

    canned = json.dumps({
        "named_baselines": ["PQ", "RaBitQ"],
        "benchmark_dataset": "BEIR-nfcorpus",
        "metric_primary": "recall@10",
        "metric_secondary": ["wall_time_s"],
        "min_meaningful_effect_size": "+0.02 recall@10",
        "expected_runtime": "60 min single CPU",
        "likely_failure_mode": "at 32b the calibration noise dominates",
        "stop_rule": "abort if seed-0 recall < 0.45 at 64b",
    })

    def fake_codex(prompt, reasoning):
        return True, canned, ""

    monkeypatch.setattr(llm_io, "_invoke_codex", fake_codex)

    composition = {
        "id": "C1",
        "synthesis_claim": "PCA-banded PQ beats uniform PQ at 64 bits",
        "papers": [
            {"paper_id": "P1", "role": "PQ primer"},
            {"paper_id": "P2", "role": "PCA decomposition"},
        ],
        "verification": {
            "L3": {"verdict": "weak", "verdict_reason": "needs more evidence"},
        },
    }
    state = {"problem": "compress embeddings", "corpus_size": 5}

    result = recipe.generate_recipe(composition, state, run_dir=tmp_path)
    assert result["named_baselines"] == ["PQ", "RaBitQ"]
    assert result["metric_primary"] == "recall@10"
    assert (tmp_path / "llm_calls" / "recipe_C1").is_dir()


def test_generate_recipe_returns_error_on_codex_failure(tmp_path, monkeypatch):
    from pipeline import llm_io, recipe

    def fake_codex(prompt, reasoning):
        return False, "", "subprocess died"

    monkeypatch.setattr(llm_io, "_invoke_codex", fake_codex)

    composition = {"id": "C1", "synthesis_claim": "x", "papers": [],
                   "verification": {"L3": {"verdict": "weak", "verdict_reason": ""}}}
    state = {"problem": "x"}
    result = recipe.generate_recipe(composition, state, run_dir=tmp_path)
    assert "error" in result
