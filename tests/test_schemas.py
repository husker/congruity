"""Tests for Pydantic models covering LLM stage outputs."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from pipeline.schemas import (
    Composition,
    CompositionPaper,
    Dimension,
    DimensionsOutput,
    L1Item,
    L3Verdict,
    L4Verdict,
    PaperScore,
)


def test_dimension_happy_path():
    d = Dimension(
        name="latency",
        description="end-to-end inference latency",
        example_question="how fast?",
    )
    assert d.name == "latency"


def test_dimension_missing_required_field_raises():
    with pytest.raises(ValidationError):
        Dimension(name="latency")  # missing description, example_question


def test_dimensions_output_wraps_list():
    out = DimensionsOutput(dimensions=[
        Dimension(name="a", description="b", example_question="c"),
    ])
    assert len(out.dimensions) == 1


def test_paper_score_happy_path():
    s = PaperScore(
        paper_id="P1",
        title_short="An attention mechanism",
        dimensions={"latency": {"score": 3, "contribution": "subquadratic"}},
    )
    assert s.paper_id == "P1"
    assert s.dimensions["latency"].score == 3


def test_scores_output_rejects_empty_list():
    """ScoresOutput.scores requires min_length=1: an empty list is a known
    silent-failure mode where the model burns reasoning budget on a long
    corpus and bails with `{"scores": []}`."""
    from pydantic import ValidationError
    from pipeline.schemas import ScoresOutput
    with pytest.raises(ValidationError):
        ScoresOutput(scores=[])


def test_paper_score_rejects_score_above_5():
    with pytest.raises(ValidationError):
        PaperScore(
            paper_id="P1",
            title_short="x",
            dimensions={"latency": {"score": 7, "contribution": "x"}},
        )


def test_composition_happy_path():
    c = Composition(
        id="C1",
        synthesis_claim="A+B beats either alone",
        papers=[
            CompositionPaper(paper_id="P1", role="provides X", supporting_quote="..."),
            CompositionPaper(paper_id="P2", role="provides Y", supporting_quote="..."),
        ],
    )
    assert c.id == "C1"
    assert len(c.papers) == 2


def test_composition_accepts_full_prompt_shape():
    """Regression for the schema/prompt mismatch that silently stripped
    supporting_quote, compatibility_argument, confidence, etc. during
    repair and broke L2 source grounding."""
    from pipeline.schemas import CompositionsOutput
    payload = {
        "compositions": [
            {
                "id": "C1",
                "synthesis_claim": "Paper A's X + Paper B's Y → new capability",
                "papers": [
                    {
                        "paper_id": "P3",
                        "role": "provides X",
                        "supporting_dimensions": ["latency", "memory"],
                        "supporting_quote": "we achieve 2.4x speedup on benchmark Z",
                    },
                    {
                        "paper_id": "P7",
                        "role": "provides Y",
                        "supporting_dimensions": ["accuracy"],
                        "supporting_quote": "the model matches state-of-the-art on Z",
                    },
                ],
                "compatibility_argument": "Both operate over dense matrices at "
                                          "comparable scale; speedups compose.",
                "covered_dimensions": ["latency", "memory", "accuracy"],
                "open_questions": ["what is the memory ceiling?"],
                "confidence": 7,
                "confidence_reason": "concrete numbers reported in both papers",
            }
        ]
    }
    out = CompositionsOutput.model_validate(payload)
    c = out.compositions[0]
    assert c.synthesis_claim.startswith("Paper A")
    assert c.compatibility_argument.startswith("Both operate")
    assert c.covered_dimensions == ["latency", "memory", "accuracy"]
    assert c.open_questions == ["what is the memory ceiling?"]
    assert c.confidence == 7
    assert c.confidence_reason.startswith("concrete")
    assert c.papers[0].supporting_quote.startswith("we achieve")
    assert c.papers[0].supporting_dimensions == ["latency", "memory"]


def test_composition_paper_accepts_legacy_quote_alias():
    """Old persisted state files use `quote`; the alias normalizes them
    to the canonical `supporting_quote` so reprocess still works."""
    p = CompositionPaper(paper_id="P1", role="x", quote="hello world")
    assert p.supporting_quote == "hello world"
    # Validation from dict (model_validate) takes the same code path.
    p2 = CompositionPaper.model_validate({
        "paper_id": "P2", "role": "y", "quote": "verbatim sentence",
    })
    assert p2.supporting_quote == "verbatim sentence"


def test_composition_accepts_missing_rich_fields():
    """Historical compositions (pre-schema-alignment) lack the rich
    fields — they should still validate without explicit values."""
    c = Composition(
        id="C9",
        synthesis_claim="legacy minimal claim",
        papers=[CompositionPaper(paper_id="P1", role="x")],
    )
    assert c.compatibility_argument == ""
    assert c.covered_dimensions == []
    assert c.open_questions == []
    assert c.confidence is None


def test_l3_verdict_happy_path():
    v = L3Verdict(verdict="weak", verdict_reason="not novel; similar to prior work")
    assert v.verdict == "weak"
    assert "novel" in v.verdict_reason


def test_l3_verdict_rejects_unknown_verdict():
    with pytest.raises(ValidationError):
        L3Verdict(verdict="MAYBE", verdict_reason="x")


def test_l3_verdict_accepts_extra_fields_from_real_llm_output():
    # Real LLM outputs include falsification_attempts and other fields; the
    # schema uses extra='allow' so they pass through without raising.
    v = L3Verdict(
        verdict="invalid",
        verdict_reason="the proposed scheme conflates two unrelated mechanisms",
        falsification_attempts=[
            {"angle": "rate-distortion", "argument": "...", "would_invalidate": True},
        ],
    )
    assert v.verdict == "invalid"


def test_l4_verdict_happy_path_with_summary():
    v = L4Verdict(
        level=4,
        level_name="predictions_and_tests",
        objection_classifications=[],
        summary={
            "total_objections": 3,
            "fundamental_count": 0,
            "empirical_count": 3,
            "net_testability_score": "PURELY_TESTABLE",
            "verdict_for_unsolved_problem": "worth a bounded computational test",
        },
    )
    assert v.summary.net_testability_score == "PURELY_TESTABLE"
    assert v.summary.total_objections == 3


def test_l4_verdict_accepts_numeric_testability_score():
    # The prompt template emits a float ratio when fundamental_count > 0.
    v = L4Verdict(
        summary={
            "total_objections": 4,
            "fundamental_count": 1,
            "empirical_count": 3,
            "net_testability_score": 0.75,
        },
    )
    assert v.summary.net_testability_score == 0.75


def test_l4_verdict_minimal_with_defaults():
    # Even an empty L4 (no objections from L3) should validate.
    v = L4Verdict()
    assert v.level == 4
    assert v.summary.total_objections == 0


def test_l1_item_happy_path():
    item = L1Item(consistent=True, issues=[])
    assert item.consistent
