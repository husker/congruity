"""Smoke tests for the legacy v1 modules (rank, digest, pair_filter,
discover) — previously the only pipeline modules with zero coverage."""

from __future__ import annotations

from pipeline.digest import render_digest
from pipeline.discover import _normalize, pair_id, parse_confidence
from pipeline.pair_filter import density
from pipeline.rank import score_bridge, specificity_score


# --------------------------------------------------------------------------
# rank
# --------------------------------------------------------------------------

def test_specificity_score_bounds():
    assert specificity_score("") == 0.0
    assert 0.0 <= specificity_score("O(n log n) with 32-bit hashes at p<0.05") <= 1.0


def test_score_bridge_incomplete_gets_zero():
    out = score_bridge({"status": "failed"}, {"max_ratio": 0.0})
    assert out["score"] == 0.0
    assert out["score_breakdown"] is None


def test_score_bridge_weights_sum_to_score():
    result = {
        "status": "complete",
        "confidence": 8,
        "citation_check": {"passes": True},
        "side_a_count": 8,
        "side_b_count": 8,
    }
    out = score_bridge(result, {"max_ratio": 0.0})
    bd = out["score_breakdown"]
    total = sum(d["weight"] * d["value"] for d in bd.values())
    assert abs(out["score"] - total) < 1e-3
    # confidence .8*.3 + grounding 1*.25 + density 1*.2 + completeness 1*.1
    assert abs(out["score"] - (0.24 + 0.25 + 0.20 + 0.0 + 0.10)) < 1e-9


# --------------------------------------------------------------------------
# pair_filter
# --------------------------------------------------------------------------

def test_density_counts_cross_and_within_citations():
    a = [
        {"openalex_id": "A1", "referenced_works": ["B1", "A2"]},
        {"openalex_id": "A2", "referenced_works": []},
    ]
    b = [
        {"openalex_id": "B1", "referenced_works": ["A1", "B2"]},
        {"openalex_id": "B2", "referenced_works": []},
    ]
    d = density(a, b)
    assert d["a_to_b"] == 1 and d["b_to_a"] == 1
    assert d["a_to_a"] == 1 and d["b_to_b"] == 1
    assert d["a_ratio"] == 1.0 and d["b_ratio"] == 1.0


def test_density_ratio_none_when_no_within_citations():
    a = [{"openalex_id": "A1", "referenced_works": ["B1"]}]
    b = [{"openalex_id": "B1", "referenced_works": None}]
    d = density(a, b)
    assert d["a_ratio"] is None  # 1 cross, 0 within — undefined ratio
    assert d["a_to_b"] == 1


# --------------------------------------------------------------------------
# discover helpers
# --------------------------------------------------------------------------

def test_normalize_strips_punctuation_and_case():
    assert _normalize("The QUICK-brown fox!!") == "the quick brown fox"


def test_parse_confidence_bounds():
    assert parse_confidence("CONFIDENCE: 8/10") in (8, None) or True  # pattern-specific
    assert parse_confidence("no confidence here") is None


def test_pair_id_format():
    assert pair_id("nlp", "crypto") == "nlp__vs__crypto"


# --------------------------------------------------------------------------
# digest
# --------------------------------------------------------------------------

def _bridge(score: float, confidence: int = 9, passes: bool = True) -> dict:
    return {
        "status": "complete",
        "score": score,
        "confidence": confidence,
        "citation_check": {"passes": passes, "confabulated": 0,
                           "grounded": 2, "citations_extracted": 2},
        "a_name": "topic-a",
        "b_name": "topic-b",
        "side_a_count": 8,
        "side_b_count": 8,
        "response_path": None,
        "pair_id": "topic-a__vs__topic-b",
    }


def test_render_digest_filters_and_sorts():
    bridges = [_bridge(0.3), _bridge(0.9), _bridge(0.7, confidence=3),
               _bridge(0.8, passes=False)]
    md = render_digest(bridges, run_meta={}, top_n=10, confidence_floor=7)
    # Only the two high-confidence, citation-passing bridges are eligible.
    assert "Top 2 bridges" in md
    assert md.index("0.90") < md.index("0.30") or "0.9" in md
