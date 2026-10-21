"""Tests for the prioritize-don't-veto verification ranking.

Encodes the gate-precision finding: L3=invalid / fundamental objections must NOT
veto a composition out of the recommended-to-verify set. Confidence (the only
outcome-predictive signal) dominates the priority score.
"""

from __future__ import annotations

from pipeline.compose import (
    DEFAULT_RECIPE_TOP_K,
    _assign_verify_priority,
    _priority_score,
)


def _comp(cid, *, confidence, l3_verdict, purely_testable, dims=2,
          l1=True, l2=True, fundamental=0):
    return {
        "id": cid,
        "confidence": confidence,
        "covered_dimensions": [f"d{i}" for i in range(dims)],
        "verification": {
            "L1": {"passed": l1},
            "L2": {"passed": l2},
            "L3": {"verdict": l3_verdict},
        },
        "verification_summary": {
            "purely_testable": purely_testable,
            "objections_fundamental": fundamental,
        },
    }


def test_priority_does_not_veto_invalid():
    """A high-confidence L3=invalid comp must outrank a low-confidence weak one.

    This is the C2/C3 lesson: L3=invalid is not a 'won't work' signal.
    """
    invalid_strong = _comp("C2", confidence=9, l3_verdict="invalid",
                           purely_testable=False, fundamental=1)
    weak_weak = _comp("C9", confidence=3, l3_verdict="weak", purely_testable=True)
    assert _priority_score(invalid_strong) > _priority_score(weak_weak)


def test_confidence_dominates():
    hi = _comp("A", confidence=10, l3_verdict="invalid", purely_testable=False)
    lo = _comp("B", confidence=2, l3_verdict="weak", purely_testable=True)
    assert _priority_score(hi) > _priority_score(lo)


def test_assign_marks_exactly_top_k_and_scores_all():
    comps = [
        _comp("A", confidence=9, l3_verdict="invalid", purely_testable=False),
        _comp("B", confidence=8, l3_verdict="weak", purely_testable=True),
        _comp("C", confidence=4, l3_verdict="weak", purely_testable=True),
        _comp("D", confidence=1, l3_verdict="invalid", purely_testable=False),
    ]
    _assign_verify_priority(comps, top_k=2)
    recs = [c for c in comps
            if c["verification"]["reconciled"]["verify_recommended"]]
    assert {c["id"] for c in recs} == {"A", "B"}
    # Every composition gets a numeric priority score recorded.
    assert all(
        isinstance(c["verification"]["reconciled"]["priority_score"], float)
        for c in comps
    )


def test_invalid_winner_recommended_over_low_conf_weak():
    """C2-like: invalid + fundamental objection but high confidence beats a
    low-confidence weak+testable composition for a scarce recipe slot."""
    comps = [
        _comp("C2", confidence=9, l3_verdict="invalid", purely_testable=False,
              fundamental=1),
        _comp("weak1", confidence=3, l3_verdict="weak", purely_testable=True),
        _comp("weak2", confidence=2, l3_verdict="weak", purely_testable=True),
    ]
    _assign_verify_priority(comps, top_k=1)
    recommended = next(c for c in comps
                       if c["verification"]["reconciled"]["verify_recommended"])
    assert recommended["id"] == "C2"


def test_assign_handles_missing_reconciled_and_top_k_zero():
    comps = [_comp("A", confidence=5, l3_verdict="weak", purely_testable=True)]
    _assign_verify_priority(comps, top_k=0)
    assert comps[0]["verification"]["reconciled"]["verify_recommended"] is False
    assert "priority_score" in comps[0]["verification"]["reconciled"]


def test_default_top_k_is_positive():
    assert DEFAULT_RECIPE_TOP_K >= 1
