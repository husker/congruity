"""Tests for L3/L4 verdict reconciliation."""

from __future__ import annotations


def _l3(verdict: str, reason: str = "x") -> dict:
    return {"verdict": verdict, "verdict_reason": reason,
            "level": 3, "level_name": "adversarial_check", "passed": False}


def _l4(score, fundamental: int = 0, empirical: int = 7) -> dict:
    return {
        "level": 4,
        "level_name": "predictions_and_tests",
        "summary": {
            "total_objections": fundamental + empirical,
            "fundamental_count": fundamental,
            "empirical_count": empirical,
            "net_testability_score": score,
        },
    }


def test_actionable_when_weak_and_purely_testable():
    from pipeline.compose import _reconcile_verdict
    r = _reconcile_verdict(_l3("weak"), _l4("PURELY_TESTABLE"))
    assert r["status"] == "actionable"
    assert r["l3_l4_consistent"] is True


def test_invalid_when_l3_invalid_even_if_l4_says_testable():
    """The TSP C3 case: L3 found a fatal flaw, L4 marked all objections
    empirical. The reconciled status is invalid (L3 wins) and the
    consistency flag is False so the dashboard surfaces it."""
    from pipeline.compose import _reconcile_verdict
    r = _reconcile_verdict(_l3("invalid"), _l4("PURELY_TESTABLE", fundamental=0))
    assert r["status"] == "invalid"
    assert r["l3_l4_consistent"] is False
    assert "L4 disagreed" in r["reason"]


def test_invalid_consistent_when_l4_also_has_fundamentals():
    from pipeline.compose import _reconcile_verdict
    r = _reconcile_verdict(_l3("invalid"), _l4("PARTIALLY_TESTABLE", fundamental=2))
    assert r["status"] == "invalid"
    assert r["l3_l4_consistent"] is True


def test_sound_consistent_when_l4_has_zero_fundamentals():
    from pipeline.compose import _reconcile_verdict
    r = _reconcile_verdict(_l3("sound"), _l4("PURELY_TESTABLE", fundamental=0))
    assert r["status"] == "sound"
    assert r["l3_l4_consistent"] is True


def test_sound_inconsistent_when_l4_finds_fundamentals_l3_missed():
    from pipeline.compose import _reconcile_verdict
    r = _reconcile_verdict(_l3("sound"), _l4("PARTIALLY_TESTABLE", fundamental=2))
    assert r["status"] == "sound"
    assert r["l3_l4_consistent"] is False
    assert "L4 found 2 fundamental" in r["reason"]


def test_weak_untestable_when_l4_finds_fundamentals():
    from pipeline.compose import _reconcile_verdict
    r = _reconcile_verdict(_l3("weak"), _l4("PARTIALLY_TESTABLE", fundamental=1))
    assert r["status"] == "weak_untestable"
    assert r["l3_l4_consistent"] is True


def test_weak_untestable_numeric_score():
    """L4 can emit a numeric ratio instead of PURELY_TESTABLE; that's not
    actionable for recipe gating."""
    from pipeline.compose import _reconcile_verdict
    r = _reconcile_verdict(_l3("weak"), _l4(0.5, fundamental=1))
    assert r["status"] == "weak_untestable"


def test_incomplete_when_l3_missing():
    from pipeline.compose import _reconcile_verdict
    r = _reconcile_verdict(None, _l4("PURELY_TESTABLE"))
    assert r["status"] == "incomplete"
    assert r["l3_l4_consistent"] is False


def test_incomplete_when_l3_has_error():
    from pipeline.compose import _reconcile_verdict
    r = _reconcile_verdict({"error": "codex_failed", "verdict": ""}, _l4("PURELY_TESTABLE"))
    assert r["status"] == "incomplete"


def test_l3_invalid_l4_missing_still_resolves_to_invalid():
    """L3's invalid verdict is conclusive even if L4 didn't run."""
    from pipeline.compose import _reconcile_verdict
    r = _reconcile_verdict(_l3("invalid"), {})
    assert r["status"] == "invalid"
    assert r["l3_l4_consistent"] is True


def test_l4_missing_under_non_invalid_l3_is_incomplete():
    from pipeline.compose import _reconcile_verdict
    r = _reconcile_verdict(_l3("weak"), {})
    assert r["status"] == "incomplete"
