"""Pin the L3 ADVERSARIAL_PROMPT_TEMPLATE contract.

The original prompt told the model to be "ruthless ... default skeptical"
without ever defining what `invalid` vs `weak` meant. Audited runs showed
38% of compositions ending up as L3=invalid + L4=PURELY_TESTABLE — i.e.
L3 was calling things "fundamentally refuted" when L4 (correctly)
classified every objection as empirical. This file pins the calibration
that closes that gap: L3 verdict definitions must mirror L4's
FUNDAMENTAL/EMPIRICAL axis so the two stages don't drift independently.
"""

from __future__ import annotations


def test_l3_prompt_defines_each_verdict_explicitly():
    """Each verdict literal must appear in the prompt body with a
    definition the model can act on — not just in the OUTPUT block."""
    from pipeline.compose import ADVERSARIAL_PROMPT_TEMPLATE
    body = ADVERSARIAL_PROMPT_TEMPLATE
    assert '"sound"' in body
    assert '"weak"' in body
    assert '"invalid"' in body
    # Definitions should appear above the OUTPUT block.
    above_output = body.split("OUTPUT", 1)[0]
    assert '"sound"' in above_output
    assert '"weak"' in above_output
    assert '"invalid"' in above_output


def test_l3_prompt_anchors_invalid_in_logical_impossibility():
    """`invalid` should be reserved for cases where no experiment can
    rescue the composition. The keywords pin the calibration so any
    future prompt edit that softens this bar trips the test."""
    from pipeline.compose import ADVERSARIAL_PROMPT_TEMPLATE
    body = ADVERSARIAL_PROMPT_TEMPLATE.lower()
    assert "logically or physically impossible" in body
    assert "no experiment can rescue" in body
    # Explicit not-invalid examples.
    assert "not novel" in body
    assert "weak" in body


def test_l3_prompt_includes_calibration_question():
    """The 'would a definitive experiment change my mind' guardrail is
    what makes L3's verdict match L4's FUNDAMENTAL/EMPIRICAL split."""
    from pipeline.compose import ADVERSARIAL_PROMPT_TEMPLATE
    body = ADVERSARIAL_PROMPT_TEMPLATE.lower()
    assert "would a definitive experiment change my mind" in body


def test_l3_prompt_explicitly_cross_references_l4_axis():
    """The whole point of the calibration is that L3 verdicts line up
    with L4 classifications. The prompt should say so out loud."""
    from pipeline.compose import ADVERSARIAL_PROMPT_TEMPLATE
    body = ADVERSARIAL_PROMPT_TEMPLATE
    assert "FUNDAMENTAL" in body
    assert "EMPIRICAL" in body
