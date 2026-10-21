"""Pydantic v2 models for every structured LLM output in the compose pipeline.

Field names and types track the existing prompt contracts in pipeline/compose.py.
Models reject unknown fields by default (extra='forbid') so silent schema drift
between prompt and parser is impossible.

L3 verdict literals verified against compose.py prompt template (line 690):
    "sound" | "weak" | "invalid"  — lowercase, as emitted by the LLM.
    (The plan spec listed "PASS"/"WEAK"/"INVALID" which do NOT match actual usage.)

L4 verdict literals used for the summary net_testability_score string
    "PURELY_TESTABLE" | "PARTIALLY_TESTABLE" | "NOT_TESTABLE"
    These are for parse_or_repair (Task 6) / future schema wiring (Task 8).
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator


class _StrictBase(BaseModel):
    model_config = ConfigDict(extra="forbid")


# Stage 2: dimensions
class Dimension(_StrictBase):
    name: str = Field(min_length=1)
    description: str = Field(min_length=1)
    example_question: str = Field(min_length=1)


class DimensionsOutput(_StrictBase):
    # Empty dimensions is a silent-failure mode (same as scoring): if the
    # model returns `{"dimensions": []}` the run continues with no
    # dimensions and downstream stages collapse. Forcing min_length=1
    # turns it into a validation error parse_or_repair retries.
    dimensions: list[Dimension] = Field(min_length=1)


# Stage 3: scores
class DimensionScore(_StrictBase):
    score: int = Field(ge=0, le=5)
    contribution: str = ""


class PaperScore(_StrictBase):
    paper_id: str = Field(min_length=1)
    title_short: str = ""
    dimensions: dict[str, DimensionScore]


class ScoresOutput(_StrictBase):
    # Empty scores is a known failure mode: the model can spend its
    # reasoning budget on a 95-paper scoring task and bail with
    # `{"scores": []}`. Forcing min_length=1 turns that silent give-up
    # into a validation error that parse_or_repair retries.
    scores: list[PaperScore] = Field(min_length=1)


# Stage 4: compositions
#
# Schema mirrors COMPOSE_PROMPT_TEMPLATE in compose.py exactly. The prompt
# asks the LLM for richer evidence (supporting_quote, supporting_dimensions,
# compatibility_argument, covered_dimensions, open_questions, confidence,
# confidence_reason) than the earlier schema accepted. With `extra='forbid'`
# the strict-narrow schema turned every valid prompt-following response into
# a validation error, the generic repair coerced it back to {id,
# synthesis_claim, papers[].quote}, and L2 source grounding then failed
# every composition because the quote it looks for (`supporting_quote`) had
# been stripped. All fields default-soft so historical state files without
# the rich fields still validate cleanly during reprocess.
class CompositionPaper(_StrictBase):
    paper_id: str
    role: str
    supporting_quote: str = ""
    supporting_dimensions: list[str] = Field(default_factory=list)

    @model_validator(mode="before")
    @classmethod
    def _alias_legacy_quote(cls, data):
        # Back-compat: accept the legacy `quote` field name as an alias for
        # `supporting_quote`. Old persisted compositions and any LLM output
        # that still emits the shorter name keep loading; `extra='forbid'`
        # would otherwise reject `quote` as unknown.
        if isinstance(data, dict) and "quote" in data:
            data = dict(data)
            legacy = data.pop("quote")
            data.setdefault("supporting_quote", legacy)
        return data


class MechanismMap(_StrictBase):
    """What actually transfers, and why it should beat the destination SOTA.

    Required by the mechanism-first composition prompt (roadmap #4): both
    historical transfer wins were structure transplants, not topic mashups.
    Optional on Composition so older runs and weaker outputs still parse.
    """

    transferring_structure: str = ""
    sota_failure_mode_addressed: str = ""


class Composition(_StrictBase):
    id: str
    synthesis_claim: str
    papers: list[CompositionPaper]
    compatibility_argument: str = ""
    mechanism_map: MechanismMap | None = None
    covered_dimensions: list[str] = Field(default_factory=list)
    open_questions: list[str] = Field(default_factory=list)
    # Prompt asks for a 1-10 integer but the model occasionally emits 0 or
    # omits the field on weak candidates; keep the type loose to preserve
    # whatever the model reports and let downstream surface the raw value.
    confidence: int | None = None
    confidence_reason: str = ""


class CompositionsOutput(_StrictBase):
    # Same min_length protection as DimensionsOutput / ScoresOutput against
    # the silent-failure mode where the model returns an empty array.
    compositions: list[Composition] = Field(min_length=1)


# Stage 5 verification: L1 internal consistency
# Verified against compose.py CONSISTENCY_PROMPT_TEMPLATE (lines 512-519):
# issues is a list of dicts {severity, detail, paper_refs}, not strings.
class L1Item(BaseModel):
    model_config = ConfigDict(extra="allow")
    consistent: bool
    issues: list[dict] = Field(default_factory=list)


# Stage 5 verification: L3 adversarial cross-check
# Verified against compose.py ADVERSARIAL_PROMPT_TEMPLATE (line 690) and
# verify_adversarial return shape (line 727): output has lowercase verdict
# plus a verdict_reason paragraph. Real LLM outputs also include a
# falsification_attempts list; we allow extras to accept them.
class L3Verdict(BaseModel):
    model_config = ConfigDict(extra="allow")
    verdict: Literal["sound", "weak", "invalid"]
    verdict_reason: str = ""


# Stage 5 verification: L4 testability classification
# Verified against compose.py PREDICTIONS_PROMPT_TEMPLATE (lines 584-608) and
# extract_testable_predictions return shape (line 652-657). L4 has no
# top-level "verdict" field — the verdict-equivalent lives in
# summary.net_testability_score (string literal OR numeric ratio).
class L4Summary(BaseModel):
    model_config = ConfigDict(extra="allow")
    total_objections: int = 0
    fundamental_count: int = 0
    empirical_count: int = 0
    # Per the prompt template, this is "PURELY_TESTABLE" if fundamental_count == 0,
    # otherwise a numeric ratio (empirical / total). Accept both shapes.
    net_testability_score: str | float | None = None
    verdict_for_unsolved_problem: str = ""


class L4Verdict(BaseModel):
    model_config = ConfigDict(extra="allow")
    level: int = 4
    level_name: str = "predictions_and_tests"
    objection_classifications: list[dict] = Field(default_factory=list)
    summary: L4Summary = Field(default_factory=L4Summary)


# Stage 6: verification recipe for actionable compositions
# Generated only for WEAK + PURELY_TESTABLE compositions; gives the user a
# concrete experiment plan instead of a "this might be worth testing" note.
class VerificationRecipe(BaseModel):
    model_config = ConfigDict(extra="allow")
    # The destination field's own best-known method — the bar that defines a real
    # cross-disciplinary transfer win. Should also appear in named_baselines.
    home_field_sota: str = ""
    named_baselines: list[str] = Field(min_length=1)
    benchmark_dataset: str = Field(min_length=1)
    metric_primary: str = Field(min_length=1)
    metric_secondary: list[str] = Field(default_factory=list)
    min_meaningful_effect_size: str = Field(min_length=1)
    expected_runtime: str = Field(min_length=1)
    likely_failure_mode: str = Field(min_length=1)
    stop_rule: str = Field(min_length=1)


# Stage 7: post-L3 novelty check
class NoveltyHit(BaseModel):
    model_config = ConfigDict(extra="allow")
    source: str
    title: str
    similarity: float = Field(ge=0.0, le=1.0)
    url: str = ""


class NoveltyReport(BaseModel):
    model_config = ConfigDict(extra="allow")
    classification: Literal["known", "adjacent-known", "not-found", "unverified"]
    key_phrases: list[str] = Field(default_factory=list)
    hits: list[NoveltyHit] = Field(default_factory=list)
    cocite_density: float = Field(ge=0.0, default=0.0)
    queries_issued: list[str] = Field(default_factory=list)
    note: str = ""
