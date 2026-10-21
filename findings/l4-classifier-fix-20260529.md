# L4 fundamental/empirical classifier — baseline-relative recalibration

**date:** 2026-05-29
**component:** `pipeline/compose.py` `PREDICTIONS_PROMPT_TEMPLATE` (L4)
**type:** pipeline fix (verified empirically with codex)

## Problem

The gate-precision series found L4 mis-flagging **C2** (the best composition in
the project) as `purely_testable=False` because of a single FUNDAMENTAL objection
— the pigeonhole bound, *"a compressor cannot shrink every arbitrary input."* That
is mathematically true, but it (a) applies equally to every baseline
(gzip/zstd/brotli all expand on random data) and (b) says nothing about beating
baselines on *typical* inputs. The old prompt defined FUNDAMENTAL as "logically or
physically impossible," which the pigeonhole objection satisfies literally — so L4
wasn't malfunctioning, its *criterion* was wrong.

## Fix

Redefine FUNDAMENTAL around the practical metric — *does the objection prevent the
composition from beating the relevant baselines on realistic inputs?* — with two
hard rules applied before anything can be FUNDAMENTAL:

1. **Baseline-relative:** if the same objection applies equally to the incumbent
   baselines, it cannot explain why this composition would lose → EMPIRICAL.
2. **Charitable narrowing:** if it only refutes an overbroad "universal/arbitrary"
   phrasing, narrow the claim to the realistic scope and re-judge.

Each objection now records `applies_to_baselines_too` and `narrowed_claim_survives`;
FUNDAMENTAL requires both false. The pigeonhole case is included as an explicit
negative example in the prompt.

## Verification (re-ran L4 via codex on saved runs)

**False-positive corrected — C2 (run 20260520-053938):**
old L4 `fundamental_count=1` (`purely_testable=False`) → new `fundamental_count=0`,
`empirical_count=10`, **`purely_testable=True`**. The pigeonhole objection is now
EMPIRICAL with `applies_to_baselines_too=True, narrowed_claim_survives=True`.

**True-positive preserved — matrix-multiplication (run 20260519-024943 C3):**
old `fundamental_count=6` → new `fundamental_count=3`. The 3 retained FUNDAMENTAL
objections are genuine baseline-relative impossibilities ("no mechanism to reduce
the matmul exponent ω"; "surrogate modeling, not faster multiplication";
"accelerates a workflow, not n×n matmul"), each `base_too=False, narrowed=False`.
Verdict stays "not worth prototyping as a candidate for faster matrix multiplication."

So the recalibrated classifier discriminates — it stops firing on universal-claim
impossibilities that hit every baseline, while still catching real,
baseline-relative refutations. Combined with the prioritize-not-veto ranking
(which already ignores both L3 verdict and `purely_testable`), the gate no longer
buries winners like C2.
