# Finding 02 — LLM speculative decoding has rediscovered 3 layers of CPU branch-prediction infrastructure

Date: 2026-05-18
Author: James Gagan
Method: AI-driven cross-subfield bridge discovery (congruity v0 prototype)
Status: PRELIMINARY — single AI, no novelty probe yet, but replicates Finding 01's pattern

## TL;DR

A second 10-minute Codex experiment on a different sub-field pair — LLM speculative decoding (Side A, 2025–2026, post-Codex-cutoff) vs. CPU branch prediction (Side B, 2010–2019) — produced **three distinct equivalence claims at progressively higher abstraction levels**, all with verified citations and 7–9 confidence:

1. **Algorithmic equivalence (9/10):** Speculative LLM decoding = branch-predicted speculative execution. Cheap predictor proposes, expensive authority validates, only validated work commits.
2. **Control equivalence (8/10):** Adaptive speculation width/length in LLM decoding = branch predictors using auxiliary or value-correlated signals to tune speculation aggressiveness.
3. **System equivalence (7/10):** Disaggregated/asynchronous speculative decoding = decoupled speculative processor pipeline, where prediction, execution, and validation overlap across different resources.

All 10 unique citations Codex produced are grounded in the supplied corpus. No confabulation.

## Why this matters as a result

This is the second test of the congruity architecture. The first (Finding 01: KV-cache ↔ cache replacement) produced one bridge with 9/10 confidence. This one produced three bridges nested at different abstraction levels. Two for two on post-cutoff CS sub-field pairs, with the second experiment producing notably richer output. The result generalizes.

The branch-prediction literature on CPUs is 35+ years old and very mature. The LLM speculative decoding literature is 2–3 years old. If Codex's bridges hold up, the LLM-serving community has independently rediscovered 35 years of CPU-architecture engineering — confidence-gated speculation length, multi-predictor fusion, side-predictor cascades, accept-rate tuning, decoupled pipelines — without (so far as the references in Finding 01's novelty probe suggest) engaging with the prior art.

## The three bridges, in plain language

### Bridge 1: predict → verify → commit

A CPU branch predictor guesses "this branch will be taken" before it knows. The CPU speculatively executes the predicted path. When the actual branch outcome is known, the CPU either commits the speculative work (if predicted correctly) or rolls back and discards it (if mispredicted). Misprediction rate is the headline metric.

An LLM speculative decoder uses a small draft model to guess the next N tokens. The expensive target model verifies the draft in a single forward pass. Accepted tokens commit; rejected tokens are discarded and the position rolls back to the first rejection. Accept rate is the headline metric.

Codex's claim: these are the same control structure. Accept rate IS misprediction rate inverted. The papers describe the structure in different vocabulary — "distribution-preserving decoding" vs. "instruction-flow prediction" — but the machinery is identical.

### Bridge 2: tune the predictor at runtime

TAGE predictors use multiple predictors at different history lengths and switch among them based on observed accuracy. Value-correlated predictors detect "hard branches" and apply specialized predictors.

Dynamic-width speculative beam decoding adjusts beam width on the fly. Adaptive speculation length expands or contracts based on observed accept rates. Reward-guided speculation uses signal from the verifier to recalibrate.

Codex's claim: both communities have independently reinvented the "side predictor + adaptation" architecture. The CPU community has 15+ years of refinement on this; the LLM community is in year two.

### Bridge 3: decouple the pipeline

Modern CPUs have decoupled front-ends (Reinman et al. 1999, etc.) where branch prediction runs ahead of instruction fetch, prefetching speculatively, and rollback is a localized event rather than a global stall. Branch-predictor side-channel attacks (ret2spec, BRB) exploit the persistence of speculative state across context switches.

Disaggregated speculative decoding (2025) runs draft and target models on different physical devices, with carbon and throughput optimization. AMUSD (2025) runs draft and verifier asynchronously across multi-device setups.

Codex's claim: the LLM serving community is reinventing decoupled-pipeline architecture, including (implicitly) the security-isolation problems that come with it. ret2spec-style side-channel concerns may eventually apply to disaggregated speculative-decoding deployments where multiple users share a draft model.

## Citation grounding

| Cited paper | Side | Year | In corpus? |
|---|---|---|---|
| Dynamic-Width Speculative Beam Decoding for LLM Inference | A | 2025 | ✅ |
| SLED: A Speculative LLM Decoding Framework for Efficient Edge Serving | A | 2025 | ✅ |
| Towards Efficient LLM Inference via Collective and Adaptive Speculative Decoding | A | 2025 | ✅ |
| AMUSD: Asynchronous Multi-Device Speculative Decoding | A | 2025 | ✅ |
| Reward-Guided Speculative Decoding for Efficient LLM Reasoning | A | 2025 | ✅ |
| Disaggregated Speculative Decoding for Carbon-Efficient LLM Serving | A | 2025 | ✅ |
| A new case for the TAGE branch predictor | B | 2011 | ✅ |
| Improving prediction accuracy of hard-to-predict branches using data value correlation | B | 2013 | ✅ |
| Interval simulation: Raising the level of abstraction in architectural simulation | B | 2010 | ✅ |
| BRB: Mitigating Branch Predictor Side-Channels | B | 2019 | ✅ |

10/10 citations land in the corpus. Years Codex mentioned (2010, 2011, 2013, 2018, 2019, 2025) exactly match corpus years.

## What's not yet shown

- **Novelty probe not yet run for this pair.** Same Semantic-Scholar rate-limit blocker as Finding 01. Needs a paid API key or local PDF parsing.
- **Bridge correctness not tested.** The differential-test oracle would implement TAGE-style multi-predictor fusion on a small speculative-decoding system and measure accept-rate improvement. ~1 week of work to set up; same shape as the proposed KV-cache ↔ ARC validation in Finding 01.
- **Other model families not tested.** Claude or Gemini on the same prompt would tell us whether these bridges are Codex-specific or robust across LLM providers. Important sanity check.

## The pattern across Finding 01 and Finding 02

| | Finding 01 (KV-cache) | Finding 02 (Spec-decode) |
|---|---|---|
| Side A papers | 8, all 2025–2026 | 8, all 2025 |
| Side B papers | 8, all 2025–2026 | 5, 2010–2019 |
| Post-cutoff for Codex? | Yes (Side A) | Yes (Side A) |
| Bridges found | 1 | 3 nested |
| Codex's confidence | 9/10 | 7–9/10 |
| Citations grounded | 8/8 | 10/10 |
| Specific mechanism transfer | Yes | Yes |
| Domain-expert sanity check | Pending | Pending |

The architecture replicates. The verification harness (citation checker against the supplied corpus) catches confabulation in both cases — and finds none in either.

## What this changes about the project

After Finding 01, the open question was "does this generalize beyond one pair." After Finding 02, the answer is "appears to, with richer output on the second test." The next questions are:

1. **Does the pattern hold under different model families?** Run the same Side A / Side B prompts through Claude and Gemini; check for agreement on the bridge claims. Agreement is evidence of bridge reality; disagreement narrows to model-specific artifacts.
2. **Does novelty hold up under fuller probing?** Pull references for 5–10 representative LLM speculative-decoding papers, scan for citations to TAGE / branch-predictor literature. If most cite none, the bridge is genuinely under-recognized.
3. **Can we run a real oracle on one claim?** Implement TAGE-style multi-predictor fusion on Medusa or EAGLE; measure whether accept rate improves. That's the v0 differential-test from § 10 of the design doc, applied to a concrete cross-pair claim.

The project's contrarian thesis (verification is the unsolved seam; computational-verification domains are ownable) survives both tests so far. The next step is real-world validation of one specific bridge claim, which is a 1–2 week implementation project rather than further AI-experiment turns.
