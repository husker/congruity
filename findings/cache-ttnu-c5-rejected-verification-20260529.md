# C5 Time-to-Next-Use cache policy — rejected-composition verification

**run_id:** 20260521-023514 (composition C5)
**experiment:** `experiments/cache_replacement/` (`time_to_next_use.py`)
**date:** 2026-05-29
**outcome:** **clean-negative**

## Why this composition

This is the first entry in a **gate-precision** series. The calibration
meta-analysis (`findings/outcome-calibration-2026-05-29.md`) found that the
L1–L4 cascade has never been tested against outcomes, because only *accepted*
("actionable") compositions were ever verified — no `L3=invalid` composition had
been built. To learn whether the gate's rejections are correct or whether L3
over-fires, we verify rejected compositions and compare their outcome rate to the
accepted set.

C5 from run `20260521-023514` (cache replacement) was rejected with **L3=invalid**
but flagged **purely_testable**. Its accepted sibling C2 (Succ-SIEVE) from the
same run was verified clean-negative, so the domain is held constant.

## The composition (verbatim claim)

> "P8's constrained temporal scheduling + P45's next-event prediction + P21's
> high-rate trigger filtering suggests a **time-to-next-use eviction scheduler**:
> predict each cached object's next request window, then evict the object whose
> scheduled future contribution is weakest under capacity. Approximates
> Belady-style replacement using cheap predictors and decayed next-use estimates
> rather than neural inference."

**L3 verdict (invalid):** "collapses into a well-known idea: approximate Belady by
predicting next reuse… not novel relative to Belady approximations, reuse-distance
prediction, Hawkeye-style policies… the cited papers do not supply the missing
cache-specific machinery: **an online, cheap, accurate next-use predictor under
tiny metadata budgets.**"

## Implementation (faithful to the stated design)

`TimeToNextUse` in `experiments/cache_replacement/time_to_next_use.py`:
- Virtual clock; per-key EWMA of inter-arrival gaps (the reuse-distance estimate),
  kept in a **bounded LRU `gap_table`** (fixed metadata budget) — exactly the
  "tiny metadata budget" L3 said was missing.
- Predicted next-use = last_access + EWMA gap; first-seen keys use the running
  global-mean gap.
- Eviction = resident key with the **largest** predicted next-use (farthest
  future), via a lazy max-heap (O(log n) amortized).
- No neural inference, no offline planning — the cheapest honest realization.

## Results (hit rate; baselines from the existing harness)

| trace | cap frac | LRU | SIEVE | ARC | S3FIFO | **TTNU** |
|---|---|---|---|---|---|---|
| zipf_s099 | 0.01 | 0.397 | 0.511 | 0.502 | 0.499 | **0.290** |
| zipf_s099 | 0.05 | 0.581 | 0.657 | 0.652 | 0.648 | **0.412** |
| zipf_s080 | 0.05 | 0.340 | 0.429 | 0.431 | 0.427 | **0.271** |
| scan_hot | 0.05 | 0.514 | 0.609 | 0.606 | 0.607 | **0.310** |
| paired_zipf | 0.05 | 0.541 | 0.630 | 0.620 | 0.619 | **0.357** |
| loop_4x | 0.10 | 0.000 | 0.000 | 0.000 | 0.000 | **0.095** |

TTNU loses to every baseline (including plain LRU/FIFO) on all realistic
web/CDN regimes. Its only edge is `loop_4x`, an adversarial scan where the entire
LRU family scores exactly 0; even there TTNU is far below the cache-size ceiling.

**Knob sweep (robustness check):** swept `alpha ∈ {0.2,0.5,0.8}` ×
`metadata_budget ∈ {2048,8192,32768}` on the worst regimes. Best TTNU still loses
to SIEVE by 10–20 points (e.g. zipf_s099/0.05: best TTNU 0.470 vs SIEVE 0.657).
The result is not a tuning artifact.

## Interpretation

Clean-negative. The cheap EWMA next-use estimate is too noisy to rank residents
well at small cache ratios — popular and one-shot objects get mis-ranked, and the
global-mean prior for first-seen keys is poor. This is **exactly L3's substantive
objection**: the value lives in "an online, cheap, accurate next-use predictor,"
and a cheap predictor is not accurate enough. (L3's *headline* reason was
"not novel," but the operative technical claim — cheap predictor insufficient —
is what the measurement vindicates.)

## Gate-precision tally (running)

| composition | gate verdict | empirical outcome | gate correct? |
|---|---|---|---|
| C5 TTNU (this) | rejected (L3=invalid) | clean-negative | ✅ yes |

First rejected composition ever verified → it indeed fails. One data point; the
series needs more rejected samples (e.g. embed-quant C2, short-string C3) before
estimating gate precision. But this is the opposite of over-firing: here L3's
rejection was directionally correct.
