# Expert blind hit-rate study — can a blind rater predict which compositions survive?

**Date:** 2026-07-07. **The Design / Metrics / Interpretation / Scope sections
below were committed (git 8f6163c) BEFORE any rater was run or any output
inspected** — a genuine pre-registration; `## Results` was appended afterward
and changed none of the pre-registered plan.

**One-line verdict:** a blind expert-proxy rater gives highly self-consistent
"looks promising" judgments (Spearman +0.79, Fleiss κ +0.60 across 5 samples)
that are **uncorrelated** with empirical outcome (AUC≈chance 0.52, accuracy =
base rate) — the sharpest misses being a unanimous favorite that cleanly failed
and the engine's single best result dismissed 0/5. Strong evidence that promise
can't be read off the idea, which is what justifies verify-don't-predict. (The
5 raters are independent samples of ONE model, so the agreement is that model's
self-consistency, not independent-expert consensus.)

## The question

congruity's premise (README): *"The unsolved problem is calibrating which
hypotheses are worth pursuing — that's what the verification cycle is for."*
The verification cycle earns its keep only if promise is **hard to predict
from the composition alone** — if a rater can eyeball a composition and call
the winners, you don't need to run the experiments; you just ask the rater.

`findings/outcome-calibration-2026-05-29.md` already showed the *pipeline's own*
pre-registration signals don't predict outcome (L3/L4 constant across the
verified set; `confidence` only weakly tracks: ≥8 → 57% POS vs <8 → 38%). This
study adds the missing baseline: an **independent blind rater** (expert proxy).

## Design (pre-registered)

- **Sample:** the 24 verified compositions in `findings/INDEX.md` that join to a
  composition object and carry a recognized outcome (13 POS / 11 NEG; ~54% base
  rate). Excludes 3 snake compositions (unresolved outcome) and derived
  follow-ups (no pre-registration).
- **Ground-truth label (POS/NEG):** clean-positive / partial-positive /
  partial-replication → POS; clean-negative / partial-negative → NEG. (Same
  mapping as `outcome_calibration.py`.)
- **Prediction target — aligned to the label:** raters predict whether a
  composition will **survive empirical test as positive-or-better (beat generic
  baselines / not be cleanly falsified)** — NOT "beats home-field SOTA"
  (partial-positives beat only generic baselines, so a beats-SOTA question would
  score against a construct the label doesn't encode; beats-SOTA ground truth is
  n=2 and unestimable here).
- **Raters:** 5 independent Claude subagents (a different model family from the
  gpt-5.5 composer → genuinely independent). Each sees ONLY the composition
  content — anonymized `synthesis_claim`, constituent papers + roles,
  `compatibility_argument`, problem/domain — in a per-rater shuffled order,
  with NO access to the outcome, pipeline confidence, L1–L4 verdict, finding
  docs, or tools. Per composition each returns: `promise` (1–10),
  `positive_or_better` (bool), one-line reason.

## Metrics (pre-registered; primary vs supporting fixed in advance)

**Primary (robustly estimable at n=24):**
1. **Inter-rater agreement** — mean pairwise Spearman ρ on the 1–10 promise
   ratings across the 5 raters, + Fleiss' κ on the binary calls. High = raters
   share a coherent notion of "promising"; low = "promising" is itself noisy.
2. **Proxy-consensus vs pipeline confidence** — Spearman ρ between mean-proxy
   promise and the pipeline's `confidence` per composition. High = the blind
   proxy and the pipeline agree on what *looks* promising.

**Supporting (noisy at n=24 — reported as point estimate + bootstrap 95% CI,
never as the verdict):**
3. AUC of mean-proxy promise vs POS/NEG.
4. AUC of pipeline `confidence` vs POS/NEG (the existing baseline).
5. Binary accuracy of proxy-consensus (majority positive_or_better) vs the 54%
   base rate.

## Interpretation (pre-registered decision table)

Read primary + supporting together against `outcome-calibration` (nothing
predicts):

- **High agreement AND/OR high proxy↔confidence, BUT proxy-AUC ≈ 0.5 (consensus
  near the 54% base rate on outcome):** a confident, coherent "looks promising"
  signal **decoupled from what works** → the strongest case for
  *verify-don't-predict*: you cannot tell from the idea, you must run it.
- **High agreement AND proxy-AUC clearly > confidence-AUC and > 0.5:** the blind
  proxy *can* pre-select winners → the verification cycle's value is in
  **execution**, not selection; a cheap proxy pre-filter would help.
- **Low inter-rater agreement:** no coherent expert signal exists at all →
  verification is the arbiter (supports the framework, differently).
- **Proxy-AUC < 0.5:** systematically anti-predictive → flag for investigation.

## Scope bounds (hard limits, stated up front — not caveats)

1. **Verified-subset only.** Outcomes exist only for the 24 compositions that
   were *verified* (a priority-filtered, non-random subset of 192). This study
   measures **within-verified-set re-ranking by outcome** — it CANNOT speak to
   the full-funnel question "could a rater replace the pipeline's selection."
2. **Proxy null ≠ human null.** The rater is a strong LLM, not a human expert.
   If it can't predict, the claim is "a blind strong LLM can't predict this,"
   NOT "no human expert could." The strong pro-framework conclusion survives a
   null only in that weaker form.
3. **n=24 → wide CIs** on every outcome-correlation metric. This is a pilot.
4. **POS/NEG, not transfer-win.** The framework's real target (distant-field ×
   beats-home-SOTA) is n=2 and not estimable; this measures the coarser
   positive-or-better only.

## Results

5 independent blind Claude raters (a different model family from the gpt-5.5
composer) each rated all 24 compositions. Their binary splits: 14/15/14/16/12
positive-or-better (mean fraction POS 0.59 vs the 0.54 base rate — mildly
optimistic, well-calibrated in aggregate). Analysis in `data/hitrate/`
(`analysis.json`, `ratings/r*.json`), reproducible via `scripts/hitrate_*.py`.

**PRIMARY — the ratings are highly self-consistent:**
- mean pairwise **Spearman on promise = +0.785** (range +0.73..+0.92)
- **Fleiss' κ on the binary calls = +0.603**

"Promising" is a coherent, stable construct — the raters largely see the same
compositions as strong and weak. **Important framing:** the five raters are
independent *samples of a single model* (Claude), not five different models or
human experts, so this number measures that model's judgment *self-consistency*,
not the convergence of independent experts. It is NOT noise — the promise signal
is stable and confident; that is what makes its decoupling from outcome telling.

**PRIMARY — proxy consensus vs the pipeline's own confidence:**
- **Spearman(mean promise, confidence) = +0.236** — only weakly aligned. The
  blind panel and the pipeline's self-rating agree a little on what *looks*
  promising, but reach it differently.

**SUPPORTING — and this is the crux — that confident consensus is decoupled
from what actually works:**
- **AUC(proxy promise → POS) = 0.517** [95% CI 0.270, 0.757] — a coin flip.
- **AUC(pipeline confidence → POS) = 0.559** [0.340, 0.779] — also chance
  (consistent with `outcome-calibration-2026-05-29.md`).
- **Spearman(mean promise, outcome) = +0.030** — essentially zero.
- **proxy-majority binary accuracy = 0.542 = the base rate exactly.** The
  panel's confident calls do no better than always guessing the majority class.

(The AUC CIs are wide at n=24, so the honest claim is "indistinguishable from
chance," not "significantly worse." The strength is the *combination*: tight,
high agreement + point estimates sitting on chance + accuracy exactly at the
base rate + the inversion below.)

**The ranking is uncorrelated with outcome (AUC≈chance) — not inverted; the
vivid individual misses carry the point:**

| composition | mean promise | POS votes | actual outcome |
|---|---|---|---|
| COMP-13 per-block codec selector | **7.4 (highest)** | 5/5 | **clean-NEGATIVE** |
| COMP-14 adaptive-delta numeric | 6.6 | 5/5 | clean-negative |
| COMP-06 embedding-quant bit-allocation | 6.4 | 5/5 | clean-negative |
| COMP-04 cache next-use eviction | 6.0 | 5/5 | clean-negative |
| … | | | |
| COMP-05 **Heilbronn** (transfer win) | 5.2 | 3/5 | clean-positive (≈median rank) |
| COMP-02 **short-string C2** (the project's BEST result) | **4.2 (near-lowest)** | **0/5** | **clean-positive** |

The single most-loved composition (per-block codec, unanimous, "wins by
construction") is a clean failure; four of the eight highest-promise picks are
clean-negatives. Most striking: **all five raters unanimously dismissed
short-string C2 (0/5, near-lowest promise) — the single best result the engine
has produced.** Being honest about the shape: this is *decorrelation with
salient misses*, NOT a systematic inversion — the AUC (0.52) sits just above
chance, not below it; the other transfer win (Heilbronn) was middling not
bottom (5.2, ~rank 12/24); and the two *lowest*-rated compositions are both
correctly negative, so the raters do get the obvious duds right. What they can't
do is separate the real winners from the confident-looking failures in the
middle-and-up band where it matters. (That the rater *missed the biggest win*
also indicates the blinding held — a leak would have surfaced it.)

## Verdict

Per the pre-registered decision table, this is the **"high self-consistency +
proxy-AUC ≈ base rate" cell — the strongest case for verify-don't-predict.** A
strong model, blind to the results, produces a *stable, confident* "this looks
promising" judgment — and that judgment is **statistically decoupled from what
actually survives empirical test** (AUC≈chance, accuracy = base rate), with
individual misses as sharp as unanimously dismissing the project's best result.
Neither the blind rater nor the pipeline's own confidence beats a coin flip at
calling the winners.

This is the affirmative evidence the framework rested on but never had: **you
cannot tell which cross-disciplinary compositions will work by inspecting the
idea — even a confident, agreeing expert panel can't — so the verification
cycle that actually runs them is not redundant with expert judgment; it is the
only thing that separates the winners from the confident-but-wrong.** The
project's own history is the proof: the two biggest wins would have been thrown
away, and a unanimous favorite would have been greenlit into a clean failure.

### Scope bounds (as pre-registered — these are limits, not hedges)

1. **Verified-subset only + range restriction.** Measured within the 24
   *verified* compositions (priority-filtered from 192). Says nothing about the
   full funnel — NOT "a rater could replace the pipeline's selection stage,"
   only "within what got verified, blind promise doesn't track outcome." Because
   these already passed a promise filter, their promise is range-restricted, so
   *some* attenuation of any promise→outcome correlation is expected by
   construction. The finding is therefore best read as "no **residual**
   predictability within the verified set" — which is exactly the question that
   matters for whether the verification stage earns its keep.
2. **Proxy null ≠ human null.** The panel is a strong LLM. The claim is "a blind
   strong-LLM expert panel can't predict this," which is suggestive of but not
   identical to "no human expert could." A human study is the next step.
3. **n=24 → wide CIs** on every outcome metric; this is a pilot. The *robust*
   findings (high inter-rater agreement; accuracy = base rate) do not depend on
   the noisy AUCs.
4. **POS/NEG, not transfer-win.** Measures positive-or-better; the framework's
   real target (beats-home-SOTA) is n=2 and unestimable — though anecdotally
   both transfer wins were bottom-rated, the sharpest possible version of the
   decoupling.
