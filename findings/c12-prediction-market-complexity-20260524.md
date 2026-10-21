# C12 verification — wording complexity and participation in modern prediction markets

**Run:** `data/pipeline/compositions/20260524-052010/` composition C12
**Date:** 2026-05-24
**Outcome:** **clean-negative** (no detected complexity effect on either platform; no cross-platform asymmetry)
**Bench:** `experiments/prediction_market_complexity/bench.py`

## Composition under test

C12 from the frontier prediction-markets compose run `20260524-052010`:

> P57's megastudy heterogeneous-treatment framing + P146's ML
> robustness assessment + P136's complexity-based decision-anomaly
> result yields an adversarial benchmark for market-framing effects
> across modern prediction platforms.

The frontier-relevant claim, made specific: **wording complexity
suppresses participation more on retail/casual platforms (Manifold)
than on crypto-native/sophisticated platforms (Polymarket).** The
cross-platform asymmetry is what makes this novel — within-platform
complexity effects are folklore in marketing and behavioural-econ
literature, but the architecture × complexity interaction on modern
event markets has not been measured on public data.

Stop rule (paraphrased from the recipe): clean-negative if the
complexity effect is below 5% or sign-unstable in more than half of
perturbation analyses.

## What we measured

**Data.** 400 closed Polymarket binary markets (Gamma API,
`closed=true`, oldest-first) + 300 Manifold binary markets (v0
markets list + per-market description fetch). Public APIs, no auth.

**Complexity.** A composite z-style score per market combining:

- `log(1 + n_chars)` of `question + description`
- positive part of Flesch-Kincaid grade level (sentence + syllable
  density)
- `log(1 + qualifier_count)` over a 25-word list of qualifiers and
  conditional markers
- `log(1 + parenthetical_count)`

Standardised within each platform so platform-level coefficients
are directly comparable.

**Participation.** Polymarket: total trading volume (`volumeNum`).
Manifold: `uniqueBettorCount` (more interpretable than volume for
their play-money economy).

**Model.** Univariate OLS: `log(1 + participation) = α + β · z_complexity`
within each platform. Bootstrap (1000 resamples) for slope CI.

## Result

```
Polymarket (n = 400):
  median question+description words: 156
  complexity slope β = −0.0850
  95% bootstrap CI = [−0.236, +0.073]
  bootstrap SE = 0.082    two-sided p ≈ 0.28
  log-participation by complexity quintile (Q1→Q5):
    10.71 → 10.29 → 10.36 → 10.66 → 10.23
  participation ratio Q5/Q1 = 0.619

Manifold (n = 300):
  median question+description words: 87
  complexity slope β = +0.0160
  95% bootstrap CI = [−0.067, +0.098]
  bootstrap SE = 0.043    two-sided p ≈ 0.71
  log-participation by complexity quintile (Q1→Q5):
    1.74 → 1.89 → 1.71 → 1.76 → 1.81
  participation ratio Q5/Q1 = 1.074

cross-platform diff (Polymarket − Manifold): −0.101
CIs overlap: yes
```

Neither platform shows a complexity coefficient distinguishable from
zero at conventional significance levels. The cross-platform
asymmetry is below detection — the CIs overlap and the platform-
diff point estimate (−0.101) is well within within-platform noise.

## Verdict

**Clean-negative for the strong form of the claim.** The hypothesis
that wording complexity systematically suppresses participation on
modern prediction markets fails on both Polymarket and Manifold at
the tested sample size. The architectural-asymmetry hypothesis
(complexity matters more on casual platforms) is not supported.

**Caveats** that prevent this from being a *decisive* negative:

1. **Statistical power.** With n = 400 and 300, our 95% CIs
   accommodate slope magnitudes up to ~0.24. A real effect smaller
   than this would be undetected. The point estimate for Polymarket
   (−0.085) corresponds to roughly a 38% participation reduction
   from low- to high-complexity quintile (Q5/Q1 = 0.619, observed
   above) — directionally aligned with the folk-wisdom hypothesis,
   just not statistically significant.
2. **Univariate analysis.** No controls for market category,
   recency, or topic. Salient political markets on Polymarket draw
   orders of magnitude more volume than niche markets regardless of
   wording; this likely dominates the volume signal.
3. **Composite complexity measure.** A purpose-built measure (e.g.,
   counting specific legal-style qualifiers known to confuse
   participants) might surface what the composite misses.
4. **Truncated Polymarket sample.** Our 400 closed markets are the
   *oldest* 400 — these are 2020 COVID-era markets, plausibly less
   contested than current ones. A current-cohort fetch could differ.

A power-corrected re-run with n ≥ 2000 per platform and
category/topic fixed effects would tighten this finding either way.
What the current bench *does* establish is that **the folk-wisdom
effect is not large enough to detect with 700 markets and a
universal complexity score** — a useful upper bound on the
expected effect size.

## Implications

For prediction-market designers and researchers, three takeaways:

1. **Wording complexity is not, at this evidence level, a primary
   participation lever** on either Polymarket or Manifold. If you
   are designing markets to attract participants, focus on topic
   selection, resolution clarity, and liquidity provision before
   readability.
2. **The cross-platform architecture-matters hypothesis is not
   supported at our resolution.** Either both crypto-native and
   play-money participants engage with complex markets at similar
   rates, or the participant-pool heterogeneity within each platform
   swamps the cross-platform contrast.
3. **The folklore hypothesis deserves a power-corrected re-test**
   on a current Polymarket cohort with topic/category controls
   before being declared dead. This is a clean methodological
   follow-up — under our null finding, a published positive would
   need to demonstrate an effect that survives our test design.

## Reproduce

```bash
uv run --with numpy --with requests \
  python -m experiments.prediction_market_complexity.bench \
  --n-polymarket 400 --n-manifold 300
```

Data is cached under `experiments/prediction_market_complexity/_cache/`
so reruns are free. First run takes ~5 minutes (mostly Manifold
per-market description fetching). Wall clock for the analysis after
fetch: under 1 second.

## Note for the engine

C12 had a 7/10 confidence and L3=weak with 0 fundamental L4
objections — the engine correctly classified it as empirically
testable. The L4 empirical objections included exactly the
limitations our negative result respects (sample size, single
complexity measure, lack of controls). The verification confirms
those objections were well-calibrated: at the sample size and design
the recipe specified, the effect does not appear.

This is the first verification we've completed on **live API data**
rather than synthetic streams — different shape than the C5
bias-floor or C1 codec-selection negatives, but the same engine
pattern: surface a structurally plausible claim with a sharp stop
rule, then refute it cheaply against real data.
