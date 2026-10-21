# Polymarket Resolution-Latency Follow-Up — Current Cohort (2024+)

**Run:** follow-up to `findings/polymarket-latency-structure-20260524.md`
**Verdict:** **partial-replication — mechanism drift**
**Bench:** `experiments/polymarket_resolution_source/bench_current_cohort.py`

## Motivation

The original latency-structure analysis (clean-positive in the
verification track record) used a Gamma-API fetch that delivered
markets oldest-first, yielding a 2020-2022 cohort. The handoff
flagged the open question: does the 41h modal-delay finding and
the US-business-hours hour-of-day signature hold on the current
high-volume political/sports cohort?

This bench refetches via `end_date_min=2024-01-01T00:00:00Z` and
runs the same latency-structure analysis on 2,500 current-cohort
markets (endDate 2024-2025, closure 2023-2025), then compares
directly against the 2020-2022 cohort's numbers.

## Result table

| metric                       | 2020-22 cohort (orig.) | 2024+ cohort (this)  | Δ |
|---|---|---|---|
| sample size (with delay)     |              ~1,969 | 2,499 | — |
| modal delay (KDE peak)       |              41.0h | **31.1h** | **−10h** |
| modal concentration (±4h)    |              52.9% | 23.7% | **less sharp** |
| median delay overall         |              ~32h | 28.3h | similar |
| peak closure-hour (UTC)      |              18:00 (3.0× uniform) | **06–08 (3.19× uniform)** | **shifted 12h** |
| 18:00 UTC concentration       |              3.0× uniform | 0.69× uniform | **collapsed** |
| weekend dip (Sat ratio)      |              0.66× | 0.62× | similar |
| Monday vs uniform            |              1.35× | 1.19× | similar |

## What replicated

- **Weekend dip persists.** Saturday and Sunday closure ratios are
  still well below uniform (0.62× and 0.68×), and Monday is
  slightly above (1.19×). Whatever human-in-the-loop component
  drives weekly seasonality remains in the current mechanism.
- **A sharp modal peak exists.** The 0–200h delay window has a
  clear KDE peak (at 31.1h now). The shape "platform has a fixed
  resolution timer, not smooth per-market drift" is unchanged.
- **Median delay is in the same band** (28.3h vs ~32h). The order
  of magnitude is stable.

## What drifted

- **Modal value dropped 10 hours**: 41.0h → 31.1h. The fixed
  timer (or whatever process imposes the mode) is now ~25% faster.
- **Peak hour shifted by 12 hours**: from 18:00 UTC (US East
  Coast afternoon — workday peak) to 06–08 UTC (US East Coast
  early morning / Asia afternoon). The 18:00 UTC slot collapsed
  from 3.0× uniform to 0.69× uniform — *below* baseline now.
- **Modal concentration weakened**: 52.9% within ±4h → 23.7%
  within ±4h. The current mechanism produces a less-sharp mode,
  consistent with either a smoothed batch-cadence or multiple
  overlapping resolution paths.

## Mechanism reinterpretation

The original finding identified "US-business-hours review queue"
as the mechanism behind the 41h mode and the 18:00 UTC peak. That
specific story is **not current-state accurate**. The 2024+
pattern suggests one of:

- **Asia-based review queue**: 06–08 UTC is 15–17:00 Tokyo /
  14–16:00 Shanghai — afternoon business hours in East Asia. If
  Polymarket added an Asia-based MIC (Markets Integrity Committee)
  shift, this pattern would emerge.
- **Overnight automated batch**: 06–08 UTC is 01–03 EST / 22–00
  PST — natural off-hours for a scheduled bot/batch job that
  resolves accumulated markets while operators sleep.
- **Distributed reviewer pool**: if reviewers are now spread
  across timezones, the 18:00 UTC concentration would dilute and
  the dominant hour would reflect whichever cohort is largest
  (plausibly an APAC operations team).

The weekend dip rules out a purely-automated mechanism; some human
gating is still present. So the most likely story is **reviewer
pool moved (or grew) to include APAC business hours**, with the
US-business-hours peak no longer dominant.

## Side observation: volume → delay correlation

Higher-volume markets get *slightly longer* delays:

| volume tertile | n | median delay |
|---|---:|---:|
| low (<$13.8K)  | 843 | 19.10h |
| mid ($13.8K–$72.9K) | 840 | 29.53h |
| high (≥$72.9K) | 816 | 30.56h |

Counter-intuitive — naively one might expect high-stakes markets
to be resolved faster. The pattern is consistent with high-volume
markets receiving more careful (slower) review, possibly because
they have more potential for dispute or more capital at stake.

## Verdict

**Partial-replication, mechanism drift.** The general structural
claim — Polymarket has a platform-level resolution mechanism that
produces a clear modal peak, hour-of-day concentration, and
weekend seasonality — replicates on the current cohort. The
*specific numbers* in the original finding (41h mode, 18:00 UTC
peak, "US-business-hours review queue" story) are **historical
characterizations valid for 2020-2022, not current-state**.

Track-record implication: the original finding stays in INDEX
as `clean-positive (mechanism identified)` because the mechanism
was correctly identified *for that cohort and time period*. This
follow-up adds a row that documents the drift and clarifies the
mechanism interpretation.

## Open follow-ups

- **Pinpoint when the drift happened.** A year-by-year sweep
  (2020/21/22/23/24/25) of modal value and peak hour would isolate
  the drift event(s). Could correlate with publicly-known
  Polymarket changes (UMA contract upgrades, operational changes).
- **Test the Asia-business-hours hypothesis.** If reviewer
  geography is the driver, the 06–08 UTC concentration should be
  stronger on weekdays and weaker on weekends — which is exactly
  what the weekend dip shows. A regression of hour-effect ×
  weekday would quantify.
- **High-volume slower-resolution effect.** The volume-delay
  correlation should be checked for confounds (high-volume markets
  may be in slower-resolving categories like politics).

## Reproducibility

```
uv run --with numpy --with requests python \
    experiments/polymarket_resolution_source/bench_current_cohort.py \
    --n-target 2500
```

Cached fetch to `experiments/polymarket_resolution_source/_cache/`
(gitignored). Runs in ~2 minutes including the initial fetch
(~8 seconds per 100 markets on Gamma API).
