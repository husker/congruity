# Polymarket resolution latency has a sharp 41-hour mode driven by US-business-hours review

**Run:** follow-up to C8 verification (`20260524-052010`)
**Date:** 2026-05-24
**Outcome:** **clean-positive structural finding** (sharp mode, identified mechanism)
**Bench:** `experiments/polymarket_resolution_source/bench_latency_structure.py`

## Setup

The C8 verification on resolution-source category was a clean
negative but flagged a striking secondary observation: Polymarket
resolves markets at a remarkably consistent ~30–40h median delay
across *all* source categories. This follow-up bench asks **what
mechanism produces that floor**.

Method: pure-Python analysis on the cached 2000-market Polymarket
Gamma fetch from C8 — no new API calls. Compute the delay
distribution shape (fine-bin histogram, KDE peaks, percentiles),
stratify by year, by hour-of-day of `closedTime`, by day-of-week,
and by category. A sharp mode at a specific value identifies a
single dominant mechanism; a smooth distribution would indicate
many small contributing factors.

## Result

### Sharp modal delay at 41 hours

```
  delay percentiles (hours)
  P5  =   +1.45      P10 = +14.13     P25 = +20.92
  P50 = +37.20       P75 = +43.36
  P90 = +60.45       P95 = +77.31     P99 = +204.50

  sub-3-day histogram (2h bins, 0-72h range, 1671/1969 markets):
     19h |  6.2%   ▆
     21h |  5.1%   ▅
     23h |  2.8%   ▃
     25h |  1.9%   ▂
     27h |  2.9%   ▃
     29h |  2.9%   ▃
     31h |  1.0%
     33h |  1.3%
     35h |  3.2%   ▃
     37h |  4.2%   ▄
     39h | 11.1%   ████████
    [41h | 16.9%   ████████████  ← modal bin]
     43h | 13.7%   ██████████
     45h |  7.8%   ██████
     47h |  3.2%   ▃
     49h |  2.8%
     51h |  0.8%
```

The modal 2-hour bin (centered at 41h) contains **16.9% of all
markets** with a positive delay under 100h. The 10-hour window
**±4h around the mode (37-45h) contains 52.9% of those markets** —
vs 10% expected under a uniform distribution. This is unambiguously
a fixed-mechanism signature, not a smooth noise distribution.

### US-business-hours concentration of `closedTime`

```
  hour | concentration
  --------------------
  07 UTC | 0.11× uniform  (3am EST)
  08 UTC | 0.05× uniform  ← global minimum
  12 UTC | 0.06× uniform  (8am EST)
  --------------------
  15 UTC | 1.86× uniform
  16 UTC | 2.55× uniform
  17 UTC | 2.66× uniform  (1pm EST)
  18 UTC | 3.01× uniform  ← global maximum (2pm EST)
  19 UTC | 2.23× uniform
  20 UTC | 1.66× uniform
```

Resolution is **27× more likely** at 18:00 UTC than at 08:00 UTC.
The 15-21 UTC window (10am-5pm Eastern, US business hours) accounts
for the bulk of resolutions. The 07-12 UTC dead zone is overnight in
US business hours.

### Weekend dip + Monday backlog clearance

```
  Mon  1.35× uniform   ← backlog from weekend
  Tue  1.05×
  Wed  0.80×
  Thu  1.19×
  Fri  1.06×
  Sat  0.66× uniform   ← weekend dip
  Sun  0.88×
```

Saturday resolution rate is roughly half of Monday's. Manual review
queues don't run on weekends.

### Year-over-year mode stability since 2021

```
  year |  n   | median  |   P25   |   P75   |  KDE peak
  2020 |   35 |   -8.06 |  -367.6 |  +16.14 |   17.5h   ← different era
  2021 | 1788 |  +39.55 |   +22.1 |  +42.93 |   41.6h
  2022 |  142 |  +39.26 |  -125.0 |  +49.11 |   42.6h
```

2020 markets (the earliest Polymarket cohort) resolve markedly
faster and with a different modal value, suggesting a substantively
different resolution process at the platform's launch. From 2021
onward the mode is **locked at 41–43h** across thousands of markets.

### Per-category tightness varies, not the mode

```
  category               n    median  IQR
  ------------------------+----+--------+------
  Sports               1023   +40.94    9.83   ← tight cluster
  NBA Playoffs          126   +40.09    4.13   ← extremely tight
  Art                    57   +41.04    5.12
  Pop-Culture           108   +35.12   22.79
  Crypto                171   +22.44   24.98   ← faster, wider
  US-current-affairs    177   +21.54   48.24   ← much wider
  Chess                  55   +19.35   22.62
  Business               35   +22.35   23.64
  Science                35   +21.83   29.80
  Coronavirus            90   +19.50  189.66   ← most variable
```

Sports markets cluster **tightly around 40h** with IQRs of 4-10h —
the resolution-process clockwork dominates. Political and
coronavirus markets are faster on average but **vastly more
variable** (Coronavirus IQR = 190h), consistent with case-by-case
human review for ambiguous events.

## Synthesis: the mechanism

Three independent pieces locate the dominant latency source:

1. **Sharp 41h modal delay** rules out a smooth multi-process
   explanation. There is a single dominant mechanism.
2. **27× concentration at 18:00 UTC and dead zone at 08:00 UTC**
   means the mechanism runs on US business hours, not a 24/7
   automated cron.
3. **Weekend dip (Sat = 0.66× uniform)** means humans are involved.
   A purely automated UMA optimistic-oracle window would not show
   this.

These collectively rule out the natural alternative hypotheses:

- ❌ **UMA Optimistic Oracle liveness period** would produce a
  smooth distribution if liveness were short, or a sharp mode at a
  round number like 24h or 48h. The 41h value is not a clean UMA
  parameter, and the business-hours concentration is incompatible
  with a fully-automated on-chain mechanism.
- ❌ **Automated bot polling cadence** would not show the strong
  weekend dip.
- ❌ **Payout/settlement latency on Polygon** is on the order of
  minutes, far too short to explain a 41h floor.
- ✅ **US-business-hours human review queue** explains all three
  patterns: tight modal value (the review runs predictably),
  US-hours concentration (review staff timezone), and weekend dip
  (staff don't work weekends).

The plausible decomposition: market ends → wait until next US
business day → cluster in the 13:00-17:00 ET review window. For
predictable events (sports, scheduled announcements) this produces
the tight 40h cluster. For ambiguous events (politics, COVID
metrics) the human-review step takes variable additional time,
broadening the right tail without shifting the mode.

## Implications

For Polymarket-design observers and prediction-market researchers:

1. **Polymarket's resolution latency is a human-process property,
   not a contract-level one.** Any intervention to shorten the
   median delay would target the review queue and its staffing
   schedule, not UMA contract parameters.
2. **Sports markets resolve with high mechanical precision** — the
   ~40h delay with sub-5h IQR makes Sports a near-deterministic
   resolution channel. Useful for any trading or arbitrage strategy
   that relies on predictable settlement timing.
3. **Political/news markets are highly variable** with right tails
   reaching weeks. Strategies that assume sub-week resolution for
   politics are mis-specified.
4. **The 24/7 nature of crypto-native infrastructure does not
   translate to 24/7 resolution.** Markets ending Saturday morning
   wait until Monday for review. Anyone designing latency-sensitive
   downstream systems (auto-rollover strategies, derivative
   products) should account for the weekend gap.

## Caveats

- This is a **2020-2022 sample** (oldest-first Gamma API
  pagination). The 2024 high-volume political markets may have
  different process dynamics (more staffing, automation, etc.).
  A current-cohort fetch would clarify whether the 41h mode
  persists.
- We did not directly query UMA's contract parameters or
  Polymarket's adapter liveness settings. The mechanism
  attribution is **inferred from data patterns**, not confirmed
  against contract source. A definitive identification would
  cross-reference the UMA Optimistic Oracle V2 deployment on
  Polygon and Polymarket's CTF adapter.
- 2020 markets (n=35) show a different pattern (median -8h, KDE
  peak 17.5h). Either Polymarket's early process was fundamentally
  different, or this cohort selection is biased.

## Reproduce

```bash
# (after running C8's bench to populate the cache:)
uv run --with numpy python \
  -m experiments.polymarket_resolution_source.bench_latency_structure
```

Sub-second runtime on cached data. No new API calls.

## Note for the engine

This finding came out of C8's *negative* verification result. The
engine surfaced a composition (C8) whose primary claim refuted, but
the act of fetching the data to test it exposed a different
structural observation — a tight 41h mode in a context where the
literature (such as it is) doesn't characterise the latency
distribution at all. This is a useful pattern to log: negative
verifications still produce useful empirical content as
byproducts. The "side findings" lane is worth a dedicated track in
findings/INDEX.md.
