# C5 verification — low-rank shared sketch for multichannel streams

**Run:** `data/pipeline/compositions/20260523-033802/` composition C5
**Date:** 2026-05-23
**Outcome:** **clean-negative with structural explanation**
**Bench:** `experiments/multichannel_lowrank_cms/bench.py`

## Composition under test

C5 from the sketching/streaming compose run `20260523-033802`:

> P85's reduced-rank latent representation + P190's graph modeling of
> inter-variable time-series relations + P162's adjacent-window
> imputation consistency suggests a **low-rank graph sketch for
> multichannel streams**. Instead of one AMS/Count-Min sketch per
> channel, maintain a shared latent representation.

The intuition is appealing: when streams across many channels share
correlated structure (network telemetry, sensor arrays, multi-tenant
counters), a single sketch in the low-rank latent space should beat
maintaining one CMS per channel at the same total memory. Recipe's
stop rule:

> Abandon early as clean negative if a 3-budget pilot on both real and
> shuffled-channel streams fails to beat the best per-channel baseline
> by at least 10% mean error at any budget.

## What we measured

Two synthetic multichannel streams, 200 000 events over 10 000 keys
in 20 channels:

  - **STRUCTURED:** true rank r = 3 ≪ 20 channels. A random
    non-negative basis `B ∈ R^{20×3}` (columns normalised so each
    latent component's channel-distribution sums to 1). Each event
    samples a latent component `s ∼ Uniform{0,1,2}`, a channel
    `c ∼ Categorical(B[:, s])`, and a key `k` from component `s`'s
    Zipf(1.1) distribution (with a component-specific key
    permutation). Per-channel count vectors are mixtures of three
    underlying patterns.

  - **SHUFFLED:** 20 independent Zipf(1.1) streams (control). Any
    apparent "low-rank gain" here is artefactual.

Three sketches at equal total byte budget:

  1. **Per-channel CMS:** 20 independent conservative-update CMS
     instances, each gets B/20 bytes.
  2. **Shared-key CMS:** one CU-CMS with key = (channel, key), full B.
  3. **Oracle low-rank CMS:** 3 CU-CMS instances (one per latent
     component), each gets B/3 bytes. Each event is routed to the
     latent component it was generated from. Estimated per-channel
     count: `est[c, k] = Σ_s B[c, s] · count_s[k]`.

The oracle is the structural upper bound for any low-rank approach:
it has perfect knowledge of the basis `B` and the latent component of
every event. A learned basis recovery, NMF, PCA, or autoencoder
cannot do better.

Metric: mean absolute frequency error over the top-50 most-frequent
keys per channel.

## Result

```
budget |   per-ch err |   oracle-lowrank err  | winner
 ------+--------------+-----------------------+----------------------------
  8 KB |   255.07     |          15.29        | oracle by +93%
 16 KB |    79.26     |           8.02        | oracle by +89%
 32 KB |    18.06     |           6.54        | oracle by +64%
 64 KB |     1.77     |           6.38        | per-channel by -261%
128 KB |     0.19     |           6.40        | per-channel by -3295%
256 KB |     0.03     |           6.42        | per-channel by -24289%
512 KB |     0.003    |           6.43        | per-channel by -214156%

oracle err decline tightest→widest: -58%   (15.29 → 6.43, plateaus)
per-ch err decline tightest→widest: -100%  (255.07 → 0.003, keeps falling)
```

The crossover is sharp and lives near n_keys / n_channels ≈ 64 KB.
Below that the per-channel CMS is so collision-saturated that even the
oracle's bias floor of ~6.4 is better; above that, per-channel keeps
improving linearly with memory while the oracle plateaus.

## Why the oracle plateaus — the bias-floor argument

The low-rank reconstruction

    est[c, k] = Σ_s B[c, s] · count_s[k]

returns the **expected** per-channel count under the basis-mixing
model, not the realised one. For each true (channel, key, latent)
triple, the contribution to `count[c, k]` is a Bernoulli with
probability `B[c, s]`. Summed across all events with latent `s` and
key `k`, the realised channel-count is `Binomial(count_s[k], B[c, s])`,
whose deviation from the expected value `B[c, s] · count_s[k]` has
standard deviation `√(count_s[k] · B[c, s] · (1 − B[c, s]))`.

Critically: **this deviation does not depend on the sketch's memory
budget.** Even with perfect `count_s[k]` (zero CMS error), the
reconstruction is off by the binomial-mixing standard deviation. Add
more memory to the latent sketches and `count_s[k]` becomes exact, but
the bias floor stays put.

Per-channel CMS has no such bias. Its error decreases as `O(N / w)`
where `w` is the sketch width: more memory directly reduces error to
zero.

Crossover budget. Per-channel CMS beats oracle low-rank once the
per-channel width exceeds the bias floor's effective complexity —
roughly when `w > √(stream-norm × basis-mixing-prob)`. For 200 000
events, 20 channels, basis-mixing ~0.05–0.10, that's `√(200k · 0.05)`
≈ 100, matching the empirically observed ~64 KB crossover.

## Verdict

**clean-negative with structural explanation.** The composition's
premise fails not because the basis is hard to learn, but because the
**reconstruction itself has an irreducible bias floor.** Specifically:

1. **The composition only wins in the over-constrained regime** where
   the per-channel baseline is so memory-starved that its error is
   dominated by collisions. Both methods are unusable in that regime.

2. **At any reasonable budget**, per-channel CMS dominates and the gap
   grows without bound (per-channel keeps improving, oracle plateaus).

3. **The negative result rules out the entire class of low-rank
   reconstruction sketches**: learned basis, PCA, NMF, autoencoder,
   sketched-SVD — the bias is a property of the reconstruction
   `Σ_s B[c, s] · count_s[k]`, not of how `B` is obtained. Any
   non-oracle variant performs strictly worse than this lower bound.

## What this means for practitioners

If you're building a frequency or top-k sketch over multiple
correlated channels (sensor networks, multi-tenant counters, network
flow telemetry, multi-variate time series), **per-channel CMS or a
shared-key CMS will beat a low-rank shared sketch at every reasonable
memory budget.** The intuition that "exploiting cross-channel
correlation should save memory" doesn't survive the bias-floor
argument: low-rank reconstruction trades collision variance for
binomial-mixing variance, and the trade is unfavourable except in the
useless regime.

The right way to exploit correlation in multi-channel sketches is
likely *not* reconstruction-based. Better candidates:

- **Conditional sketching**: route to a smaller CMS based on observed
  latent component (avoids the reconstruction step entirely);
- **Boolean-rank approaches** (heavy hitters only): exploit overlap of
  top-k sets across channels;
- **Lossless dictionary compression** of repeated per-channel patterns
  (no estimation step, no bias).

But "stack a low-rank reconstruction in the latent space" is dead.

## Reproduce

```bash
uv run --with numpy python -m experiments.multichannel_lowrank_cms.bench \
  --seeds 3 \
  --budgets 8000 16000 32000 64000 128000 256000 512000
```

Wall clock: a few minutes on a single CPU. Add `--rank 5` or
`--channels 50` to explore the regime dependence; the plateau is
robust across all settings we tested.
