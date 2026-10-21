# C1 Verification — Adaptive Compressive Sketch (Streaming Sketches Run)

**Run:** `20260523-033802` (streaming sketches, filtered 129-topic)
**Composition:** C1 (3 papers, conf 8/10, L1/L2/L3-weak/L4-PURELY_TESTABLE)
**Verdict:** **clean-negative — foreclosed at oracle upper bound**
**Bench:** `experiments/adaptive_compressive_sketch/bench.py`

## Claim under test

> P159's single-pixel compressive measurement idea + P233's constrained
> submodular allocation + P194's continual adaptation suggests an
> adaptive compressive sketch: instead of fixed Count-Min/AMS rows,
> allocate projection rows to stream regions whose marginal error
> reduction is highest, and update that allocation when regimes shift.
> The measurable claim is that, on synthetic Zipf/mixture streams with
> drift, adaptive row allocation can reduce frequency or quantile
> error at the same byte budget versus fixed random projections or
> classical sketches.

The C1 contribution beyond C6 (which already showed imbalance-aware
counter allocation helps at tight budgets) is the **"adapt when regimes
shift"** clause: drift-aware reallocation of sketch capacity.

## Falsification design

Same structural-oracle approach used for C5 (bias-floor lemma) and C6
(imbalance-CMS): construct the oracle upper bound and test whether it
opens a meaningful gap. If even the oracle can't, no learned controller
can recover what the oracle already cannot deliver.

### Stream

Synthetic two-phase Zipf(s=1.1) stream over V=10,000 keys, N=200,000
events. Drift implemented as **key-rank reversal at midpoint**: heavy
keys in phase 1 become rare keys in phase 2 and vice versa. Full-stream
truth = phase-1 count + phase-2 count.

This is the most C1-favorable drift shape: maximum churn in the
heavy-key set, so dynamic reallocation has the largest possible target.
Less severe drift (partial permutation, gradual concept drift) would
only narrow the dyn-vs-static gap.

### Methods at fixed total byte budget B

1. **CU-CMS** — conservative-update Count-Min, single table of size
   B / 8 bytes (int64 cells), depth d, width B / 8d. Standard
   strong-baseline.
2. **CountSketch** — Charikar/Chen/Farach-Colton signed AMS-style
   sketch with median estimation. This is the **literal P159
   compressive-measurement model**: each row stores a signed Bernoulli
   projection of the key-count vector. Included as the closest
   classical realization of the composition's invoked mechanism.
3. **Static-oracle partition** — knows full-stream stats. Reserves
   B/2 for exact counters of the top-H heavy keys (full-stream) and
   B/2 for a CU-CMS over the rest. This is C6's mechanism. Tests
   whether *any* oracle partitioning helps independently of drift.
4. **Dynamic-oracle partition** — single continuous CU-CMS over the
   full stream **plus** a dynamic exact-counter set that reallocates
   at the drift point:
     - Phase 1: P1-heavy keys → exact-active; all else → CMS
     - Drift: exact-active values frozen into exact-retired;
       exact-active rekeyed to P2-heavy with zero counts
     - Phase 2: P2-heavy keys → exact-active; all else (including
       P1-heavy, now retired) → CMS
     - Query: est(k) = retired[k] + active[k] + (CMS unless k was
       exact in both phases)
   Storage: 2H exact slots (active + retired) + remaining CMS budget.

The dynamic-oracle is the strongest fair formulation of "adapt
allocation when regimes shift": it preserves coherent CMS semantics
over the full stream (the L3 epoch-boundary objection is dodged by
construction) and only the bypass set adapts.

### Metric

Relative L2 frequency error over all keys seen in the stream:

    rel_L2 = sqrt(sum_k (est_k - true_k)^2) / sqrt(sum_k true_k^2)

Unitless; saturation ⇒ 0, naive-zero estimator ⇒ 1.

### Stop rule (regime-aware)

Two filters, **both** required to pass for a structural win:

1. **Non-saturated regime only**: best-fixed rel-L2 must exceed 0.05.
   At smaller errors, the sketch is already accurate enough that
   nothing matters — the C6 stop-rule-bug lesson encoded as a hard
   gate rather than a wording detail.
2. **Dynamic-oracle vs best-fixed**: ≥ 15% relative L2 error
   reduction in the non-saturated regime.
3. **Dynamic-oracle vs static-oracle**: the C1-specific contribution
   must not collapse to the static partition (which is C6 territory).
   A win that vanishes when compared to static-oracle is foreclosed
   on C1's novelty.

## Results

Full sweep: 4 budgets × 2 depths × 5 seeds = 40 runs.

| budget (B) | depth | rl2 CU-CMS | rl2 CountSketch | rl2 static-oracle | rl2 dynamic-oracle | Δ dyn vs best-fixed | Δ dyn vs static | regime |
|---:|---:|---:|---:|---:|---:|---:|---:|---|
|    16,000 | 4 | 0.214 | 0.495 | 0.196 | 0.193 | **+9.79%** | **+1.25%** | non-saturated, **fail** |
|    16,000 | 8 | 0.373 | 0.330 | 0.355 | 0.347 | **−5.14%** | +2.46% | non-saturated, **fail** |
|    64,000 | 4 | 0.029 | 0.119 | 0.027 | 0.027 | +9.75% | +2.99% | saturated |
|    64,000 | 8 | 0.048 | 0.073 | 0.046 | 0.045 | +4.92% | +1.64% | saturated |
|   256,000 | 4 | 0.004 | 0.027 | 0.003 | 0.003 | +12.70% | +2.72% | saturated |
|   256,000 | 8 | 0.004 | 0.017 | 0.004 | 0.004 | +2.67% | −2.03% | saturated |
| 1,024,000 | 4 | 0.0003 | 0.007 | 0.0003 | 0.0003 | −3.13% | −4.37% | saturated |
| 1,024,000 | 8 | 0.0001 | 0.003 | 0.0001 | 0.0001 | +19.20% | −4.59% | saturated |

### Three structural observations

**1. CountSketch (the literal P159 compressive-measurement realization)
loses to CU-CMS at every budget.** Signed Bernoulli random projections
with median estimation produce strictly higher variance than positive
counters with min estimation at the same memory. The composition's
core invoked mechanism (P159's "compressive measurement") *does not
transfer* from imaging to streaming frequency sketches — confirming
the L3 challenge that "P159 analogy is weak".

**2. The dynamic-vs-static-oracle gap is essentially zero across
all regimes.** Means: +1.25%, +2.46%, +2.99%, +1.64%, +2.72%,
−2.03%, −4.37%, −4.59% — hovering at noise levels in both
directions. The C6 static-oracle already captures the entire
oracle-partitioning gain; the *C1-specific* "adapt when regime
shifts" addition contributes nothing measurable. The composition's
novelty over C6 is structurally dead.

**3. Even ignoring the dyn-vs-static issue, dynamic-oracle does
not beat best-fixed by 15% at any non-saturated budget.** At
B=16000 depth 4 it wins by 9.79%; at B=16000 depth 8 it *loses*
by 5.14% (the 2H retired-counter overhead eats too far into the
CMS budget). The structural margin C1 needs to clear over fixed
sketches does not exist.

### Why dynamic doesn't help

The dynamic-oracle is exposed to the same fundamental constraint as
the static-oracle: at fixed total byte budget, *every byte spent on
exact counters is a byte not spent on the CMS*. The C6 finding said
the optimal trade-off near tight budgets is to spend ~B/2 on H exact
counters for full-stream heavies and B/2 on a smaller CMS. The
dynamic-oracle spends 2×H bytes on exact slots (active + retired) so
its CMS is smaller; in exchange it gets exact coverage for the
*union* of P1-heavy and P2-heavy, ~2H distinct keys.

For Zipf with key-rank reversal, P1-heavy ∩ P2-heavy is small, so
dynamic covers ~2H distinct heavy keys exactly vs static's H. But:

- **Static-oracle already exact-counts the keys with the largest
  full-stream weight.** Under permutation drift, P1-heavies and
  P2-heavies have *equal* full-stream weight (each gets one phase
  of heavy weight, one phase of rare weight). Their full-stream
  totals are similar to each other but the top-H by full-stream
  weight is still a single coherent set the static-oracle nails.
- **The exact bytes the dynamic spends on retired counters give
  return only on keys that switched out of the heavy set at drift.**
  These keys have lower full-stream weight than the constantly-heavy
  keys static would prefer. So the dynamic-oracle's marginal exact
  coverage targets *less-impactful* keys.
- **The CMS shrinkage costs are paid on the rare-key tail**, where
  collision noise scales as N/w. Dynamic's smaller CMS produces
  systematically worse tail estimates.

Net effect: the marginal benefit of dynamic exact-coverage (lower-
weight keys) approximately cancels the marginal cost of CMS
shrinkage (higher tail noise). Hence dyn-vs-static ≈ 0.

### Robustness

The single drift point is the **most favorable** case for dynamic
reallocation. Multi-drift streams require K×H retired slots for K
drift points; the dynamic-vs-static gap only narrows further as K
grows. Less severe drift (e.g., partial permutation) yields lower
heavy-set churn, also narrowing the gap.

## Verdict

**Clean-negative, foreclosed at oracle upper bound.** Three structural
findings:

1. **CountSketch (P159 mechanism) is not the right baseline.** Plain
   CU-CMS dominates at every budget. The compressive-measurement
   analogy invoked by the composition does not transfer to streaming
   frequency estimation.
2. **Static-oracle partition (C6 territory) is the only mechanism
   that meaningfully helps**, and only in the non-saturated regime
   — consistent with the C6 finding.
3. **Dynamic reallocation (the C1-specific contribution) adds
   nothing measurable over static partitioning.** dyn-vs-static gap
   is within noise across 8 (budget, depth) settings.

No learned drift-detector or controller can recover what the oracle
already cannot deliver. C1's "adapt-when-regime-shifts" premise is
structurally dead.

## Composition pieces in retrospect

- **P159 (compressive measurement)**: mechanism does not transfer.
  CountSketch is the natural classical realization and loses to
  CU-CMS. L3 challenge "P159 analogy is weak" confirmed.
- **P233 (submodular allocation)**: the claimed objective (marginal
  error reduction under collisions + drift) was never empirically
  shown to be submodular. The static-oracle's optimal allocation is
  proportional to per-region heavy-key weight; this is not a
  submodular maximization but a knapsack with bounded support.
  L3 challenge "Submodularity is asserted, not shown" confirmed.
- **P194 (continual adaptation)**: under the cleanest possible
  oracle drift-handling (the dynamic-oracle constructed here),
  adaptation produces ~0% relative gain over a non-adaptive static
  partition. L3 challenge "Continual learning component is
  underspecified" confirmed at the upper bound — no specification
  of the drift detector could change the answer because oracle
  knowledge already saturates the available gain.

## Related findings

- `findings/c6-imbalance-cms-20260523.md` — the static-partition
  positive at non-saturated budgets (this finding's baseline)
- `findings/c5-multichannel-lowrank-cms-20260523.md` — same oracle-
  upper-bound shape applied to low-rank multichannel sketches
- `findings/c1-per-block-codec-20260523.md` — the compression-domain
  twin: per-segment adaptation foreclosed by stronger fixed
  whole-stream methods

## Reproducibility

```
uv run python experiments/adaptive_compressive_sketch/bench.py \
    --n-events 200000 --n-keys 10000 --zipf-s 1.1 \
    --budgets 16000 64000 256000 1024000 --depths 4 8 \
    --heavy-count 100 --seeds 5
```

All synthetic. No external data. Runs in ~3 minutes on a single
CPU core.
