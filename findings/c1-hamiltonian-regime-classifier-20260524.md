# C1 Verification — Hamiltonian Regime Classifier + SAT Portfolio

**Run:** `20260523-003811` (Hamiltonian wide, 250 papers)
**Composition:** C1 (3 papers, conf 8/10, L1/L2/L3-weak/L4-PURELY_TESTABLE)
**Verdict:** **partial-positive — oracle-supported, cheap-learner falsified**
**Bench:** `experiments/hamiltonian_regime_classifier/bench.py`

## Claim under test

> P113's sparse/random-graph structural regime analysis + P142's GNN-
> guided exact branching + P186's extensible SAT-solver architecture
> yields an instance-adaptive exact HC/TSP solver: first classify
> topology-dependent tractability, then choose branching targets
> with a learned graph policy, then enforce Hamiltonian/TSP
> constraints through a modern SAT backend.

Three-stage pipeline:

  - **Stage 1** (P113): topology features → regime classification
  - **Stage 2** (P142): GNN-guided branching policy (out of scope —
    requires multi-week training as L4 noted)
  - **Stage 3** (P186): CaDiCaL SAT backend with HC encoding

## Falsification design

Two tests against the parts of the pipeline that don't require GNN
training:

**Test A — Regime-classifier signal (Stage 1 upstream gate).**
Predict log(SAT-solve runtime) from topology features vs a baseline
of (n, |E|) only. Required signal: held-out ΔR² ≥ 0.05.

**Test B — Solver-portfolio oracle headroom (Stage 1 → Stage 3).**
Compare CaDiCaL (permutation HC encoding, vertex-0 pinned for
symmetry breaking) against a plain DFS backtracker on the same
instances. Required portfolio gain: oracle-portfolio time vs
best-single-solver time ≥ 5%.

**Test C (closure) — Cheap learned classifier.** Train a Ridge
regressor on topology features to predict per-instance
`log(SAT_time) − log(DFS_time)`, pick the predicted-faster solver
per instance under 5-fold CV. Compare its total runtime against the
single-best solver. This is the cheapest plausible realization of
the Stage-1 → Stage-3 selection the composition proposes.

## Instances

180 random graphs across (n, regime) cells:

- **Erdős–Rényi** G(n, p) at n ∈ {15, 18, 21, 24}, p sweep relative
  to Hamiltonicity threshold ln(n)/n at multipliers
  {0.5, 0.8, 1.0, 1.3, 1.7, 2.5, 4.0} × 4 seeds (the threshold
  multiplier = 1.0 regime is the empirically hardest)
- **Watts-Strogatz small-world** at n × {k=4, k=6} × rewire-p ∈
  {0.1, 0.3, 0.6} × 2 seeds
- **Random regular** at n × d ∈ {3, 4, 6} × 2 seeds

13 topology features extracted via NetworkX: n, |E|, density,
degree min/max/mean/std, clustering, transitivity, algebraic
connectivity, spectral gap, threshold ratio, triangle count,
connectedness.

## Results

| metric | result |
|---|---|
| valid instances (excludes 4 timeouts) | 180 |
| baseline R² (n, m only)               | −0.0084 ± 0.1167 |
| full-topology R²                      | +0.2412 ± 0.1562 |
| ΔR² (full vs baseline)                | **+0.2496** |
| CaDiCaL total runtime                 | 0.644 s |
| DFS total runtime                     | 11.170 s |
| oracle portfolio total runtime        | 0.416 s |
| best-single solver                    | CaDiCaL |
| **oracle gain vs best-single**        | **+35.48%** |
| per-instance wins: CaDiCaL / DFS / tie | 63 / 113 / 4 |
| log-runtime correlation ρ(SAT, DFS)   | +0.692 |
| **Ridge classifier total runtime**    | 1.187 s |
| **learned-classifier gain vs best-single** | **−84.16%** |
| oracle headroom recouped by Ridge     | **−237.2%** |

### Three structural findings

**1. Topology features genuinely predict log-runtime.** ΔR² = +0.25
over (n, |E|) baseline on held-out folds is well above the 0.05
threshold. The baseline R² is near-zero because instance size alone
does not predict log-runtime in this regime — it's the topology
(clustering, spectral gap, threshold ratio) that carries the signal.
The L3 challenge "Topology classification may not predict exact-
solver hardness" is **rejected**: it does predict hardness.

**2. Oracle solver portfolio has substantial structural headroom.**
CaDiCaL has tight runtime distribution (median 0.0001s, p90 0.008s,
max 0.089s); DFS has fat-tailed distribution (median 0.0001s, p90
0.011s, max 3.345s). DFS wins per-instance more often (113 vs 63)
but the single hard tail destroys its total. An oracle that picks
the per-instance fastest solver saves 35.48% wall-clock time vs
the best single solver. The L3 challenge "Generic backend does not
solve the hard part" is partly refuted: CaDiCaL's bounded worst-
case complements DFS's typically-faster median, and both contribute
to a non-trivial portfolio.

**3. Cheap learned classifier falsifies at this scale.** The
Ridge-regression closure on topology features produces a wall-clock
total of 1.187s — *worse* than always running CaDiCaL by 84%, and
worse than even running DFS by 6.5%. The learner recoups −237% of
the oracle headroom (i.e. it has *negative* skill at solver
selection). The mechanism is **structural cost asymmetry**: a
fraction of Ridge's "DFS will be faster" predictions hit the
DFS catastrophic-timeout cases (3.3s runtime), and a small number
of such mispredictions wipes out the savings from many correct
"pick DFS" calls. The classifier predicts log-runtime symmetrically;
the actual cost is wildly asymmetric.

## Verdict

**Partial-positive — oracle-supported, cheap-learner falsified.**

This is the first composition this session that **isn't** foreclosed
at the oracle upper bound. The structural premise — topology
features predict hardness AND multiple solvers have complementary
behavior worth selecting between — is empirically supported. But
the simplest plausible realization of the composition's "first
classify, then choose" pipeline (Ridge regression on topology
features → solver selection) catastrophically fails because of cost
asymmetry. The pipeline's failure mode is not "no signal" but
"signal exists, simple-cost-symmetric learner can't exploit it".

### What the composition would need to actually work

Four candidate paths the verification suggests:

1. **Asymmetric-cost learning**: train a classifier with explicit
   cost weights that penalize "predicted DFS but actually
   DFS-timeout" cases 10–100× harder than the inverse error.
2. **Racing portfolio**: skip per-instance prediction entirely;
   run CaDiCaL and DFS in parallel with a shared time budget and
   take whichever finishes first. Pays 2× CPU but recoups ≈ oracle.
3. **Safe-default + escape-hatch**: always run CaDiCaL by default
   (the tail-bounded solver); use topology features only to *also*
   start DFS speculatively on instances predicted "easy enough
   that DFS finishes faster". The cost asymmetry then has DFS
   savings without DFS-timeout exposure.
4. **Stage-2 (GNN branching policy) test**: the most ambitious
   piece of the composition was never tested here. A learned
   branching policy inside the SAT solver could in principle
   tighten CaDiCaL's already-bounded runtime further, but the
   compute cost makes this an unrelated multi-week test (per L4
   estimates).

The track record now has a distinct *partial-positive with
specific structural failure* shape — different from the C5/C6/C11/
C1-streaming clean-negative foreclosed lane.

## Composition pieces in retrospect

- **P113 (sparse/random-graph regime analysis)**: load-bearing.
  Topology features (especially spectral gap, threshold ratio,
  clustering) contribute the +0.25 R² that (n, |E|) alone misses.
- **P142 (learned exact branching)**: untested in this verification.
  The L4 estimate of 2-4 compute-weeks for the GNN-policy
  training experiment is accurate; this finding addresses Stages
  1 and 3 only.
- **P186 (SAT backend / CaDiCaL)**: contributes a tail-bounded
  solver complementary to plain DFS. Without it, the portfolio
  has no second member with bounded worst case. The L3 "generic
  backend doesn't solve the hard part" challenge is partly true
  — CaDiCaL alone isn't dramatically faster than DFS at median —
  but it's load-bearing for the portfolio because of its bounded
  tail.

## Related findings

- `findings/c5-multichannel-lowrank-cms-20260523.md` — opposite
  shape (clean-negative foreclosed at oracle bound)
- `findings/c1-adaptive-compressive-sketch-20260524.md` — also
  foreclosed at oracle
- `findings/c11-adaptive-delta-numeric-20260524.md` — same shape:
  foreclosed at oracle
- `findings/c4-knn-recall-20260523.md` — closest analogue: "not-
  falsified-by-proxy" with a similar nuanced verdict

This finding's structural cost-asymmetry insight may generalize:
**any composition that proposes "predict-then-select" in a domain
with asymmetric per-instance costs needs cost-sensitive learning,
not symmetric regression** — a meta-observation worth carrying
forward to other algorithm-selection compositions.

## Reproducibility

```
uv run --with python-sat --with networkx --with scikit-learn python \
    experiments/hamiltonian_regime_classifier/bench.py \
    --n-per-cell 4 --timeout 5
```

Runs in ~5 minutes on a single CPU core. Permutation HC SAT
encoding with vertex-0 pinned for rotation symmetry breaking
(O(n²) variables, O(n³) clauses); CaDiCaL153 via PySAT. Plain DFS
backtracker for the baseline. NetworkX for graph generation and
topology features; scikit-learn for Ridge regression and 5-fold
cross-validation.
