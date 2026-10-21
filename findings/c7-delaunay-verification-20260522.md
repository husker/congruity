# C7 verification — Delaunay-restricted candidate edges for Euclidean TSP

**Run:** `data/pipeline/compositions/20260522-215133/` composition C7
**Date:** 2026-05-22
**Outcome:** **clean-negative**
**Bench:** `experiments/tsp_delaunay/bench.py`

## Composition under test

C7 from compose run `20260522-215133` (after L3 reprocess with the tightened
adversarial prompt landed in `f1dcae0`):

> P26's Delaunay triangulation can restrict Euclidean TSP edges, P96's spectral
> hypergraph partitioning can decompose the resulting geometric neighborhood
> graph while preserving global structure, and P72's min-cost-flow methods can
> evaluate or repair cross-partition connectivity. This yields a sparse
> geometric TSP decomposition with cut/flow-aware boundary handling, something
> none of the individual papers provides.

The composition's actionability rests on a single premise: that Delaunay
edges form a sufficient candidate set for the optimal TSP tour. Every
downstream stage (hypergraph partitioning, flow-based boundary repair) is
operating on the Delaunay edge subgraph; if the optimal tour requires edges
outside that subgraph, the composition's stack can never reach the optimum.

The recipe Codex generated for C7 named its own stop rule:

> Abandon as clean negative if Delaunay candidate recall is <99% on any
> small instance with known optimum

That floor is the cheapest falsification path — we don't have to implement
the spectral partition or min-cost-flow stages to refute the composition.

## What we measured

For each (n, seed) in {10, 12, 14, 16, 18} × {0…9}:

1. Generate n points uniformly in [0,1]² with a fixed RNG seed.
2. Solve TSP exactly via Held-Karp DP (O(2ⁿ·n²); fine through n=18).
3. Compute the Delaunay triangulation via `scipy.spatial.Delaunay`.
4. Candidate recall = |optimal tour edges ∩ Delaunay edges| / n.

Tight uniform-random Euclidean instances are the *best case* for Delaunay's
geometric prior — TSPLIB instances (real road networks, layouts, etc.)
typically have less favourable geometry, so a uniform-random falsification
is decisive.

## Result

```
sizes: [10, 12, 14, 16, 18], seeds per size: 10
50 instances total

mean recall:         99.11%
min recall:          91.67%
max recall:          100.00%
instances at 100%:   43/50
instances < 99%:     7/50
instances < 95%:     7/50

per-size mean recall:
  n=10: 100.00%
  n=12:  99.17%
  n=14:  99.29%
  n=16:  98.75%
  n=18:  98.33%
```

> **Note (2026-05-23 correction):** earlier versions of this finding reported
> mean recall ~92% because `tour_edge_set` iterated one step too many over
> the Held-Karp output, adding a bogus `(0, 0)` self-loop to the optimal-edge
> set. The denominator was inflated by 1; fixing it raises every reported
> recall to its true value. The verdict (clean-negative) is unchanged but
> the headline numbers are.

The honest reading: Delaunay does contain the full optimal tour on most
small uniform-random instances (43 of 50 are perfect), but a meaningful
minority (7 of 50, or 14%) miss at least one optimal-tour edge. The
recipe's stop rule gates on per-instance recall ("on **any** small
instance with known optimum"), so 7 sub-99% instances out of 50 is
enough to fire it.

This is consistent with what's actually known about Delaunay and the
TSP: the classical containment result is **MST ⊆ Delaunay**, not
TSP ⊆ Delaunay. The TSP tour is allowed to take longer "shortcut" edges
between distant clusters that the Delaunay triangulation does not
include because those edges would cross other Delaunay edges. The
composition's premise generalised the wrong theorem; the violation just
happens to be rarer than the buggy bench first suggested.

## Verdict

**clean-negative.** C7's stop rule fires on the cheapest possible
falsification: a Delaunay-restricted candidate set cannot represent the
optimal tour even on toy uniform-random instances, so the
hypergraph-partition + min-cost-flow stages would be optimising over a
strictly suboptimal subgraph from the start.

This is the right kind of negative for a verification-first synthesis
engine: the engine surfaced a structurally plausible cross-disciplinary
candidate (geometry + spectral methods + flows), the verification stage
ran the cheapest falsification first, and we got a decisive answer in
seconds of CPU. No effort was spent on the harder spectral and flow
stages that would have inherited the broken premise.

## Implications for the pipeline

This is also a useful data point for the L3 adversarial verifier. Under
the *old* L3 prompt this composition was correctly tagged `invalid`,
likely on the grounds that "Delaunay does not contain TSP" is a known
result. Under the *new* L3 prompt (commit `f1dcae0`), this composition
moved to `weak` because the empirical-vs-fundamental distinction puts
"the cited geometric prior is too weak" in the EMPIRICAL bucket
(definitively testable via the recall measurement we just ran).

Both verdicts are defensible. The relevant calibration question is
whether L3 should bake well-known math results into the FUNDAMENTAL
bucket more aggressively. For now the reconciler's behaviour is correct:
the recipe was generated, the verification ran cheaply, and the negative
landed in this document. That's the system working as intended.

## Reproduce

```bash
uv run --with scipy --with numpy python -m experiments.tsp_delaunay.bench \
  --sizes 10 12 14 16 18 --seeds 10
```

Wall clock: a few seconds end to end. Held-Karp DP dominates at n=18
(~0.7 s/instance); everything else is sub-millisecond.
