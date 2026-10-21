# C4 verification — K-NN candidate-edge recall as a baseline for the diffusion proposer

**Run:** `data/pipeline/compositions/20260523-003811/` composition C4
**Date:** 2026-05-23
**Outcome:** **not falsified by proxy** (K-NN at K=10 satisfies the stop rule on 50/50 instances)
**Bench:** `experiments/tsp_knn_recall/bench.py`

## Composition under test

C4 from the wide Hamiltonian-TSP compose run `20260523-003811`:

> P88's generative diffusion sampling + P187's formal safety/constraint
> layer + P186's SAT solver suggests a generate-filter-solve HC/TSP
> pipeline: a learned sampler proposes candidate edges, formal safety
> filters them, SAT solver completes.

Recipe Codex generated for C4 named this stop rule:

> Abandon early if candidate edge recall for known optimum tours is
> below 99% on 20 pilot TSPLIB instances at the chosen pruning budget,
> because exact completion cannot recover missing optimum edges.

The recipe leaves "the chosen pruning budget" unspecified. To bound the
composition without training a diffusion model, we test the cheapest
possible candidate-edge proposer — symmetric K-nearest-neighbors — at
several budgets. K-NN is what any learned proposer has to beat: a
diffusion model that's no better than K-NN at the same budget is not
worth building.

## What we measured

For each (n, seed) in {10, 12, 14, 16, 18} × {0…9}:

1. Generate n uniform random points in [0,1]² with a fixed RNG seed.
2. Solve TSP exactly via Held-Karp DP.
3. For each K ∈ {3, 5, 8, 10, 12}, build the symmetric K-NN graph
   (edge (i, j) exists iff j is in i's K nearest OR i is in j's K nearest).
4. Candidate recall = |optimal tour edges ∩ K-NN edges| / |optimal tour edges|.

The candidate-edge recall is the fraction of optimal-tour edges that
survive K-NN pruning. A diffusion proposer would have to match or beat
this at the same budget to add value.

## Result

```
sizes: [10, 12, 14, 16, 18], seeds per size: 10
50 instances total

per-K aggregate:
  K= 3: mean recall 93.28%, min 77.78%, 27/50 below 99%, 23/50 perfect
  K= 5: mean recall 99.06%, min 91.67%,  7/50 below 99%, 43/50 perfect
  K= 8: mean recall 99.89%, min 94.44%,  1/50 below 99%, 49/50 perfect
  K=10: mean recall 100.00%, min 100.00%, 0/50 below 99%, 50/50 perfect
  K=12: mean recall 100.00%, min 100.00%, 0/50 below 99%, 50/50 perfect

per-size mean recall (K=10):
  n=10: 100.00%
  n=12: 100.00%
  n=14: 100.00%
  n=16: 100.00%
  n=18: 100.00%
```

**K-NN at K=10 hits 100% recall on every single one of 50 instances.**
For n ≤ 18, the optimal tour is always a subgraph of the K=10 nearest-
neighbor graph; K=8 misses one instance; K=5 misses 7.

## Verdict

**Not falsified by proxy.** C4's stop rule ("candidate recall ≥ 99%
at the chosen pruning budget") is *trivially satisfiable* by symmetric
K-NN at K = 10 on small uniform-random instances. The composition's
generate-filter-solve premise survives this floor; the open question is
whether a diffusion-based proposer can do better than K-NN — which
means using **fewer** candidate edges than 10×n while keeping ≥ 99%
recall, or finding non-local optimal edges that K-NN misses.

That is a much sharper question than the recipe stated. Phrased
correctly:

  > Can a learned (diffusion) sampler outperform a K=10 symmetric K-NN
  > candidate filter on the candidate-recall × budget tradeoff?

Falsifying *that* requires either:

- a published K-NN-vs-diffusion benchmark on TSP candidate edges, or
- actually training the diffusion model and running the comparison.

Both are out of scope for a one-session falsification.

## Implications

This is a different shape of result than the C7-Delaunay verification:

- **C7** had a misuse of a known theorem (MST ⊆ Delaunay → TSP ⊆
  Delaunay was the wrong generalisation). The premise was structurally
  wrong; falsification was decisive.
- **C4** has a premise that's loose-but-not-broken: candidate-edge
  proposers exist (K-NN works on toy instances). The composition's
  marginal value depends on whether a learned proposer can beat the
  trivial baseline.

The verification-first engine handled both cases the right way: it
generated a recipe with a concrete stop rule, the cheapest proxy
implementation cleared (C4) or failed (C7) that rule, and we got the
real answer in seconds of compute. The C4 result re-shapes the question
the composition has to answer rather than killing it.

## Bug-fix note

This bench and `experiments/tsp_delaunay/bench.py` originally both
included a `(0, 0)` self-loop in the optimal-edge set because Held-Karp
returns a tour with a duplicate start vertex. `tour_edge_set` iterated
one step too many. The bug inflated reported recall denominators by 1,
which biased every recall number downward and made the C7 bench's
clean-negative look more universal than it is. Both benches now skip
self-loops; the C7 finding doc has been corrected.

## Reproduce

```bash
uv run --with numpy python -m experiments.tsp_knn_recall.bench \
  --sizes 10 12 14 16 18 --seeds 10 --ks 3 5 8 10 12
```

Wall clock: a few seconds end to end.
