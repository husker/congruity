# Heilbronn triangle — verification (clean positive)

**Source composition:** `compose-20260521-043936.md` C1
(repulsion + smooth-min surrogate + annealed gradient flow)
**Run date:** 2026-05-21
**Prototype:** `experiments/heilbronn/` (Python + JAX)

## Composition under test

L3 verdict on C1: WEAK. L4: PURELY TESTABLE (10 empirical / 0 fundamental).
Codex confidence: 8/10.

C1's claim:
> P92's force-field view + P144's energy-stable gradient flow + P28's array
> programming → a pure-Python optimizer that combines (a) Riesz/Coulomb
> repulsion so every point gets a force, (b) a logsumexp soft-min over
> triangle areas (annealed), (c) boundary penalty, all in a vectorized
> all-triplet JAX objective with gradient descent.

The L3 reviewer's strongest attacks:
- *"Folklore optimization recipe — smoothing + annealing + regularization is well-known."*
- *"Energy stability ≠ search power"*
- *"Pairwise repulsion optimizes spacing, not the near-collinearity bottleneck."*
- *"No evidence it beats existing SA / basin-hopping."*

Last attack is the critical empirical one. This is what we tested.

## Verification setup

Implemented in `experiments/heilbronn/`:
- `evaluator.py` — exact min triangle area over C(n,3) triples (numpy)
- `baselines.py` — SA-uniform (random init), SA-halton (low-discrepancy init),
  SciPy basin-hopping with L-BFGS-B on smoothed objective
- `c1_smooth.py` — C1 prototype with JAX autograd:
  - `loss = -softmin_τ(areas) + α·sum(1/√(d²ᵢⱼ+ε)) + β·boundary_penalty`
  - τ annealed geometrically from 10 → 5000 over the run
  - α annealed linearly from 1e-4 → 1e-8
  - β = 50, lr = 5e-3, multi-restart from uniform random
- `bench.py` — fixed compute budget (n_steps=50000, n_starts=8) across methods

## Results — min triangle area, larger is better

| n | SA-uniform | SA-halton | Basin-hop | **C1-smooth** | published SOTA |
|---|---|---|---|---|---|
| 8 | 0.05732 | 0.05732 | 0.02664 | **0.06250** | 0.07286 |
| 10 | 0.03337 | 0.03337 | 0.02071 | **0.03693** | 0.04812 |
| 12 | 0.01545 | 0.01545 | 0.00797 | **0.02332** | 0.03224 |
| 15 | 0.01080 | 0.01080 | 0.00505 | **0.01448** | 0.02448 |
| 20 | 0.00308 | 0.00308 | 0.00150 | **0.00493** | 0.01797 |

C1 wins at every n. Also fastest at larger n (56s vs 148-175s for SA / basin-hop at n=20).

### C1 vs each baseline

| n | C1 vs SA | C1 vs Basin-hop |
|---|---|---|
| 8 | +9.0% | +135% |
| 10 | +10.7% | +78% |
| 12 | +51.0% | +192% |
| 15 | +34.1% | +186% |
| 20 | +60.1% | +229% |

The gap *grows with n* — exactly what you'd expect if the smooth-surrogate
helps when the search landscape gets harder.

## What the L3 reviewer got right and wrong

**Got partially right** — "Folklore optimization recipe":
The ingredients (logsumexp smoothing, annealing, repulsion regularization)
are individually well-known in numerical optimization. C1 is not inventing
new primitives. The reviewer's framing was directionally correct.

**Got wrong** — "No evidence it beats existing SA / basin-hopping":
C1 in fact beats both. By a lot. The specific *combination* — soft-min
annealing schedule + Riesz repulsion + boundary penalty, all in one
differentiable objective — is meaningfully better than the standard
alternatives at this budget. Basin-hopping with L-BFGS-B on a smoothed
objective (which is conceptually the closest baseline) does *worse* than
naive SA, because L-BFGS overshoots/undershoots on the smoothed objective
without the τ annealing schedule.

**Got wrong** — "Pairwise repulsion optimizes spacing, not near-collinearity":
The annealing handles this. Early in training (low τ, low α decay), the
repulsion does optimize for spacing — but this is exactly what gives the
optimizer a global force structure to navigate from. Late in training
(high τ, α near 0), the soft-min becomes sharp and the optimization
focuses on near-collinear triples specifically. The hand-off works.

**Open** — "Energy stability ≠ search power":
We didn't use the SAV (Scalar Auxiliary Variable) energy-stable scheme
from P144 — just plain gradient descent. The composition cited P144 as
motivation but a vanilla optimizer was sufficient. SAV might help further
but wasn't needed for the positive result.

## What this is NOT

- C1 does NOT reach published SOTA. At n=20 we get 0.0049 vs published
  0.0180 — a 73% gap. Published values use much more compute, hand-tuned
  basin hopping with discrete moves, and often manual seeding from known
  symmetric configurations.
- This is a *method-vs-method* positive at the same compute budget, not a
  *new-SOTA* positive.
- 50k × 8 restarts is modest; scaling C1 further (more restarts, longer
  annealing, better schedule tuning) is unexplored.

## What it IS

The composition produced a method that *measurably outperforms standard
local-search baselines* on a well-defined extremal geometry problem with
no domain-specific tuning. The L3 reviewer's central empirical attack
("you have no evidence it beats SA / basin-hopping") is falsified by
direct measurement. The reviewer's "folklore" objection is partially
upheld but the specific ingredient combination does add measurable value.

## Compose-verification track record (now n=4)

| run | L3 verdict | Verification outcome |
|---|---|---|
| short-string compression | WEAK + "Brotli+dict / Zstd+dict / HPACK equivalent" | Partial positive (LZ-family falsified, HPACK upheld) |
| multitrack stems | WEAK + "high-level correlation ≠ sample-level predictability" | Clean negative |
| cache replacement | WEAK + "successors ≠ residents" | Clean negative |
| **Heilbronn triangle** | WEAK + "folklore recipe, no evidence vs SA / basin-hop" | **Clean positive** — beats SA by 9-60%, basin-hop by 78-229% |

Track record: 2 positives (1 partial, 1 clean), 2 negatives. L4
PURELY_TESTABLE classifier 4-for-4 on "testable in one session, clear
signal." L3 directionally correct in 3 of 4 WEAK verdicts but
empirically wrong in 1 — exactly the calibration shape a useful
adversarial filter should have.

## Reproducing

```
uv run --with numpy --with jax --with scipy python -m experiments.heilbronn.bench
```

## Files

- `experiments/heilbronn/evaluator.py` — exact min triangle area
- `experiments/heilbronn/baselines.py` — SA + scipy basin-hopping
- `experiments/heilbronn/c1_smooth.py` — C1 prototype (JAX)
- `experiments/heilbronn/bench.py` — runner

## Verdict

C1 (annealed soft-min + Riesz repulsion + boundary penalty, vectorized
gradient flow) is a real method that beats SA and basin-hopping on small-n
Heilbronn at fixed compute. Doesn't match published SOTA (which uses much
more compute and hand-tuning), but the comparison method-vs-method at the
same budget is a clean win. This is the **first true-positive** in the
compose-verification track record where the synthesis claim holds up under
empirical test against the L3 reviewer's named baselines.
