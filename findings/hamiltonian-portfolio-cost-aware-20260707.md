# Hamiltonian solver-portfolio — cost-aware follow-up (open thread closed)

**Date:** 2026-07-07
**Follows:** `findings/c1-hamiltonian-regime-classifier-20260524.md`
(partial-positive — oracle-supported, cheap Ridge learner falsified at −84%)
**Bench:** `experiments/hamiltonian_regime_classifier/{collect,portfolio}.py`
**Verdict:** **negative for practical value — the transferred mechanism
(topology → solver choice) has real signal and real oracle headroom, but
no cheap realization beats the tail-bounded single solver. Not a transfer
win.**

## What this tests

The original finding left an explicit open thread: the naive Ridge
selector lost 84% because it used a *symmetric* loss while the per-instance
costs are wildly asymmetric (a mispredicted DFS timeout wipes many correct
picks). It named four candidate fixes. This follow-up tests the two cheap,
on-target ones — **path #1 cost-aware learning** and **path #3 safe-default
+ speculative escape-hatch** — against the project's actual bar: does a
distant-field *mechanism* beat the home-field best single solver?

(Path #2, racing, was considered and rejected before building: racing
*abandons* the transferred classifier and "wins" only by 2× CPU brute
force, so a racing win would be evidence the classifier is *unnecessary* —
against the composition, not for it. It also only re-measures
`oracle + spawn_overhead`, computable from solo timings.)

## Method

`collect.py` solves both CaDiCaL (permutation HC SAT encoding, PySAT) and a
plain DFS backtracker once over 270 instances (ER + Watts-Strogatz + random
regular, n ∈ {15,18,21,24} spanning the Hamiltonicity threshold, timeout
4 s, seed 42) and saves solo wall-clock + 14 topology features. **0
solver disagreements** (both agree on HC existence — correctness sanity
passes), so selection is purely a speed question. `portfolio.py` then
evaluates selection policies offline over the recorded timings (5-fold CV);
no re-solving, no parallel infra.

## Results (N = 270)

Solo timing structure — the whole story is in the tails:

| solver | total | median | p99 | max | timeouts |
|---|---|---|---|---|---|
| CaDiCaL | 2.641 s | 0.00025 s | 0.126 s | **0.412 s** | 0 |
| DFS | 46.33 s | 0.00028 s | 4.000 s | **4.000 s** | 7 |

**Mutuality — occurrence is mutual, absolute catastrophe is one-sided:**
- occurrence (≥2× faster): CaDiCaL on 94 instances, DFS on 123 — both
  solvers genuinely win a large share (mutual complementarity).
- absolute catastrophe (≥0.5 s): CaDiCaL **0**, DFS **18**. CaDiCaL
  rescues all 18 DFS catastrophes; DFS rescues 0 CaDiCaL catastrophes
  (there are none). **CaDiCaL is the unambiguous safe default.**

This refines the original's implicit "one-sided" reading: the solvers *do*
trade wins constantly, but only DFS ever hits an absolute disaster, so the
cost the portfolio must respect is one-sided.

**Selection policies — single-core / total-CPU metric** (best-single is
the honest bar; nothing running one solver per instance can beat it by
running two):

| policy | total | gain vs best-single |
|---|---|---|
| best-single (CaDiCaL) | 2.641 s | 0.00% (the bar) |
| oracle (unreachable) | 2.144 s | +18.82% |
| Ridge symmetric | 3.146 s | **−19.09%** (reproduced failure) |
| cost-aware Ridge (safety margin) | 2.631 s | **+0.40%** |
| cost-aware GBM (safety margin) | 2.641 s | **+0.00%** |

Stable across CV seeds {1,7,42,123}: Ridge symmetric always loses (−4% to
−35%); **cost-aware (both linear Ridge and nonlinear gradient boosting)
always lands at breakeven (+0.00% to +0.40%), never a meaningful win.**
With a safety margin, both learners correctly learn to *almost always pick
CaDiCaL* — because the diffuse DFS occurrence-wins (tiny savings) are not
worth the risk of the 18 DFS catastrophes, and topology features don't
separate "DFS fast" from "DFS catastrophe" sharply enough to change that.

**2-core latency metric** (safe-default + speculative-DFS, path #3): always
run CaDiCaL, also run DFS on gated instances (CaDiCaL always backstops, so
never exposed to a DFS timeout): latency 2.509 s = **+5.01%** vs
best-single, but total-CPU 3.122 s = **−18.21%** (the 2× price). Only pays
off if single-instance latency matters and a spare core is free.

## Verdict

**The composition's structural premise holds** — topology features predict
hardness (ΔR² ≈ +0.28, reproduced) and an oracle portfolio has real
headroom (+18.82%). **But its practical value does not:** no cheap
predict-then-commit realization of "classify topology, then choose solver"
beats just running CaDiCaL, because the headroom is dominated by DFS's
one-sided catastrophe risk and the safe response is to always pick the
tail-bounded solver. Cost-aware learning fixes the −84% catastrophe of the
symmetric learner but only reaches breakeven — it recovers *CaDiCaL*, not a
win. The practical answer is **"just use CaDiCaL."**

This closes the open thread as a **clean negative for the transfer-win
question**, while confirming and sharpening the original diagnosis:

1. Cost asymmetry was the right diagnosis — symmetric Ridge robustly loses.
2. The proposed fix (cost-aware learning) was tested and reaches only
   breakeven; it does not rescue the composition.
3. The headroom is occurrence-mutual but catastrophe-one-sided; the portfolio's
   only genuine payoff is *safe speculation on a spare core* (+5% latency,
   2× CPU), which does not reduce total work and does not use the learned
   classifier for anything a fixed "speculate when idle" rule couldn't do.

Meta-observation carried forward (reinforces the original): **predict-then-
commit algorithm selection in a domain with one-sided catastrophe cost is
structurally hard to make pay — the safe default dominates, and the
learner's best move is to reproduce it.** The value, if any, is in
*safe hedging* (speculation with a backstop), not *selection*.

## Reproducibility

```
uv run --with python-sat --with networkx --with scikit-learn python \
    -m experiments.hamiltonian_regime_classifier.collect \
    --n-per-cell 6 --timeout 4 --out data/hamiltonian/timings.json
uv run --with scikit-learn --with networkx python \
    -m experiments.hamiltonian_regime_classifier.portfolio \
    --data data/hamiltonian/timings.json --save data/hamiltonian/portfolio_result.json
```

Windows note: force `PYTHONUTF8=1` (the original bench prints a `Δ` that
crashes on cp1252 consoles). Solving ~1 min; analysis a few seconds.
