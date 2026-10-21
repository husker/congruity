# Solver-Portfolio C1 Verification

- Originating run: `20260525-011650`
- Composition: `C1`
- Outcome: **partial-positive**
- Instances: 160
- Cross-validation folds: 5

## Question

Can cheap structural fingerprints of routing instances help choose a solver strategy better than simple baselines?

## Method

The harness generated local Euclidean TSP-style instances across uniform, clustered, line-like, grid-like, and ring-like families. Each instance was run through seven deterministic heuristics: nearest neighbor, angle sweep, axis sweep, farthest insertion, cheapest insertion, light 2-opt, and fuller 2-opt. The selector loss charged both tour-length gap and a normalized effort penalty of 0.04.

Selectors compared:

- single best solver from the training fold
- generator-family selector
- direct supervised random-forest selector
- density-peak cluster-first selector

## Results

| selector | mean regret vs oracle | regret reduction vs single-best |
|---|---:|---:|
| single best | 0.0116 | 0.0% |
| family | 0.0024 | +79.6% |
| direct supervised | 0.0066 | +42.2% |
| density cluster | 0.0118 | -6.4% |

- Density-cluster best-solver purity: 78.6%
- Random-cluster best-solver purity: 71.9%
- Density-cluster loss variance: 0.10679
- Random-cluster loss variance: 0.14970

## Interpretation

This is a partial result. The generated feature space contains solver-selection signal, but the density-cluster mechanism is not yet the strongest way to use it. C1 remains interesting, but the next version should treat clustering as an explanation layer or feature rather than the primary selector.

## Reproduction

```bash
uv run python -m experiments.solver_portfolio_c1.bench --effort-penalty 0.04 --write-findings
```
