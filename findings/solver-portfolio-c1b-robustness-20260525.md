# Solver-Portfolio C1b Robustness Verification

- Originating run: `20260525-011650`
- Parent composition: `C1`
- Outcome: **clean-negative**
- Base-family instances: 160
- Hybrid holdout instances: 160

## Question

Did C1 discover a structural route-instance signal, or did it mostly learn obvious generator labels such as grid, ring, and clustered?

## Method

C1b withholds whole families from training. It uses leave-one-family-out tests on the base generators, then trains on all base generators and tests on unseen hybrid generators: clustered rings, jittered grids, line-plus-cluster mixtures, anisotropic blobs, and compact clusters with outliers.

The main selector is cost-sensitive: it predicts each solver's expected loss and chooses the solver with the lowest predicted loss. This is tested against single-best, direct winner classification, and density-cluster selection.

## Results

| split | single-best regret | direct ML | density cluster | cost-sensitive |
|---|---:|---:|---:|---:|
| leave_out:clustered | 0.0020 | 0.2124 | 0.2657 | 0.0020 |
| leave_out:grid | 0.0025 | 0.0025 | 0.0025 | 0.0025 |
| leave_out:line | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| leave_out:ring | 0.0477 | 0.0477 | 0.0477 | 0.0477 |
| leave_out:uniform | 0.0057 | 0.1134 | 0.0057 | 0.0057 |
| hybrid_holdout | 0.0069 | 0.0156 | 0.0069 | 0.0087 |

- Mean cost-sensitive regret reduction vs single-best: -4.1%
- Leave-one-family-out cost reduction: +0.0%
- Hybrid-holdout cost reduction: -24.7%

## Most Useful Structural Features

- `cov_anisotropy`: 0.2241
- `pair_dist_cv`: 0.1910
- `nn_mean`: 0.1066
- `mst_edge_max`: 0.0911
- `centroid_radius_std`: 0.0851
- `mst_edge_mean`: 0.0625
- `mst_length`: 0.0529
- `nn_p90`: 0.0498

## Interpretation

This weakens C1. Once whole families or hybrid families are withheld, the tested structural fingerprints do not improve enough over the single-best baseline. The first result was likely too dependent on easy generator-family cues.

## Reproduction

```bash
uv run python -m experiments.solver_portfolio_c1b.bench --write-findings
```
