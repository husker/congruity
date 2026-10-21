# C2 Discrete Geometry Verification: Delaunay-Local Active Repair

- Originating run: `20260525-042258`
- Composition: `C2`
- Outcome: **partial-positive**
- Sizes tested: 10, 14, 18, 22
- Seeds per size: 3
- Search steps per run: 350

## Question

Can a Delaunay-local, force-like repair loop improve finite point layouts, or does it optimize the wrong local surrogate and miss the true worst near-collinear triples?

## Method

The objective is the Heilbronn-style score: maximize the minimum triangle area over all triples of points in the unit square. The benchmark compares:

- `SA-full`: simulated annealing using full exhaustive triangle scans.
- `C2-full`: the same active-repair move generator, but with all triples visible.
- `C2-delaunay`: active repair using only Delaunay cell triangles.
- `C2-augmented`: Delaunay cells plus Delaunay one-ring triples, k-nearest-neighbor triples, and periodic global scans that inject missed bad triples.

This directly tests the L3 objection: pure Delaunay locality may not see the actual combinatorial bottleneck for no-near-collinearity.

## Results

| method | mean best min-triangle area | mean seconds | mean active-set recall | mean candidate triples |
|---|---:|---:|---:|---:|
| SA-full | 0.00162735 | 0.087 | 1.000 | 710.0 |
| C2-full | 0.00623924 | 2.149 | 1.000 | 710.0 |
| C2-delaunay | 0.00145011 | 0.365 | 0.005 | 23.7 |
| C2-augmented | 0.00621805 | 2.028 | 0.727 | 384.4 |

## Active-Set Recall Stress Test

| candidate set | top-5 recall | candidate count / full triples |
|---|---:|---:|
| Delaunay cells | 11.7% | 5.3% |
| Augmented C2 | 88.3% | 63.4% |

## Verdict

C2 is useful but not fully proven. The pure Delaunay version confirms the reviewer's warning, while the augmented loop recovers much more of the true bottleneck set and is competitive enough to justify a stronger follow-up with CMA-ES, differential evolution, and larger n.

## What This Says About the Original Claim

- The most important issue is not Delaunay geometry itself; it is whether the candidate set sees the actual bottleneck triples.
- Pure Delaunay cells are too narrow for no-near-collinearity-style objectives.
- Periodic global scans are not optional. They are the control loop that keeps the cross-domain idea honest.
- The next decisive test is a larger head-to-head against DE, CMA-ES, and basin hopping using equal wall-clock budgets.

## Reproduce

```bash
uv run python -m experiments.discrete_geometry_c2.bench --write-findings
```
