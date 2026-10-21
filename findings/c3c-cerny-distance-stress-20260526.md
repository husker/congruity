# Černý C3c verification — distance stress test

- Originating run: `20260526-062120`
- Composition: `C3c` follow-up
- Outcome: **clean-negative**
- n values: `[8, 9, 10]`
- distance thresholds: `[4, 6]`
- trials per family: `2`
- budget per family/trial: `420` exact evaluations

## Claim Tested

C3c asks whether the C3b signal survives after forcing candidates out of the immediate Černý neighborhood. Candidates must still be strongly connected, synchronizing, and free of closed-core attachments.

## Result

The near-Černý advantage collapses under distance stress: once candidates must be farther from Černý, far cycle evolution no longer beats the baselines.

| threshold | family | best reset length | mean acceptance yield | mean elapsed s |
|---:|---|---:|---:|---:|
| distance >= 4 | far_random | 30 | 0.996 | 0.180 |
| distance >= 4 | far_hill_climb | 33 | 0.980 | 0.292 |
| distance >= 4 | far_seeded_hill_climb | 34 | 0.968 | 0.289 |
| distance >= 4 | far_cycle_evolution | 33 | 0.973 | 0.264 |
| distance >= 6 | far_random | 30 | 0.971 | 0.183 |
| distance >= 6 | far_hill_climb | 35 | 0.961 | 0.296 |
| distance >= 6 | far_seeded_hill_climb | 35 | 0.952 | 0.286 |
| distance >= 6 | far_cycle_evolution | 36 | 0.956 | 0.254 |

## Key Numbers

- Best far evolutionary reset length: `36`
- Best far baseline reset length: `35`
- Minimum evolutionary delta across thresholds: `-1`
- Threshold deltas: `d>=4:-1, d>=6:1`
- Mean acceptance yield: `0.969`

## Best Far Candidate

- Family: `far_cycle_evolution`
- n states: `10`
- distance threshold: `distance >= 6`
- seed: `20364736`
- exact reset length: `36`
- exact BFS expanded subsets: `1014`
- transition edits from formula Černý table: `8`
- reset witness: `100010100000010100001110000110100001`

The transition table uses rows as states and columns as input letters:

| state | letter 0 | letter 1 |
|---:|---:|---:|
| 0 | 1 | 4 |
| 1 | 2 | 3 |
| 2 | 3 | 0 |
| 3 | 4 | 1 |
| 4 | 5 | 0 |
| 5 | 6 | 8 |
| 6 | 7 | 5 |
| 7 | 8 | 7 |
| 8 | 9 | 6 |
| 9 | 0 | 9 |

## Verdict

C3c is not a proof of a new Černý-family construction. It tests whether the generator's value remains after a simple structural novelty bar. A next step would need an interpretable parametrized family or a much stronger distance-from-Černý requirement.

## Artifacts

- `experiments/cerny_c3c/bench.py` — distance-threshold benchmark
- `tests/test_cerny_c3c.py` — regression tests for the distance gate
