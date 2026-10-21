# Černý C3b verification — strict hard-automata generator

- Originating run: `20260526-062120`
- Composition: `C3b` follow-up
- Outcome: **partial-positive**
- n values: `[8, 9, 10]`
- trials per family: `3`
- budget per stochastic family/trial: `360` exact evaluations

## Claim Tested

C3b asks whether the C3 evolutionary idea still works after rejecting closed-core attachments and requiring strong connectivity. This is a stricter discovery test than the original C3 run.

## Result

The strict evolutionary generator beat both strict random sampling and strict hill climbing after excluding closed-core attachments.

| family | best strict reset length | mean strict yield | mean elapsed s |
|---|---:|---:|---:|
| cerny_reference | 81 | 1.000 | 0.001 |
| strict_random | 20 | 0.131 | 1.200 |
| strict_hill_climb | 35 | 0.693 | 0.146 |
| strict_cycle_evolution | 65 | 0.971 | 0.238 |

## Key Numbers

- Best strict random reset length: `20`
- Best strict hill-climb reset length: `35`
- Best strict evolutionary reset length: `65`
- Best Černý reference reset length: `81`
- Strict evolutionary delta vs best strict baseline: `30`
- Mean strict candidate yield: `0.598`

## Best Strict Novel Candidate

- Family: `strict_cycle_evolution`
- n states: `10`
- seed: `20360842`
- exact reset length: `65`
- exact BFS expanded subsets: `1013`
- transition edits from formula Černý table: `2`
- reset witness: `10000000100000001000000010000000100000001000000010000000100000001`

The transition table uses rows as states and columns as input letters:

| state | letter 0 | letter 1 |
|---:|---:|---:|
| 0 | 1 | 0 |
| 1 | 2 | 4 |
| 2 | 3 | 2 |
| 3 | 4 | 3 |
| 4 | 5 | 4 |
| 5 | 6 | 5 |
| 6 | 7 | 6 |
| 7 | 8 | 7 |
| 8 | 9 | 8 |
| 9 | 0 | 9 |

Caveat: this is a strict candidate, but it is still near-Černý: the best table differs from the formula Černý table by only `2` transition(s). This supports a stronger hard-instance generator claim, not yet a new structural family claim.

## Structural Filters

- Strong connectivity is required, so one-way attached states are rejected.
- Proper closed subautomata with the same reset length are rejected as closed-core attachments.
- Exact state/letter relabelings of the formula Černý automaton are not counted as novel.

## Verdict

C3b is a stricter local test, not a proof about the full conjecture. A compelling next result would show a repeatable family of strongly connected non-Černý automata whose reset ratios scale better than random or local mutation baselines.

## Artifacts

- `experiments/cerny_c3b/bench.py` — strict structural filters and benchmark
- `tests/test_cerny_c3b.py` — regression tests for the stricter novelty definition
