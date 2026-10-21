# Černý C3 verification — evolutionary hard-automata generator

- Originating run: `20260526-062120`
- Composition: `C3`
- Outcome: **partial-positive**
- n values: `[6, 7, 8]`
- alphabet size: `2`
- trials per family: `4`
- budget per stochastic family/trial: `220` exact evaluations

## Claim Tested

C3 claimed that mutation-style generation plus multi-objective evolutionary selection could produce harder synchronizing DFAs than ordinary random sampling or simple local mutation, when every candidate is scored with an exact reset-word oracle.

## Result

The evolutionary generator found a non-Cerny candidate that beat the best random or hill-climb baseline by at least one reset step.

| family | best reset length | best novel reset length | mean sync yield | mean elapsed s |
|---|---:|---:|---:|---:|
| cerny_seed | 49 | n/a | 1.000 | 0.000 |
| random_sampling | 28 | 28 | 0.878 | 0.021 |
| hill_climb | 29 | 29 | 0.914 | 0.026 |
| pareto_evolution | 49 | 36 | 0.766 | 0.035 |

## Key Numbers

- Best random reset length: `28`
- Best hill-climb reset length: `29`
- Best known Černý-family reset length in this benchmark: `49`
- Best evolutionary reset length including seeds: `49`
- Best evolutionary novel/non-Černý reset length: `36`
- Novel evolutionary delta vs best non-Černý baseline: `7`
- Mean synchronizing yield: `0.853`

## Best Novel Candidate

- Family: `pareto_evolution`
- n states: `8`
- seed: `20340630`
- exact reset length: `36`
- exact BFS expanded subsets: `123`
- transition edits from formula Černý table: `2`
- reset witness: `100000010000001000000100000010000001`

The transition table uses rows as states and columns as input letters:

| state | letter 0 | letter 1 |
|---:|---:|---:|
| 0 | 1 | 1 |
| 1 | 2 | 1 |
| 2 | 3 | 2 |
| 3 | 4 | 3 |
| 4 | 5 | 4 |
| 5 | 6 | 5 |
| 6 | 0 | 6 |
| 7 | 2 | 7 |

Caveat: this length equals the Černý threshold for one fewer state. The transition table should be treated as a useful hard test case, but it may be a lower-dimensional Černý-like core with an extra state attached rather than a genuinely new extremal construction.

## L3 Attacks Checked

| L3 attack | Verdict from this run |
|---|---|
| Trivial recombination | Tested directly against random sampling and single-mutation hill climbing. |
| Exact scoring bottleneck | Exact BFS was cheap at this scale, but that does not remove the scaling risk for n=10-12. |
| Mutation operators underspecified | The harness used repair-by-rejection, merge-biased edits, duplicate suppression, and a novelty objective. |
| Risk of rediscovering Černý | Tracked separately with distance from the formula-generated Černý automaton and a novel-candidate metric. |

## Verdict

This is a bounded local test, not a proof about the full conjecture. A true discovery signal would be a reproducible non-isomorphic automaton family with unusually high exact reset thresholds, or a generator that reliably beats random/local baselines without leaning on the known Černý seed.

## Artifacts

- `experiments/cerny_c3/bench.py` — exact oracle, generators, benchmark, findings writer
- `tests/test_cerny_c3.py` — oracle and benchmark regression tests
