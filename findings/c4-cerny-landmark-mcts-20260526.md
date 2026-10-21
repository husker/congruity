# Černý C4 verification — landmark-biased MCTS reset solver

- Originating run: `20260526-062120`
- Composition: `C4`
- Outcome: **partial-negative**
- n values: `[7, 8, 9, 10]`
- instances per n: `4`
- MCTS iterations: `350`
- rollout depth factor: `4`

## Claim Tested

C4 claimed that MCTS plus DFA-specific planning landmarks and bounded certification could find short reset words better than unguided search. This harness uses pair-merge landmarks as the planning facts and a bounded exact-BFS certificate as the local substitute for SAT checks.

## Result

Landmark MCTS was competitive with plain MCTS, but it did not beat the strong greedy pair-compression baseline.

| variant | mean optimality gap | optimal rate | mean elapsed s |
|---|---:|---:|---:|
| greedy_pair | 1.800 | 0.267 | 0.001 |
| plain_mcts | 5.333 | 0.600 | 0.016 |
| landmark_mcts | 4.733 | 0.667 | 0.045 |

## Key Numbers

- Greedy mean optimality gap: `1.800`
- Plain MCTS mean optimality gap: `5.333`
- Landmark MCTS mean optimality gap: `4.733`
- Landmark MCTS optimal rate: `0.667`
- Instances tested: `15`

## Representative Cases

| instance | optimal | greedy | plain MCTS | landmark MCTS |
|---|---:|---:|---:|---:|
| cerny-10 | 81 | 87 | 103 | 103 |
| cerny-9 | 64 | 69 | 84 | 84 |
| cerny-8 | 49 | 52 | 67 | 61 |
| cerny-7 | 36 | 38 | 52 | 52 |
| random-cycle-8-0 | 13 | 14 | 13 | 14 |
| far-seed-9 | 11 | 11 | 13 | 11 |
| random-cycle-10-1 | 10 | 11 | 12 | 10 |
| random-cycle-9-0 | 10 | 11 | 10 | 10 |

## L3 Attacks Checked

| L3 attack | Verdict from this run |
|---|---|
| Landmarks may be vacuous | Pair-merge landmarks are necessary reset obligations, but their guidance largely reduces to pair-compression progress. |
| MCTS may be mismatched to deterministic shortest-word search | Tested directly against greedy and plain MCTS under fixed rollout budgets. |
| SAT checks may only certify | The bounded exact-BFS certificate validated optimality gaps; it did not feed back into MCTS discovery. |

## Verdict

This is a local algorithmic benchmark, not a theorem about reset-word search. A stronger C4 would need landmarks that add information beyond ordinary pair-compression and a genuine SAT/UNSAT feedback path.

## Artifacts

- `experiments/cerny_c4/bench.py` — landmark extraction, solvers, exact certificates
- `tests/test_cerny_c4.py` — regression tests for landmarks and benchmark validation
