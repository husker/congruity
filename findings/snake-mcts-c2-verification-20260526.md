# Snake-in-the-Box C2 Verification

- Originating run: `20260526-043036`
- Composition: `C2`
- Outcome: **partial-positive**
- Dimensions: 6, 7, 8
- Seeds per dimension: 8

## Question

Does a novelty/metadynamic MCTS policy find longer exact snake paths than plain MCTS or greedy randomized restarts under matched local compute?

## Method

The harness builds partial paths in hypercube graphs from vertex `0`. Every candidate is checked as an induced path: consecutive vertices must differ in one bit, vertices may not repeat, and nonconsecutive vertices may not be hypercube neighbors. The C2 variant uses endpoint mobility, occupied fraction, and forbidden-neighborhood density as coarse descriptor bins; repeated bins receive a metadynamic penalty while rare bins receive a novelty bonus.

Compared variants:

- greedy randomized restarts with onward-mobility weighting
- vanilla UCT MCTS with random valid rollouts
- novelty/metadynamic MCTS with the same iteration budget

## Results

| variant | mean best vertices | median | max |
|---|---:|---:|---:|
| greedy | 44.458 | 41.000 | 71.000 |
| plain_mcts | 45.708 | 42.000 | 73.000 |
| novelty_metadynamic_mcts | 46.083 | 43.500 | 73.000 |

- Mean novelty delta vs plain MCTS: 0.375 vertices
- Mean novelty delta vs best baseline per seed: 0.208 vertices
- Novelty win rate vs plain MCTS: 0.500
- Novelty win rate vs best baseline: 0.500

## Interpretation

This is a partial result. The novelty/metadynamic mechanism improved on plain MCTS, but it did not clearly beat the strongest cheap baseline. C2 remains interesting only if the next pass adds stronger descriptors or domain propagation.

## Reproduction

```bash
uv run python -m experiments.snake_mcts_c2.bench --write-findings
```
