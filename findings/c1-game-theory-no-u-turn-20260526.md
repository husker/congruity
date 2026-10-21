# C1 game-theory verification: no-u-turn restart dynamics

Originating compose run: `20260526-205855` `C1`.

## Verdict

**clean-negative**.

C1 predicted that a finite-game no-u-turn/cycle trigger would improve 
no-regret or extragradient dynamics by restarting or regularizing from 
the best Nash residual point. This harness implements the cheap local 
version: all solvers receive the same game instances, iteration budget, 
and step-size sweep; C1 must beat the strongest tuned baseline on 
NashConv rather than just beat an undertuned fixed run.

## Benchmark

- Families: `matching_pennies, cyclic, coordination, near_zero_sum, random_general_sum`
- Sizes: `[20, 40]`
- Seeds per size/family: `[0, 1, 2, 3, 4, 5]`
- Iterations per solver: `450`
- Step sizes swept: `[0.02, 0.05, 0.1]`
- Instances: `60`
- Primary metric: best NashConv/exploitability observed during the run
- Baselines: multiplicative weights, regret matching, projected gradient, 
  extragradient, and generic residual restart

## Overall Results

| solver | mean best | median best | mean final | triggers | seconds |
|---|---|---|---|---|---|
| multiplicative weights | 0.0580 | 0.0321 | 0.0968 | 0.00 | 0.0388 |
| regret matching | 0.0624 | 0.0284 | 0.6940 | 0.00 | 0.0312 |
| projected gradient | 0.0221 | 0.0205 | 0.4009 | 0.00 | 0.0610 |
| extragradient | 0.0094 | 0.0089 | 0.1263 | 0.00 | 0.1073 |
| generic residual restart | 0.0367 | 0.0335 | 0.0405 | 10.60 | 0.0661 |
| C1 no-u-turn restart | 0.0221 | 0.0205 | 0.2826 | 1.00 | 0.0738 |

Best baseline by mean best NashConv: `extragradient` 
at `0.0094`.
C1 mean ratio vs best baseline: 
`2.362`.
C1 instance win rate: `0.333`.

## Family Breakdown

| family | C1 mean best | best baseline | C1 ratio |
|---|---|---|---|
| coordination | 0.0000 | 0.0000 | 1.000 |
| cyclic | 0.0172 | 0.0171 | 1.005 |
| matching_pennies | 0.0292 | 0.0113 | 2.591 |
| near_zero_sum | 0.0491 | 0.0074 | 6.645 |
| random_general_sum | 0.0149 | 0.0109 | 1.373 |

## Interpretation

The C1 trigger lost clearly to the strongest tuned baseline. This falsifies the cheap version of the composition: on these local generated games, trajectory reversal detection did not add useful equilibrium-search signal beyond standard first-order/no-regret dynamics and generic restart.

## Reproducibility

```bash
uv run python -m experiments.game_theory_c1.bench --write-findings
uv run pytest tests/test_game_theory_c1.py -q
```
