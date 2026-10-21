# Heilbronn record attempt #1 — strong vs SOTA, ~91% of the 1972 record

_2026-06-12. Autosearch on composition C2 of run `20260525-042258` (Delaunay
local-cell force relaxation, compose confidence 9/10 — the highest on file)
against `small_points_optimization` with the cached record table. Ledger
label: **partial, valid, confidence medium** — the first valid outcome in a
record domain._

## Search phase (train: n=11,13; 80k evals; seeds 1-2)

Ten candidates over four generations; fitness = mean-area delta vs equal-eval
CMA-ES. The loop climbed monotonically for the first time on any domain:
+0.0085 → +0.0094 → +0.0109 (winner `g2_e1.py`, 376 lines: Halton/border
seeding → active-set force relaxation over the currently-smallest triangles
with Delaunay neighborhoods → active-set SLSQP polish; implements
`solve_ablated`). Deterministic: the seed re-evaluation reproduced
+0.010886 exactly, twice.

## Confirmation (holdout seed 101; n=11,13; 80k evals)

| | n=11 | n=13 | mean |
|---|---|---|---|
| candidate (recomputed from coords) | 0.03182 | 0.01950 | 0.02566 |
| CMA-ES (equal evals) | — | — | 0.01647 |
| best-known record | 0.03704 (Goldberg 1972) | 0.0270 (Karpov 2011) | |

- vs SOTA: **+0.0092** — far above CMA-ES, but graded "partial" because the
  legacy recipe's min-effect parses to an absurd 20.0 (recipe predates the
  schema fix; regenerate before the next run).
- vs records: **below-record at both** (auto-compared): −0.0052 at n=11,
  −0.0075 at n=13. Best validated area ever achieved in-search: 0.03363 at
  n=11 (seeds 1-2) — **90.8% of the 54-year-old record**.
- Ablation: probe at 240k evals showed the active-set mechanism contributing
  +0.0015 over its own ablation; the confirm at 80k/seed-101 showed
  +0.00004 — i.e. the mechanism's marginal value is budget-dependent and most
  of the edge over CMA-ES comes from the overall seed-and-polish pipeline.

## Infrastructure findings (all fixed + committed)

1. **Eval budgets are a leaky fairness unit**: at the same 240k evals,
   CMA-ES took 6 seconds and the candidate 75 minutes (SLSQP per-step
   overhead). Two confirmation timeouts (1800s, 5400s) were this, not hangs.
   Confirm commands must be sized by measured wall-clock.
2. **Timeout left orphans**: `subprocess.run` kills `uv run` but not the
   bench python (one grandchild survived 6+ hours). execute_pack_bench now
   kills the whole process group.
3. The gates never lied: two timeouts → two quarantined-invalid entries,
   zero invented labels.

## Next steps for an actual record

- Give the winner a serious budget run on n=11 only (hours of wall-clock,
  many seeds/restarts) — the gap is 0.0034 from the best validated config.
- Regenerate C2's recipe under the new schema so the benchmark grade can be
  earned honestly.
- The points files persist coordinates per claim; any future `new-record`
  outcome should be hand-checked against Friedman's page before claiming.
