# Small Points Optimization Gotchas

- Recompute the objective from emitted coordinates; never trust a method's own
  reported score.
- Compare against tuned optimizers, not default `scipy.optimize.basinhopping`.
- Use multiple fixed seeds and report mean plus spread.
- Treat the smoke command as a quick CLI health check; use `full_command` for
  candidate claims.
- A surrogate objective win is not a result unless it improves the true metric.
- Published records may be for subtly different objectives or constraints.
- Function-evaluation budgets are a leaky fairness unit: a candidate whose
  inner loop has heavy per-step overhead (e.g. SLSQP polish) can take 700x
  CMA-ES's wall-clock at the same eval budget (measured 2026-06-12: 6s vs
  75min at 240k evals, n=11). Size confirm commands by measured wall-clock,
  and treat wall-clock-normalized comparisons as the stricter check.
