# Graph Coloring Gotchas

- Recompute conflicts from the emitted assignment; never trust a solver's
  reported color count or "valid" flag.
- Compare at an equal iteration budget. TabuCol with 10x the iterations is a
  different bar, and a faster inner loop is an engineering win, not an
  algorithmic one.
- Beating greedy or DSATUR means nothing; TabuCol is the bar. Most published
  "improvements" lose to plain TabuCol at equal budget.
- Test across all instance families. G(n,0.5) tuning often regresses on queen
  graphs (structured) and planted instances (known chromatic upper bound).
- Planted k-partite instances bound the chromatic number from above by
  construction — a method that cannot recover the planted count is weak.
- Published DIMACS best-known values are for the exact benchmark files, often
  with hours of CPU; do not compare seeded generator output against them.
