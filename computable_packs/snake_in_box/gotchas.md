# Gotchas — snake_in_box

- **Length = EDGES, not vertices.** A sequence of k+1 vertices has length k.
  Off-by-one here is the most common scoring bug.
- **The skin condition is on non-consecutive pairs.** It is NOT enough that
  consecutive vertices differ by one bit; no vertex may be Hamming-adjacent
  to any path vertex other than its immediate neighbors. The validator
  checks all pairs.
- **Start canonically at [0, 1].** Q_n is edge-transitive; fixing the first
  edge collapses the symmetry group with no loss of generality.
- **Charge your budget honestly.** 1 unit per successor generated/scored.
  Baselines are budget-matched; exceeding it is unfair, undershooting
  wastes headroom.
- **Do not hardcode literature snakes.** The holdout uses unseen seeds and
  identical output across seeds is treated as suspicious; the value comes
  from the SEARCH METHOD, not a cached sequence.
- **Snake (open path) != coil (closed cycle).** This pack scores snakes.
