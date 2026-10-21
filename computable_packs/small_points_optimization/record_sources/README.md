# Record sources — small_points_optimization (Heilbronn, unit square)

Fetched 2026-07-06 from Peter Karpov's Ascension page
(http://inversed.ru/Ascension.htm), which publishes the coordinate files
behind the n=13 and n=15 best-known entries on Erich Friedman's Heilbronn
page (verified the same day: Friedman still lists 0.0270+ / 0.0211+,
Karpov, August 2011; no 2025-2026 improvement found — arXiv:2512.14505
is certification work up to n=9, not new best-knowns).

| file | n | Karpov's score (file header) | re-measured via `experiments.heilbronn.evaluator.min_triangle_area` |
|---|---|---|---|
| `Heilbronn_S13.txt` | 13 | 2.70192722391139E-2 | 0.027019272239114 (delta +6.9e-18) |
| `Heilbronn_S15.txt` | 15 | 2.11056818784151E-2 | 0.021105681878415 (delta +2.6e-16) |

- Source URLs: http://inversed.ru/Ascension/Heilbronn_S13.txt ,
  http://inversed.ru/Ascension/Heilbronn_S15.txt (verbatim copies here).
- The `.json` siblings are machine-readable versions used by tests and
  the exact-polish calibration harness.
- Structure notes (measured): S13 has 6/13 points on the boundary and 16
  triangles within 1e-6 relative of the min (22 within 1%); S15 has 7/15
  boundary points, 18 tight (24 within 1%). Max-min optima are heavily
  degenerate — polish must equalize the full active set.
- Record semantics: a NEW RECORD must strictly exceed the STATIONARY
  values of these configurations under `experiments.heilbronn.
  exact_polish` — n=13: 0.027019916669111 (+6.4e-7 over the file),
  n=15: 0.021106168589172 (+4.9e-7) — which is what `records.json` now
  carries as `value`. Rationale: a claim between Karpov's file score and
  its polished stationary value is not defensible, because re-polishing
  his own file beats it. The published 0.0270+/0.0211+ are truncations
  of the file scores.
