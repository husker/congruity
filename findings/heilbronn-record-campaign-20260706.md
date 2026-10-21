# Heilbronn record campaign — day 1: exact polish, verified bars, basin-hop miner

_2026-07-06. Resumes the record push from
`findings/heilbronn-record-attempt-20260618.md` ("beating the record
needs a stronger candidate, not more seeds"). This log is the campaign's
chronological record; the durable machinery is
`experiments/heilbronn/exact_polish.py`, `calibrate_exact_polish.py`,
`polish_hopper.py`._

## M0 — record targets live-verified (and sharpened)

- Friedman's Heilbronn page re-checked 2026-07-06: n=13 = 0.0270+ and
  n=15 = 0.0211+ (both Karpov, Aug 2011, both "not symmetric") still
  stand; n=14/16 still Beyleveld 2006. No 2025-2026 improvement found.
  arXiv:2512.14505 (Monji/Modir/Kocuk, Dec 2025) is certification work —
  MIQCP/QCP + global solvers, certified n=9 optimal in one day vs Chen
  et al.'s 31 days — it does NOT move the n>=10 best-knowns. Its
  symmetry-breaking/bound-tightening tricks are relevant to our NLP.
- **Karpov publishes his coordinate files.** Fetched + committed
  verbatim (`computable_packs/small_points_optimization/record_sources/`),
  re-measured through our evaluator: S13 = 0.027019272239114 (file
  header matches to 7e-18), S15 = 0.021105681878415 (3e-16).
  Convention match confirmed.
- Structure of the record configs: S13 has 6/13 points exactly on the
  boundary and **16 triangles tight within 1e-6 relative** of the min
  (22 within 1%); S15: 7/15 boundary, 18 tight. Max-min optima are
  heavily degenerate — any polish that only tracks the currently
  smallest triangles will chatter instead of equalizing the active set.

## M1 — exact epigraph polish: built, calibrated; verdict = basin-gap

`exact_polish.py`: maximize t s.t. s_ijk * signed_area_ijk(x) >= t over
ALL C(n,3) triples (orientations fixed at the iterate), x in [0,1]^2,
SLSQP with analytic jacobians, outer re-fix loop, area always recomputed
from clipped coordinates. 7 tests in
`tests/test_heilbronn_exact_polish.py` (jacobian FD check, n=4 corner
recovery, never-degrades, Karpov-holds).

**Calibration ladder** (400 starts/n, alternating uniform and
boundary-biased; `calibrate_exact_polish.py`):

| n | proven optimum | best of 400 | % | hits | s/start |
|---|---|---|---|---|---|
| 5 | 0.1924500897 | 0.1924500897 | 100.00 | 77 | 0.004 |
| 6 | 0.1250000000 | 0.1250000000 | 100.00 | 69 | 0.005 |
| 7 | 0.0838590090 | 0.0838590090 | 100.00 | 23 | 0.008 |
| 8 | 0.0723764243 | 0.0723764243 | 100.00 | 5 | 0.013 |
| 9 | 0.0548759992 | 0.0517502282 | 94.30 | 0 | 0.021 |

The polish recovers proven optima EXACTLY whenever a start lands in the
right basin; the right-basin frequency decays ~x0.4-0.5 per n. For
scale: 400 dumb starts reach 94.3% at n=9; the entire 2026-06 attempt-#2
(17,650 seeds x ~19 min g2_e1 attempts) reached 93.8% at n=13. Per-basin
cost dropped ~4 orders of magnitude (0.07 s vs ~19 min at n=13).

**Probe 1 — Karpov configs are (nearly) stationary:** polish refines
S13 by +6.4e-7 to **0.027019916669111** and S15 by +4.9e-7 to
**0.021106168589172**. Same basin = refinement, not record. These
polished stationary values are now the pack's record bars
(`records.json` updated): a claim between Karpov's file score and its
stationary value is not defensible, since re-polishing his file beats it.

**Probe 2 — attempt-#2 bests gain exactly 0.0** under the polish: n=13
0.0253360415 and n=15 0.0189022154 are already stationary. **The
2026-06 campaign was basin-limited, not optimizer-limited.** Its basins
top out at 93.8%/89.6%; no polish rescues them. All compute goes to
basin discovery.

## M2 — polish_hopper candidate (basin discovery)

`polish_hopper.py`, record_run-compatible `solve(n, seed, budget)`,
deterministic in (n, seed): phase-1 multistart (uniform / 0.35 / 0.6
boundary fractions), phase-2 basin-hopping with structured kicks
(teleport the most-constrained vertex, global jitter, edge snap, corner
pull), greedy accept, stagnation restarts; for n in {13,15} a quarter of
restarts hop from Karpov's config (record-neighborhood exploration —
any strictly-better find wins regardless of provenance). budget//400
polish calls (80000 -> 200 ~ 14 s at n=13, 31 s at n=15).

Smoke: single seed, 200 polishes: n=13 -> 92.79% of record; n=15 ->
refound the polished-Karpov stationary value exactly (via the seeded
restart, as designed). Deterministic re-run byte-identical.

## Environment notes (this machine, 2026-07-06)

- **uv is GONE from this Windows box** (not on any PATH, no .venv);
  campaign now drives `python`
  (3.14.3) directly with PYTHONPATH=repo root. Installed there this
  session: ruff, cma, pydantic (numpy 2.4.3 / scipy 1.17.1 / pytest 9.0.2
  were already present).
- Test suite on native Windows: **373 passed, 13 failed, 13 skipped** —
  every failure (and 4 ignored collection errors) roots in
  `pipeline/autoverify.py`'s Unix-only `import fcntl` (+ subprocess
  variants of the same). Pre-existing platform issue, unrelated to this
  campaign; heilbronn modules and tests are clean. Worth a guarded
  import + msvcrt fallback someday.
- Long background jobs die on this machine — mining runs as chunked
  foreground stints (<= ~8 min) against the resumable
  `record_run.py` state, per the snake campaign's protocol.

## Stints 1-3 (day 1, ~21 min wall-clock, 336 attempts ~ 67k basins)

State: `data/heilbronn/polish_hopper.record_run.json` (resumable,
next_seed=1336). Schema note: the pack's RecordsFile model forbids
extra keys, so the coords pointer lives in `precision_note`, not a
`coords_source` field.

| n | best validated | % of record | verdict | note |
|---|---|---|---|---|
| 10 | 0.04575318 | 98.31 | below-record | ~23 cold attempts, no seeding — softest cell |
| 11 | 0.03458098 | 93.37 | below-record | record is Goldberg 1972 (1/27) |
| 12 | 0.02927290 | 89.80 | below-record | |
| 13 | 0.02701992 | 100.00 | **matches-record** (3e-16) | record stationary value re-found from cold in minutes |
| 14 | 0.02042307 | 84.03 | below-record | record config is rectangle-symmetric; hopper has no symmetry moves |
| 15 | 0.02110617 | 100.00 | **matches-record** (3e-16) | |
| 16 | 0.01448612 | 70.57 | below-record | record config is 180-degree-symmetric |

Reading: the machinery is record-grade (both Karpov cells sit exactly at
their bars; the comparator's matches-record verdict and the absence of a
false new-record banner confirm the sharpened bar semantics). Nothing
ABOVE the bars in 45k n=13-heavy basins yet — Karpov's basins may be
optimal, or deeper ones are rare. Adjacent cells got only a ~23-attempt
probe each and n=10 is already at 98.31%.

## Stints 4-8 + the symmetry program (day 1, evening)

- **n=10 MATCHED from cold** (stint 4, 573 attempts): 0.0465374196,
  matches-record to 1.7e-11 — the Comellas-Yebra 2001 configuration,
  found with no seeding. Three cells now sit at their records (10, 13,
  15).
- n=11 -> 95.84%, n=12 -> 98.06% (free lane, stint 5); still below.
- **v2 symmetric lane** (`polish_hopper_sym.py`, own state file):
  symmetric start families + symmetrize kicks. n=14 jumped 84.0% ->
  96.35% immediately, then STUCK on a magnetic shelf at 0.02341751 —
  two independent seeds (1016, 1206) converged to the identical
  stationary value; deep chains (budget 240k) did not move it. n=16
  crept 70.6% -> 77.3%.
- **`sym_polish.py` — symmetry-constrained epigraph polish** (affine
  fundamental-domain reparameterization, chain-ruled jacobians):
  structures with >=3 points pinned to one axis line are
  collinear-dead by construction (the sweep showed this instantly —
  most rect structures score exactly 0). Viable menu at n=14: 5
  structures.
- **n=14 RECONSTRUCTED: 0.024303979621 = Beyleveld's record (100.000%,
  agrees with the exact form to 2e-11)** via rect(q3,x0,y1,c0) at 800
  starts in 7 dims; independently confirmed via rect(q3,x1,y0,c0).
  Coordinates (8/14 on the boundary, full rectangle symmetry) banked in
  `record_sources/heilbronn_s14_reconstructed2026.json` (provenance:
  reconstructed, NOT fetched). Free search's 0.02341751 shelf is a
  different, worse basin — structure knowledge beat 200+ free attempts.
- n=16 resists reconstruction so far: best rect(q3,x1,y1) 80.7%,
  rot180(p8) 70.0% at 800 starts and only 75.4% at 2500 more
  (seed0=7). Note rot180's space CONTAINS the rect configs yet scores
  lower — coverage thinning in 16 dims; the record basin is small even
  reduced. 7/341 exact value hints at a highly structured (grid-like?)
  configuration — grid-biased starts are the untried family.

## Stints 9-12 + the three-lane push (day 1, night)

Lane results (v3 = `polish_hopper_v3.py`, continuation starts from
record_sources configs at m = n+-1/+-2, own state file):

- **n=11 MATCHED (stint 10): 0.03703704 = Goldberg 1972's 1/27** (v2
  lane, seed 1483, 3.7e-11). The recovered coordinates are EXACT
  rationals — x in {0,1/6,1/3,1/2,2/3,5/6,1}, y in {0,2/9,4/9,2/3,
  7/9,1}, mirror-symmetric — i.e. Goldberg's lattice construction
  itself. Notably the mirror(p5,a1) SWEEP had missed it (91.5% best at
  500 starts in 11 dims); free hopping found it.
- **n=12 MATCHED (stint 10): 0.03259886** (C-Y 2001, 8.2e-12), config
  is rot180-symmetric.
- **SIX of seven cells now sit at their records** (10-15). Every
  reachable published value is exactly re-attained, never exceeded —
  strong evidence the 2001-2011 best-knowns for 10-15 are (at least)
  formidable local ceilings.
- Coordinates for 10/11/12 banked in record_sources
  (heilbronn_s{10,11,12}_matched2026.json) — the v3 source pool now
  covers 10-15.
- **n=16 via v3 continuation: 80.7% -> 95.56% in one stint** (insert a
  point into S15/S14, re-equalize; seed 1134). Then a second
  independent seed (1200) converged to the IDENTICAL stationary value
  0.01961682 — n=16 now has its own magnetic shelf (the n=14 pattern
  repeating). Gap 0.00091.
- n=16 negative probes, both informative: (a) integer-lattice SA on
  the (i/11, j/31) grid (7/341 = 14/682 numerology) topped out at
  min_det 9/14 — hypothesis neither confirmed nor killed, annealer
  underpowered; (b) **the 0.01961682 shelf rot180-projects to area
  0.0** (collinear triples) — the shelf basin is structurally
  incompatible with the record's published symmetry, and rot180
  reduced-space search seeded near it dies at 66%. Beyleveld's basin
  is elsewhere; brute rot180(p8) multistart (dim 16, 3300+ starts)
  reaches only 75.4%.
- sym_polish gained diagonal + anti-diagonal mirror structures. n=11
  structure sweep = clean negative (all symmetric families <= 92.4% <
  free search's own 95.8% at the time). Bonus structure theorem: for
  ODD n, rot180/rect configs need the center point, but center + any
  antipodal pair is collinear — odd-n records can never be
  rotationally symmetric. (Consistent: 11 is mirror, 13/15
  asymmetric.)

## n=16 endgame probes (day 1, late night — both negative, both load-bearing)

- **Boundary-pinned rot180 sweep** (`rot180pin` structures added to
  sym_polish: pairs pinned to edges, 1 dof each, dims 12-14, max 2
  points per edge line): best 77.4% (L2,B1,f5) at 700 starts/structure.
  No crack.
- **Shelf deepening** (route-then-deepen, no restarts): 900 compound
  kicks from the 0.01961682 incumbent, **0 accepts** — the shelf is
  deepen-rigid (the snake campaign's 181-shelf phenomenon exactly).
- Full n=16 attack ledger: free 77% | rect 81% | rot180 75% |
  grid-jitter 76% | (i/11,j/31) lattice SA 64%-eq | shelf-seeded rot180
  66% | pinned rot180 77% | continuation 95.56% (rigid shelf). The
  record basin has resisted every family tried in ~1 day of chunked
  compute.

## Day 2 (2026-07-07) — n=16 deep dive: reduced-space hopper + lattice CP-SAT

The day-1 anomaly resolved into a method upgrade and a sharp verdict.

- **Reduced-space symmetric hopper** (`sym_polish.py reduced_hop`/`hop`):
  basin-hops inside a symmetry's fundamental domain (perturb theta,
  re-polish REDUCED — every iterate stays exactly symmetric), unlike v2
  which polishes full-space and breaks symmetry immediately. **Validated
  decisively: it reaches the n=12 rot180 record EXACTLY (100.000%,
  0.032598858692) where pure multistart stalled at 91.5%** — and the
  winner is a PINNED structure rot180pin(L1,B2,f3), confirming
  boundary-pinned families carry these records.
- **n=16 under the validated hopper still caps at 85.3%**
  (rot180pin(L1,B2,f5), budget 120k x 10 restarts x 7 structures;
  complete family rot180(p8) = 81.1%, every pinned variant 77-85%).
  This is the load-bearing negative: the SAME tool that nails n=12's
  rot180 record cannot get n=16's rot180 family above 85% — so either
  the n=16 record basin is a needle even in reduced space, or **the
  n=16 record is not actually a continuous rot180 optimum** (note the
  best continuous config overall, the 0.01961682 shelf, is ASYMMETRIC
  and beats every symmetric attempt).
- **Lattice hypothesis tested** (7/341 = 14/682; on grid x=i/11,y=j/31
  every area = |D|/682, record needs min|D| >= 14). n=16's exact form
  is a clean rational "7/341" while n=10/12/14 are cubic irrationals —
  a real tell it's a lattice/algebraic construction. CP-SAT
  maximization (ortools, rot180, 300s, `lattice_maxmin_16.py`) climbed
  monotonically to **min|D| = 10** (feasible, 71% of record) and did
  NOT reach 14 or prove it unreachable; strong integer ILS
  (`lattice_strong_16.py`) got only 7-9 (collinear triples flood the
  11x31 grid). So 14 is unproven either way but looks hard on this
  grid — and other grids (22x31 with |D|=28, etc.) remain untested.
- Targeted insertion probe (400 S15+1 spots + 500 S14+2 pairs, each
  polished): converges to the SAME 0.01961682 shelf. Now 6+ independent
  routes hit that identical value — it is the dominant continuous
  attractor.

**n=16 attack ledger (final, this campaign):** free 77% | rect 81% |
rot180 multistart 75% | reduced-space rot180 hop 85.3% | grid-jitter
76% | continuation 95.56% (deepen-rigid shelf) | pinned rot180 85% |
integer lattice (CP-SAT) min|D|=10/14 | integer lattice (ILS) 7-9/14.
Best continuous = 0.01961682 (95.56%), asymmetric, reached everywhere,
rigid. Record 0.0205278592 unbeaten and unmatched.

**Confirmation run:** the reduced-space rot180 hop on the best structure
rot180pin(L1,B2,f5) at 3.3x budget (24 restarts x 1000 calls) reached
0.017645 (85.96%) — +0.67 pts over the 120k run. The rot180 family is
firmly plateaued ~86%; the record (100%) is decisively not a continuous
rot180 optimum this search reaches. This is the load-bearing negative:
the SAME tool nails n=12's rot180 record exactly.

## Day 2 addendum — the lattice solver, tried for real (dead end)

Ran CP-SAT max-min|D| (`lattice_search.py`, 8 workers, upper-bound
reporting fixed) properly on both candidate grids:

| grid | record needs min\|D\| | CP-SAT best (~5-9 min) | proven UB | % of record |
|---|---|---|---|---|
| 11x31 | 14 | 9 | 198 (trivial) | 64.3% |
| 22x31 | 28 | 19 | 488 (trivial) | 67.9% |

**It does not converge and it underperforms continuous search.** The
bilinear-determinant model gives CP-SAT no useful objective bound (UB
stays ~trivial), so it can neither find a record config nor prove one
absent — non-decisive by construction. And the lattice configs it does
find (64-68%) are WORSE than the continuous shelf (95.56%), so the
lattice frame is not the route to n=16.

**The reframing that matters (x682 units, record = 14/682 exactly):**

- record ................... 14.00 / 682   (100%)
- best continuous overall .. 13.38 / 682   (95.56%, asymmetric shelf)
- best continuous rot180 ... 12.04 / 682   (85.96%, validated hopper)
- best 22x31 lattice ....... ~9.5 / 682-eq (67.9%)
- best 11x31 lattice ....... ~9.0 / 682    (64.3%)

The record sits **0.62 units above my best continuous config** — a
genuine needle basin at the top of the ladder, NOT a coarse-lattice
construction (the coarse lattices can't even reach the continuous
shelf). n=16's clean rational value 7/341 most likely reflects the
exact continuous optimum being rational, not a low-denominator lattice
we can solve exactly.

## Verdict (revised, honest)

**Six of seven cells (n=10-15) sit exactly at their published records;
none exceeded.** n=16 is the lone holdout at 95.56%. It is NOT a proven
ceiling and NOT a solvable lattice — it is a **search-coverage wall**:
the highest-dimension cell (32 coords) whose record has the weakest
useful symmetry (order-2 rot180 -> ~13-dim fundamental domain, vs
n=14's order-4 rectangle -> 7-dim domain that WAS crackable). Both the
continuous hopper (86-95.56%) and the lattice solver (64-68%) fall
short of a needle basin 0.62/682 above the shelf. The record
configuration exists (Beyleveld 2006) and my polish would nail it from
the right start — the only failure is landing in its basin.

Live threads if the campaign continues:
1. **Stronger CONTINUOUS search of the n=16 basin** (the productive
   lever): much larger reduced-space rot180 campaign, smarter
   basin-hop schedule / annealing, or continuation volume (v3 state
   next_seed=1229). The lattice solver is a confirmed dead end here.
2. If n=16 stays capped, the honest endpoint is a publishable-grade
   negative ("all seven 2001-2011 best-knowns re-attained +
   characterized at 10^4x lower cost; none beaten") plus the reusable
   exact-polish / fundamental-domain / hopper / lattice machinery — the
   campaign's real product. Characterizations banked: n=11 = Goldberg's
   exact rational lattice, n=14 = Beyleveld reconstructed from its
   symmetry group, shelf rigidity, the odd-n no-rotational-symmetry
   theorem.
3. Record semantics reminder: new record = strictly exceed the pack
   bars (13/15: polished-Karpov stationary values; others: published
   values) with a distinct configuration; miner banner -> hand-check
   before any claim.
