# Windowed Wang-Landau oracle — design

_Date: 2026-07-06. Author: campaign session (snake-in-the-box d9)._
_Status: design approved; pending spec review → implementation plan._

## Purpose

Decide the d9 record campaign's strategic fork with data instead of hope:
**keep hunting for a snake > 185 in Q9, or pivot to the census / upper-bound
game.** The instrument is the density of states S(L) = log N(L) (log-count of
directed snakes of length L in Q_n) in the tail L ∈ [180, 190]. The world
record is Wynn's 190; our best is 185; every generative lane has converged on
a statistical "181 shelf" (6 distinct 181s, zero 182+), so we cannot tell by
search alone whether 182–190 is *scarce* (census) or merely *hard to hit*
(keep hunting). S(L) in the tail settles it.

## Background: why the current (global) oracle cannot see the tail

`experiments/snake_in_box/wang_landau.py` runs a flat-histogram Wang-Landau
walk over **all** snakes with grow/shrink moves at either end. The move set is
ergodic in principle (any snake shrinks to a point and regrows) and the
acceptance math is Hastings-corrected and calibrated (trust chain verified
exact through Q6; see the Day-6 campaign log).

The failure is *reachability*, not correctness. Over the full range `[0, 200]`
the histogram is dominated by the astronomically populous mid-length bins
(~L 100–140). The walk satisfies the flat-histogram criterion on the reachable
band `[0, ~138]`, decays `f` there, and never climbs into `[139, 200]`. The
same failure was measured at Q6: the global chain **never reached L=26**, the
optimum bin. So the region the campaign needs — 182–190 — is exactly the
region the global oracle is blind to.

Verified state at handoff (commit `81d27c7`): Q9 WL checkpoint, 3.40B moves,
S(L) converged through L≈130, `best_seen`=139.

## Design

### Principle

Do not re-measure S(L) from scratch. The global WL already has trustworthy `g`
over `[0, ~130]`. Measure only the **tail** with windowed runs, then anchor
those runs to the global chain where it is still trusted (which itself anchors
to the exact bin N(0) = 2^n).

### 1. Windowing — surgical change to `WLState`

- **`lmin` floor.** `shrink` clamps at `lmin` instead of 0 (a single added
  comparison in the inner loop). The validated grow/shrink acceptance
  expressions are otherwise **unchanged** — this is what protects the trust
  chain: the sampler's core math is not re-derived, only its length domain is
  bounded.
- **Seed constructor.** Load a known snake from a certificate JSON, rebuild the
  `on` (occupancy) and `nc` (occupied-neighbor-count) arrays from its vertex
  path, and truncate to land inside `[lmin, lmax]`. The chain therefore
  **starts inside the tail** rather than trying to random-walk up to it.
- Hastings factors, flat-histogram test, entropy anchoring, and kill-safe
  chunked checkpoints all carry over untouched.

With `lmin > 0`, a single window's `g` is only defined up to an additive
constant (it no longer contains the exact N(0) bin). The constant is fixed
externally by stitching (§2).

### 2. Window ladder + stitching

Overlapping tail windows. Example for Q9:

- `[120, 160]` seeded from a ~155-length router snake,
- `[150, 190]` seeded from the 185 certificate.

Each window runs to flat-histogram within its band. **Stitch** by matching the
`g`-offset across the overlap bins (least-squares over the shared L range),
chaining down to the global WL's `g` at L≈130, which is anchored to the exact
N(0) = 2^n. Result: a single S(L) curve over `[0, 190]`, tail included.

The **overlap mismatch is a first-class bias diagnostic.** If two windows agree
on the shape of `g` across their shared bins (small residual after offset fit),
that is empirical evidence of within-window ergodicity. A large, structured
mismatch is the basin-trapping alarm (see §3 fallback).

### 3. Calibration protocol — the trust gate

Extends the existing exact-through-Q6 trust chain. Ordered, with a hard gate:

1. **Q5 windowed sanity.** Window `[8, 13]` seeded from an L=13 snake; must
   reproduce the exact tail counts. Runtime: seconds.
2. **Q6 tail acid test — GO/NO-GO.** Window `[16, 26]` seeded from a known L=26
   snake (emitted cheaply by one greedy/first-found DFS descent). Must
   reproduce `data/snake_in_box/wl_q6_exact_counts.json` on bins `[20, 26]`
   within **~0.3 nats — including L=26**, the bin the global chain never
   reached. This is the whole thesis of windowing under test: can a seeded,
   bounded walk measure a bin the unbounded walk could not?
   - **Pass** → proceed to Q9 production.
   - **Fail with a trapping signature** (systematic under-count at the top
     bins, large overlap mismatch, replicate spread that does not shrink with
     more moves) → escalate to the **pre-agreed fallback**: memo #5's λ-ladder
     soft-constraint moves (temporarily relax the chord-free / induced-path
     constraint with an annealed penalty λ to tunnel between fixed-length
     basins). The fallback is *earned by a concrete calibration failure*, not
     built speculatively.
3. **Q9 production.** Tail windows stitched to global `g` at L≈130. Run **≥2
   seeds / RNG streams per window** so replicate spread yields error bars on
   the tail S(L).

### 4. Verdict protocol — what the oracle outputs

Report S(185), S(188), S(190) with error bars (from replicate spread), read
against the measured Q6 cliff as the reference shape (Q6: S fell 21.9 → 10.7
over the last 6 lengths; the 25→26 step alone was −4.6 nats, ≈100×).

- **Collapse** — S(190) small (N(190) on the order of hundreds), slope
  steepening into the tail → Wynn's 190 is near-unique and 191 almost surely
  does not exist → **pivot to the census / upper-bound game** (master-LP 191
  bound, impossibility ledgers, SDP/Lasserre d9 upper bound).
- **Fat tail** — S(185) still large (say N(185) ≫ 10³) → the 181-shelf is a
  *search* failure, not scarcity → **keep hunting d9**; and log N(185)
  measured against our observed hit-rate even yields a rough effort estimate
  for reaching 186+.

The output is a written verdict in the campaign log with the numbers and their
error bars, not a bare recommendation.

### 5. Ergodicity knob — window width

Within-window ergodicity holds only if `lmin` is low enough that all in-band
snakes connect through the `lmin` shell. Window **width** (not any single
parameter) is therefore the real ergodicity control:

- **Wide** windows (low `lmin`, e.g. `lmin ≈ 0.6·lmax`): more ergodic, slower
  to fill the extreme-tail bins.
- **Narrow** windows: faster tail fill, higher basin-trapping risk.

Start **wide** and narrow only if tail-fill is too slow, with Q6-tail
calibration and the overlap-mismatch diagnostic both watching for trapping.

### 6. Code layout

- **`experiments/snake_in_box/wang_landau.py`** — extend in place. Add `--lmin`
  and `--seed-snake PATH`. `WLState.__init__` accepts `lmin`; `run()` clamps
  shrink at `lmin`; add a `WLState.from_snake(path, lmin, lmax)` /
  `seed_from(path)` that rebuilds `on`/`nc` and truncates into band. The inner
  acceptance loop is otherwise byte-for-byte the current, validated code.
- **`experiments/snake_in_box/windowed_wl.py`** — new orchestration module:
  window-ladder driver, the stitch (offset least-squares + anchor to a
  reference bin), the Q5/Q6 calibration harness, and the verdict computation.
  A `max-length-snake` helper (one DFS descent returning the first max-length
  snake found) supplies calibration seeds.

### 7. Testing

- `lmin` clamp: a walk with `lmin=k` never records a length < k after seeding
  in-band; acceptance math unchanged at L > lmin (regression vs current).
- Seed-occupancy rebuild round-trips: `on`/`nc` rebuilt from a path equal the
  arrays produced by growing that path move-by-move.
- Stitch recovers a known synthetic offset between two overlapping `g` arrays.
- **Q6-tail calibration assertion** (the gate, §3.2) as an executable test with
  the ~0.3-nat tolerance, so a regression in the sampler surfaces immediately.

## Non-goals / YAGNI

- The λ-ladder is **not** in scope unless Q6-tail calibration fails (§3.2). No
  speculative soft-constraint machinery.
- No re-measurement of the mid-range S(L) already trusted from the global run.
- No change to the grow/shrink acceptance math.

## Machinery reused

C core / occupancy invariant (`snake_core.c`, `fastcore.py`), certificates in
`data/snake_in_box/certificates/` and `seed6003_q9_181.json` /
`ours185_window_ledger.json` (seeds), `wl_q6_exact_counts.json` (calibration
target). Chunked-foreground kill-safe pattern (background jobs die on this
machine) is already how `wang_landau.py` runs.
