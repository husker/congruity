# Windowed WL — Q6 calibration tension (quasi-ergodicity in clamped high windows)

_Date: 2026-07-06. Snake-in-the-box d9 density oracle._
_Companion to: findings/windowed-wl-oracle-design-20260706.md._
_Status: machinery built + green (10 tests); calibration gate open — this note
records the failure mode and the discriminator plan._

## What was built

Windowed extension of the validated flat-histogram sampler (`wang_landau.py`),
all TDD-green: `lmin` window floor (+ dump/load persistence), `seed_from`
(occupancy rebuilt from a certificate snake), windowed `entropy` (raw g over
the band, offset deferred to stitch), plus `windowed_wl.py` with `find_snake`
(calibration-seed generator) and `stitch` (least-squares offset alignment of
overlapping windows, chained to an anchored reference). The validated
grow/shrink acceptance loop is byte-for-byte unchanged.

## The failure mode (measured against exact Q6 counts)

The sampler is **unbiased globally and in windows that reach down toward the
density peak**, but **biased at the top of windows clamped entirely above the
peak**, with the bias growing in the window's log-density range:

| Config | top-bin error | notes |
|---|---|---|
| Global Q4 `[0,7]` | +0.01 | all bins < 0.01 nats |
| Global Q5 `[0,13]` | +0.01 | all bins < 0.01 nats |
| Windowed Q5 `[8,13]` (reaches peak) | −0.05 | accurate across band |
| Windowed Q5 `[10,13]` (above peak) | +0.70 | L12 +0.61, L13 +0.70 |
| Windowed Q6 `[21,26]` (above peak) | **+5.91** | grows monotonically: 24:+1.1 25:+2.0 26:+5.9 |

The Q6 `[21,26]` error is **stable across f from 1e-6 to 6e-8 with every band
bin well-visited (min h ≈ 75k)** — it converges to a systematically wrong
value. Wide Q6 `[16,26]` has the dual problem: the walk escapes to the peak
(~L17) and almost never reaches L=26, so the top bin stays starved (and the
default flatness test, which excludes never-visited bins, lets f run away while
g[26] never develops).

## Mechanism (working hypothesis, ~70/30 intrinsic vs schedule-artifact)

**Quasi-ergodicity, not an f-schedule artifact.** A walker clamped to
`[21,26]` can never disassemble and globally rebuild the snake, because the
major structural-rearrangement routes pass through lengths *below* the window
floor. It recirculates within whatever structural basin its seed belongs to,
and WL faithfully measures the density of states of **that basin**, not of the
ensemble. No 1/t schedule repairs this — schedules fix convergence of f, not
connectivity of the restricted state space.

The implication is precise: the defect is in the **move set within the
window**, not in windowing itself. The fix short of the λ-ladder is a move that
rearranges structure **at fixed length** — the **pivot** (`pivot_prototype.py`,
already on the shelf). Escalation ladder, cheapest first:
**pivots → replica-exchange WL → λ-ladder** (the ladder is the most new
machinery for the same medicine; it stays last).

Why this matters for scaling: even if a wide `[16,26]` window could be forced
to converge, it would **not** transfer to Q9 — Q9's peak (~L120–140) to 190
spans a far larger log-density range than any single window can tunnel. The
scaling architecture is necessarily **narrow stitched windows**, so the bias
must be solved *there*, not avoided by widening.

## Discriminators to run (all foreground-budget cheap)

1. **Seed-dependence** — run `[21,26]` from several structurally distinct
   max-snake seeds; g estimates disagreeing beyond noise confirms broken
   ergodicity (no schedule work saves the clamped walk as-is).
2. **Round-trip census** — count full bottom↔top window traversals; near-zero
   round trips is the same verdict from the dynamics side.
3. **Pivot-augmented window (headline)** — mix pivot moves into the `[21,26]`
   walker, rerun vs exact Q6. If the +5.9-nat bias collapses, the diagnosis is
   confirmed and fixed in one stroke with existing machinery.

## Gate reframing (adopt on merits, not as retreat)

The single max bin is both the hardest to estimate and the least
decision-relevant: sampling can never see states that don't exist, so g(191)
was never measurable, and the "is 190 a wall" verdict always rested on the tail
**slope** S(185..189) + extrapolation. Redefine the gate around the slope —
kept **truth-anchored**: validate the slope estimator against exact Q6 (and
ideally Q7) counts *including* the near-max bins (the same bias contaminates
L=24–25, less), targeting demonstrated ±1-nat/bin accuracy on known truth. A
slope gate with proven accuracy on truth is a stronger instrument than a
max-bin gate that cannot converge.

## Resolution — discriminator #3 (pivot) favorable

Ran the headline discriminator: mix fixed-length pivot moves (`pivot(path, i,
sigma)`, a coordinate permutation of the suffix about a joint — length- and
snake-preserving, accept-iff-legal, ΔL=0 so no WL g-factor) into the clamped
walker. On the **smallest biased case, Q5 `[10,13]`** (fast to converge):

| p_pivot | maxerr over tail | L13 error |
|---|---|---|
| 0.00 (baseline) | 0.695 | +0.70 |
| 0.20 | **0.024** | −0.02 |
| 0.50 | **0.023** | −0.01 |

The bias **collapses ~30× to sampling noise**, in the same 12M moves, robust to
pivot rate. This confirms the quasi-ergodicity diagnosis and fixes it with a
move already in the repo. Q6 `[21,26]` baseline reproduces +5.97; the
pivot-augmented Q6 confirmation is **compute-bound in a foreground probe** and
belongs in a checkpointed background run using the production kill-safe sampler
(the scaling check to a ~10.5-nat range, en route to Q9's far larger range).

Discriminators #1 (seed-dependence) and #2 (round-trip census) were alternative
confirmations of the *same* diagnosis; #3 both confirmed it and fixed it, so
they are held as optional corroboration rather than run.

**Next:** productionize the pivot as an optional move in the windowed sampler
(TDD, keeping the validated length-move loop intact), then run the Q6 `[21,26]`
scaling confirmation as a checkpointed background job. If it also collapses,
proceed to the Q9 campaign under the tail-slope gate.

## Shipped — campaign runner (commit 44b890d)

The fix is productionized and the campaign is launchable. `WLState` gained an
optional `p_pivot` move (ΔL=0, accept-iff-legal, persisted through dump/load —
a resume keeps its pivots). `windowed_wl.py` gained: `calibrate_window`
(truth-anchored gate), `run_window_checkpointed` (resumable single window,
checkpoints every N moves — kill-safe against the machine's job-killing),
`run_windowed_campaign` (window ladder stitched to a reference, with the lowest
window **re-measured** and cross-checked via `overlap_residual` rather than
trusting old global g at the seam), and `tail_verdict` (collapse vs fat_tail).
18 tests green, incl a two-window Q5 campaign that recovers the exact tail to
<0.15 nats and the Q5 `[10,13]` pivot gate to <0.1.

**Next step — Q6 [21,26] scaling confirmation (background, on the box):**

```
uv run python -m experiments.snake_in_box.windowed_wl --p-pivot 0.3
# defaults to Q6 [21,26], checkpoints to data/snake_in_box/wl_confirm_q6_21_26.json
# re-invoke the same command to resume after a kill; prints tail err + PASS/FAIL @0.3
```

Convergence needs ~10²–10³ M moves (the ~10.5-nat range), so run it detached
and read the checkpoint, not foreground. PASS confirms the fix scales past Q5.

**Then Q9 (open-record run):** build a `run_windowed_campaign` ladder — a low
window straddling ~130 (seeded from a mid-length certificate) that is
re-measured and cross-checked against the global-g reference (James's anchor
choice), then `[120,160]`/`[150,190]` seeded from the 181/185 certificates.
Read the verdict off `tail_verdict(S, our_best=185, record_len=190)`: fat_tail →
keep hunting; collapse → census/upper-bounds.

## Budget discipline

This thread exists to inform strategy on the 190 program, not to produce
records. If the discriminators come back ugly (seed-dependent, pivots
insufficient, exchange still biased), take the slope gate at reduced precision
and push freed effort back to the column-generation main line rather than
funding a third generation of sampler machinery. The DOS estimate is worth a
few more sessions; it is not worth becoming the project.

## Result: Q6 confirmation PASSED (2026-07-06 evening)

Launched after review-hardening the runner (62c1c57: atomic checkpoint
writes, resume guard, per-window seeds). Q6 [21,26], p_pivot=0.3, seed 7:
converged to f=1e-6 in 335M moves (~12 min wall).

| L  | exact S | WL S   | err    |
|----|---------|--------|--------|
| 21 | 21.257  | 21.273 | +0.016 |
| 22 | 20.529  | 20.513 | -0.016 |
| 23 | 19.286  | 19.363 | +0.077 |
| 24 | 17.851  | 17.978 | +0.127 |
| 25 | 15.363  | 15.597 | +0.234 |
| 26 | 10.738  | 10.764 | +0.026 |

**max tail err = 0.234 nats — PASS @ 0.3.** The top bin L=26, which the
pivot-free sampler starved (the quasi-ergodicity failure this file
documents), lands at +0.026. The scaling gate is cleared: pivots restore
within-window ergodicity at Q6, not just Q5. Checkpoint kept at
`data/snake_in_box/wl_confirm_q6_21_26.json` as the run artifact.

**Next: the Q9 ladder** — windows straddling the g(130) region upward,
[120,160] and [150,190] seeded from the 181/185 certificates, verdict via
tail_slope at 190 per `run_windowed_campaign`.
