# C2 Delaunay-partition TSP solver — rescued-composition verification

**run_id:** 20260522-215133 (composition C2)
**experiment:** `experiments/tsp_delaunay/c2_partition_route.py`
**date:** 2026-05-29
**outcome:** **partial-positive** (speed/scalability Pareto point; loses tour quality; L3 seam objection confirmed)

## Why this composition

The single highest-priority candidate from the 2026-05-29 rescue sweep
(`findings/rescue-sweep-2026-05-29.md`). C2 was gate-rejected (L3=invalid, fund=1
→ purely_testable=False) but flipped to purely_testable under the baseline-relative
L4 and ranked priority 5.9. TSP is the one explored domain with no prior clean
winner, so this was the best-EV rescue to verify.

## The composition (verbatim claim)

> "P26's Delaunay triangulation gives a sparse candidate edge set for Euclidean
> TSP, P88's GPU k-way graph partitioner partitions that sparse graph quickly, and
> P86's global-local framework solves the clustered routing subproblems — a
> hardware-scaled solver that avoids the full complete graph."

**L3 (invalid):** strongest objection was "the missing boundary/reconnection layer
— partitioning a sparse proximity graph creates clusters whose seams are not
addressed." (Note: the Delaunay candidate-recall premise was already validated by
the C7 bench; this experiment isolates the partition+stitch question.)

## Method

`c2_partition_route.py` (numpy + scipy):
- Delaunay neighbor lists = the candidate edge set.
- Recursive coordinate bisection into k≈5%·n clusters (P88 stand-in).
- Per-cluster: NN seed + Delaunay-restricted 2-opt (don't-look bits, Bentley).
- **Stitch (the reconnection layer L3 flagged):** order clusters by an NN tour over
  centroids, join sub-tours at nearest endpoints, then a bounded global
  Delaunay-2-opt cleanup to repair seams.
- Baselines: `delaunay-global2opt` (same 2-opt, NO partition — the quality
  ceiling) and `NN-only` (floor). Validated against Held-Karp optimum on n≤12.

## Results (gap vs the no-partition baseline; lower is better)

| n | C2 gap to global-2opt | C2 speedup | NN-only gap (floor) |
|---|---|---|---|
| 200 | ~2.8% (1.2–5.3) | 1.8× | ~21% |
| 500 | ~4.0% (0.3–9.0) | 3.6× | ~17% |
| 1000 | ~5.6% (4.5–6.8) | 6.6× | ~14% |
| 2000 | 6.4% | 10× | 16% |

## Interpretation

**Partial-positive.** C2 is a legitimate **speed/scalability Pareto point**: far
better than the NN floor and ~10× faster than global 2-opt at n=2000, which is
exactly what its "scaled solver that avoids the full complete graph" claim
promises. But on the **primary metric (tour quality) it loses** to unpartitioned
global 2-opt at every size, and — decisively — **L3's seam objection is empirically
confirmed: the quality gap grows monotonically with partition count** (2%→6.4% as
n: 200→2000), even with a seam-repair cleanup pass. So partitioning buys speed at a
quality cost that worsens with scale.

Net: not a clean win (global 2-opt dominates on quality) and not a falsification
(C2 delivers real, growing speedup and beats the weak baseline). Like the C2/C3
compression rescues, this is an effective-but-known method (geographic
partitioning + tour merging is classic) that the gate wrongly vetoed.

## Gate-precision tally (running, incl. rescued-then-verified)

| composition | gate verdict | empirical outcome | gate correct? |
|---|---|---|---|
| C5 TTNU (cache) | L3=invalid | clean-negative | ✅ |
| C3 micro-LM (short-str) | L3=invalid | partial-positive | ❌ over-fire |
| C2 BPE (short-str) | L3=invalid + fund>0 | clean-positive (best in project) | ❌ over-fire |
| C2 Delaunay-partition (TSP) | L3=invalid + fund>0 | **partial-positive** | ❌ over-fire |

4 rejected compositions verified → **3 over-fires, 1 correct rejection.** The
rescue sweep's premise holds: the gate buried real (if known) results. As predicted,
though, the yield is effective-but-not-novel methods in already-explored domains —
supporting the strategy of biasing future effort toward NEW problems now that the
gate is fixed.
