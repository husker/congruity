# Heilbronn record attempt #2 — parallel mass-seed miner, 93.8% of the n=13 record

_2026-06-18. Follow-up to attempt #1 (`findings/heilbronn-record-attempt-20260612.md`).
Composition C2 of run `20260525-042258`, candidate `g2_e1.py` (Delaunay +
active-set SLSQP family), driven by a new parallel record miner over ~17.6k
random seeds instead of the small confirm sweep. Pack
`small_points_optimization`, retargeted to the soft records n=13,15._

> **Provenance caveat.** The `g2_e1.py` that ran here is **not** the original C2
> candidate documented in attempt #1 (committed at `3eba07b`). Earlier this
> session the seeding experiment's *composition* arm re-ran autosearch on C2,
> which by design overwrites the canonical `search/` dir — so the on-disk
> `g2_e1.py` (mtime Jun 17 19:57) is a regenerated, literature-seeded sibling of
> the original, ~300 lines different though the same Delaunay+SLSQP family. The
> miner loaded that version. The original is preserved in git history at
> `3eba07b`. Treat comparisons to attempt #1 as candidate-and-seed-count
> confounded, not a clean ablation.

## Outcome: no record. Clean Tier-A near-miss.

| n | best validated area | seed | best-known record | margin | % of record | verdict |
|---|---|---|---|---|---|---|
| 13 | **0.02533604** | 13182 | 0.0270 (Karpov 2011) | −0.00166 | **93.8%** | below-record |
| 15 | **0.01890222** | 7623 | 0.0211 (Karpov 2011) | −0.00220 | **89.6%** | below-record |

17,650 attempts, 0 new-records, 0 invalid/errored trials. Coordinates for both
bests live in `experiments/auto/20260525-042258-C2/search/g2_e1.py.record_run.json`;
the search is resumable (`next_seed=18655`).

## What mass-seeding bought (confounded by candidate regen — see caveat)

Attempt #1's confirm (seeds 1–2, then holdout seed 101) topped out at **0.01950**
at n=13; its best in-search was 0.02424. The mass-seed run reached **0.02534**.
This is **not** a clean "seeds alone" gain: the candidate that ran was the
regenerated `g2_e1.py`, not attempt #1's, so the improvement mixes a (possibly
better) candidate with ~17k× more seeds. What the curve does show cleanly is the
*within-run* climb as seeds accumulate (n=13 best vs cumulative seeds):

- ~0.0214 after 80 seeds (an early, misleading "ceiling")
- 0.0235 → 0.0240 → 0.0252 by a few thousand seeds
- 0.0253 by ~13k seeds, then flat

Improvements decelerated to ~2e-5 per breakthrough, and the final gain needed
~7k seeds. Closing the remaining −0.00166 (needs +6.6%) by brute-force seeding of
**this** candidate is not realistic — the distribution has a hard right tail well
short of 0.0270. Beating the record needs a stronger candidate, not more seeds.

## Method

`python -m experiments.heilbronn.record_run --candidate <g2_e1.py> --ns 13,15
--budget 80000 --hours 8 --workers 6`. Candidate-only miner: round-robins
`solve(n, seed, budget)` over a `ProcessPoolExecutor`, validates every point set
through `bench._validated_pts_area` (shape/bounds/finiteness, then
`min_triangle_area` recomputed from coordinates — the candidate's self-reported
value is never trusted), and keeps a cumulative, resumable per-n best
(area+seed+coords) that only ever improves and never reuses a seed. On a per-n
improvement it auto-calls `pipeline.computable.compare_to_record`; a `new-record`
verdict would have triggered a loud hand-check banner. None did.

## Infrastructure finding (fixed + committed)

**BLAS/OpenMP thread oversubscription masqueraded as a 6-hour compute hang.**
The first serial launch showed a single `solve(n=13, budget=80000)` consuming
5h44m at 97% CPU with zero completed attempts. It was not compute: each
numpy/scipy worker spawns ~8 OpenBLAS threads by default, so 6 workers (and,
earlier, a concurrent seeding experiment) put dozens of threads on 8 cores and
thrashed. Capping `OMP/OPENBLAS/MKL/VECLIB/NUMEXPR_NUM_THREADS=1` before numpy
imports (these matrices are tiny; threading only hurts) turned 6-hour "attempts"
into ~6-second ones — ~17k attempts in 8h on 6 cores. Now set at the top of
`record_run.py` so no launch can forget it.

Related: `record_run.py` gained `--workers` (process pool; main process owns the
state file, so out-of-order completion still tracks the cumulative max with no
cross-process lock). Pack `small_points_optimization` `search_command`/
`confirm_command` retargeted from n=11,13 to the softer n=13,15 entries.

## Seeding experiment (roadmap #5) — captured, then stopped

Run alongside the miner to answer: does literature-seeded composition
out-discover its alternatives at equal budget? Three arms on C2 /
`small_points_optimization` (population 4, generations 3):

| arm | best holdout area | Δ vs home-field SOTA (0.00549) | beats home-field |
|---|---|---|---|
| composition (literature-seeded) | 0.01757 | +0.0121 | false |
| unseeded (first-principles) | 0.01730 | +0.0118 | false |
| evolve-sota | ~0.0113 (train; confirm not run) | — | — |

Seeded ≈ unseeded (~1.5% apart); evolving the SOTA directly did worst. **On this
problem, literature seeding gives no meaningful edge over first-principles
design.** The evolve-sota arm was killed mid-confirm — its confirm had inherited
the slow n=13 path and would have run for hours — so its holdout summary and the
auto-written findings table are absent; the arm comparison above is the result.
