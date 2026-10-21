# Handoff — Heilbronn record push (interrupted by full disk, 2026-06-12)

Session ended with ENOSPC while creating the long-run record mode. Everything
durable is committed on `handoff-doc` through `3eba07b` (+ fixes `50a5c7c`,
`ec6ae6d`). Nothing is running; the orphaned bench process was killed.

> **RESOLVED 2026-06-18.** All five items below are done. Results:
> `findings/heilbronn-record-attempt-20260618.md`.
> 1. `record_run.py` recreated + parallelized (`--workers`, process pool).
> 2. Pack retargeted to soft records n=13,15.
> 3. Record run completed: 17,650 attempts, **no record** — n=13 0.02534
>    (93.8% of 0.0270), n=15 0.01890 (89.6% of 0.0211). Tier-A near-miss.
> 4. Seeding experiment captured: composition ≈ unseeded > evolve-sota
>    (literature seeding gives no edge over first-principles here).
> 5. Hand-check / confirm — N/A, no record fell.
> Key fix: BLAS thread oversubscription (not compute) caused the apparent
> 6-hour attempt hangs; threads now capped to 1 in `record_run.py`.
> Conclusion: this candidate is not record-capable for n=13/15; beating the
> records needs a stronger candidate, not more seeds.

## State

- **Best candidate**: `experiments/auto/20260525-042258-C2/search/g2_e1.py`
  (deterministic; active-set Delaunay relaxation + SLSQP polish; exposes
  solve(n, seed, budget) and solve_ablated). Best validated areas: 0.03363
  at n=11 (record 0.03704, Goldberg 1972), 0.02424 at n=13 (record 0.0270).
- **Ledger**: C2 labeled partial/valid/medium (+0.0092 over CMA-ES on
  unseen-seed holdout). Full story: findings/heilbronn-record-attempt-20260612.md
- **Wall-clock gotcha**: the candidate takes ~19 min per (n, seed) at 80k
  evals (CMA-ES: seconds). Size everything by measured wall-clock.
- Strategy + status of all roadmap items: docs/roadmap-discovery-2026-06.md

## Interrupted work (next session, in order)

1. **Create `experiments/heilbronn/record_run.py`** (the write hit ENOSPC;
   may exist empty/truncated — recreate). Spec: candidate-only record miner;
   args --candidate --ns 13,15 --budget 80000 --hours 8 --max-attempts
   (testing) --seed0 1000 --state; loops seeds round-robin over ns until the
   wall-clock deadline; validates via bench._validated_pts_area; keeps a
   CUMULATIVE state file ({candidate}.record_run.json) holding best
   area+seed+points per n that only ever improves across invocations (and
   resumes next_seed so seeds are never reused); on improvement calls
   pipeline.computable.compare_to_record(pack 'small_points_optimization')
   and announces a new-record loudly with a hand-check warning.
2. **Retarget the pack at the soft records**: in
   computable_packs/small_points_optimization/evaluator.json set
   search_command --ns 13,15 (--n-steps 10000 --n-starts 4, seeds 2) and
   confirm_command --ns 13,15 (--n-steps 20000 --n-starts 4, --seeds 1
   --seed0 101). n=11 (1/27) may be optimal — don't bang on it; n=13/15 are
   4-digit Karpov 2011 values, the softest entries.
3. **Launch the record run** (background, ~8h):
   `uv run --with numpy --with scipy python -m experiments.heilbronn.record_run
   --candidate experiments/auto/20260525-042258-C2/search/g2_e1.py
   --ns 13,15 --budget 80000 --hours 8`
4. **Launch the seeding experiment** (background; answers whether literature
   seeding beats unseeded/evolve-sota at equal budget):
   `uv run python scripts/seeding_experiment.py --run 20260525-042258
   --cid C2 --pack small_points_optimization`
5. If a record falls: hand-check coordinates against Karpov
   (http://inversed.ru/Ascension.htm) and Friedman's Heilbronn page before
   claiming; then run a confirm so the ledger gets the new-record label.

Also queued (lower priority): regenerate C2's recipe (legacy min-effect
parses to 20.0, blocking honest transfer-win grades).
