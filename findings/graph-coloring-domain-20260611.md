# Graph coloring domain test — paper-direct corpus works, autoverify can't see packs

_2026-06-11. Compose run `20260611-022616`; autoverify C1 outcome in
`data/pipeline/outcomes.jsonl` (quarantined)._

## What was tested

Graph coloring as the fourth computable domain: minimum vertex coloring vs a
descending-k TabuCol bar, chosen because the home-field SOTA is cheap, local,
and faithful — the exact opposite of the uncheap-SOTA wall that stalled the
flywheel (`findings/autoverify-flywheel-verdict-20260530.md`).

## New infrastructure

- `experiments/graph_coloring/` — seeded instances (G(n,p), queen, planted
  k-partite), greedy/DSATUR/TabuCol baselines, validator that recomputes
  conflicts from the emitted assignment. Sanity: TabuCol finds queen8=9,
  queen10=11 (known chromatic numbers), recovers planted k=9, and gets 18 on
  a DSJC125.5-shaped instance (published best-known on the real file: 17).
- `computable_packs/graph_coloring/` — preflight `ai_verifiable: true`.
- **Paper-direct corpus mode** (`compose --paper-search "q1;q2;..."`):
  topic-sampled corpora failed the relevance gate twice (raw harvest 2/100,
  hand-curated core list 5/90) because OpenAlex has no graph-coloring topic —
  same failure class as merge sort. Direct works-search with
  coloring-anchored queries passed 27/120 on the first try. Note
  `assemble_corpus` re-sorts topics by works_count internally, so curating
  topic-list *order* does nothing.

## Compose results

5 compositions, all L1/L2 PASS, all L3 weak (no signal per
`findings/outcome-calibration-2026-05-29.md`), all PURELY_TESTABLE, all with
recipes naming the TabuCol bar as `home_field_sota`. Confidence ranking:
C1=7 (RL/GNN TabuCol + chromatic concentration k-window + SAT certificate),
C2=6 (adaptive-k Ising + parallel annealing), C5=6, C3=5, C4=4.

## Autoverify verdict: invalid/quarantined — but for a NEW reason

`pipeline autoverify --run 20260611-022616 --cid C1` (4 audit rounds):

- The codex scaffold **ignored the local pack bench** and invented its own
  SOTA arrangement: tried to download DIMACS files, looked for HEA/MACOL or a
  `COLORING_SOTA_CMD` env var, and crashed in its own DSJC-fallback name
  parser (`queen8` doesn't split into 3 parts).
- The fairness audit correctly refused it: `sota_faithful: false` — "the
  home-field SOTA is not faithfully established."
- Gates held: 0 false labels leaked; the outcome is quarantined.

**The wall is not (only) uncheap SOTA.** Even with a cheap, faithful,
already-implemented SOTA sitting in `experiments/graph_coloring/baselines.py`
and declared in the pack, autoverify cannot use it, because the
pack→autoverify integration was explicitly deferred in the pack spec
("Allow autoverify to call pack-declared baseline adapters instead of asking
codegen to invent baselines",
`docs/superpowers/specs/2026-05-31-computable-problem-packs-design.md`).

## Resolution (same day): codex scaffold removed, pack-bench mode shipped

James's call: the codex scaffold is a liability — removed entirely rather than
integrated around. `pipeline/autoverify.py` was rewritten: scaffold, LLM
fairness audit, and both repair loops deleted; the harness, baselines,
validator, and home-field SOTA now come from a computable pack (required
`--pack` flag), and codegen is ONE contained step that may only emit the
candidate solver behind the pack's declared interface
(`evaluator.candidate_command` / `candidate_interface`; preflight gains
`candidate_ready`). The graph coloring bench grew `--candidate` (plug-in
module, NaN-poisoned score on any invalid coloring) and `--json`
(RESULT_CONTRACT). A human-written candidate can bypass codegen via
`--candidate <path>`. No repair loop: a broken candidate is an invalid,
quarantined label.

First results — the flywheel's first valid, non-quarantined auto-labels ever:

- Smoke (human candidate, TabuCol at half budget): grade **partial**, valid,
  confidence **medium**, tied SOTA 10.71 vs 10.71 mean colors over 21 trials.
- **C1 (codex candidate)**: grade **partial**, valid, confidence **medium**.
  Scores: greedy 14.05, DSATUR 13.29, TabuCol 10.71, candidate 10.71. The
  composition's learned-controller/k-window additions collapsed to SOTA — it
  beat the weak baselines but added nothing over DSATUR-seeded TabuCol at
  equal budget, exactly as the L3 challenge predicted ("the marginal value of
  RL-guided move selection may vanish... when the baseline is DSATUR-seeded").

Verdict: the flywheel spins when the harness is human-owned and codegen is
fenced to the candidate. The May 30 zero-yield was the scaffold, not just
uncheap SOTA. C2-C5 are one command each:
`pipeline autoverify --run 20260611-022616 --cid C2 --pack graph_coloring`.
