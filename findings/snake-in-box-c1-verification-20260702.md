# Snake-in-the-box C1 — autosearch verification (2026-07-02)

**Verdict: positive (Tier A transfer-win, auto label, confidence medium).**
First flywheel label on the `snake_in_box` pack, produced end-to-end in one
session: pack authored → compose → autosearch → single holdout confirm.

## Setup

- **Problem:** longest induced (chord-free) open path in Q_n at an equal
  candidate-extension evaluation budget. Metric = snake length (edges),
  exact re-validation of every returned sequence.
- **Pack:** `computable_packs/snake_in_box/` (new today; doctor-green).
  Home-field bar = restarted stochastic beam search (width 48) at equal
  budget. Record bar unavailable pending human-transcribed records.json.
- **Compose run:** `20260702-072744` — topic-sampled corpus (32 topics via
  `/topics` search; OpenAlex anonymous `/works?search` was 503-throttled
  all day), 146 papers → 38 after the codex relevance gate, 6 compositions.
  Total compose cost: 32 codex calls, 26 codex-min, ~104k est tokens,
  across 3 kill/resume cycles (`--resume` re-paid nothing).
- **Candidate composition C1** (confidence 8, top priority): SAT-style
  exact infeasibility pruning + rollout/tree policy guidance + multimodal
  live-frontier maintenance. L1 PASS, L2 PASS, L3 weak (6 empirical /
  0 fundamental objections). Source papers: P20 (SAT restart/look-ahead
  machinery), P17 (rollout-guided tree policy), P29 (multimodal frontier
  preservation) — none snake-specific.

## Search trace

`pipeline autosearch --run 20260702-072744 --cid C1 --pack snake_in_box
--population 4 --generations 3 --elites 2` — 10 candidates total.

| gen | best train delta vs beam |
|---|---|
| 0 | −0.67 (all four below SOTA) |
| 1 | −0.67 |
| 2 | −0.67 |
| 3 | **+1.5** (g3_e1) |

## Holdout confirmation (spent once)

Dims (6,7,8) × seeds 101–103 (never seen in search), budget 1.2M evals:

| method | mean snake length |
|---|---|
| greedy rollouts (naive) | 47.22 |
| stochastic beam (home-field bar) | 47.33 |
| **candidate g3_e1** | **48.33** |

Ledger row: `grade=transfer-win, valid=true, provenance=auto,
confidence=medium, delta_vs_home=+1.0, n_trials=9, transfer_distance=2`.

## What the winning candidate actually does

`experiments/auto/20260702-072744-C1/search/g3_e1.py` implements the
composition's three mechanisms rather than a generic tweak:

- **Exact pruning:** bitmask connected-component reachability upper bound
  from the tail (`component_upper_bound`) plus one-step trap/forced-move
  analysis — children that cannot survive validation are never expanded.
- **Rollout guidance:** move ordering by scored lookahead features
  (onward degree, endpoint gap, parity balance, dimension-repeat penalty).
- **Frontier diversity:** `mode_id` hashes the last ~8 move dimensions so
  the frontier holds distinct prefix families instead of collapsing onto
  one — directly the "multimodal live-frontier" mechanism.

## Caveats

- +1.0 mean edges over 9 trials is a real but modest margin; no
  statistical grading was applied to this label (single confirm by
  design). Treat as Tier A win vs a fair reproducible bar, not a record.
- The home-field bar is this pack's own beam baseline. The stronger claim
  (vs published records: proven optima through n=7) awaits records.json —
  the candidate's holdout snakes are well below the known-optimal regime
  (e.g. Q7 best-known is 50; candidate means sit in the 40s), so no record
  proximity yet.
- Auto label, confidence capped at medium per ledger policy. A human rerun
  of the confirm command reproduces it deterministically.

## Follow-ups

- Transcribe records.json from the literature (sources required) to arm
  the record bar; then point `autosearch` at the per-n record gap.
- C2/C3/C5/C6 also carry recipes; each is one
  `pipeline autosearch --run 20260702-072744 --cid <C#> --pack snake_in_box`.

## Addendum — Milestone A record calibration (same day)

Record miner stint on Q7 (proven optimum 50): 40 attempts x 2M evals each.
stochastic_beam best = 46; g3_e1 best = 45. Neither recovers the optimum,
so per the record ladder, Q9+ mining is NOT yet warranted — stronger
methods first. Note the mean/tail inversion: g3_e1 wins on mean (the
transfer-win) but loses best-of-40 to the beam; record hunting optimizes
the tail, so the next autosearch iteration should use record-gap (max
length at target n) fitness, not mean delta.
