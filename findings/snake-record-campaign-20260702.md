# Snake-in-the-box record campaign — day 1 (2026-07-02)

Campaign goal: climb from Tier-A wins (beat the pack's beam baseline) to
record territory (best-known lower bounds start at Q9 = 190). Gate:
**Milestone A — recover the proven Q7 optimum (50) before spending
wall-clock on Q9.** Status: **not passed; plateau at 47.**

## Ladder (best Q7 snake, 40 attempts x 2M evals unless noted)

| method | Q7 best | holdout mean delta vs beam |
|---|---|---|
| g3_e1 (C1 winner, mean-bred) | 45 | +1.00 (transfer-win) |
| stochastic_beam (pack baseline) | 46 | — |
| g0_c2 (C5 winner, mean-bred) | 47 | +3.33 (transfer-win) |
| record-mode evolution seeded from g0_c2 | 47 (11 candidates) | n/a (no ledger by design) |
| **proven optimum** | **50** | |

Also this day: C2 autosearch → partial (+0.67, below min effect).
All three auto labels (C1, C5 transfer-wins; C2 partial) are in
`data/pipeline/outcomes.jsonl`.

## Findings

1. **Mean/tail inversion confirmed.** g3_e1 beats the beam on mean but
   loses best-of-40; record hunting rewards tails. This motivated
   `autosearch --record-dim` (record-gap fitness; shipped + tested).
2. **The 47 plateau is method-class, not budget.** Three independent
   method families and a tail-bred generation all stop at 47; a 5x-budget
   stint is running to characterize the budget curve, but the working
   hypothesis is a knowledge gap: the machinery that closed Q7 in the
   literature (canonical forms / isomorph rejection / symmetry-aware
   pruning) is absent from all our candidates.
3. **Record-mode plumbing works end-to-end**: record_command, tail
   fitness, certificate persistence, new-record early stop, no ledger
   contamination.

## Next

- Symmetry-targeted compose run launched (corpus: symmetry breaking,
  isomorph-free generation, canonical forms, orderly generation) — the
  untried lever aimed directly at the hypothesized gap.
- Overnight 60 x 10M-eval Q7 stint with g0_c2 (budget-curve data point).
- If a symmetry composition beats 47 → re-run Milestone A; at 50 →
  graduate to Q9 (`--record-dim 9` evolution + long mining).

## Day-1 close

- 5x-budget stint (60 x 10M evals): Q7 best 47 -> **48**. Budget still
  buys progress but the curve is brutally flat (+1 edge per ~5x compute).
- Symmetry-targeted compose (13-paper corpus after gate, min-gate-papers
  10; /works search throttled all day so topic-mode only): 6 compositions,
  C1 = product structure + stabilizers + propagation + ordered
  enumeration (conf 8). Record-mode evolution on C1 seeded with g0_c2:
  best 46 = the seed itself; none of 10 symmetry candidates beat the
  incumbent. One-shot codegen of group-theoretic machinery did not
  outperform mature tuned heuristics at this population/generation scale.
- Final day-1 ladder: mean-bred 45-47 | tail-bred 47 | budget-mined 48 |
  symmetry-bred 46 | proven optimum 50.

Next levers, in rough order of expected value: (a) more mining budget on
g0_c2 (cheap, resumable, +1 edge per ~5x — could grind to 49); (b) deeper
evolution: larger populations/generations in record mode, possibly
seeding from the symmetry C1 *composition prompt* with the g0_c2 *code*
as parent; (c) full-text mining of the actual snake-hunting papers
(roadmap-deferred full-text mode) so codegen sees real canonical-DFS
pseudocode instead of abstracts; (d) paper-direct symmetry corpus once
OpenAlex /works search recovers.
