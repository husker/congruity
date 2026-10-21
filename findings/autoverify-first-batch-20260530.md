# Automated-verification first batch — the honest result (2026-05-30)

First audited batch through the brick-#2 flywheel (`pipeline autoverify`):
scaffold a bench → **fairness-audit it** → run → grade vs the home-field SOTA →
record an auto-label. 4 compositions attempted.

## Results

| composition | domain | grade | confidence | audit | beats home-field? |
|---|---|---|---|---|---|
| voice C5 (20260530-065501) | source separation | partial | low | fair, **strawman SOTA** | no |
| discrete-geom C2 (20260525-042258) | point layouts | partial | **quarantined** | **unfair harness** | no |
| Hamiltonian C1 (20260523-003811) | SAT/graph | partial | low | fair, strawman SOTA (Δ≈0 tie) | no |
| TSP-Delaunay C2 (20260522-215133) | TSP | errored | — | — (no recipe resolved) | — |

**0 trustworthy (medium-confidence) labels; 0 transfer-wins; 0 false wins leaked.**

## What this means (the important part)

The flywheel runs end-to-end and the **fairness audit is doing exactly its job**:
it quarantined the one unfair bench and down-weighted every strawman-SOTA bench to
`low`, so nothing false entered the ledger as a trustworthy negative or win. The
rigor guarantee holds.

But the **label yield is poor**, and the bottleneck is now precisely identified:

> Automated verification at scale is limited by the **fairness AND faithfulness of
> auto-generated benchmarks**, not by compute. Cheap codex codegen rarely produces
> a *faithful* implementation of the destination field's real best-known method —
> it tends to invent a weaker stand-in (strawman SOTA) or, occasionally, an unfair
> harness. Against a strawman the "result" is meaningless, so the audit correctly
> refuses to trust it.

This is the honest state of the moat thesis: the audited flywheel will not spin
fast on naive auto-codegen. The compounding asset (a catalog of verified transfers
that beat the home specialist) still requires either (a) materially better bench
codegen that faithfully implements home-field SOTAs, or (b) a human-confirmed
sample to calibrate and promote auto-labels.

## Implication for next steps

1. **Harden codegen faithfulness** — feed the audit's strawman findings back into
   the scaffold prompt (name and cite the real home-field SOTA; forbid inventing a
   weaker stand-in) and re-scaffold until the audit clears `sota_faithful`. A
   scaffold→audit→repair loop is the natural next brick.
2. **Triage by cheap-SOTA availability** — domains whose real SOTA needs GPU
   training (voice) are out of scope for cheap auto-verification; the audit detects
   this (strawman flag) and should route those to "needs heavy verification."
3. **Human-confirm a sample** — spot-check a handful of audited labels manually to
   measure auto↔manual agreement before trusting `medium` labels in the predictor.

The flywheel's plumbing is sound; the water is the bench faithfulness, and that's
where the next effort goes.

## Update: scaffold → audit → repair loop (brick #3)

Ran the repair loop (max 2 rounds) on discrete-geometry C2 — the composition the
batch QUARANTINED for an unfair harness. Result:

- **The repair loop did its job:** after 2 rounds the audit CLEARED
  (`fair=true, sota_faithful=true, issues=[]`). Feeding the audit's concrete issues
  back to codex successfully converted an unfair/strawman bench into a fair one with
  a faithful home-field SOTA. The faithfulness mechanism is validated.
- **But a faithful bench is not yet a valid one:** the repaired bench ran and
  produced `NaN` scores (a numerical bug in the generated code), so it graded
  `invalid` → `quarantined`. The validity gate caught it — no garbage label leaked.

So clearing the audit (fairness/faithfulness, judged from code) is necessary but
not sufficient; there is a second axis — does the faithful bench actually execute
and produce finite scores? The repair loop fixes fairness; it does not yet fix
runtime/numerical correctness.

**Next sub-brick:** an EXECUTION-repair loop, analogous to the fairness-repair
loop, that feeds runtime errors / NaN / degenerate output back to codex until the
bench runs cleanly — composed with the fairness loop (repair-until fair AND
faithful AND runs-valid). Only then does auto-label yield become real. The two
loops together are the path to medium-confidence labels at scale.

