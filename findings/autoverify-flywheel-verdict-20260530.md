# Verdict: can the autonomous verification flywheel spin? (2026-05-30)

Brick #2–#4 built the full automated-verification loop:
**scaffold → repair-until (fair AND faithful-SOTA AND runs-valid) → grade vs the
home-field SOTA → record**, with a fairness audit gating every label and two repair
loops (fairness + execution). The question this answers: can it produce
*trustworthy* "beats the home-field SOTA" labels **autonomously and cheaply**?

## The data (every autonomous attempt)

| composition | rounds | final audit | result | label |
|---|---|---|---|---|
| voice C5 | 0 | fair, **strawman SOTA** (real = neural sep.) | partial | low |
| Hamiltonian C1 | 0 | fair, **strawman SOTA** (Δ≈0) | partial | low |
| discrete-geom C2 | 4 | **not fair, strawman** (real = Packomania record) | NaN→invalid | quarantined |
| TSP-Delaunay C2 | — | (no recipe resolved) | error | — |

**Across all attempts: 0 trustworthy (medium) labels, 0 transfer-wins, 0 false
labels leaked.**

## Verdict: no — not autonomously, and the reason is structural

The repair loops *work mechanically* (fairness-repair cleared the audit once;
execution-repair is wired and tested). But the loop repeatedly hits the same wall:
**the destination field's REAL best-known method is usually not cheaply
re-implementable.** It is one of:

- a **GPU-trained model** (voice separation), or
- a **published record / database** built from decades of specialized search
  (Packomania/Specht packings), or
- ambiguous enough that a quick stand-in ties it (Hamiltonian).

When codex can't faithfully reproduce the true SOTA, it substitutes a weaker one —
and the fairness audit *correctly* refuses to trust a win against a strawman. So the
gates hold (no false labels), but autonomous yield is ~zero. The rigor that makes
the catalog valuable is exactly what blocks cheap autonomous labeling.

## A real calibration tension this surfaced

The auto-audit demands the *literal* field SOTA (e.g. the Packomania record). The
manual verification of this same composition (`transfer-catalog-20260530.md`) used
**strong standard algorithms** (SA, full active-repair) as the bar — and graded it
partial-positive. So "home-field SOTA" is not one fixed thing: the *record* vs the
*best reproducible standard method* are different bars, and the audit currently
insists on the former. That strictness is defensible (it's the true bar) but it
makes most problems un-auto-verifiable.

## Decision

Per the fork posed before this brick — "try the auto-path once more; if it can't
converge, labels need humans" — it did not converge, for structural reasons. So:

1. **The flywheel will not spin fully autonomously on cheap codegen.** Stop adding
   auto-repair bricks; the wall is the uncheap real-SOTA, not the loop.
2. **Two viable paths to real labels:**
   - **Human-in-the-loop**: the loop drafts a fair bench; a human supplies/【confirms】
     the faithful home-field SOTA (or imports the record) and confirms the run.
     Lower throughput, real `medium`/`high` labels.
   - **Curated SOTA library**: maintain faithful, reusable implementations (or
     cached records) of home-field SOTAs per domain; codegen must *call* them rather
     than re-implement. Turns "implement the SOTA" (hard) into "call the canonical
     SOTA" (easy) — and is itself a moat asset (process power).
3. **Allow a configurable SOTA bar**: "literal record" vs "best reproducible
   standard method", so cheap auto-verification can run against the latter when the
   former is uncheap — with the label tagged accordingly.

## What stands

The machinery (audit + both repair loops + ledger + grading) is sound and, crucially,
**never manufactured a false win** — it told the truth at every step. That honesty is
the asset; the next investment is in *faithful SOTA supply* (humans or a curated
library), not more auto-repair.
