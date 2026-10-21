# Cross-disciplinary transfer catalog (2026-05-30)

The goal congruity is for: **surface a technique from a distant field that beats
the destination field's own best-known method.** This catalog grades every
verified finding on the two axes that define that goal:

- **Transfer distance** — how far apart the source fields of the composition's
  papers are (from `scripts/transfer_distance.py`: distinct OpenAlex fields +
  field entropy). Low = within-field engineering; high = no single human spans it.
- **Beats home-field SOTA** — did the imported technique beat the *destination
  field's own best-known method*, not just generic baselines? (Judged from the
  finding docs; this is the bar that makes a result worth paying for.)

The valuable quadrant is **high transfer × beats home-field SOTA**. "Beats
generic baselines but loses to the home-field specialist" is a near-miss;
"high transfer but the win is speed-only / not quality" is a near-miss;
"low transfer" is in-field work regardless of outcome.

## Grading

| finding | fields (dist) | outcome | beats home-field SOTA? | grade |
|---|---|---|---|---|
| **Heilbronn C1** (20260521-043936) | CS+Eng+Physics (1.58) | clean-positive | **Yes** — physics gradient-flow/SAV beat SA *and* basin-hopping (the point-config field's own optimizers) +9–60% / +78–229% | ★ **transfer win** |
| **Short-string C2** (20260520-053938) | Bio+CS (1.58) | clean-positive | **Yes** — BPE+entropy coding (bio-sequence tokenization import) beat HPACK-stateful, HTTP's *specialized* tool, on real HTTP (0.561 vs 0.608) and generalized off-protocol | ★ **transfer win** (and the old gate had *rejected* it) |
| Short-string C3 (20260520-053938) | Bio+CS (0.92) | partial-positive | Partial — beat general compressors; lost to HPACK on HTTP, won only off home turf | ◐ near-miss |
| Discrete-geometry C2 (20260525-042258) | CS+Materials+Math (1.58) | partial-positive | Partial — beat SA strongly but only *matched* the strongest home method (full active-repair) | ◐ near-miss |
| Hamiltonian C1 (20260523-003811) | Bio+CS+… (1.58) | partial-positive | Partial — oracle classifier beat solver-selection, but the cheap learner was falsified | ◐ near-miss |
| TSP-Delaunay C2 (20260522-215133) | CS+Eng (0.92) | partial-positive | No (quality) — 10× faster but *lost* tour quality to global 2-opt (a within-field method) | ○ speed-only |
| Polymarket wallet C4 (20260524-052010) | Business+Decision+Social (1.58) | partial-positive | N/A — empirical market analysis, not a technique-vs-SOTA contest | ○ different shape |
| Snake-MCTS C2 (20260526-043036) | CS+Physics (0.81) | partial-positive | Marginal — +0.4 vertices over plain MCTS | ○ marginal |
| Solver-portfolio C1 (20260525-011650) | CS+Eng (0.81) | partial-positive | Partial — selection signal real, clusters weak; C1b robustness clean-negative | ◐ near-miss |
| Voice-isolation C5 (20260530-065501) | CS only (0.0) | partial-positive | Yes vs beamformers — but **low transfer** (IVA is native to audio) | ○ in-field |
| (negatives: embed-quant C3, cache, multitrack, TSP C7, game-theory C1, …) | various | clean/partial-negative | — | falsified |

## What this shows

- **Only 2 of ~24 verified findings are clean transfer wins** (high distance ×
  beat the home-field specialist): **Heilbronn** (physics → discrete geometry) and
  **short-string C2** (genomics → compression). That is the genuine product — and
  it's rarer than the headline "~50% positive" rate suggests, because most
  positives either don't clear the home-field bar or aren't cross-field.
- **One of the two golds (C2) was rejected by the old gate** — the moat-relevant
  output was being thrown away. (Now fixed; see the gate-precision findings.)
- The near-misses (C3, discrete-geometry C2, Hamiltonian C1) are the second tier:
  real cross-field imports that beat *generic* baselines but not the home
  specialist. Closing that last gap is exactly where verification effort should go.

## Implication for the metric

The success criterion going forward must be **"beats the destination field's
best-known method," with the recipe naming that home-field SOTA explicitly** — not
a generic baseline. And ranking should weight **transfer distance**: a win that
spans Bio→CS or Physics→Geometry is worth far more than a within-field tweak,
because it's exactly what no human reliably finds. `scripts/transfer_distance.py`
supplies the distance; the recipe change (below) supplies the home-field bar.
