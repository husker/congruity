# Roadmap: from pipeline to discovery engine (2026-06)

Goal: find and prove novel discoveries. Diagnosis (2026-06-11, full analysis in
session notes + `findings/graph-coloring-domain-20260611.md`): congruity is a
hypothesis generator with a one-shot tester. Systems that produce machine-found
results are search loops wrapped around verifiers. The pack-bench rewrite gave
us a trustworthy cheap fitness function; nothing exploits it yet. Track record:
~26 verified compositions, 4 clean positives, 2 transfer wins; first valid
auto-label (C1) collapsed exactly to SOTA.

Proof tiers: **A** benchmark wins vs local bar (weak claim, hard to win);
**B** certifiable records — object + checker = proof, novelty = beat the
published table (Heilbronn, our one undisputed win, was already this shape);
**C** conjectures (skip). Rebalance toward B.

## Items, in priority order

1. **Search loop in autoverify** — population of candidates per composition,
   bench-as-fitness, refine from per-instance traces. Strict train/holdout
   instance+seed splits; confirm once on untouched holdout; that result is the
   recorded label. *(DONE 2026-06-11 — `pipeline autosearch`; first live run:
   C2 partial at TabuCol parity, 10 candidates)*
2. **Implementation fidelity + ablation** — behavioral diff vs SOTA (identical
   outputs = "degenerate", not "partial"); mechanism ablation via optional
   `solve_ablated`; per-trial traces persisted. *(DONE 2026-06-11)*
3. **Record pack type (Tier B)** — cached best-known table per pack; grade
   `new-record` iff the claim verifies and beats a best_known entry (beating a
   proven optimum quarantines as objective mismatch). Heilbronn unit-square
   records n=5-16 cached with sources. *(framework DONE 2026-06-11; remaining:
   wire experiments/heilbronn bench for candidate+record_claims mode, then
   snake-in-box and DIMACS coloring tables)*
4. **Mechanism-first composition** — `compose --computable-pack` injects pack
   failure modes/gotchas into prompts; compositions asked for a mechanism_map
   (transferring structure + SOTA failure mode addressed). *(DONE 2026-06-11;
   full-text/methods mining still deferred)*
5. **Seeding experiment** — `autosearch --arm` composition|unseeded|evolve-sota
   + scripts/seeding_experiment.py; control arms skip the ledger. *(harness
   DONE 2026-06-11; the experiment itself not yet run)*
6. **Throughput + economics** — scripts/flywheel.py (compose→autosearch, one
   cron-able command) + scripts/funnel_report.py (gate rates, ledger
   distribution, codex cost proxy; first reading: 137 codex-min per valid
   label). *(DONE 2026-06-11; deferred: cache_replacement candidate mode needs
   a harness-owned-state redesign for capacity verification;
   short_string_compression candidate mode unwired)*
7. **Statistical grade for Tier A** — paired bootstrap CI over per-trial
   deltas; transfer-win requires mean ≥ min effect AND CI excluding zero.
   *(DONE 2026-06-11)*
8. **Calibration loop** — scripts/ledger_calibration.py joins the ledger with
   compose-time features (confidence, transfer distance, mechanism_map);
   informative once n ≥ ~10 valid labels. *(script DONE 2026-06-11)*

Standing rule from the calibration data: **no new gates** (zero discriminative
power, 84% over-rejection). Spend tokens on search, not judgment.

---

## Addendum — 2026-07-01 engineering pass

Tooling landed this pass (see git log for details): `pipeline pack doctor`
(structural + live-smoke pack health check), `pipeline pack new` (scaffolder
producing contract-compliant pack + bench skeletons on
`experiments/_harness.py`), typed `evaluator.candidate_signature`,
`compose --resume`, parallel scoring/verification fan-out, per-run
`llm_usage.json` cost tracking, real novelty title-search
(S2 + OpenAlex), and nightly scheduling scaffolding
(`docs/flywheel-scheduling.md` — not installed by default).

Deliberately NOT auto-generated, because each requires human-verified
domain truth and fabricating it would poison the ledger:

1. **Curated home-field-SOTA library** (HANDOFF's top next step) — needs a
   human to select/verify faithful SOTA implementations and cached records
   per domain. `pack new` + `pack doctor` now make the packaging cheap;
   the verification of the SOTA itself cannot be automated away.
2. **cache_replacement / short_string_compression candidate modes** — the
   roadmap already scoped cache_replacement as needing a harness-owned-state
   redesign; short_string needs its candidate interface designed. Both are
   an afternoon each of careful bench work, now startable from the
   scaffolder templates.
3. **Snake-in-box + DIMACS record tables** — records.json entries must be
   human-verified published values with sources (the schema enforces
   provenance fields). Populate from the literature, never from memory.

### Update, same day: `pipeline pack forge` (2026-07-01)

Item 1 is now semi-automated by moving trust from generation to checking:
`pack forge <id> --domain "..."` drafts a pack with **cited** SOTA
provenance, clones + pins the official implementation to a commit SHA,
**requires it to reproduce its own published number within tolerance**
(the acceptance test a strawman can't pass), battery-tests the drafted
validator with good/bad attack cases, snapshots record-table sources with
provenance, and emits an evidence bundle (`evidence.md`). The forge
workspace (`data/pipeline/forge/<id>/`) can never touch the ledger —
autoverify/autosearch only load from `computable_packs/`.
`pack promote <id> --approve "<name>"` is the human signature: it copies
the pack in after a final doctor smoke gate (rolled back on failure) and
records approver + pin + reproduction evidence in the pack's
provenance.json. Record extraction into records.json remains deliberately
human — the snapshots turn it into transcription rather than research.
