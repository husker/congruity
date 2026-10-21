# Phase 3 product layer - 2026-05-21

Implementing the Phase 3 plan from `findings/project-deep-analysis-20260521.md`.

## Shipped

- **Verification recipes per actionable composition.** New `pipeline/recipe.py` and `VerificationRecipe` pydantic model. On `--with-recipes`, every composition with L3 verdict `weak` and L4 `net_testability_score == "PURELY_TESTABLE"` gets a structured plan: named baselines, benchmark dataset, primary + secondary metrics, minimum meaningful effect size, expected runtime, likely failure mode, stop rule. Artifacts under `llm_calls/recipe_<cid>/NN/`. Default off; opt-in because it costs an extra LLM call per actionable composition.
- **Automated novelty check.** New `pipeline/novelty.py` and `NoveltyReport` pydantic model. Classifies each non-invalid composition as `known` / `adjacent-known` / `not-found` / `unverified` by searching S2 + OpenAlex for the claim's key phrases and measuring co-citation density across the constituent papers. The classification thresholds (`KNOWN_HIT_THRESHOLD=3`, `KNOWN_SIM_THRESHOLD=0.85`, `ADJACENT_COCITE_THRESHOLD=0.3`) are heuristic — first 10 real `--with-novelty` runs should inform tuning. S2 + OpenAlex helpers are stub-only here; tests pin the classification logic; wiring the live HTTP calls (modeled on `scripts/step3_novelty_check_s2.py`) is the obvious follow-up.
- **Calibration dashboard as a markdown report.** New `pipeline/calibration.py` walks `data/pipeline/compositions/<run_id>/state.json` for every saved run and emits a markdown report covering run summary, L3 verdict mix, L4 testability mix, skipped relevance gates, total LLM wall time, and a verified-outcomes section rendered verbatim from `findings/INDEX.md`. Smoke against the real saved runs immediately surfaces: 16 runs, 64 compositions, 34 weak + 30 invalid in L3. New `findings/INDEX.md` is the manually-maintained track record table.
- **CS / algorithms domain pack.** `domain_packs/cs_algorithms/` contains topic queries, problem templates, canonical baselines, evaluation datasets, transient deps, and a `gotchas.md` capturing recurring traps from the existing experiments. New `pipeline/domain_pack.py` CLI with `list` and `apply --dry-run` subcommands. `apply` prints a structured summary; live integration with `compose_one` is a follow-up.

## Test count

Phase 2 finished at 76 passing tests. Phase 3 ships **97 passing tests** (+21):
- +4 verification recipe
- +6 novelty check
- +5 calibration dashboard
- +6 domain pack

`uv run ruff check .` green throughout.

## Plan-vs-reality notes

- **Novelty live search is stubbed.** The plan called for live S2 + OpenAlex calls; the implementation ships the schema, classification logic, opt-in plumbing, and tests but leaves the actual HTTP calls as no-op stubs. The reason: the classification logic and CLI plumbing land cleanly without the live wiring, and the live calls have rate-limit + retry + caching considerations that deserve a focused follow-up commit. The existing `scripts/step3_novelty_check_s2.py` already has the retry/backoff pattern we'd copy in.
- **OpenAlex topic IDs in the CS pack are placeholders.** `domain_packs/cs_algorithms/topic_queries.json` includes an explicit ENGINEER TODO marker. Real IDs need a one-time OpenAlex lookup before the apply CLI is wired into a live compose run.
- **`pipeline.domain_pack apply` is dry-run only.** The plan envisioned the apply CLI stamping the pack into a fresh compose run automatically; this phase ships the structured summary output, leaving the live `compose_one` integration to a follow-up.
- **Calibration smoke immediately re-confirms the deep-analysis numbers.** The 16 runs / 64 compositions / 34 weak / 30 invalid breakdown lines up with the deep-analysis paragraph that listed "14 complete runs, 64 surfaced compositions, 34 WEAK and 30 INVALID L3 verdicts" — the calibration report makes that summary regenerable on every run.

## Commits

| SHA | Summary |
|---|---|
| d3bfc99 | docs: Phase 3 product-layer spec |
| e0ec7ad | docs: Phase 3 implementation plan |
| 0aa7704 | feat(pipeline): verification recipe stage |
| 1ac9860 | feat(pipeline): post-L3 novelty check stage |
| 77dd3f1 | feat(pipeline): calibration dashboard markdown report |
| 3de3010 | feat(domain_packs): first domain pack (CS / algorithms) |

## Deferred

- **Item 5 (expert hit-rate study).** Needs blind domain experts and a rating form. Can be set up as scaffolding (form template, anonymized sampler, scoring spec) in a follow-up.
- **Live novelty network calls.** Schema + classifier are in; wire S2 + OpenAlex HTTP + cache the responses under `data/pipeline/novelty_cache/`.
- **`pipeline.domain_pack apply` → live compose_one integration.** Today `apply` is dry-run only; the next step is to thread the pack's topic queries and problem-template fill into a real compose invocation.
- **Multi-domain packs.** Only CS / algorithms ships this phase. Chemistry / biology / physics packs are templates the next user can fill from the CS pack's structure.
- **Calibration dashboard as a live UI.** Markdown is enough for now; a Streamlit / static HTML view is a later affordance.

## Open items inside Phase 3

- **Novelty classification thresholds are guesses.** 3 hits / 0.85 similarity / 0.3 cocite-density were picked to be defensible from first principles; first 10 real `--with-novelty` runs should validate or shift them.
- **The OpenAlex topic IDs in the CS pack need verification.** Explicit ENGINEER TODO marker is in `topic_queries.json` so the first user who runs `apply` knows to fix them before live use.
- **`findings/INDEX.md` must be maintained as new finding docs land.** The calibration report renders it verbatim; if it goes stale the verified-outcomes counts drift.
