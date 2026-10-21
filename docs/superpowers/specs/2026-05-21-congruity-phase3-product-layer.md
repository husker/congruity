# Spec: Congruity Phase 3 — Product Layer

**Date:** 2026-05-21
**Status:** approved, ready for implementation planning
**Source:** [findings/project-deep-analysis-20260521.md](../../../findings/project-deep-analysis-20260521.md), Phase 3 of the prioritized remediation plan. Builds on Phase 1 ([spec](2026-05-21-congruity-phase1-remediation.md), [findings](../../../findings/phase1-remediation-20260521.md)) and Phase 2 ([spec](2026-05-21-congruity-phase2-benchmark-hardening.md), [findings](../../../findings/phase2-benchmark-hardening-20260521.md)).

## Goal

Turn Congruity from "idea generator with a verification arm" into "research triage and test planner." Phase 1 made every stage auditable. Phase 2 made every harness reproducible. Phase 3 makes every surfaced composition **actionable** — there's a verification recipe attached, novelty has been checked, and the running track record is visible at a glance.

## Scope

Four items shipped this phase. The fifth deep-analysis item (expert hit-rate study) needs external participants and is deferred.

1. **Verification recipes per composition** — for every WEAK + PURELY_TESTABLE composition, emit a structured recipe: baseline(s), dataset, metric, minimum meaningful effect size, expected runtime, likely failure mode, stop rule.
2. **Automated novelty check** — new post-L3 stage that searches for exact synthesis terms, constituent-paper pairings, and references; classifies each composition as `known`, `adjacent-known`, `not-found`, or `unverified`. Saves evidence links and queries.
3. **Calibration dashboard (markdown report)** — single command aggregates all `data/pipeline/compositions/<run>/state.json` files and emits a markdown report covering run counts, L3 verdict mix, L4 testability mix, verified outcomes, skipped gates, cost per run, top recurring L3 failure modes.
4. **Domain pack: CS / algorithms** — the first domain pack, since that's where the existing verification track record lives. Includes topic-harvest queries, problem statement templates, known baselines per problem family, evaluation datasets, gotchas, recommended transient deps.

## Item 1: Verification recipes per composition

**Problem.** Today the pipeline outputs WEAK + PURELY_TESTABLE compositions but says nothing about *how to test them*. The user has to design the experiment themselves. That extra step is where most candidates die.

**Solution.** New stage that fires on every composition with L3 verdict WEAK (or sound, but those are rarer) and L4 net_testability_score == "PURELY_TESTABLE":

- New module `pipeline/recipe.py` exporting `generate_recipe(composition, state) -> VerificationRecipe`.
- LLM call (codex) prompted with the composition's claim, the constituent papers, and the L3/L4 outputs. The prompt asks for a structured JSON with these fields:
  - `named_baselines`: 2-4 named methods the composition must beat
  - `benchmark_dataset`: existing public dataset name + retrieval link (or note if synthetic generation is the right call)
  - `metric`: primary metric (single string) + secondary
  - `min_meaningful_effect_size`: numeric (depends on metric — e.g. "+0.02 recall@10", "−15% wall time")
  - `expected_runtime`: range like "30 minutes single GPU" or "no GPU, 5 minutes CPU"
  - `likely_failure_mode`: one paragraph describing the most likely way this falls apart
  - `stop_rule`: condition under which the test can be abandoned early as a clean negative
- Pydantic model `VerificationRecipe` in `pipeline/schemas.py`; validated via `parse_or_repair` like every other stage; artifacts saved under `llm_calls/recipe_<cid>/NN/`.
- New CLI command `compose recipe <run_id>` to (re)generate recipes for a saved run.
- Recipe attached to the composition's `verification.recipe` field in `state.json`.

**Acceptance.** New run that surfaces a WEAK+PURELY_TESTABLE composition emits a non-empty recipe. Recipe validates against `VerificationRecipe`. A test using stubbed codex confirms the recipe is recorded and the schema validates the canned response.

## Item 2: Automated novelty check

**Problem.** The strongest current gap is novelty. The Rydberg-fusion novelty check was manual; the early CS bridge novelty checks were partial. WEAK + PURELY_TESTABLE compositions get expensive empirical work — wasting that work on a known result is the worst case.

**Solution.** New post-L3 stage that runs only on compositions whose L3 verdict is not `invalid`:

- New module `pipeline/novelty.py` exporting `check_novelty(composition, state) -> NoveltyReport`.
- The novelty check has two arms:
  - **Exact-term search.** Take the synthesis claim's key phrases (extracted via codex), search S2 (Semantic Scholar) and OpenAlex for matching titles/abstracts. Existing project scripts (`scripts/step3_novelty_check_s2.py`, `scripts/step3_novelty_check_foundational.py`, `scripts/step3_novelty_check.py`) already do parts of this — refactor the most reusable bits into `pipeline/novelty.py`.
  - **Constituent-paper-pair search.** For each pair `(P_i, P_j)` of constituent papers, query S2/OpenAlex for papers that cite both. The number of co-citing papers is a proxy for "this combination is well-known."
- Classification:
  - `known`: ≥3 hits with high title/abstract similarity to the claim.
  - `adjacent-known`: 1-2 hits, OR co-citation density above a threshold.
  - `not-found`: zero hits and low co-citation density.
  - `unverified`: search APIs failed or no API key.
- Output `NoveltyReport` is saved on the composition under `verification.novelty` and recorded in `state.json`.
- Skip the LLM-based key-phrase extraction if no codex available; fall back to TF-IDF over the claim text.

**Acceptance.** New `pipeline/novelty.py` and `NoveltyReport` pydantic schema. A unit test with stubbed S2 + stubbed OpenAlex confirms the classification logic (known/adjacent/not-found/unverified) given canned responses. Integration with compose CLI behind an opt-in flag `--with-novelty` (default off, since the search API hits cost time).

## Item 3: Calibration dashboard (markdown report)

**Problem.** The track record is the product's trust engine but it's currently scattered across findings docs. There's no single place to see "how many compose runs have we done, what was the L3 mix, what did L4 say, how many verified positive."

**Solution.** New CLI: `python -m pipeline.calibration` (or `compose calibration` subcommand). It walks `data/pipeline/compositions/<run_id>/state.json` for every saved run and emits a markdown report:

- **Run summary table**: run_id, problem, corpus_size, status, dimensions count, compositions count, run date.
- **L3 verdict mix**: counts of sound / weak / invalid across all compositions, plus per-problem-family breakdown if we can infer family from topic_list.
- **L4 testability mix**: counts of PURELY_TESTABLE / numeric-ratio / NOT_TESTABLE.
- **Verified outcomes**: counts of true-positive, true-negative, partial-positive, clean-positive, clean-negative drawn from the existing `findings/*.md` track record. The mapping from finding doc → verdict can be done with a small `findings/INDEX.md` that lists each verification finding with its outcome label.
- **Skipped gates**: count of runs where `relevance_skipped: true` in state.json (Phase 1 surfaced this field).
- **Cost per run**: sum of `duration_s` across all `llm_calls/*/NN/meta.json` files. Reports total wall-time per run + per-stage averages.
- **Top recurring L3 failure modes**: stems of the `verdict_reason` field clustered by simple keyword extraction (or skipped if simple text-frequency is too noisy).

Output is one markdown file at `findings/calibration-<today>.md`. Re-run replaces the file.

**Acceptance.** Calibration CLI runs cleanly against the existing `data/pipeline/compositions/*` and emits the report. New test with a tiny fixture of saved runs verifies the report contains all required sections.

## Item 4: Domain pack — CS / algorithms

**Problem.** The pipeline is domain-agnostic but useful runs depend heavily on domain-specific query design and known baselines. Without a starting point, every new problem requires the user to do the legwork.

**Solution.** First domain pack lives at `domain_packs/cs_algorithms/`:

- `topic_queries.json` — curated list of OpenAlex topic IDs + display names + suggested years_min for the CS / algorithms domain. Drawn from the topics already used in the existing experiments (embedding compression, cache replacement, Heilbronn, etc.).
- `problem_templates.json` — list of problem statements scaffolded for CS work, e.g. "Compress sentence embeddings to under N bits per vector with recall@10 ≥ X."
- `known_baselines.json` — mapping from problem family to the canonical baselines: e.g. embedding compression → [PQ, OPQ, RaBitQ, scalar quantization]; cache replacement → [LRU, ARC, SIEVE, S3FIFO]; small-points optimization → [SA, basin hopping, CMA-ES, DE].
- `evaluation_datasets.json` — public datasets per family: BEIR for retrieval, SNIA / MSR Cambridge for cache traces, Heilbronn published reference values, etc.
- `gotchas.md` — recurring traps (e.g. embedding bench using non-normalized vectors, argpartition off-by-one — same one Phase 2 fixed).
- `transient_deps.json` — recommended dev-only deps per family (numpy + scikit-learn for embeddings; cma + scipy for optimization; etc.).

New CLI: `python -m pipeline.domain_pack list` and `python -m pipeline.domain_pack apply cs_algorithms --problem "X"`. The `apply` command stamps the chosen domain pack into a fresh compose run: pre-fills topic list, fills the problem template, attaches the known baselines as a comment in the recipe stage, references the gotchas in the run's README.

**Acceptance.** Domain pack files exist and validate against simple pydantic schemas. `pipeline.domain_pack list` shows the pack. `pipeline.domain_pack apply cs_algorithms --problem "..." --dry-run` prints what the compose run would do without invoking the LLM.

## Non-goals (deferred)

- **Item 5: Expert hit-rate study.** Needs blind domain experts and a rating form. Can be set up as scaffolding (form template, anonymized sampler, scoring spec) in a follow-up session if the user wants.
- **Calibration dashboard as a live UI.** Markdown is enough for now. A Streamlit / static HTML dashboard is a later affordance.
- **Multi-domain packs.** Only CS / algorithms ships this phase. Other domains (chemistry, biology, physics) are templates the next user can fill.
- **Live novelty checks at compose time.** The novelty check is opt-in (`--with-novelty`) so default runs stay cheap and offline-capable.

## Risks

- **Item 1 prompt design.** Bad recipe prompts will produce vague "test it somehow" outputs. Mitigation: derive the prompt iteratively from real saved compositions; require structured JSON output via the schema.
- **Item 2 API limits.** S2 has rate limits; without an API key it's even tighter. Mitigation: opt-in flag, cache responses under `data/pipeline/novelty_cache/`, exponential backoff already exists in the prior scripts.
- **Item 3 finding doc parsing.** Verified-outcomes counting depends on a maintained `findings/INDEX.md`. Mitigation: write the index in this phase; future findings docs add a single line there. If the index isn't maintained, the dashboard skips that section.
- **Item 4 domain coverage.** A single CS pack will cover ~80% of existing experiments but won't generalize. Mitigation: explicit scope; other domains documented as templates not implementations.

## Definition of done

- All four items shipped or explicitly deferred in `findings/phase3-product-layer-20260521.md`.
- `uv run pytest` green (new schema tests, novelty classification tests, calibration fixture tests).
- `uv run ruff check .` green.
- A real run with `--with-novelty` and a saved-run calibration markdown report exist as concrete evidence.
