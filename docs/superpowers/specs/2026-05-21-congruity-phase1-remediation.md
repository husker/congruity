# Spec: Congruity Phase 1 — Correctness & Auditability

**Date:** 2026-05-21
**Status:** approved, ready for implementation planning
**Source:** [findings/project-deep-analysis-20260521.md](../../../findings/project-deep-analysis-20260521.md), Phase 1 of the prioritized remediation plan.

## Goal

Raise the floor on Congruity's compose-then-verify pipeline: every stage traceable, every parser strict, every gate honest about whether it ran. This phase is purely about correctness and auditability — no new generative capability, no benchmark hardening (Phase 2), no product layer (Phase 3).

## Scope

Five items, in roughly the order they should be implemented:

1. Ruff hygiene — clear 20 lint failures.
2. Corpus persistence + stable paper IDs.
3. LLM artifact persistence for every stage.
4. Schema validation for L1/L3/L4 + dimensions/scores/compositions.
5. Strict relevance gate for non-dry-run compose.

## Item 1: Ruff hygiene

**Problem.** `uv run ruff check .` reports 20 errors (17 auto-fixable, 3 F841 dead assignments). Lint is not green.

**Solution.**
- Run `uv run ruff check . --fix` to clear F541 / F401 mechanically.
- Manually remove the 3 F841 dead assignments:
  - `experiments/embed_quant/bench.py:47` (`exact = topk_cosine(...)`) — flagged in deep-analysis as actual bug; Phase 2 will reintroduce correctly.
  - `experiments/embed_quant/bench.py:51` (`calib_db = db[:1000]`) — same.
  - The remaining F841 (a `d` in `pipeline/pair_filter.py`) — remove.

**Acceptance.** `uv run ruff check .` → 0 errors. `uv run pytest` still green.

## Item 2: Corpus persistence + stable paper IDs

**Problem.** `compose_one` writes `corpus.json` *before* the corpus quality gate drops bad papers, then reindexes survivors (`pipeline/compose.py:814` then `:850–855`). Later reprocessing (`:1074`) reloads the pre-drop corpus, so `P1`, `P2`, … in saved artifacts can map to the wrong papers when re-audited.

**Solution.**

- Add a `paper_uid` field on every paper: `sha256(openalex_id || "title:" + title + ":" + str(year))[:16]`.
  - Computed once per paper during corpus assembly.
  - Stable across runs given the same identifying inputs.
  - Distinct from display IDs (`P1`, `P2`) used in prompts.
- In `compose_one`:
  - **Before gate:** write `corpus_raw.json` (all papers, with `paper_uid`).
  - **After gate:** reindex survivors to `P1…Pn`, preserve `paper_uid`, write `corpus_validated.json`.
  - Write `corpus.json` as a copy of `corpus_validated.json` (back-compat for any existing tools that load by that name).
- Update reprocess loader (`pipeline/compose.py:1074`) to prefer `corpus_validated.json` and fall back to `corpus.json`.

**Acceptance.**
- New test: same `(openalex_id, title, year)` → identical `paper_uid` across two runs.
- New test: distinct papers → distinct uids.
- New test: after a compose dry-run on a fixture corpus with one structure-failing paper, `corpus_validated.json` excludes it and `paper_uid` is preserved on survivors.

## Item 3: LLM artifact persistence

**Problem.** `run_codex` returns `(ok, stdout, stderr)`. Successful calls save parsed JSON, but raw prompt / raw stdout / raw stderr / model config are not consistently persisted per stage. Audit and replay are difficult.

**Solution.**

- New module `pipeline/llm_io.py` with `run_codex_with_artifacts(stage, prompt, run_dir, reasoning, seq=None) -> (ok, stdout, stderr, artifact_dir)`.
- Artifact layout per call: `<run_dir>/llm_calls/<stage>/<NN>/`
  - `prompt.txt` — verbatim prompt text.
  - `stdout.txt` — verbatim stdout.
  - `stderr.txt` — verbatim stderr.
  - `parsed.json` — parsed structured output (if parse succeeded).
  - `meta.json` — `{stage, model, reasoning, exit_code, duration_s, parse_status, validation_errors?}`.
- `<NN>` is a zero-padded sequence number so multiple calls within the same stage stay ordered (e.g. retry after schema-validation failure → `01/`, `02/`).
- Refactor the seven `run_codex` call sites in `pipeline/compose.py` (51, 172, 235, 339, 481, 645, 717) to use the wrapper, passing `run_dir` and a meaningful stage name.

**Acceptance.**
- New test: a dry-run compose against a stubbed codex produces the expected directory layout.
- After a real compose run, every stage has at least one `<NN>/` directory under `llm_calls/<stage>/`.

## Item 4: Schema validation

**Problem.** Codex outputs are currently parsed with regex/brace heuristics. Malformed responses can silently produce empty arrays, wrong field types, or partial structures that pass through to L3/L4 and downstream rendering.

**Solution.**

- Add `pydantic>=2` to `[project.dependencies]`.
- New `pipeline/schemas.py` with Pydantic v2 models for every structured LLM output:
  - `Dimension`, `PaperScore`
  - `Composition`, `CompositionPaper`
  - `L1Item`, `L3Verdict`, `L4Verdict`
  - Field names and types derived from existing prompt contracts in `pipeline/compose.py`.
- Parse flow:
  1. `parse_or_repair(raw_stdout, model_cls, llm_caller, max_repairs=1)`.
  2. First attempt: existing JSON extraction + `model_cls.model_validate`.
  3. On validation failure: serialize the validation error and call the LLM **once** with a repair prompt that includes the original prompt, the original response, and the error message.
  4. On second failure: write `parse_status: failed` and `validation_errors: [...]` into `meta.json`, return None to caller; caller decides whether to skip the stage or abort.
- Existing `run_codex` parsers in `compose.py` migrate to call `parse_or_repair`.

**Acceptance.**
- New tests: each model accepts a happy-path fixture; rejects each documented bad shape (missing field, wrong type, extra unknown field if model is strict).
- New integration test: a stubbed LLM that returns invalid JSON triggers exactly one repair retry; if both fail, `meta.json` has `parse_status: failed`.

## Item 5: Strict relevance gate

**Problem.** `pipeline/corpus_gate.py:141` short-circuits with `skipped=True` when no `OPENAI_API_KEY` is set, and treats embedding failures as non-fatal warnings (`:157`, `:163`). Saved `corpus_gate.json` files from real runs show the relevance check often did not run. Step 1.5 then reduces to structure + encoding checks.

**Solution.**

- Add `--strict-gate / --no-strict-gate` flag to compose CLI; default **strict** when not a dry-run, **non-strict** when `--dry-run`.
- In `validate_corpus`:
  - Accept a `strict: bool` parameter.
  - If `strict and rel_skipped`: set `report.aborted = True`, `report.abort_reason = "relevance check skipped (no API key or transient error) and --strict-gate is enabled"`.
- Compose CLI: refuse to advance to LLM stages if `report.aborted` is true.
- Digest renderer: when relevance was skipped (even in non-strict mode), surface a prominent badge: `⚠️ relevance check skipped — corpus not topic-filtered`.
- **Out of scope (Phase 3 candidate):** local embedding fallback (sentence-transformers). User decision recorded 2026-05-21: keep deps thin.

**Acceptance.**
- New tests: `validate_corpus(strict=True, api_key=None)` → `report.aborted = True`.
- New tests: `validate_corpus(strict=False, api_key=None)` → `report.aborted = False`, `report.skipped_checks` contains `"relevance"`.
- Compose CLI integration: running a non-dry-run compose with no API key exits non-zero before any LLM stage.

## Non-goals (deferred)

- Phase 2 benchmark hardening (round-trip tests, embed_quant fixes, Heilbronn baselines, real cache traces, fake-LLM integration tests).
- Phase 3 product layer (verification recipes, automated novelty checks, calibration dashboard, domain packs, expert hit-rate study).
- Local embedding fallback for the relevance gate.
- Backfill of `paper_uid` into existing saved run state directories (new runs only).

## Risks

- **Wider blast radius than expected:** refactoring 7 `run_codex` call sites in `compose.py` may touch behavior subtly. Mitigation: keep the original `run_codex` available, layer the wrapper on top, and migrate call sites one at a time with tests around each.
- **Pydantic schemas diverge from prompts:** if a prompt asks for a field name and the model expects another, every run fails. Mitigation: derive each schema directly from an existing happy-path saved response; commit those responses as fixtures.
- **Strict gate breaks dev loops:** anyone running compose without an API key will be blocked. Mitigation: clear error message includes the `--no-strict-gate` opt-out and the obvious `OPENAI_API_KEY=...` fix.

## Definition of done

- All five acceptance criteria above are met.
- `uv run pytest` green (new tests included).
- `uv run ruff check .` green.
- Track-record finding entry updated in `findings/` summarizing what changed and any newly observable risks.
