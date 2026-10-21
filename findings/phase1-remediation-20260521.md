# Phase 1 remediation - 2026-05-21

Implementing the Phase 1 plan from `findings/project-deep-analysis-20260521.md`.

## Shipped

- Ruff lint clean (was 20 errors; 17 auto-fixed + 3 dead assignments removed).
- `paper_uid` stable identifier (16-hex SHA-256 prefix) attached to every paper via `pipeline/identity.py`.
- Dual-write: `corpus_raw.json` (pre-gate) and `corpus_validated.json` (post-gate). `corpus.json` retained as a back-compat copy of the validated set.
- Per-call LLM artifacts under `<run_dir>/llm_calls/<stage>/<NN>/` (`prompt.txt`, `stdout.txt`, `stderr.txt`, `meta.json` with stage, reasoning, exit_code, duration_s, parse_status, validation_errors). New module `pipeline/llm_io.py`.
- Pydantic v2 schemas for every LLM-emitted shape in `pipeline/schemas.py`. `DimensionsOutput`, `ScoresOutput`, `CompositionsOutput`, and the composite models inside them use `extra="forbid"`. `L1Item`, `L3Verdict`, and `L4Verdict` use `extra="allow"` because real LLM outputs are richer than the original prompt contracts suggested.
- One-shot repair retry on schema validation failure via `parse_or_repair`. Both attempts persist artifacts so audit can see the original and repaired outputs.
- All six LLM-call functions in `compose.py` migrated:

  | Function | Stage name | Schema |
  |---|---|---|
  | extract_dimensions | `dimensions` | `DimensionsOutput` |
  | score_papers | `scoring` | `ScoresOutput` |
  | find_compositions | `compositions` | `CompositionsOutput` |
  | verify_consistency | `verify_l1_<cid>` | `L1Item` |
  | verify_adversarial | `verify_l3_<cid>` | `L3Verdict` |
  | extract_testable_predictions | `verify_l4_<cid>` | `L4Verdict` |

  Old `run_codex` and `_parse_json_block` helpers deleted; `subprocess` import auto-pruned.
- Strict relevance gate (`--strict-gate` on by default for non-dry-run). Aborts before any LLM stage if `OPENAI_API_KEY` is missing or the embedding call fails. `--no-strict-gate` opt-out documented in `--help`.
- `print_report` now prefixes skipped checks with `⚠️ SKIPPED CHECKS:`. `state.json` gains a top-level `relevance_skipped` boolean for downstream tools.

## Validation at end of phase

- `uv run pytest`: **58 passed** (was 30 at start). New coverage: paper_uid stability (5), corpus dual-write (1), schemas (13), parse_or_repair + artifact wrapper (6), strict gate (2), digest badge (1).
- `uv run ruff check .`: clean (was 20 errors).
- `uv run python -m pipeline.compose --help` shows the new `--strict-gate / --no-strict-gate` flags adjacent to the existing `--skip-gate`.

## Plan-vs-reality corrections made during implementation

- **L3 verdict literals are lowercase** in real saved data (`sound|weak|invalid`), not the uppercase the plan listed. Schema literal corrected.
- **L4 has no top-level `verdict` field**. The verdict-equivalent lives in `summary.net_testability_score`, which is either the literal `"PURELY_TESTABLE"` or a numeric ratio. `L4Verdict` and `L4Summary` restructured accordingly with `extra="allow"`.
- **L1Item.issues is a list of dicts**, not strings — the consistency prompt asks for `{severity, detail, paper_refs}`. Schema relaxed to `list[dict]`.
- **`find_compositions` migration kept a defensive `run_dir=None` fallback** initially; cleaned up before commit so the wrapper path is the only path.

These corrections argue for adding a "sample real saved outputs before locking schemas" step to Phase 2 / Phase 3 design work.

## Commits

| SHA | Summary |
|---|---|
| 4561897 | docs: phase 1 remediation spec |
| 9f5b34b | docs: phase 1 implementation plan |
| e519441 | chore: clear all ruff failures |
| e2ced6d | feat: paper_uid stable identifier |
| cfdfb79 | feat: dual-write corpus artifacts |
| a8e807a | chore: add pydantic v2 |
| a828785 | feat: pydantic schemas |
| bad6b07 | feat: parse_or_repair |
| 9976c56 | feat: run_codex_with_artifacts wrapper |
| cb0cf47 | refactor: migrate dimensions/scoring/compositions |
| 2848ff8 | refactor: migrate L1/L3/L4 verifiers + delete old helpers |
| b8a939e | feat: strict relevance gate |
| dd320f7 | feat: surface skipped relevance check prominently |

## Deferred to Phase 2

- Embed-quant top-k fix, calib_db reuse, bit-budget labeling.
- Round-trip tests for compression experiments.
- Stronger Heilbronn baselines + seed sweeps + CIs.
- Real cache traces (SNIA / MSR Cambridge).
- Compose integration tests with fake OpenAlex + fake codex.

## Deferred to Phase 3

- Verification recipes per WEAK+PURELY_TESTABLE composition.
- Post-L3 novelty check stage.
- Calibration dashboard.
- Domain packs.
- Expert hit-rate study.

## Open items inside Phase 1

- **Aborted-gate corpus persistence:** when the gate aborts (e.g., fail rate too high), the new code does NOT write `corpus_raw.json` or `corpus_validated.json`. The audit data on aborted runs is therefore weaker than on successful runs. Accepted trade-off for this phase; `corpus_gate.json` still captures the failure reason. Revisit if aborted-run audits become important.
- **Local embedding fallback for the relevance gate:** intentionally not added per the 2026-05-21 spec decision (keep deps thin). Revisit if missing API key blocks too many dev runs.
- **Backfill of paper_uid into pre-existing saved run state directories:** out of scope by design. New runs only.
