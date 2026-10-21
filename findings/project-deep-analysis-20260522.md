# Congruity project deep analysis - 2026-05-22

## Bottom line

The project is materially better than the 2026-05-21 snapshot. The important
upgrades are real: ruff is clean, the test suite grew from 30 to 154 tests,
experiment harness bugs were fixed, LLM calls are now persisted for audit, corpus
artifacts are split into raw and validated forms, scoring is batched, empty-array
LLM failures are guarded at the schema layer, and the calibration dashboard now
joins compose runs to verified outcomes.

However, the newest code has two high-priority accuracy regressions in the main
compose product path:

1. The `CompositionsOutput` schema no longer matches the compose prompt. Real
model output includes `supporting_quote`, `supporting_dimensions`,
`compatibility_argument`, `covered_dimensions`, `open_questions`, `confidence`,
and `confidence_reason`, but the strict schema only accepts `id`,
`synthesis_claim`, and `papers[].quote`. The repair step then strips the richer
fields. As a result, L2 source grounding fails every repaired composition with
`NO_QUOTE_PROVIDED`, digests lose compatibility/confidence/open questions, and
recipe quality is weakened.

2. The documented top-level CLI, `uv run python -m pipeline compose`, does not
expose the new compose controls. The richer CLI lives at
`uv run python -m pipeline.compose`. That means the common path cannot pass
`--relevance-mode`, `--strict-gate`, `--no-strict-gate`, `--skip-gate`,
`--with-recipes`, or `--with-novelty`, and it does not load `.env` before
calling `compose_one`.

The overall product is improving, but the next change should be a correctness
pass on the compose contract, not more features.

## Validation run

Commands run on the current tree:

```bash
uv run pytest
uv run ruff check .
uv run python -m pipeline compose --help
uv run python -m pipeline.compose --help
uv run python -m pipeline.calibration --format md --out /private/tmp/congruity-calibration-20260522.md
uv run python -m pipeline.calibration --out /private/tmp/congruity-calibration-20260522.html
uv run python -m oracle.runner --n-requests 50 --capacity 50 --policies lru,arc,paged_stub
uv run python -m experiments.cache_replacement.bench
uv run --with brotli --with zstandard --with hpack python -m experiments.c1_short_string.bench --small
```

Results:

- `pytest`: 154 passed in 3.52s.
- `ruff`: all checks passed.
- Top-level compose help only exposes basic compose args.
- Module compose help exposes the new relevance, strict-gate, recipe, and novelty args.
- Calibration markdown generated successfully for 24 compose runs.
- Calibration aggregate: 24 runs, 17 complete, 5 aborted at corpus gate, 2 dry runs,
  74 total compositions, L3 mix 38 invalid / 36 weak, L4 mix 62 PURELY_TESTABLE /
  8 numeric_ratio / 4 missing.
- Current saved-state files have no persisted `verification.reconciled` fields yet;
  the dashboard derives reconciliation on the fly for older runs.
- Oracle runner: LRU 86.5%, ARC 85.4%, PagedEvictionStub 86.5% on the small trace.
- Cache replacement benchmark still confirms the clean-negative story: Succ-SIEVE
  is mostly tied with SIEVE and slower.
- C1 short-string small benchmark still confirms the partial-positive story:
  C1 wins on JSON, is competitive on real HTTP headers, loses to HPACK-stateful
  on HTTP, and loses to zstd+dict on plain `text128`.

No `tsc --noEmit` was run because this is a Python project with no TypeScript
build surface.

## Improvements since the previous analysis

### 1. Basic code health improved

The previous analysis found ruff failures. The current tree is ruff-clean.
Several experiment-harness issues have also been addressed:

- C1 now uses deterministic BLAKE2 hashes instead of process-randomized
  `hash()` for minimizers. See `experiments/c1_short_string/c1.py:40`.
- C1 has round-trip and determinism tests. See `tests/test_c1_roundtrip.py:1`.
- Embed-quant fixed the `argpartition` top-k off-by-one. See
  `experiments/embed_quant/bench.py:20` and `tests/test_embed_quant_topk.py:1`.
- Embed-quant now computes the exact top-k once and relabels C3 methods with
  realized bit counts after divisor rounding. See
  `experiments/embed_quant/bench.py:64` and `experiments/embed_quant/bench.py:90`.
- Multitrack stems now documents that overflow makes the prototype bounded-error,
  not lossless, and keeps quantized weights as int32 to avoid sign wrap. See
  `experiments/multitrack_stems/predictor.py:13` and
  `experiments/multitrack_stems/predictor.py:124`.

### 2. LLM auditability improved

The new `pipeline/llm_io.py` wrapper saves `prompt.txt`, `stdout.txt`,
`stderr.txt`, and `meta.json` under `llm_calls/<stage>/<NN>/`. This is a major
product improvement because failures are now inspectable without reconstructing
terminal logs. See `pipeline/llm_io.py:156`.

One gap remains: repair artifacts are created, but their `meta.json` often stays
at `parse_status: "pending"`. This happened in both current TSP runs:

- `data/pipeline/compositions/20260522-062501/llm_calls/compositions_repair/01/meta.json`
- `data/pipeline/compositions/20260522-154341/llm_calls/compositions_repair/01/meta.json`

The primary composition artifact says `parse_status: "repaired"`, but the repair
artifact itself does not record parse success/failure. That makes the audit trail
partially opaque exactly where the most interesting failure happened.

### 3. Corpus persistence improved

The raw/validated split fixes the previous save-before-gate problem:

- `corpus_raw.json` captures the pre-gate corpus.
- `corpus_validated.json` captures the post-gate corpus.
- `corpus.json` mirrors validated corpus for backward compatibility.

See `pipeline/compose.py:1108` and `tests/test_corpus_persistence.py:1`.

### 4. Scoring is much more robust

The code now batches scoring into chunks of 20, preserves corpus-level paper IDs,
performs ID-aware repair, deduplicates repaired scores, and writes coverage
metadata. See `pipeline/compose.py:238`, `pipeline/compose.py:275`, and
`pipeline/compose.py:371`.

The 2026-05-22 run sequence shows why this mattered:

- `20260522-062501`: pre-hardening complete run, 95-papers, but `scores.json`
  contained 0 scores. It still generated 7 compositions, which is dangerous.
- `20260522-154341`: post-hardening smaller run, 31/31 papers scored across
  two batches, with `scoring_coverage: 1.0`.

The new tests cover the original failure modes well. See
`tests/test_score_papers_batched.py:1`.

### 5. Calibration dashboard is now useful

`pipeline/calibration.py` now aggregates saved runs, L3/L4 verdicts, verified
outcomes from `findings/INDEX.md`, LLM wall time, skipped relevance gates, and
derived reconciled verdicts. The HTML report has sortable tables and chart
tooltips. See `pipeline/calibration.py:323` and `pipeline/calibration.py:621`.

This is the right product direction. The product needs calibration and
traceability more than it needs more speculative generation.

## Main remaining risks

### P0: Composition schema does not match the prompt contract

The compose prompt asks for this richer shape:

- `papers[].supporting_dimensions`
- `papers[].supporting_quote`
- `compatibility_argument`
- `covered_dimensions`
- `open_questions`
- `confidence`
- `confidence_reason`

See `pipeline/compose.py:480`.

The schema accepts only:

- `CompositionPaper.paper_id`
- `CompositionPaper.role`
- `CompositionPaper.quote`
- `Composition.id`
- `Composition.synthesis_claim`
- `Composition.papers`

See `pipeline/schemas.py:62`.

Because `_StrictBase` forbids extra fields, valid prompt-following output is
schema-invalid. The generic repair prompt coerces the response into the narrower
schema, and the surviving objects lack the fields that the rest of compose
expects.

Observed effect in `20260522-154341`:

- Raw composition output had full `supporting_quote`, `supporting_dimensions`,
  compatibility, open questions, and confidence.
- Repaired output retained only `quote`.
- `verify_source_grounding()` looks for `supporting_quote`, not `quote`, so L2
  failed all compositions as `NO_QUOTE_PROVIDED`.
- The digest rendered empty quote/dimension/confidence sections.

Relevant code:

- Prompt contract: `pipeline/compose.py:447`.
- L2 quote lookup: `pipeline/compose.py:732`.
- Narrow schema: `pipeline/schemas.py:62`.
- Digest expects rich fields: `pipeline/compose.py:1357`.

Recommended fix:

- Make `Composition` and `CompositionPaper` match the prompt exactly.
- Add backward-compatible aliases so `quote` maps to `supporting_quote`.
- Add a regression test that feeds a real prompt-shaped composition into
  `CompositionsOutput` and asserts L2 grounding uses the quote.
- Add a digest snapshot test so compatibility/confidence/open questions cannot
  silently disappear again.

### P0: Stage 4 can run after low or zero scoring coverage

The current hardening records coverage warnings, but `compose_one()` does not
gate on coverage before Stage 4. That allowed `20260522-062501` to complete with
0/95 scored papers and still surface 7 compositions.

Relevant code:

- `score_papers()` only returns scores, no status: `pipeline/compose.py:371`.
- `compose_one()` proceeds immediately after printing count:
  `pipeline/compose.py:1240`.

Recommended fix:

- Return `coverage`, `empty_batches`, and `n_batches` from `score_papers()`.
- Abort with `state.status = "insufficient_scoring_coverage"` if coverage is
  below a configured floor or any batch is empty under strict mode.
- Make `find_compositions()` refuse to run when `scores` is empty unless the user
  explicitly passes a debug override.

### P0: Top-level CLI does not expose the new product controls

`uv run python -m pipeline compose --help` exposes only basic args. The new
flags are only available through `uv run python -m pipeline.compose --help`.

Relevant code:

- Top-level runner compose command: `pipeline/runner.py:202`.
- Module compose parser: `pipeline/compose.py:1493`.

Impact:

- README and AGENTS.md document `python -m pipeline compose`.
- Users on the documented path cannot request recipes, novelty checks, relevance
  backend selection, threshold tuning, strict gate behavior, or gate bypass.
- `.env` loading is only wired in `pipeline.compose.main()`, not the top-level
  runner path.

Recommended fix:

- Either make `pipeline.runner.cmd_compose()` delegate to `pipeline.compose.main`
  semantics or copy the full argument surface into the top-level subcommand.
- Load `.env` at the start of `pipeline.runner.main()`.
- Add a CLI parity test comparing key flags between the two help outputs.

### P1: Reconciliation exists but current run states do not persist it

The code now writes `verification.reconciled` for future runs. Current saved
states show `rec: missing` for all 74 historical compositions; the dashboard
derives reconciliation on the fly.

That is acceptable for backward compatibility, but it means reports and raw
state files disagree on whether reconciliation is present. The current compose
digests also do not display the reconciled verdict.

Relevant code:

- Reconciler: `pipeline/compose.py:1015`.
- Future-state write: `pipeline/compose.py:1267`.
- Dashboard fallback: `pipeline/calibration.py:85`.
- Digest omission: `pipeline/compose.py:1314`.

Recommended fix:

- Add a safe reprocess command for reconciliation only, without rerunning LLMs.
- Render reconciled status in compose digest.
- In future reports, separate "raw L3/L4" from "actionable status" so users do
  not over-select L3 invalid plus L4 testable candidates.

### P1: L4 still over-classifies invalid objections as empirical

The dashboard/handoff reports 28 of 74 compositions, about 38 percent, with
L3=invalid plus L4=PURELY_TESTABLE disagreement. The reconciler prevents those
from becoming recipes, but the underlying classifier drift remains.

Recommended fix:

- Change the L4 prompt so it receives the L3 final verdict and must explain how
  an objection can be empirical without contradicting an invalid verdict.
- Add calibration examples where L3 invalid should produce fundamental L4
  objections.
- Track disagreement rate as a dashboard quality metric with a target below
  10 percent.

### P1: Relevance gate is still hard to calibrate

The new OpenAI/Anthropic/Codex relevance backends are a step forward, but the
May 22 TSP runs show threshold brittleness:

- Several broad/focused runs aborted with 93-100 percent relevance failures.
- The successful `20260522-062501` run used `--relevance-mode none`.
- Later small runs also had `relevance_skipped: true`.

Relevant code:

- Relevance dispatch: `pipeline/corpus_gate.py:274`.
- LLM judge prompt: `pipeline/relevance.py:72`.
- `auto` mode preference: `pipeline/relevance.py:375`.

Recommended fix:

- Store relevance scores, not just failures, so threshold choices are inspectable.
- Add per-backend calibration docs for score ranges.
- In the dashboard, split "skipped because user chose none" from "skipped because
  backend failed or auth was missing".
- Consider two thresholds: a soft warning band and a hard drop threshold.

### P2: Novelty and domain packs are still stubs

The product-layer direction is right, but the current implementations are not
yet product-grade:

- Novelty search helpers return empty lists unless tests monkeypatch them. See
  `pipeline/novelty.py:45`.
- Co-citation density returns 0.0. See `pipeline/novelty.py:55`.
- Domain pack `apply` is dry-run only and says live compose integration is a
  follow-up. See `pipeline/domain_pack.py:53`.

These should be described as scaffolding, not as complete product features.

### P2: Documentation is now stale in important places

`README.md` still points users to `python -m pipeline compose`, which misses the
new flags. `OVERVIEW.md` says "Status as of 2026-05-19" and describes older
validation totals and claims that are not aligned with the current calibration
index.

Recommended fix:

- Update README quickstart with the full preferred compose command.
- Update OVERVIEW after the schema/CLI correctness pass.
- Add a short "current command surface" section documenting `pipeline` versus
  `pipeline.compose` until those are unified.

## Product assessment

The best product is not "more compositions." It is a calibrated verification
workbench:

1. A run page that makes corpus quality, scoring coverage, L3/L4 agreement,
   source grounding, and recipe eligibility visible.
2. A strict compose contract that preserves evidence fields end-to-end.
3. A verification queue of actionable candidates, each with named baselines and
   a one-session benchmark plan.
4. A dashboard that tracks verified hit rate by domain and failure mode.

The new dashboard, recipes, corpus persistence, and LLM artifacts move in this
direction. The current schema/CLI issues are fixable and should be treated as
blocking because they directly affect user trust.

## Prioritized remediation plan

### Phase 1: Restore compose contract integrity

- Align `CompositionsOutput` with `COMPOSE_PROMPT_TEMPLATE`.
- Accept old `quote` as an alias for `supporting_quote`.
- Make L2 use the normalized quote field.
- Add tests using real TSP raw composition output.
- Mark repair artifact meta with parse status.

Expected effect: L2 will become meaningful again, digests regain evidence and
confidence fields, and recipes get richer input.

### Phase 2: Gate downstream stages on scoring health

- Return scoring coverage metadata from `score_papers()`.
- Abort or require explicit override when coverage is too low.
- Add an integration test where scoring returns empty and assert Stage 4 does
  not run.

Expected effect: no more complete runs with generated compositions from an
unscored corpus.

### Phase 3: Unify the CLI

- Mirror all compose args in `pipeline.runner`.
- Load `.env` in the top-level runner.
- Add CLI parity tests.
- Update README and AGENTS quickstart examples.

Expected effect: documented commands will exercise the actual product.

### Phase 4: Recalibrate L4 and relevance

- Add L3/L4 disagreement examples to tests/prompts.
- Store relevance score distributions.
- Separate skipped relevance reasons in state and dashboard.

Expected effect: fewer false-actionable signals and clearer corpus quality
decisions.

### Phase 5: Finish product scaffolding

- Implement real S2/OpenAlex novelty search.
- Implement co-citation density.
- Wire domain packs into compose.
- Add run-detail HTML or markdown pages that summarize audit fields per
  composition.

Expected effect: the product becomes easier to use without reading raw JSON.

## Verdict

The project is on the right track and substantially more robust than yesterday's
snapshot. The strongest new evidence is the clean 154-test suite, the successful
post-hardening small TSP run with 31/31 scoring coverage, and the new dashboard
that makes calibration visible.

The key caution is that the newest code also introduced a contract mismatch that
turns rich composition output into impoverished repaired output. That is the next
thing to fix. Once the compose schema, scoring gate, and CLI are aligned, the
project will be in a much better position to verify the next candidate rather
than merely generate one.
