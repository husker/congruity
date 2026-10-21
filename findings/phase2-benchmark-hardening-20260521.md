# Phase 2 benchmark hardening - 2026-05-21

Implementing the Phase 2 plan from `findings/project-deep-analysis-20260521.md`.

## Shipped

- **C1 short-string deterministic hash + round-trip tests.** Replaced Python's randomized `hash()` with a stable BLAKE2 64-bit prefix; minimizer signatures (and thus encoded outputs) are now identical across processes. New `tests/test_c1_roundtrip.py` round-trips the JSON, HTTP-header, and code corpora plus out-of-training and tiny/empty inputs (9 tests).
- **Multitrack stems lossy invariants + decoder + bug fix.** Renamed the codec as lossy (the docs claimed lossless but the encoder clipped overflow samples). Added a `decode_residual` function that mirrors the encoder's integer math. Found and fixed a real bug: `W_q.astype(np.int16)` silently wrapped near-unit weights to negative values, which the new round-trip test catches via a zero-overflow lossless case. New `tests/test_multitrack_roundtrip.py` pins: non-overflow round-trips bit-exact, overflow error bounded by int16 range, overflow count agrees with overflow mask, zero-overflow case is fully lossless (4 tests).
- **Embed-quant harness fixes.** Hoisted the exact top-k computation out of the per-method loop (it was recomputed identically once per method). Fixed an `np.argpartition` off-by-one (`kth = k - 1`, not `k`). Relabeled C3 methods with the actual bit count after divisor-rounding (a "C3-var-32b" that lands on 30 bits is now correctly labelled "C3-var-30b"). New `tests/test_embed_quant_topk.py` (3 tests) pins the argpartition / argsort equivalence.
- **Project dep hygiene.** numpy, scikit-learn (pulls scipy transitively), cma, jax are now declared in `pyproject.toml`. The default `uv run pytest` and `uv run python -m experiments.<x>.bench` commands work out of the box.
- **Heilbronn: CMA-ES + DE baselines + multi-seed runner.** `experiments/heilbronn/baselines_cmaes.py` wraps the `cma` package; `baselines_de.py` wraps `scipy.optimize.differential_evolution`. Both follow the same `(n_points, n_steps, n_starts, seed) -> (pts, area)` signature as the existing SA baselines. The bench now accepts `--ns`, `--seeds`, `--n-steps`, `--n-starts`, `--methods` and reports mean ± 95% Student-t CI plus per-seed best and total time per (method, n).
- **Compose integration tests.** New `tests/test_compose_integration.py` runs `compose_one` end-to-end against a fixture corpus and a stubbed `_invoke_codex` that pattern-matches on prompt text to return canned outputs per stage. Two tests: full pipeline produces the expected `state.json`, `corpus_raw.json`, `corpus_validated.json`, and per-stage `llm_calls/<stage>/NN/` artifacts; strict gate aborts cleanly when `OPENAI_API_KEY` is missing.

## Test count

Phase 1 finished at 58 passing tests. Phase 2 ships **76 passing tests** (+18):
- +9 C1 round-trip
- +4 multitrack round-trip
- +3 embed-quant top-k
- +2 compose integration

`uv run ruff check .` green throughout.

## Plan-vs-reality findings during implementation

- **Multitrack predictor had a latent bug.** The encoder truncated quantized weights to `int16` before returning them, which silently wrapped values >= 32768 to -32768. The new lossless round-trip test (target == sources[0]) revealed this immediately; weights are now returned as int32. Side-info size roughly doubles for that branch but the negative conclusion stands.
- **`hash()` randomization affects more than just lookup.** The C1 deterministic-hash fix also stabilizes the encoded output bytes across processes, not just the minimizer-index keys. The new tests assert byte-level equality between two codecs trained on the same corpus.
- **The bench's exact top-k recomputation didn't change recall numbers** (the input was the same per method) but did waste ~5–10s per method on large `main_db`. Moving it outside the loop is a pure speedup; the recall conclusions in `findings/embed_quant-c3-verification-20260520.md` still stand.
- **Cache replacement experiment already had realistic traces.** `experiments/cache_replacement/traces.py` defines `zipf_trace`, `loop_trace`, `scan_with_hot_trace`, and `paired_trace`; the bench already runs every policy against every trace at multiple cache sizes. The deep-analysis recommendation to "add realistic traces" was largely already satisfied; remaining work (SNIA adapter, prefetcher variant) is deferred.

## Commits

| SHA | Summary |
|---|---|
| 5a07b1d | docs: Phase 2 benchmark-hardening spec |
| 95b985a | docs: Phase 2 implementation plan |
| 918ed5e | fix(c1_short_string): stable hash + round-trip tests |
| b948ec6 | fix(multitrack_stems): rename codec lossy + decoder + tests |
| 991ac34 | chore: add numpy as a project dep |
| dc3d489 | fix(embed_quant): correct top-k + hoist exact + accurate labels |
| 6c1f160 | chore: add cma + scipy deps |
| 9cf2127 | feat(heilbronn): CMA-ES + DE + multi-seed bench with CIs |
| de287f5 | test(compose): end-to-end integration tests with stubbed LLM |

## Deferred to a follow-up

These were in the Phase 2 plan but are deferred because the value gradient was clearly diminishing within this session:

- **Tuned basin-hopping baseline** for Heilbronn (existing scipy_basinhopping is already a reasonable baseline).
- **C1 schedule + sharpness ablation** for Heilbronn (a useful follow-up to characterize C1's robustness; not load-bearing for the win-or-lose claim).
- **SNIA-style trace adapter** + **prefetcher variant** for cache replacement. The existing `traces.py` already covers Zipf / loop / scan_with_hot / paired patterns, which span most of what the deep-analysis flagged as missing. The prefetcher addition is the highest-value remaining item because the deep-analysis correctly noted that successor prediction is a prefetching mechanism, not a replacement mechanism.

## Deferred to Phase 3

- Verification recipes per composition.
- Automated novelty checks.
- Calibration dashboard.
- Domain packs.
- Expert hit-rate study.
- Local embedding fallback for the relevance gate.

## Open items inside Phase 2

- **Bench output baselines may shift.** The off-by-one fix in `topk_cosine` and the relabelled C3 bit counts could change the embed_quant table in `findings/embed_quant-c3-verification-20260520.md` slightly. A re-run with the corrected code is worth doing before treating the published numbers as canonical.
- **JAX is now a hard dep.** The Heilbronn `c1_smooth.py` requires jax; previously this was invoked via `uv run --with jax`. Adding jax to `[project] dependencies` makes the bench reproducible out of the box but inflates the test/CI environment by ~100MB. Reconsider if a JAX-optional refactor would be worthwhile.
