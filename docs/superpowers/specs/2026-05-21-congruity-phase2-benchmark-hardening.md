# Spec: Congruity Phase 2 — Benchmark Hardening

**Date:** 2026-05-21
**Status:** approved, ready for implementation planning
**Source:** [findings/project-deep-analysis-20260521.md](../../../findings/project-deep-analysis-20260521.md), Phase 2 of the prioritized remediation plan. Builds on Phase 1 ([spec](2026-05-21-congruity-phase1-remediation.md), [findings](../../../findings/phase1-remediation-20260521.md)).

## Goal

Tighten the experimental harnesses so every published result is reproducible and the baselines are strong enough that negative results are credible and positive results are not artifacts of weak comparisons. This phase is purely about making the existing experiments more trustworthy — no new generative pipeline capability, no product layer (Phase 3).

## Scope

Five items, in roughly the order they should be implemented (1, 2, 5 are tight test/correctness work; 3 and 4 add experimental depth):

1. Round-trip tests for compression experiments (C1 short-string, multitrack stems).
2. Embed-quant harness fixes: top-k off-by-one, calibration-split reuse, bit-budget labeling.
3. Stronger Heilbronn baselines: CMA-ES, differential evolution, tuned basin hopping, multi-seed runs with confidence intervals, schedule ablations for C1.
4. Real cache traces: SNIA / MSR Cambridge–style block traces (synthesized in the same style if downloading the full datasets is impractical).
5. Compose integration tests with stubbed OpenAlex + stubbed codex.

## Item 1: Round-trip tests for compression experiments

**Problem.**

- C1 short-string (`experiments/c1_short_string/c1.py:70`) uses Python's randomized `hash()` for minimizer signatures, so candidate lookups are non-reproducible across processes. No round-trip tests exist over the corpora the experiment uses, so we can't tell whether the "compression" actually preserves the input bits.
- Multitrack stems (`experiments/multitrack_stems/predictor.py:95`) clips overflow samples and records `overflow_count` but never encodes a real lossless fallback for them. The codec is documented as lossless when it is not.

**Solution.**

- Replace `hash()` in C1 with a stable hash (BLAKE2 64-bit prefix, or a deterministic xxhash-style int).
- Add `tests/test_c1_roundtrip.py`: for each input corpus the experiment uses, encode → decode → assert original equals decoded. Cover the JSON, HTTP-header, and code-snippet corpora.
- For multitrack stems, either implement a true overflow fallback (raw-pass-through for overflow samples) **or** rename the codec as lossy and add `tests/test_multitrack_roundtrip.py` that asserts the documented lossy invariants (e.g., bit-exact on non-overflow samples, max absolute error bounded for overflow samples).
- Add a hierarchy ablation for C1 (the composition named hierarchy but the current prototype uses flat templates).

**Acceptance.** Round-trip tests pass for every corpus C1 evaluates on. Multitrack codec either survives a true lossless round-trip OR is renamed lossy with bounded-error tests passing. C1's hashing is deterministic across processes.

## Item 2: Embed-quant harness fixes

**Problem.** From the deep-analysis:

- `experiments/embed_quant/bench.py:20` — exact top-k recomputed inside every loop (was `exact = topk_cosine(queries, db, k=K)` flagged as F841 in Phase 1; removed but the underlying need for exact top-k remains).
- `experiments/embed_quant/bench.py:47` — calibration split (`calib_db = db[:1000]`) was assigned and unused; C3 was supposed to use it for utility allocation.
- `np.argpartition(-sims, k)` should be `np.argpartition(-sims, k - 1)` for top-k (off-by-one).
- C3 methods labeled `128b`/`256b` sometimes use fewer bits after allocation rounding; the label is misleading.

**Solution.**

- Reintroduce `exact = topk_cosine(queries, db, k=K)` **once, outside the loop**. Use it as the ground-truth top-k for recall.
- Pass `calib_db` into C3 methods that use `allocation_mode="utility"`, and verify the calibration is actually consumed.
- Fix the `np.argpartition` call sites to use `k - 1`.
- Compute the actual bit budget after allocation rounding and use it in the method label (`C3-var-128b` becomes `C3-var-126b` if rounding lands there).

**Acceptance.** Recall@k numbers move (or hold) with documented reasons. Each C3 method's label exactly matches its post-allocation bit count. New test `tests/test_embed_quant_topk.py` asserts `np.argpartition(-sims, k - 1)[:, :k]` matches a brute-force `argsort` top-k on a tiny fixture.

## Item 3: Stronger Heilbronn baselines

**Problem.** The current Heilbronn benchmark uses simulated annealing and SciPy basin-hopping. The deep-analysis recommends adding CMA-ES, differential evolution, tuned basin hopping, longer SA sweeps, multiple seeds with confidence intervals, and schedule ablations for C1.

**Solution.**

- Add `experiments/heilbronn/baselines_cmaes.py` using `cma` (or a vendored small implementation if we want to stay deps-thin).
- Add `experiments/heilbronn/baselines_de.py` using `scipy.optimize.differential_evolution`.
- Add a tuned-basin-hopping variant with larger step size / more inner iterations.
- Refactor the benchmark runner to run each method across N seeds (default 10) and report mean ± 95% CI per method per problem size.
- Add a C1 schedule ablation: hold-everything-else, vary the annealing temperature schedule and the smooth-min sharpness parameter, plot/print the sensitivity.

**Acceptance.** Benchmark output table shows mean and CI per method per n. C1's win/loss against the strongest baseline is unambiguous (CI bars don't overlap when the claim is "C1 beats X"). New deps documented in `pyproject.toml`.

## Item 4: Real cache traces

**Problem.** The cache replacement experiment uses synthetic workload generators. Real workloads have temporal locality, scan patterns, and access skew that synthetic generators struggle to capture. The current negative result for Succ-SIEVE may not generalize.

**Solution.**

- Add `experiments/cache_replacement/traces/` with a small adapter for SNIA-style trace files (timestamp, op, lba, size).
- Either:
  - Bundle a small public SNIA-derived sample (a few hundred MB), with a download script that verifies checksums, **or**
  - Generate synthetic traces that mimic SNIA/MSR Cambridge patterns (heavy-tailed distribution, scans, hot/cold partition) without downloading anything.
- Re-run the cache benchmark on traces and compare LRU / ARC / SIEVE / Succ-SIEVE. Add a prefetcher variant since the deep-analysis correctly noted that successor prediction is a prefetching mechanism, not a replacement mechanism.

**Acceptance.** Benchmark runs on at least two trace styles (random, real-or-real-like). Prefetcher variant added. Conclusion documents whether Succ-SIEVE's negative result holds on real(-like) traces.

## Item 5: Compose integration tests with stubbed LLM + OpenAlex

**Problem.** The compose pipeline has unit tests for individual stages but no end-to-end smoke test. A regression in the orchestrator (e.g., a state.json field rename, an arg threading bug) wouldn't be caught.

**Solution.**

- New `tests/test_compose_integration.py`.
- Stub `pipeline.corpus.sample_topic` to return a deterministic fixture corpus.
- Stub `pipeline.llm_io._invoke_codex` to return fixture JSON outputs per stage (dimensions, scoring, compositions, L1, L3, L4).
- Run `compose_one` end-to-end with `dry_run=False`, `skip_gate=True`, `strict_gate=False`. Assert:
  - state.json has the expected top-level keys (`run_id`, `problem`, `corpus_size`, `dimensions`, `compositions`, `status="complete"` or similar).
  - `corpus_raw.json` and `corpus_validated.json` both exist.
  - At least one composition has L1/L2/L3/L4 verification dicts.
  - `llm_calls/dimensions/01/meta.json` exists with `parse_status="ok"`.
- A second integration test covers the `--strict-gate` no-API-key abort path.

**Acceptance.** Two new integration tests pass. They are fast (< 5s combined) because the LLM stub is in-process. Any future refactor to compose_one that breaks state.json shape will fail this test.

## Non-goals (deferred to Phase 3)

- Verification recipes per composition.
- Automated novelty checks.
- Calibration dashboard.
- Domain packs.
- Expert hit-rate study.
- Local embedding fallback for the relevance gate.

## Risks

- **Item 3 dep weight.** `cma` adds a non-trivial dep; if we vendor a small CMA-ES we own more code. Mitigation: prefer `cma>=3` as a real dep, document the choice.
- **Item 4 dataset weight.** SNIA traces can be huge. Mitigation: prefer the synthesized-style approach unless we hit a clear case where real traces would change the verdict.
- **Item 5 stub fidelity.** A fake codex output that's too perfect could mask schema drift. Mitigation: derive the fixture from a real saved compose run's `llm_calls/<stage>/01/stdout.txt` so the stub matches actual model behavior.

## Definition of done

- All five items shipped or explicitly deferred with a note in `findings/phase2-benchmark-hardening-20260521.md`.
- `uv run pytest` green (round-trip + integration + topk tests added).
- `uv run ruff check .` green.
- Track-record finding entry added describing what tightened and what the strengthened experiments now show.
