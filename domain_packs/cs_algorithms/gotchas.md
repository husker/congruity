# CS / algorithms gotchas

Recurring traps surfaced by the experiments already in this project. Each item is a real bug or weak-baseline mistake that wasted real time. If you're starting a new CS/algorithms compose run, skim this list first.

## Embedding quantization

- `np.argpartition(-sims, k)` returns the top **k+1** items; for top-k use `k - 1` as the partition index. Phase 2 fixed this in `experiments/embed_quant/bench.py:topk_cosine`.
- Normalize query and DB vectors before computing cosine similarity. The bench does this defensively now, but if you write a new variant, don't assume upstream normalization.
- After PQ allocation rounding (`_weights_to_ms` style), the realized bit count can be less than the nominal budget. Label methods by the realized count, not the constructor argument.
- Calibration splits (held-out queries for utility allocation) must NOT leak into the retrieval target set. The deep-analysis flagged a real instance of this — fixed in `experiments/embed_quant/bench.py`.

## Cache replacement

- Synthetic Zipfian workloads make every policy look fine. Loop, scan, and scan-with-hot (already in `experiments/cache_replacement/traces.py`) are where most "novel" replacement policies fall apart.
- **Successor prediction is a prefetching mechanism, not a replacement mechanism.** Designing one as a replacement policy collapses into known prior art (Succ-SIEVE was a clean-negative for exactly this reason).
- Compare against ARC and SIEVE at minimum. LRU alone is a weak baseline.

## Small-points optimization (Heilbronn family)

- Use a stable hash for any signature-based candidate filtering. Python's `hash()` is randomized per process — Phase 2 replaced it with BLAKE2 in `experiments/c1_short_string/c1.py`.
- Report mean ± 95% CI over multiple seeds. Single-seed runs hide noise large enough to flip a method's win/loss.
- Beating the implemented SA baseline is **not** beating published SOTA. Be explicit about which is the bar.
- CMA-ES and differential evolution are tough baselines now wired into `experiments/heilbronn/bench.py`. Beat them or document why.

## Short-string compression

- Test round-trip on every corpus the bench evaluates on. Without it, "compression" can silently be lossy and you'll only notice when a downstream consumer breaks.
- Stateful baselines (HPACK) are much harder to beat than stateless dictionary-based codecs (Brotli, Zstd). Note which class your method is in before claiming a win.
- Compression *ratio* and compression *latency* are different metrics. Don't measure one and claim wins on the other.

## Cross-cutting

- Verification recipes (Phase 3) force you to name baselines, dataset, metric, and stop rule up front. If you can't fill in the recipe, the composition isn't ready to test yet.
- Novelty checks (Phase 3 `--with-novelty`) catch most "this is folklore" cases before you implement. Run them.
- Calibration dashboard (`python -m pipeline.calibration`) shows the running track record. If L3 verdicts have been mostly `invalid` for a domain, that's a signal to revisit the compose prompts before another batch.
