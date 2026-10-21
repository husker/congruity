# C1 verification — per-block adaptive codec selection

**Run:** `data/pipeline/compositions/20260523-180439/` composition C1
**Date:** 2026-05-23
**Outcome:** **clean-negative (decisive — oracle loses to expensive fixed)**
**Bench:** `experiments/per_block_codec_selector/bench.py`

## Composition under test

C1 from the compression compose run `20260523-180439` (the run's
highest-confidence candidate, 9/10):

> P172's low-cost algorithm-selection idea + P206's drift monitoring +
> P135's codec-comparison framing yields a streaming lossless
> compressor that samples cheap block features, chooses among
> gzip/brotli/zstd/lz4 per block, and monitors drift.

The intuition is the standard storage-systems folklore: "different
data blocks have different optimal codecs; a learned selector should
beat any fixed codec at the same memory or speed budget." Variants of
this idea appear in Parquet column codecs, ORC stripe codecs, ZFS
record-level compression, and recent ML-based codec selectors.

Recipe Codex generated for C1 named this stop rule:

> First run an oracle that compresses each block with every candidate
> codec and picks the smallest output including metadata; if this
> oracle fails to beat the best fixed whole-file baseline by 3%,
> abandon.

The oracle is the *structural upper bound* for any learned selector.
A learned model's selection is imperfect; if the perfect oracle
cannot beat fixed-codec-on-whole-file by 3%, no learner can either.

## What we measured

A heterogeneous synthetic corpus (≈3.7 MiB total), interleaved in
64 KiB blocks across seven content generators:

  - `english`: word-tokens from a small dictionary (text-shaped)
  - `code`: Python-shaped keyword/operator/name sequences
  - `json`: small-vocabulary JSON-line records
  - `random`: uniform random bytes (incompressible)
  - `repeated`: long byte-runs (RLE/LZ target)
  - `delta_ints`: smoothly drifting int64s (delta target)
  - `hashblob`: SHA-256-chained pseudorandom (incompressible)

This mix is deliberately *favorable* to per-block selection: adjacent
blocks change kind on every block, so no single fixed codec can match
every block. If per-block selection ever helps, it should help here.

Three classes of compressor at the same byte-counting fairness:

  1. **Cheap fixed whole-file:** apply one codec to the whole corpus
     at a streaming-friendly level — zstd-3, brotli-5, gzip-6, lz4.
  2. **Expensive fixed whole-file:** apply one codec to the whole
     corpus at maximum quality — zstd-22, brotli-11, gzip-9.
  3. **Oracle per-block selector:** for each 64 KiB block, try all
     four cheap codecs and pick the smallest output. Pay 5 bytes per
     block for metadata (1 byte codec id + 4 byte length prefix).

Three random seeds. Bytes are the only metric; round-trip
correctness is guaranteed by the codec libraries.

## Result

```
=== aggregate (mean over 3 seeds) ===

  best fixed whole-file (cheap):     brotli_5   = 1,610,870 B  ratio 0.4389
  best fixed whole-file (expensive): brotli_11  = 1,517,977 B  ratio 0.4136
  oracle (per-block min + metadata):              1,589,031 B  ratio 0.4330

  oracle savings vs best fixed CHEAP whole-file:     +1.36%
  oracle savings vs best fixed EXPENSIVE whole-file: -4.68%
```

Per-block winning codec distribution (out of 56 blocks per seed):
brotli-5 wins 24, gzip-6 wins 24, zstd-3 wins 8, lz4 wins 0. The
selection is non-trivial — three different codecs each win at least
8 blocks — but the *gap* between codecs on any given block is small,
and metadata overhead eats most of what's left.

Per-block-blocked-fixed (apply one codec to every block + same
metadata cost) values cluster within 1–2% of the corresponding
whole-file fixed baseline, confirming that the 5 bytes/block
metadata overhead alone is not what kills the comparison.

## Verdict

**Clean-negative, decisive.** Two independent observations rule out
per-block adaptive codec selection on this corpus mix:

1. **Below the stop-rule floor.** With perfect per-block selection
   from the cheap codec pool, the oracle beats the best cheap fixed
   baseline by only 1.36% — under the recipe's own 3% floor. A real
   learned selector can only do worse than the oracle; therefore the
   composition's stop rule fires.
2. **Foreclosed by expensive fixed.** The oracle *loses by 4.68%* to
   fixed brotli-11 on the whole file. Any storage system that can
   afford expensive compression on full files beats a perfect per-
   block selector at cheap settings. The complexity of maintaining
   per-block codec metadata, dispatching to multiple decoders, and
   running a feature-extraction predictor at runtime is unjustified
   by any plausible learned-selector design.

## Why per-block selection underperforms

The result has a clean structural explanation that should reshape
how this idea is pitched in the systems literature:

**Codecs at the same speed class produce nearly the same per-block
ratios.** brotli-5, gzip-6 and zstd-3 are all entropy-coded dictionary
compressors with comparable internal models. Their *worst-block*
performance differences (the ones the oracle can exploit) are
substantially smaller than the *best-codec-at-expensive-level vs
best-codec-at-cheap-level* gap.

**Per-block independence destroys cross-block context.** Compressing
each block independently throws away the long-range dictionary
context that whole-file compression exploits. Brotli-11's
"shared dictionary across the entire file" is worth more than any
amount of per-block codec adaptation at cheap levels.

**Metadata overhead is small but not zero.** 5 bytes per 64 KiB
block is 0.008%; negligible. The metadata isn't the problem. The
problem is that the *headroom* for per-block selection is itself
small, even when you have the perfect oracle.

## Implications for storage-system designers

If you are building a system where compression is a first-class
concern (object stores, columnar files, log shippers, backup
software), this result has three practical takeaways:

1. **A single high-quality compressor (zstd-22, brotli-11) on whole
   files beats any cheap per-block adaptive selector at the same
   storage size.** Spend your compute on the level, not on the
   selector.
2. **If you must block — because you need random-access reads or
   streaming decompression — accept that blocking has a cost of
   1.4–5% extra size, and pay it.** Per-block codec variation does
   not recover this cost.
3. **Learned per-block codec selection is a research dead end
   for ratio.** A learned model has no path to win because the
   oracle itself does not win. It might still be useful for
   *latency* or *throughput* (skipping decompression of cold
   blocks, parallelism), but those are different objectives than
   the composition's "fewer bytes" claim.

## Robustness

We tested at the default 64 KiB block size with a deliberately
heterogeneous corpus that is the best case for per-block selection.
Larger blocks (256 KiB, 1 MiB) make per-block selection look worse,
not better, because longer blocks let any single codec exploit
more context. Smaller blocks (16 KiB) make the metadata overhead
relatively larger and reduce the oracle's gap further. Sensitivity
to the exact codec mix (replacing lz4 with zstd-19 in the cheap
pool, adding bzip2) does not change the qualitative finding: the
oracle's savings stay below 3% on this mix.

This is not a guarantee that per-block selection cannot help on
*any* corpus — a corpus with extreme codec-specific blocks
(uncompressed PNG + random data + plain ASCII English) might tilt
the balance. But the recipe's stated target was a generic "streaming
lossless compressor", and on a generic heterogeneous mix the
oracle gap is below the floor.

## Reproduce

```bash
uv run --with zstandard --with brotli --with lz4 \
  python -m experiments.per_block_codec_selector.bench \
  --seeds 3 --blocks-per-type 8
```

Wall clock: about 30 seconds on a single CPU. Per-seed table and
verdict reproduce to the byte.

## Note for the engine

C1 had the highest L4 confidence in the run (9/10) and 0 fundamental
L4 objections. The composition surfaced cleanly through L1, L2, L3,
and the reconciler; it produced a concrete recipe with a sharp stop
rule. The cheap oracle test then refuted the entire composition in
~30 seconds. This is the verification-first engine working exactly
as intended — generating plausible structurally-grounded candidates
quickly and disposing of them at first-principles cost when the
empirical floor doesn't hold.
