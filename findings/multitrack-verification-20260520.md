# Multitrack stem compression — verification (negative result)

**Source compositions:** `compose-20260520-065937.md` C4, C5, C6 (all WEAK + PURELY TESTABLE)
**Run date:** 2026-05-20
**Prototype:** `experiments/multitrack_stems/` (Python + FLAC/WavPack CLIs)

## Compositions under test

All 6 compose-pipeline outputs reduce to one shared empirical claim:

> Predict one music stem (vocals / drums / bass / other) from the other stems,
> code exact residuals; total compressed bytes should beat per-stem FLAC by
> exploiting sample-level cross-stem correlation (kick→bass timing, harmonic
> coherence, repetition structure).

The L3 adversarial reviewer's strongest attack on all 6:
> "High-level musical correlation is not sample-level predictability: tempo,
> key, song sections, and chorus repetition help predict musical events, but
> lossless compression needs sample-accurate amplitude prediction... FLAC and
> WavPack already exploit strong local waveform predictability; the proposed
> high-level correlations may not lower residual entropy enough to beat them."

L4 testability: PURELY TESTABLE (0 fundamental, 8-10 empirical per composition).

## Result

**The L3 adversarial reviewer was correct.** Cross-stem linear prediction
*loses* to per-stem FLAC by 4-10% on the MUSDB18 sample dataset (20 tracks ×
4 stems × 6.8 sec at 44.1 kHz stereo, AAC-decoded to int16 PCM).

| method | ratio (total_bytes / raw_bytes) | vs per-stem FLAC |
|---|---|---|
| **per-stem FLAC** (baseline) | **0.4076** | — |
| joint 8ch WavPack -hh | 0.4126 | +1.24% (worse) |
| joint 8ch FLAC | 0.4293 | +5.33% (worse) |
| C1 cross-stem residual, taps=0 | 0.4234 | +3.87% (worse) |
| C1 cross-stem residual, taps=2 | 0.4357 | +6.89% (worse) |
| C1 cross-stem residual, taps=20 (±0.5ms) | 0.4505 | +10.52% (worse) |

## Why it fails — variance reduction is essentially zero

| target stem | median variance reduction (taps=20) |
|---|---|
| bass | +0.1% |
| other | +0.2% |
| vocals | +0.1% |

Cross-stem linear prediction reduces residual variance by less than 0.3% in
the median case. FLAC's intra-stem LPC and entropy coding extract orders of
magnitude more predictability from each stem alone. The added side-info
overhead (W, b matrices) plus the loss of FLAC's intra-stem predictability
(because residuals look like noise) makes the chain encoding strictly worse
than per-stem FLAC.

**More taps makes it WORSE.** Going from taps=2 to taps=20 added context but
made the residual harder for FLAC to compress (+10.52% over baseline). The
linear predictor with more degrees of freedom overfits noise on this 6.8s
training window; the resulting residuals have artificial structure that
FLAC's LPC can't exploit.

## Why this is the "right" failure mode

Music stems are deliberately *separated* sources, not redundant measurements
of the same field (the L3 reviewer's "sensor-array analogy is structurally
wrong" attack on C4). A kick drum hit and a bass note may co-occur in time,
but their 44.1 kHz PCM samples don't share waveform structure: different
pitch, envelope, phase, timbre, room response. Per-channel LPC inside FLAC
already captures the strongest local predictability.

This also explains why WavPack multichannel (which does standard
cross-channel decorrelation) barely beats per-stem FLAC by 1.2%: the
operational primitive exists, but the signal isn't there in this domain.

## What this falsifies

| Compose-pipeline claim | Verification verdict |
|---|---|
| C4: distributed acoustic sensing analogy → cross-stem residual coding | ✗ stems are not redundant measurements; linear predictor finds ~0% variance |
| C5: shared event timing (LIGO-style) → onset-conditioned residual | ✗ even if event timing were free, residuals don't shrink at sample level |
| C6: analysis-by-synthesis with generative prior → exact residual | ✗ residual entropy not reduced by any linear predictor; nonlinear unlikely to recover orders-of-magnitude gap |
| All "would-invalidate: high-level musical correlation ≠ sample-level predictability" | ✓ confirmed empirically |

A nonlinear (neural) predictor could potentially do better than linear, but
to make up the 4-10% bytes-loss it would need to find *orders of magnitude*
more cross-stem signal than linear found. The mutual-information ceiling is
set by what's there in the PCM, and the linear bound suggests it's very low.

## What this means for congruity

This is the **inverse** of the C1 short-string verification result. There:
- L3 said WEAK + "operationally equivalent to Brotli+dict / Zstd+dict"
- Verification falsified the central L3 attack
- C1 idea was real

Here:
- L3 said WEAK + "high-level correlation ≠ sample-level predictability"
- Verification confirmed the central L3 attack
- The composition family is empirically dead in this regime

Both are valuable. A research-finding system should produce both true positives
and true negatives. This run is a clean **true negative** — a compose-pipeline
output that the L4 classifier correctly flagged as testable, where the test
revealed the L3 reviewer's main concern was empirically correct.

Calibration data point: in our 2 verified compose runs to date (n=2)
- 1 true positive (C1 short-string, beat baselines on JSON, tied HPACK-static)
- 1 true negative (multitrack stems, all 6 compositions falsified)
- L4 PURELY_TESTABLE classifier correct in both cases — both were testable in
  one session, both produced clear empirical signals.

The L3 adversarial reviewer was directionally right in both cases:
- For C1: L3 partially wrong (LZ-family claim falsified; HPACK claim partially upheld)
- For multitrack: L3 fully right (sample-level predictability absent)

## Caveats and what could change the result

1. **AAC-decoded PCM**: stems came from MUSDB18 sample (AAC-encoded inside MP4).
   AAC encoding may have removed cross-stem correlations that exist in raw
   studio masters. Worth retesting on MUSDB18-HQ (lossless WAV, 30GB) before
   declaring the result definitive. But: lossy codecs typically destroy fine
   correlations rather than preserve them, so the negative is unlikely to flip.
2. **6.8 second clips**: longer pieces would let a predictor exploit chorus
   repetition. But FLAC's block size is ~4096 samples (~100ms); long-range
   structure wouldn't help FLAC's own intra-stem prediction either, so the
   relative comparison should hold.
3. **Linear predictor only**: a deterministic neural predictor (fixed-point
   inference) could find nonlinear cross-stem patterns. But to make up a 4-10%
   bytes deficit, it would need to find orders of magnitude more mutual
   information than linear found — possible but unlikely given how close
   linear got to zero variance reduction.
4. **Sample-aligned stems**: real-world multitrack production has separately
   recorded takes that may have phase drift. The MUSDB stems are time-aligned
   to the mix; "studio reality" might be even worse than this test.

## Reproducing

```
brew install flac wavpack ffmpeg  # CLI codecs
uv run --with musdb --with numpy --with soundfile python \
  -m experiments.multitrack_stems.bench  # 20 tracks
```

## Files

- `experiments/multitrack_stems/corpus.py` — MUSDB18 sample loader
- `experiments/multitrack_stems/baselines.py` — FLAC / WavPack via CLI
- `experiments/multitrack_stems/predictor.py` — linear cross-stem predictor (lossless)
- `experiments/multitrack_stems/bench.py` — benchmark runner

## Verdict

The compose pipeline produced 6 cross-disciplinary multitrack stem
compositions; verification falsified all of them at the linear-predictor
level. This is the **expected behavior of a calibrated research-finding
system**: pipelines that always confirm their own outputs are not actually
testing anything. The L4 PURELY_TESTABLE flag did its job — testing was
cheap, and the answer is clear.

For the user: don't pursue multitrack stem compression further unless a
specific reason emerges to believe nonlinear cross-stem prediction would
find what linear didn't. The MI ceiling appears low.
