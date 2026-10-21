# C5 blind-demixing voice isolation — verification (first new-problem run on the fixed gate)

**run_id:** 20260530-065501 (composition C5)
**experiment:** `experiments/voice_isolation_c5/`
**date:** 2026-05-30
**outcome:** **partial-positive** (blind-prior claim confirmed; composed masking add-on counterproductive)

## Why this composition

First composition verified from a NEW-problem run on the corrected pipeline
(prioritize-not-veto + baseline-relative L4). The voice-isolation run surfaced 5
deep-heavy compositions; C5 was the most cheaply-verifiable because its core is
classical: a blind demixing prior + masking.

## The composition (verbatim claim)

> "P24's IVA/NMF determined BSS + P7's complex ratio masking + P3's deep clustering
> + P13's CHiME-style multichannel setting yields a multichannel phase-aware
> separator whose **blind demixing prior supplies spatial separation**."

We verified the cheaply-implementable CORE (AuxIVA + complex ratio mask), skipping
the deep-clustering neural component (GPU/data). The distinctive claim is that the
blind prior separates WITHOUT a steering direction — so the decisive test is vs
beamformers that use a DOA, including an imperfect one.

## Method

`experiments/voice_isolation_c5/separation.py` (pure numpy + scipy): synthetic
non-stationary super-Gaussian sources (speech-band), anechoic far-field linear
array, determined (M mics = K sources). Methods: delay-and-sum (oracle DOA), MVDR
(oracle DOA), MVDR (+10° DOA error), AuxIVA (Ono 2011, blind), C5 = AuxIVA + complex
ratio mask (blind). Metric: SI-SDR improvement of the target estimate vs the target
as received at the reference mic; 8 seeds per condition.

## Results — SI-SDR improvement (dB over input mixture)

| condition | delay-sum (oracle) | MVDR (oracle) | MVDR (+10°) | **AuxIVA (blind)** | C5 IVA+mask |
|---|---|---|---|---|---|
| 2 src / 20 dB | 4.62 | 7.28 | 6.32 | **12.28** | 8.55 |
| 2 src / 10 dB | 4.31 | 6.08 | 5.41 | **6.77** | 5.97 |
| 3 src / 20 dB | 6.01 | 9.72 | 7.04 | **10.51** | 7.71 |

## Interpretation

**Partial-positive.**
1. **Core claim CONFIRMED:** the blind AuxIVA prior beats every beamformer in every
   condition — including MVDR with *oracle* DOA — and its margin over DOA-based
   methods widens when the DOA is imperfect (mvdr_doaerr). The blind demixing prior
   does supply strong spatial separation without a steering direction.
2. **Composed masking add-on is counterproductive:** C5's complex-ratio-mask step is
   1–4 dB *worse* than AuxIVA alone in every condition — it distorts the
   already-separated output. The composition as stated overclaims; its value lives
   in the blind-prior component, not the full IVA+mask pipeline. (Consistent with
   the recurring pattern: the useful part is one component; the synthesis overreaches.)

Net: the blind-prior thesis is a real, clean win over classical beamforming on this
regime; the specific composed form is not the best realization.

## Limitations (honest scope)

- **Synthetic non-stationary sources + anechoic** (no reverb). AuxIVA is known to
  degrade in reverberation; the IVA-beats-beamformer margin would likely shrink (and
  could reverse) with real reverberant speech. This result is for the clean
  anechoic/synthetic regime.
- **Determined case only** (M mics = K sources). The ratio mask might help in the
  underdetermined regime (M<K) where IVA cannot fully separate — untested.
- **Deep-clustering component skipped** — the neural part of C5 (where modern gains
  concentrate) was not verified; this tests the classical core only.

## Note

First new-problem composition verified end-to-end on the fixed gate — a real
partial-positive, cheaply obtained. As with the rescued results, the confirmed
mechanism (blind IVA > beamforming) is effective-but-known; the engine's role was
surfacing the specific cross-paper synthesis to test.
