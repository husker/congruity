# C3 static micro-LM compressor — rejected-composition verification

**run_id:** 20260520-053938 (composition C3)
**experiment:** `experiments/c1_short_string/` (`c3.py`)
**date:** 2026-05-29
**outcome:** **partial-positive** — **gate over-fired** (L3 wrongly rejected a winner)

## Why this composition

Gate-precision series, data point 2 (see
`findings/outcome-calibration-2026-05-29.md`). C3 from run `20260520-053938`
(short-string compression) was rejected with **L3=invalid**. Its accepted sibling
C1 (minimizer + template lookup) from the same run was verified **partial-positive**
(`findings/c1-verification-20260520.md`), so the domain is held constant and we
have a direct accept-vs-reject comparison.

## The composition (verbatim claim)

> "P8's large-reference approximate mapping + P31's learned sequence regularities
> + P15's vectorized computation suggests a **two-stage short-input compressor**:
> use tiny fingerprints to pick a few candidate learned-token dictionaries, then
> use a precomputed micro language model or probability table over those tokens,
> laid out for SIMD batch decode. … avoiding online adaptation warmup."

**L3 verdict (invalid):** "does not actually derive a new lossless short-input
compressor from the cited papers… leaves the hardest parts unspecified: exact
reversible tokenization…"

## Implementation (faithful to the stated design)

`C3Codec` in `experiments/c1_short_string/c3.py`:
- **Static, precomputed** order-2→1→0 byte context model with add-one backoff
  smoothing (the "precomputed micro-LM / probability table"). Trained offline →
  **no per-message adaptation warmup**, which is the wedge for short inputs.
- **Fingerprint routing**: a cheap key (header field-name before `:`) selects
  among a few per-family models; a 1-byte model selector is stored in the stream.
- **32-bit carryless range coder** for entropy coding; frequencies rescaled to a
  bounded total so the coder stays exact. Lossless verified: 0 round-trip
  mismatches on 1127 real records + 500 arbitrary random byte strings.
- Model tables deployed once, not charged per record — identical amortized
  accounting to Brotli-dict / Zstd-dict / C1. The per-record model-id + length
  varints (~2 bytes) ARE charged (they are in the encoded output).

## Results (total compressed / total raw, lower is better)

| corpus | **C3** | C1 (accepted) | zstd-19+dict | brotli-11 | hpack-stateful |
|---|---|---|---|---|---|
| real_http_site (leakage-free) | **0.6534** | 0.6567 | 0.7610 | 0.9672 | 0.6083 |
| real_http_shuf | 0.6625 | 0.6490 | 0.7444 | 0.9613 | 0.6099 |
| json | **0.4340** | 0.4456 | 0.5712 | 1.0255 | 0.9244 |
| text128 | **0.4770** | 1.0078 | 0.4924 | 0.6303 | 0.7308 |
| http_headers | **0.6821** | 0.8833 | 1.0346 | 1.0441 | 0.6421 |
| syslog | 0.6039 | 0.7005 | **0.5279** | 0.9424 | 0.7707 |

- C3 **beats every general-purpose compressor** (gzip/zlib/brotli/zstd/zstd+dict)
  on 5 of 6 corpora (loses to zstd+dict only on syslog).
- C3 **matches or beats the accepted C1** on 5 of 6 corpora, and is far more
  general: C1 collapses to ~1.0 (useless) on text128 and 0.88 on http_headers,
  while C3 is strong everywhere.
- C3 loses only to **HPACK-stateful on HTTP** (0.6534 vs 0.6083) — but HPACK is
  HTTP-protocol-specific and useless off-protocol (json 0.92, text128 0.73),
  exactly the boundary that gave C1 its partial-positive. C3 generalizes the win.

## Interpretation

**Partial-positive — and the gate over-fired.** A composition the gate rejected
as `L3=invalid` performs as well as or better than the accepted, partial-positive
composition in the same domain, and beats general-purpose baselines broadly.

L3's "not novel" claim is *technically true* — a static context model + arithmetic
coder is classic PPM-family compression. But L3 used "invalid" to mean "won't
work / not worth testing," and that was wrong: novelty ≠ performance, and the
project's outcome metric is *beating baselines*, not originality. The gate's
novelty filter actively de-prioritized the better-performing direction. This is
the precise over-firing mode flagged in the L3-calibration memory.

## Gate-precision tally (running)

| composition | gate verdict | empirical outcome | gate correct? |
|---|---|---|---|
| C5 TTNU (cache) | rejected (L3=invalid) | clean-negative | ✅ yes |
| C3 micro-LM (short-string) | rejected (L3=invalid) | **partial-positive** | ❌ **over-fired** |

2 rejected compositions verified → 1 correct rejection, 1 over-fire. Small sample,
but it directly demonstrates that L3=invalid is **not** a reliable "won't work"
signal: it fires on novelty grounds and sometimes buries winners. Actionable: when
L3=invalid co-occurs with purely_testable, prototype before discarding — and
consider separating an L3 "novelty" axis from a "will-it-work" axis.
