# C2 BPE micro-domain compressor — rejected-composition verification

**run_id:** 20260520-053938 (composition C2)
**experiment:** `experiments/c1_short_string/` (`c2.py`)
**date:** 2026-05-29
**outcome:** **clean-positive** — **gate over-fired on BOTH axes** (L3 verdict *and* L4 `purely_testable`)

## Why this composition

Gate-precision series, data point 3. C2 was the chosen test of the **`fund>0` /
`purely_testable=False`** axis — the gate signal proposed (last session) as a more
trustworthy replacement for L3's verdict. C2 carried **L3=invalid AND 1 FUNDAMENTAL
objection** (`purely_testable=False`), so under *either* gating rule it would be
rejected.

## The composition + the fundamental objection

> C2 claim: "micro-domain entropy coder: tokenize short records into learned
> sub-byte or multi-byte units, select a tiny domain prefix/codebook for headers
> vs JSON vs identifiers, and decode many symbols through table/vector operations."

> L4 FUNDAMENTAL objection (confidence 10): "A compressor cannot make every
> arbitrary 8-256 byte string shorter… by the pigeonhole principle, if any inputs
> shrink, some must expand. No experiment can rescue a claim of uniformly
> shortening all arbitrary inputs."

## Implementation

`C2Codec` in `experiments/c1_short_string/c2.py`: learned greedy-BPE vocabulary
(all 256 bytes always in-vocab → arbitrary input representable → lossless, byte
tokens are the literal/escape path); per-domain fingerprint-routed static order-1
token model; 32-bit range coder. Amortized model accounting identical to
Brotli-dict / Zstd-dict / C1 / C3. Round-trip lossless: 0 mismatches on 1127 real
+ 803 random/edge records.

## Results (total compressed / total raw, lower is better)

| corpus | **C2** | C3 | C1 (accepted) | zstd+dict | hpack-stateful |
|---|---|---|---|---|---|
| real_http_site (leak-free) | **0.5607** | 0.6534 | 0.6567 | 0.7610 | 0.6083 |
| real_http_shuf | **0.5668** | 0.6625 | 0.6490 | 0.7444 | 0.6099 |
| json | **0.2469** | 0.4340 | 0.4456 | 0.5712 | 0.9244 |
| text128 | **0.3061** | 0.4770 | 1.0078 | 0.4924 | 0.7308 |
| syslog | **0.3886** | 0.6039 | 0.7005 | 0.5279 | 0.7707 |
| http_headers | **0.6322** | 0.6821 | 0.8833 | 1.0346 | 0.6421 |

**C2 is the best method on all six corpora** — it beats every general-purpose
compressor, beats the accepted C1 and the C3 winner, and even beats the
HTTP-specific HPACK-stateful on real HTTP (which C1 and C3 both lost to) while
generalizing off-protocol (json 0.247, text128 0.306). This is the strongest
result of any composition in the project to date.

### The fundamental objection, tested directly

Random-bytes corpus (incompressible) — ratio > 1 means expansion:

| method | C2 | gzip-9 | zlib-9 | zstd-19 | brotli-11 |
|---|---|---|---|---|---|
| random ratio | 1.058 | 1.170 | 1.080 | 1.067 | 1.030 |

The pigeonhole objection **holds for C2 — and equally for every baseline.** C2
even expands *less* than gzip/zlib/zstd. The objection is mathematically true,
universally applicable, and **non-discriminating**: it says nothing about whether
a composition beats baselines on realistic data.

## Interpretation — the gate over-fired on both axes

The gate rejected C2 via `L3=invalid` (novelty) *and* `purely_testable=False`
(one fundamental objection). Both were wrong about the outcome:

- **L3 verdict** repeated its novelty over-fire (cf. C3).
- **L4's `purely_testable=False`** fired on the pigeonhole bound — a true statement
  about the composition's *literal universal claim* that has no bearing on the
  *practical metric* (beat baselines on typical short inputs). It's the same
  category error as L3, one layer up: conflating "the universal claim is false"
  with "the composition won't work."

**This invalidates last session's proposed fix** (gate on `purely_testable`
instead of L3): that rule would still have buried the best composition in the
project. Every real compressor "fails" the pigeonhole test; using it as a veto
discards winners.

## Gate-precision tally (running)

| composition | L3 verdict | purely_testable | empirical outcome | L3 gate correct? | testable gate correct? |
|---|---|---|---|---|---|
| C5 TTNU (cache) | invalid | True | clean-negative | ✅ | (said test → negative, ok) |
| C3 micro-LM (short-str) | invalid | True | partial-positive | ❌ over-fire | ✅ |
| C2 BPE (short-str) | invalid | **False** | **clean-positive (best in project)** | ❌ over-fire | ❌ **over-fire** |

3 rejected compositions verified → **2 of 3 were over-fires, one of them the best
result in the project.** Neither L3's verdict nor L4's `purely_testable` is a safe
hard filter.

## Recommendation

With verification this cheap (~1 session each), the gate should **prioritize, not
veto.** Specifically:
1. Do not treat `L3=invalid` or `fund>0` as disqualifying.
2. Fix L4 to ask whether a fundamental objection **bears on the practical metric**
   (does it preclude beating baselines on realistic data?), not merely whether the
   *literal/universal* claim is false. A pigeonhole objection that applies to all
   baselines should be classified as non-disqualifying.
3. Rank candidates for verification; verify the top-k regardless of L3/L4 verdict
   until the cheap-test budget is exhausted.

_Accounting note: C2's win uses the harness's amortized-dictionary rule (BPE +
per-family tables deployed once, not charged per record) — identical to zstd+dict
/ brotli-dict / C1 / C3. The leakage-free site-disjoint split (0.5607) controls
for train/test site overlap; C2 still wins._
