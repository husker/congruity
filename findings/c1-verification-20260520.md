# C1 verification — short-string compression

**Source composition:** `compose-20260520-053938.md` C1
(minimizer-fingerprint + family-hierarchy + sparse-residual)
**Run date:** 2026-05-20
**Prototype:** `experiments/c1_short_string/` (pure Python)

## Claim under test

The L3 adversarial reviewer's central objection:
> "Not Novel: The proposed compressor is essentially static dictionary/template
> compression plus approximate matching plus residual delta coding. That family
> already includes VCDIFF/xdelta, Brotli static dictionary transforms, Zstd
> trained dictionaries, HPACK/QPACK indexed names and values, and record-template
> codecs."

Predicted outcome: C1 ties or loses to Brotli+dict / Zstd+dict / HPACK.

## Result, headline

**Calibrated:** C1 *generalizes* HPACK's operational primitive (table + literal)
to arbitrary corpora. It loses to HPACK in HPACK's production-deployed
configuration (stateful dynamic table) on HTTP headers, but beats every
general-purpose compressor (Brotli, Zstd, Zstd+trained-dict) on highly
templated short records and beats HPACK by 52% on JSON because HPACK is
HTTP-specific.

## Numbers — compression ratio, lower is better

n=2000 test records for synthetic corpora; real-HTTP uses scraped headers from
105 public sites (`data/real_http_headers.json`). `real_http_shuf` shuffles
header lines into 50/50 train/test (potential leakage from common headers).
`real_http_site` splits by SITE — train and test from disjoint site sets, no
line-level leakage.

| method | http_headers† | syslog | json | text128 | real_http_shuf | **real_http_site** |
|---|---|---|---|---|---|---|
| raw | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 |
| gzip-9 | 1.387 | 1.170 | 1.269 | 0.843 | 1.336 | 1.346 |
| zlib-9 | 1.151 | 1.034 | 1.060 | 0.749 | 1.084 | 1.090 |
| brotli-11 | 1.044 | 0.942 | 1.026 | 0.630 | 0.961 | 0.967 |
| zstd-19 | 1.173 | 1.026 | 1.102 | 0.764 | 1.108 | 1.116 |
| zstd-19+dict | 1.035 | **0.528** | 0.571 | **0.492** | 0.744 | 0.761 |
| hpack-static | 0.686 | 0.771 | 1.015 | 0.770 | 0.687 | 0.688 |
| **hpack-stateful** | 0.642 | 0.771 | 0.924 | 0.731 | **0.610** | **0.608** |
| **C1-proto** | 0.883 | 0.701 | **0.446** | 1.008 | 0.649 | 0.657 |

† synthetic http_headers — its templated structure favors HPACK's static table
and partially HPACK-stateful's dynamic table. Real headers (`real_http_site`)
are the meaningful comparison.

## Honest reading

| regime | winner | gap to C1 | takeaway |
|---|---|---|---|
| HTTP/2 production (HPACK stateful per connection) | hpack-stateful 0.608 | **C1 loses by 8%** | HPACK as deployed beats C1. Don't replace HPACK in HTTP/2. |
| Stateless per-record HTTP encoding | C1 0.657 | C1 wins by 5% over hpack-static (0.688) | C1 wins where dynamic-table state isn't available. |
| Templated structured records (JSON) | C1 0.446 | C1 wins by 52% over hpack-stateful (0.924) | HPACK is HTTP-specific; C1 generalizes. Real wedge. |
| Variable-template records (syslog) | zstd+dict 0.528 | C1 loses by 25% | LZ-with-trained-state wins where templates vary in non-trivial ways. |
| Heavy literal short text (Markov chunks) | zstd+dict 0.492 | C1 falls back to literal (1.008) | C1 has graceful failure mode, not catastrophic. |

## What this actually tells us

1. **L3 "Not Novel — operationally equivalent to Brotli+dict / Zstd+dict"
   is FALSIFIED on real data.** Zstd+trained-dict ratio is 0.76 on real HTTP
   headers; C1 ratio is 0.66. Different operational primitives, measurably different
   behavior.

2. **L3 "Operationally equivalent to HPACK/QPACK" is PARTIALLY UPHELD.**
   HPACK-stateful (the production mode) outperforms C1 on HTTP headers by 8%.
   They are very close cousins. C1's win condition over HPACK is *outside HTTP*,
   not inside it.

3. **L3 "Latency Target" remains UNTESTED.** Pure Python prototype. Sub-microsecond
   decompression claim is unaddressed — still the strongest open objection. Note
   that HPACK's wire format is bit-packed with Huffman coding; C1's prototype is
   byte-aligned. A native C1 with Huffman+SIMD might close more gap with HPACK-stateful
   or pull further ahead on JSON.

4. **The actual novel wedge:** Auto-mined template + sparse-residual coding,
   trained per corpus rather than hand-curated per protocol. HPACK works because
   IETF curated the static table; C1 works because the templates are learned
   from the actual corpus.

## Site-level leakage check

Comparing `real_http_shuf` (line-shuffle, possible leakage from globally-common
headers like `Connection: close`) to `real_http_site` (held-out sites, no
line-level leakage):

- C1: 0.649 → 0.657 (slight degradation, but still beats hpack-static)
- HPACK-stateful: 0.610 → 0.608 (essentially unchanged — HPACK isn't trained)
- HPACK-static: 0.687 → 0.688 (unchanged — HPACK isn't trained)

Conclusion: C1's win on real headers is not driven by leakage; it generalizes
to held-out sites. The wedge over zstd+dict (0.76) and Brotli (0.97) is real
even under the strictest split.

## Audiences who could plausibly care

- **Observability / telemetry vendors** (JSON event streams, custom record formats):
  C1 beats Zstd+trained-dict by ~25% on JSON. At Datadog/Honeycomb/Grafana Cloud
  scale, this is storage-cost money. But would need: native impl, real customer data.
- **KV store / time-series database operators** storing small values (50-256 bytes):
  similar story.
- **HTTP/2 / HTTP/3 deployments**: no value. HPACK/QPACK already deployed.
- **Academic compression researchers**: possibly interesting as a methodology
  result — auto-mined templates generalize HPACK's hand-curated approach to
  arbitrary domains.
- **Congruity backers / pilot customers**: yes, as one of several proof points
  that the compose pipeline produces falsifiable claims that survive measurement.

## What would strengthen this further (before sharing widely)

1. **Native (Rust/C++) implementation** with cycle counts. Addresses the L3
   latency objection. Without this, the L3 attack stays half-open.
2. **Larger real-headers corpus** — 500-1000 sites instead of 105. HTTP Archive
   has the data. Run on real production logs/JSON if any partner can share.
3. **QPACK comparison** — HTTP/3's QPACK is the bit-packed Huffman cousin of
   HPACK and would tighten the protocol-codec comparison.
4. **Ablation:** C1 with hierarchy-aware template miner (Drain/Spell) vs flat
   templates — the C1 composition specified hierarchy but the prototype omits it.
   Tests whether hierarchy improves the syslog case where flat templates struggled.

## Reproducing

```
uv run python -m experiments.c1_short_string.fetch_real_headers
uv run --with brotli --with zstandard --with hpack python -m experiments.c1_short_string.bench
```

## Files

- `experiments/c1_short_string/fetch_real_headers.py` — scrapes ~100 public sites
- `experiments/c1_short_string/corpora.py` — synthetic + real corpora, site-split helper
- `experiments/c1_short_string/baselines.py` — gzip, zlib, brotli, zstd, zstd+dict, HPACK (static + stateful)
- `experiments/c1_short_string/c1.py` — C1 prototype
- `experiments/c1_short_string/bench.py` — benchmark runner
- `data/real_http_headers.json` — 105 scraped sites

## Verdict

Calibrated framing: **C1 is a generic auto-mined template codec that loses to
HPACK in HPACK's HTTP-only home turf but generalizes the same primitive to
arbitrary short structured data, beating every general-purpose codec including
Zstd-with-trained-dictionary on JSON by a large margin.** The L3 reviewer's
"already covered by Brotli+dict / Zstd+dict" attack is falsified. The
"already covered by HPACK/QPACK" attack is partially upheld — HPACK is the
right cousin to flag — but the generalization to non-HTTP corpora is the
non-obvious result.
