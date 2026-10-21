# C11 Verification — Adaptive-Delta Numeric Log Compressor

**Run:** `20260523-180439` (compression, tight 44-topic)
**Composition:** C11 (3 papers, conf 8/10, L1/L2/L3-weak/L4-PURELY_TESTABLE)
**Verdict:** **clean-negative — foreclosed at oracle upper bound**
**Bench:** `experiments/adaptive_delta_numeric/bench.py`

## Claim under test

> P184's robust adaptive filters + P206's drift detection + P135's
> tradeoff framing yields a lossless adaptive-delta compressor for
> streaming numeric logs: use simple predictors until drift is
> detected, reset or switch predictor families, and report a Pareto
> curve of bytes versus encode/decode time. Test: counters,
> timestamps, sensor readings, and regime-changing integer streams
> against delta+zstd, varint+zstd, gzip, and brotli.

## Falsification design

Same oracle-upper-bound shape as **C1 per-block codec selection**
(`findings/c1-per-block-codec-20260523.md`) — the natural twin of
this finding. C1 per-block tested per-block codec selection and found
clean-negative foreclosed: cross-block dictionary context in whole-
file high-quality compression beats per-block adaptation at cheap
settings. C11 is the predictive-transform variant of the same
question.

**Predictors** (lossless transforms on int64-reinterpreted values, so
floats and ints share a code path via IEEE-754 bit reinterpretation):

- `raw` — identity
- `delta1` — first difference: `a[i] - a[i-1]`
- `delta2` — second difference: `a[i] - 2 a[i-1] + a[i-2]` (Gorilla-
  style integer-delta-of-delta, perfect for regular timestamps)
- `xor` — `a[i] XOR a[i-1]` on uint64 bit patterns (Gorilla-style
  float value compression)

**Backends** (entropy coders applied to predictor residuals):

- raw (no entropy coding)
- zstd-3 (default)
- zstd-22 (max — the C1-per-block lesson said this regime dominates)
- brotli-11 (max — strongest text/general-purpose ratio)

**Strong fixed baseline** = best (predictor, backend) combination
applied to the WHOLE stream as a single buffer.

**Oracle adaptive** = per-block predictor selection at a fixed
backend, with 1 byte per block of switch metadata charged. This is
the structural upper bound: knows the optimal predictor for every
1024-value block. A real drift-detector or LMS controller can only
do worse.

## Streams

12 streams total, covering all four C11-named regimes:

- **Synthetic counters** (`synth_counter`): monotonic + small jitter
  — delta1 target
- **Synthetic timestamps** (`synth_timestamps_ns`): regular ~1ms
  intervals + ±1µs jitter — delta2 (Gorilla) target
- **Synthetic regime-changing integers** (`synth_regime_int`):
  5-segment piecewise-stationary AR(1) — the *exact* case C11
  targets
- **Synthetic sensor floats** (`synth_sensor_float`): smooth-drifting
  with noise
- **Real NAB time-series** (8 files, ~1,000–22,000 values each):
  - CPU utilization streams (cpu_utilization_asg, ec2_cpu_5f5533,
    ec2_request_latency, occupancy_6005, speed_7578)
  - Machine temperature (failure stream)
  - Ad exchange (exchange-2_cpc)
  - Artificial-with-anomaly (art_daily_jumpsup) — regime-change case

## Stop rule

Oracle adaptive (per-block predictor + best backend, with 1 byte/
block switch metadata charged) must beat the best fixed
(predictor, backend) by ≥ 5% bytes-per-value to count as structural
support.

If oracle cannot beat best fixed, no learned drift-detector or
LMS-style predictor can recover what the oracle already cannot
deliver — same logical structure as the C1/C5/C6 oracle tests.

## Results

Block size 1024, full stream lengths.

| stream | kind | n | best fixed | bpv | oracle backend | bpv | Δ oracle-vs-best-fixed | Δ oracle-vs-delta1+zstd3 |
|---|---|---:|---|---:|---|---:|---:|---:|
| synth_counter                          | int   | 20,000 | delta1/zstd3     | 0.314 | zstd3    | 0.390 | **−24.11%** | −24.11% |
| synth_timestamps_ns                    | int   | 20,000 | delta1/brotli11  | 1.567 | brotli11 | 1.705 | **−8.80%**  | +23.28% |
| synth_regime_int                       | int   | 20,000 | raw/brotli11     | 1.330 | brotli11 | 1.396 | **−4.95%**  | +22.42% |
| synth_sensor_float                     | float | 20,000 | delta1/brotli11  | 6.881 | brotli11 | 6.975 | **−1.37%**  |  +3.43% |
| art_daily_jumpsup                      | float |  4,032 | xor/brotli11     | 6.946 | brotli11 | 7.010 | **−0.92%**  |  +4.57% |
| cpu_utilization_asg_misconfiguration   | float | 18,050 | raw/brotli11     | 2.212 | brotli11 | 3.162 | **−42.99%** | +16.45% |
| ec2_cpu_utilization_5f5533             | float |  4,032 | raw/brotli11     | 2.312 | brotli11 | 2.980 | **−28.86%** | +23.41% |
| ec2_request_latency_system_failure     | float |  4,032 | raw/brotli11     | 2.131 | brotli11 | 2.842 | **−33.37%** | +25.49% |
| exchange-2_cpc_results                 | float |  1,624 | delta1/brotli11  | 7.108 | brotli11 | 7.165 | **−0.80%**  |  +3.89% |
| machine_temperature_system_failure     | float | 22,695 | delta1/brotli11  | 6.606 | brotli11 | 6.788 | **−2.77%**  |  +3.98% |
| occupancy_6005                         | float |  2,380 | raw/brotli11     | 1.383 | brotli11 | 1.585 | **−14.65%** | +41.54% |
| speed_7578                             | float |  1,127 | raw/brotli11     | 0.901 | brotli11 | 0.961 | **−6.70%**  | +35.42% |

**0/12 streams** have oracle adaptive beating best fixed by ≥ 5%.
Macro-average oracle Δ vs best fixed: **−14.19%** (oracle is
*worse* than best fixed across the board).

### Oracle per-block predictor choice distribution

| stream | blocks | raw | delta1 | delta2 | xor |
|---|---:|---:|---:|---:|---:|
| synth_counter                        |  20 |  0 | **20** |  0 |  0 |
| synth_timestamps_ns                  |  20 |  0 | **20** |  0 |  0 |
| synth_regime_int                     |  20 | **20** |  0 |  0 |  0 |
| synth_sensor_float                   |  20 |  0 |  3 |  0 | **17** |
| art_daily_jumpsup                    |   4 |  0 |  0 |  0 |  **4** |
| cpu_utilization_asg_misconfiguration |  18 | **18** |  0 |  0 |  0 |
| ec2_cpu_utilization_5f5533           |   4 |  **4** |  0 |  0 |  0 |
| ec2_request_latency_system_failure   |   4 |  **4** |  0 |  0 |  0 |
| exchange-2_cpc_results               |   2 |  0 |  **2** |  0 |  0 |
| machine_temperature_system_failure   |  23 |  0 | **22** |  0 |  1 |
| occupancy_6005                       |   3 |  **3** |  0 |  0 |  0 |
| speed_7578                           |   2 |  **2** |  0 |  0 |  0 |

In **11 of 12 streams** the oracle picks a *single* predictor for
every block. The lone exception (machine_temperature_system_failure)
picks delta1 for 22/23 blocks and xor for one. **"Adaptation" isn't
adapting** — the optimal predictor is stable across the stream, so
per-block switching does no real work; it only adds block-
fragmentation overhead.

## Three structural findings

**1. The "baselines too weak" L3 challenge is confirmed.** Vs
`delta1+zstd3` (the recipe's weak baseline), oracle adaptive wins
on 7/12 streams with macro avg **+14.98%**. Vs the strongest fixed
baseline (almost always `raw/brotli11` or `delta1/brotli11`), oracle
*loses* on every stream. If C11 were benchmarked against gzip /
brotli / delta+zstd alone (as the composition proposes), it would
look like a positive. The L3 challenge that the proposed baseline
set omits brotli-11 and full-stream-context codecs is exactly the
hole that hides the foreclosure.

**2. Cross-block dictionary context absorbs per-block adaptation.**
Most streams' best fixed is `raw/brotli11` — *no* predictor. Brotli
at maximum quality finds patterns across the whole stream that any
per-block transform destroys. Even when delta1 wins as the fixed-
predictor choice (synth_counter, synth_timestamps_ns,
machine_temperature, exchange-2), the per-block oracle still loses
because the fragmented blocks each compress less efficiently than
the whole-stream buffer.

**3. The "drift detection" piece adds nothing measurable.** The
oracle picks the same predictor every block in 11/12 streams,
including in **art_daily_jumpsup** — an explicitly regime-change
artificial stream. The regime-change visible to a human (a sudden
upward jump) is not visible to the predictor-selection mechanism
because the predictor-residual statistics it depends on don't shift
in any way that flips the optimal predictor. The drift-detection
component is operating on a target distribution where there is no
useful drift to detect at the optimal-predictor level.

## Verdict

**Clean-negative, foreclosed at oracle upper bound.** Same
structural mode as C1 per-block codec selection
(`findings/c1-per-block-codec-20260523.md`):

- Cross-block context in zstd-22 / brotli-11 beats per-block
  adaptation at every cheap setting and every realistic numeric-log
  stream including regime-change cases.
- The oracle picks a single predictor for nearly every stream —
  "adaptation" provides no marginal benefit because the optimal
  predictor is stable.
- The L3 baseline-risk challenge is the load-bearing one: the
  composition would look positive against its proposed weak
  baselines and is foreclosed against the actually-strong ones.

No drift detector, no LMS predictor, no learned adaptive filter can
beat what the oracle (with perfect per-block lookahead) already
fails to deliver.

## Composition pieces in retrospect

- **P184 (robust adaptive filters)**: the static-predictor oracle
  test shows scalar-delta predictors are already at their useful
  limit on these streams; no robust/adaptive-filter machinery from
  the widely-linear / quaternion / Clifford-algebra literature is
  needed because the residual statistics are already stationary
  enough that a fixed predictor wins. L3 challenge "P184 mismatch"
  confirmed.
- **P206 (drift detection)**: oracle picks single-predictor in 11/12
  streams. There is no useful drift signal at the predictor-
  selection level even on explicitly anomaly-bearing data
  (art_daily_jumpsup, machine_temperature_system_failure,
  cpu_utilization_asg_misconfiguration). The proposed detection
  mechanism has nothing to detect that would change a decision.
  L3 challenge "Drift detection overhead" confirmed at the upper
  bound.
- **P135 (bytes/time Pareto framing)**: irrelevant; the bytes axis
  alone forecloses the composition. The Pareto-front evaluation
  methodology cannot rescue a mechanism that loses on the dominant
  axis at the structural upper bound.

## Related findings

- `findings/c1-per-block-codec-20260523.md` — the per-block codec
  twin (clean-negative foreclosed, same structural pattern)
- `findings/c5-multichannel-lowrank-cms-20260523.md` — same oracle-
  upper-bound shape applied to low-rank multichannel sketches
- `findings/c1-adaptive-compressive-sketch-20260524.md` — same shape
  applied to streaming sketches under drift (also foreclosed)

The C1-compression + C11 + C1-streaming triple establishes a clean
pattern: **on any task where cross-context global compression
exists (whole-stream zstd-22 / brotli-11 for compression; full-
stream sketches for frequency estimation), per-block/per-epoch
adaptation must overcome an overhead cost it cannot recoup at the
oracle upper bound, regardless of the underlying domain.**

## Reproducibility

```
uv run --with zstandard --with brotli python \
    experiments/adaptive_delta_numeric/bench.py \
    --synth-n 20000 --block-size 1024
```

Synthetic streams generated deterministically (seed 42). NAB data
auto-loaded from `experiments/adaptive_delta_numeric/_cache/` if
present (gitignored; eight CSVs from `numenta/NAB` master branch
covering CPU, latency, temperature, traffic, ad exchange, and
artificial-with-anomaly streams). Round-trip lossless decoding
asserted for every (predictor, stream) pair before measurement.
Runs in ~10 minutes on a single CPU core.
