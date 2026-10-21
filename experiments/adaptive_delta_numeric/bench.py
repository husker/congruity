"""C11 oracle verification — compose run 20260523-180439 (compression).

C11's synthesis claim:

    P184's robust adaptive filters + P206's drift detection + P135's
    tradeoff framing yields a lossless adaptive-delta compressor for
    streaming numeric logs: use simple predictors until drift is
    detected, reset or switch predictor families, and report a Pareto
    curve of bytes versus encode/decode time. Test: counters,
    timestamps, sensor readings, and regime-changing integer streams
    against delta+zstd, varint+zstd, gzip, and brotli.

Same falsification template as C1 per-block codec selection
(20260523-180439 / findings/c1-per-block-codec-20260523.md):
construct the oracle upper bound for per-block predictor selection
and ask whether it beats the strongest fixed-predictor + best-backend
baseline by enough to justify the block-level switch metadata.

C1 per-block found that cross-block dictionary context (whole-file
high-quality compression) beats per-block adaptation: oracle codec
selection lost by 4.68% vs brotli-11 on the whole file. C11 is the
predictive-transform twin of that finding — same structural question
about whether per-block adaptation can recover cross-block context.

Predictors (lossless transforms applied to int64-reinterpreted values):
  - raw:    no transform
  - delta1: a[i] - a[i-1]                           (first difference)
  - delta2: a[i] - 2 a[i-1] + a[i-2]                (delta-of-delta;
            Gorilla-style integer-delta for timestamps)
  - xor:    a[i] XOR a[i-1] on uint64 bit patterns  (Gorilla-style
            float value compression)

All predictors operate on the int64 bit pattern so the same code path
handles integers and IEEE-754 floats (round-trip via reinterpret).
Block prelude: 1 byte predictor ID + first 16 bytes of raw int64
context for the delta2/xor recursions. Negligible overhead vs the
default block size of 1024 values (≈ 0.2%).

Backends (entropy coders applied to predictor residuals):
  - raw (no entropy coding — predictor + bitpacked residuals)
  - zstd-3   (default)
  - zstd-22  (max — the C1-per-block lesson said this dominates)
  - brotli-11 (max — strongest text/general-purpose ratio)

Strong fixed baselines = best over (predictor × backend) applied to
the WHOLE stream as a single buffer. Oracle adaptive = per-block
best predictor at a fixed backend, with 1 byte / block of switch
metadata charged.

Stop rule (regime-aware, lessons from C6/C1-codec):

  - Oracle adaptive (per-block predictor + best backend, with switch
    metadata charged) must beat the best fixed (predictor, backend)
    combination by >= 5% bytes-per-value to count as structural
    support for C11.
  - If oracle adaptive does NOT beat best fixed, no learned drift-
    detector or LMS-style adaptive filter can recover what the
    oracle already cannot deliver. Foreclosed.
"""

from __future__ import annotations

import argparse
import sys
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import cast

import brotli
import numpy as np
import zstandard


# --------------------------------------------------------------------------
# Predictors (lossless transforms on int64 bit patterns)
# --------------------------------------------------------------------------

PREDICTORS = ["raw", "delta1", "delta2", "xor"]
BACKENDS = ["raw", "zstd3", "zstd22", "brotli11"]


def predict(bits: np.ndarray, predictor: str) -> np.ndarray:
    """Apply lossless predictor to an int64 array; return int64 residuals
    of the same length. First k elements are kept verbatim so the
    transform is exactly invertible (k = 1 for delta1/xor, 2 for delta2)."""
    n = len(bits)
    if n == 0:
        return bits.copy()
    if predictor == "raw":
        return bits.copy()
    out = np.empty_like(bits)
    if predictor == "delta1":
        out[0] = bits[0]
        if n > 1:
            out[1:] = bits[1:] - bits[:-1]
        return out
    if predictor == "delta2":
        out[0] = bits[0]
        if n > 1:
            out[1] = bits[1] - bits[0]
        if n > 2:
            out[2:] = bits[2:] - 2 * bits[1:-1] + bits[:-2]
        return out
    if predictor == "xor":
        as_u64 = bits.view(np.uint64)
        out_u = out.view(np.uint64)
        out_u[0] = as_u64[0]
        if n > 1:
            out_u[1:] = as_u64[1:] ^ as_u64[:-1]
        return out
    raise ValueError(f"unknown predictor {predictor!r}")


def invert_predictor(residuals: np.ndarray, predictor: str) -> np.ndarray:
    """Round-trip check: invert the predictor on residuals."""
    n = len(residuals)
    if n == 0 or predictor == "raw":
        return residuals.copy()
    out = np.empty_like(residuals)
    if predictor == "delta1":
        out[0] = residuals[0]
        for i in range(1, n):
            out[i] = out[i - 1] + residuals[i]
        return out
    if predictor == "delta2":
        out[0] = residuals[0]
        if n > 1:
            out[1] = out[0] + residuals[1]
        for i in range(2, n):
            out[i] = 2 * out[i - 1] - out[i - 2] + residuals[i]
        return out
    if predictor == "xor":
        as_u64 = residuals.view(np.uint64)
        out_u = out.view(np.uint64)
        out_u[0] = as_u64[0]
        for i in range(1, n):
            out_u[i] = out_u[i - 1] ^ as_u64[i]
        return out
    raise ValueError(f"unknown predictor {predictor!r}")


# --------------------------------------------------------------------------
# Backends (entropy coders on predictor-residual byte buffers)
# --------------------------------------------------------------------------

_ZSTD3 = zstandard.ZstdCompressor(level=3)
_ZSTD22 = zstandard.ZstdCompressor(level=22)


def encode(buf: bytes, backend: str) -> int:
    if backend == "raw":
        return len(buf)
    if backend == "zstd3":
        return len(_ZSTD3.compress(buf))
    if backend == "zstd22":
        return len(_ZSTD22.compress(buf))
    if backend == "brotli11":
        return len(brotli.compress(buf, quality=11))
    raise ValueError(f"unknown backend {backend!r}")


def compressed_size(
    values_i64: np.ndarray, predictor: str, backend: str
) -> int:
    """Bytes for (predictor → backend) on a single block/stream."""
    residuals = predict(values_i64, predictor)
    return encode(residuals.tobytes(), backend)


# --------------------------------------------------------------------------
# Stream loaders (everything reduced to int64 bit-pattern views)
# --------------------------------------------------------------------------

@dataclass
class Stream:
    name: str
    kind: str  # 'int' or 'float'
    values_i64: np.ndarray  # int64 reinterpretation of the values


def floats_to_int64(arr: np.ndarray) -> np.ndarray:
    """Reinterpret a float64 array as int64 bit patterns (lossless)."""
    return np.frombuffer(arr.astype(np.float64).tobytes(), dtype=np.int64).copy()


def load_nab_csv(path: Path) -> Stream:
    """NAB CSV with (timestamp, value) columns. We compress the value
    column as float64 reinterpreted to int64."""
    rows = []
    with open(path) as f:
        header = f.readline()
        del header
        for line in f:
            parts = line.strip().split(",")
            if len(parts) < 2:
                continue
            try:
                rows.append(float(parts[-1]))
            except ValueError:
                continue
    arr = np.array(rows, dtype=np.float64)
    return Stream(name=path.stem, kind="float", values_i64=floats_to_int64(arr))


def synth_counter(n: int, rng: np.random.Generator) -> Stream:
    """Monotonic increasing integer counter with small jitter; perfect
    delta1 target."""
    deltas = rng.integers(1, 5, size=n, dtype=np.int64)
    return Stream(
        name="synth_counter",
        kind="int",
        values_i64=np.cumsum(deltas),
    )


def synth_timestamps_ns(n: int, rng: np.random.Generator) -> Stream:
    """Regular ~1ms timestamps with small ns-scale jitter; perfect
    delta2 target (delta-of-delta is tiny)."""
    base = np.arange(n, dtype=np.int64) * 1_000_000  # 1ms regular
    jitter = rng.integers(-1000, 1001, size=n, dtype=np.int64)  # ±1µs
    return Stream(
        name="synth_timestamps_ns",
        kind="int",
        values_i64=base + jitter,
    )


def synth_regime_int(n: int, rng: np.random.Generator) -> Stream:
    """Regime-changing integer stream: piecewise-stationary AR(1) with
    abrupt mean shifts at known points. The exact case C11 targets."""
    n_regimes = 5
    seg = n // n_regimes
    chunks = []
    for r in range(n_regimes):
        mean = int(rng.integers(0, 100_000))
        amp = int(rng.integers(10, 1000))
        prev = mean
        out = np.empty(seg if r < n_regimes - 1 else n - seg * (n_regimes - 1),
                       dtype=np.int64)
        for i in range(len(out)):
            step = int(rng.integers(-amp, amp + 1))
            prev = (prev * 7 + (mean + step) * 3) // 10
            out[i] = prev
        chunks.append(out)
    return Stream(
        name="synth_regime_int",
        kind="int",
        values_i64=np.concatenate(chunks),
    )


def synth_sensor_float(n: int, rng: np.random.Generator) -> Stream:
    """Smooth-drifting float sensor reading; values stay close so
    XOR/delta1 should compress well."""
    base = 20.0 + np.cumsum(rng.normal(0, 0.05, size=n))
    noise = rng.normal(0, 0.5, size=n)
    return Stream(
        name="synth_sensor_float",
        kind="float",
        values_i64=floats_to_int64(base + noise),
    )


# --------------------------------------------------------------------------
# Bench harness
# --------------------------------------------------------------------------

@dataclass
class FixedResult:
    predictor: str
    backend: str
    bytes_total: int


@dataclass
class OracleResult:
    backend: str
    bytes_total: int
    block_bytes_choices: list[str]  # which predictor won per block


def block_iter(arr: np.ndarray, block_size: int):
    for start in range(0, len(arr), block_size):
        yield arr[start : start + block_size]


def bench_stream(
    stream: Stream, block_size: int, verify_roundtrip: bool
) -> tuple[list[FixedResult], list[OracleResult]]:
    """For one stream: compute all fixed (predictor, backend) full-stream
    sizes and all per-backend oracle-adaptive sizes."""
    if verify_roundtrip:
        for p in PREDICTORS:
            r = predict(stream.values_i64, p)
            back = invert_predictor(r, p)
            assert np.array_equal(back, stream.values_i64), (
                f"round-trip failed for predictor {p!r} on stream "
                f"{stream.name!r}"
            )

    fixed = [
        FixedResult(p, b, compressed_size(stream.values_i64, p, b))
        for p in PREDICTORS
        for b in BACKENDS
    ]

    oracles: list[OracleResult] = []
    # Per-block predictor selection at each backend, charging 1 byte / block
    # for the predictor ID (encoded into the block header).
    for backend in BACKENDS:
        total = 0
        choices: list[str] = []
        for block in block_iter(stream.values_i64, block_size):
            best_b = None
            best_p = None
            for p in PREDICTORS:
                sz = compressed_size(block, p, backend)
                if best_b is None or sz < best_b:
                    best_b = sz
                    best_p = p
            assert best_b is not None and best_p is not None
            total += best_b + 1  # 1 byte predictor ID per block
            choices.append(best_p)
        oracles.append(
            OracleResult(backend=backend, bytes_total=total, block_bytes_choices=choices)
        )
    return fixed, oracles


def best_fixed(rs: list[FixedResult]) -> FixedResult:
    return min(rs, key=lambda r: r.bytes_total)


def best_oracle(os_: list[OracleResult]) -> OracleResult:
    return min(os_, key=lambda r: r.bytes_total)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--block-size", type=int, default=1024)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--synth-n", type=int, default=20000)
    ap.add_argument(
        "--nab-dir",
        default="experiments/adaptive_delta_numeric/_cache",
    )
    ap.add_argument(
        "--required-gap",
        type=float,
        default=0.05,
        help="oracle-vs-best-fixed bytes/value gap required to count "
        "as structural support (default 5%)",
    )
    ap.add_argument("--no-roundtrip", action="store_true")
    args = ap.parse_args()

    print("\n=== C11 oracle verification — adaptive-delta numeric logs ===")
    print(f"block size: {args.block_size}  required gap: {args.required_gap:.0%}")
    print(f"predictors: {PREDICTORS}")
    print(f"backends:   {BACKENDS}")
    print(
        "stop rule: oracle-adaptive (per-block predictor + best backend, "
        "1 byte/block switch metadata charged) must beat best fixed "
        "(predictor, backend) by required gap on bytes/value."
    )

    rng = np.random.default_rng(args.seed)
    streams: list[Stream] = [
        synth_counter(args.synth_n, rng),
        synth_timestamps_ns(args.synth_n, rng),
        synth_regime_int(args.synth_n, rng),
        synth_sensor_float(args.synth_n, rng),
    ]
    nab_dir = Path(args.nab_dir)
    if nab_dir.exists():
        for csv in sorted(nab_dir.glob("*.csv")):
            streams.append(load_nab_csv(csv))

    rows = []
    for stream in streams:
        fixed, oracles = bench_stream(
            stream, block_size=args.block_size, verify_roundtrip=not args.no_roundtrip
        )
        bf = best_fixed(fixed)
        bo = best_oracle(oracles)
        n = len(stream.values_i64)
        bpv_fixed = bf.bytes_total / n
        bpv_oracle = bo.bytes_total / n
        rel_gap = (bf.bytes_total - bo.bytes_total) / bf.bytes_total

        # Track delta+zstd specifically (the L3-cited weak baseline)
        delta_zstd3 = next(
            r.bytes_total for r in fixed if r.predictor == "delta1" and r.backend == "zstd3"
        )
        bpv_delta_zstd3 = delta_zstd3 / n
        rel_gap_vs_delta_zstd3 = (
            (delta_zstd3 - bo.bytes_total) / delta_zstd3
        )

        choice_dist = Counter(bo.block_bytes_choices)
        rows.append(
            {
                "stream": stream.name,
                "kind": stream.kind,
                "n": n,
                "best_fixed": f"{bf.predictor}/{bf.backend}",
                "bpv_best_fixed": bpv_fixed,
                "best_oracle_backend": bo.backend,
                "bpv_oracle": bpv_oracle,
                "rel_gap_vs_best_fixed": rel_gap,
                "bpv_delta1_zstd3": bpv_delta_zstd3,
                "rel_gap_vs_delta1_zstd3": rel_gap_vs_delta_zstd3,
                "oracle_choice_dist": dict(choice_dist),
            }
        )

    print(
        "\n=== per-stream summary (bytes-per-value; lower = better) ===\n"
    )
    print(
        "stream                              kind   |  n     | best fixed (pred/back)    bpv  | "
        "best oracle backend  bpv  | Δ oracle-vs-best-fixed | Δ oracle-vs-delta1+zstd3"
    )
    print(
        " " * 35
        + "-" * 130
    )
    for r in rows:
        print(
            f"{r['stream']:36s} {r['kind']:6s} | {r['n']:6d} | "
            f"{r['best_fixed']:24s} {r['bpv_best_fixed']:5.3f} | "
            f"{r['best_oracle_backend']:18s} {r['bpv_oracle']:5.3f} | "
            f"{r['rel_gap_vs_best_fixed']:+7.2%}             | "
            f"{r['rel_gap_vs_delta1_zstd3']:+7.2%}"
        )

    # Oracle choice distribution
    print("\n=== oracle per-block predictor choice distribution ===")
    for r in rows:
        choice_dist = cast("dict[str, int]", r["oracle_choice_dist"])
        total_blocks = sum(choice_dist.values())
        dist = " ".join(f"{p}={choice_dist.get(p, 0)}" for p in PREDICTORS)
        print(f"  {r['stream']:36s}  blocks={total_blocks:4d}  | {dist}")

    weak_wins = [r for r in rows if r["rel_gap_vs_delta1_zstd3"] >= args.required_gap]
    strong_wins = [r for r in rows if r["rel_gap_vs_best_fixed"] >= args.required_gap]

    print("\n=== verdict ===")
    print(
        f"streams where oracle beats best fixed by >= {args.required_gap:.0%}: "
        f"{len(strong_wins)}/{len(rows)}"
    )
    print(
        f"streams where oracle beats delta1+zstd3 (weak baseline) by >= "
        f"{args.required_gap:.0%}: {len(weak_wins)}/{len(rows)}"
    )

    macro_gap_vs_best = float(np.mean([r["rel_gap_vs_best_fixed"] for r in rows]))
    macro_gap_vs_weak = float(
        np.mean([r["rel_gap_vs_delta1_zstd3"] for r in rows])
    )
    print(f"macro-avg oracle Δ vs best fixed:     {macro_gap_vs_best:+.2%}")
    print(f"macro-avg oracle Δ vs delta1+zstd3:   {macro_gap_vs_weak:+.2%}")

    if len(strong_wins) == len(rows) and macro_gap_vs_best >= args.required_gap:
        print(
            "\nORACLE-POSITIVE — per-block predictor selection beats the\n"
            "  strongest (predictor, backend) baseline on every stream by\n"
            "  >= 5% bytes/value after switch metadata. The C11 mechanism\n"
            "  has structural headroom even against zstd-22 / brotli-11."
        )
    elif not strong_wins:
        print(
            "\nFORECLOSED — oracle per-block adaptive predictor selection\n"
            "  cannot beat the best fixed (predictor, backend) baseline\n"
            "  by 5% on any stream. Cross-block context in zstd-22 /\n"
            "  brotli-11 absorbs the per-block adaptive gains; no\n"
            "  drift detector or LMS predictor can recover what oracle\n"
            "  selection already cannot deliver. Same structural mode\n"
            "  as C1 per-block codec selection (clean-negative foreclosed)."
        )
    else:
        print(
            "\nMIXED — oracle wins on a subset of streams. The composition's\n"
            "  applicability depends on stream characteristics; see per-\n"
            "  stream breakdown above."
        )
    return 0


if __name__ == "__main__":
    sys.exit(main())
