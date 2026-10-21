"""Run multitrack stem compression benchmark.

For each track in the MUSDB sample subset, compute total compressed bytes via:
    A. Per-stem FLAC (4 stereo FLACs)
    B. Joint 8-channel FLAC (cross-channel decorrelation enabled)
    C. Joint 8-channel WavPack -hh (the adversary's required baseline)
    D. Cross-stem linear residual: for each stem, predict from the other 3
       stems, FLAC-encode the residual, plus side-info bytes (W, b)

Total bytes for D = FLAC(vocals_residual) + FLAC(drums_residual)
                  + FLAC(bass_residual) + FLAC(other_residual)
                  + side_info_total
But that's wasteful: we don't need to encode every stem as residual; we can
keep one as anchor (FLAC raw) and encode the other 3 as residuals predicted
from it. The simpler "all 4 as residual from the other 3" actually means each
target is predicted using sources that themselves need decoding — circular.

Two designs:
    D1: anchor mode — keep 1 stem as raw FLAC, predict other 3 from it
    D2: 3-of-4 mode — predict each stem from the other 3 (decoder-ambiguous,
                      requires a canonical encode order); we encode this as
                      stem_1 raw, stem_2 from stem_1, stem_3 from stems 1+2,
                      stem_4 from stems 1+2+3 (causal chain)

We implement D2 (chain mode) since it lets every stem benefit from prediction.

Usage:
    uv run --with musdb --with numpy python -m experiments.multitrack_stems.bench
"""

from __future__ import annotations

import statistics
import sys
import time

from experiments.multitrack_stems.baselines import (
    encode_arr_with_flac,
    joint_multichannel_flac_bytes,
    joint_multichannel_wavpack_bytes,
    per_stem_flac_bytes,
)
from experiments.multitrack_stems.corpus import load_corpus
from experiments.multitrack_stems.predictor import encode_residual, side_info_bytes


def cross_stem_chain_bytes(
    stems,
    order=("drums", "bass", "other", "vocals"),
    taps: int = 0,
):
    """Chain encoding: encode stems in order; each subsequent stem predicted
    from the already-encoded ones. Returns dict with total bytes.
    """
    encoded_bytes = 0
    side_bytes = 0
    sources_so_far = []
    per_stem = {}
    total_overflow = 0
    for i, name in enumerate(order):
        target = stems[name]
        if i == 0:
            # Anchor: FLAC raw
            b = encode_arr_with_flac(target)
            per_stem[name] = {"role": "anchor", "flac": b, "side": 0}
            encoded_bytes += b
        else:
            residual, _overflow_mask, W_q, b_q, info = encode_residual(
                sources_so_far, target, taps=taps,
            )
            b = encode_arr_with_flac(residual)
            sb = side_info_bytes(W_q, b_q)
            per_stem[name] = {
                "role": "residual",
                "flac": b,
                "side": sb,
                "var_reduction": 1 - info["residual_var_int"] / max(info["raw_var"], 1),
                "overflow": info["overflow_count"],
            }
            encoded_bytes += b
            side_bytes += sb
            total_overflow += info["overflow_count"]
        sources_so_far.append(target)
    return {
        "total": encoded_bytes + side_bytes,
        "flac_total": encoded_bytes,
        "side_total": side_bytes,
        "per_stem": per_stem,
        "total_overflow": total_overflow,
    }


def bench_track(name: str, stems: dict) -> dict:
    """Run all methods on one track."""
    t0 = time.time()
    A = per_stem_flac_bytes(stems)
    B = joint_multichannel_flac_bytes(stems)
    C = joint_multichannel_wavpack_bytes(stems)
    D_chain_t0 = cross_stem_chain_bytes(stems, taps=0)
    D_chain_t2 = cross_stem_chain_bytes(stems, taps=2)
    D_chain_t20 = cross_stem_chain_bytes(stems, taps=20)
    raw_total = A["raw_total"]
    return {
        "name": name,
        "raw": raw_total,
        "A_per_stem_flac": A["total_flac"],
        "B_joint_flac": B,
        "C_joint_wavpack": C,
        "D_chain_taps0": D_chain_t0["total"],
        "D_chain_taps2": D_chain_t2["total"],
        "D_chain_taps20": D_chain_t20["total"],
        "D_chain_taps20_detail": D_chain_t20["per_stem"],
        "elapsed_s": time.time() - t0,
    }


def main(argv: list[str]) -> int:
    max_tracks = 20
    if "--small" in argv:
        max_tracks = 5
    elif "--all" in argv:
        max_tracks = None

    results: list[dict] = []
    print(f"Running benchmark (max_tracks={max_tracks})", flush=True)
    for i, (name, stems) in enumerate(load_corpus(max_tracks=max_tracks)):
        r = bench_track(name, stems)
        results.append(r)
        ratios = {
            "A": r["A_per_stem_flac"] / r["raw"],
            "B": r["B_joint_flac"] / r["raw"],
            "C": r["C_joint_wavpack"] / r["raw"],
            "D0": r["D_chain_taps0"] / r["raw"],
            "D2": r["D_chain_taps2"] / r["raw"],
            "D20": r["D_chain_taps20"] / r["raw"],
        }
        print(
            f"[{i + 1}] {name[:40]:40s}  "
            f"A={ratios['A']:.3f} B={ratios['B']:.3f} C={ratios['C']:.3f} "
            f"D0={ratios['D0']:.3f} D2={ratios['D2']:.3f} D20={ratios['D20']:.3f}  "
            f"({r['elapsed_s']:.1f}s)",
            flush=True,
        )

    # Aggregate
    print("\n## SUMMARY (avg ratio = total_method_bytes / total_raw_bytes)")
    methods = {
        "A_per_stem_flac": "per-stem FLAC",
        "B_joint_flac": "joint 8ch FLAC (mc decorrelation)",
        "C_joint_wavpack": "joint 8ch WavPack -hh",
        "D_chain_taps0": "C1 cross-stem residual (taps=0)",
        "D_chain_taps2": "C1 cross-stem residual (taps=2)",
        "D_chain_taps20": "C1 cross-stem residual (taps=20, ±0.5ms)",
    }
    raw_sum = sum(r["raw"] for r in results)
    print(f"  raw_total_bytes: {raw_sum:>12d}  (n_tracks={len(results)})")
    print(f"  {'method':40s} {'total_bytes':>14s} {'ratio':>8s} {'vs_A':>8s}")
    a_bytes = sum(r["A_per_stem_flac"] for r in results)
    for key, label in methods.items():
        total = sum(r[key] for r in results)
        print(
            f"  {label:40s} {total:>14d} {total/raw_sum:>8.4f} {(total/a_bytes - 1)*100:>+7.2f}%"
        )

    # Per-stem residual variance reduction (taps=20)
    print("\n## Cross-stem variance reduction (taps=20 chain), per role")
    by_role: dict[str, list[float]] = {}
    by_role_overflow: dict[str, list[int]] = {}
    for r in results:
        for name, info in r["D_chain_taps20_detail"].items():
            if info["role"] == "residual":
                by_role.setdefault(name, []).append(info["var_reduction"])
                by_role_overflow.setdefault(name, []).append(info["overflow"])
    for name, vals in by_role.items():
        ov = by_role_overflow[name]
        print(
            f"  {name:10s}  var_reduction median={statistics.median(vals):>+.3f}  "
            f"mean={statistics.mean(vals):>+.3f}  "
            f"overflow_median={statistics.median(ov):>5.0f}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
