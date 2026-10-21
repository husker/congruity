"""Calibration harness for the exact max-min polish (campaign gate M1).

Ladder: from cheap multistart (uniform + boundary-biased, ~45% of record
points sit on the boundary), ``exact_polish`` must recover the PROVEN
optima n=5-9. Hit rate against the proven values measures candidate
strength independent of record luck.

Probes:
  --probe-karpov      polish Karpov's n=13/15 record configs (expected:
                      ~zero gain — they are stationary; validates that
                      record-grade configs survive the polish).
  --probe-state PATH  polish the per-n bests from a record_run state
                      file (attempt #2 miner). The lift decides
                      optimizer-gap vs basin-gap for the campaign.

    python -m experiments.heilbronn.calibrate_exact_polish \
        --ns 5,6,7,8,9 --starts 40
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

from experiments.heilbronn.evaluator import min_triangle_area
from experiments.heilbronn.exact_polish import exact_polish, load_karpov

RECORDS_PATH = (
    Path(__file__).resolve().parents[2]
    / "computable_packs"
    / "small_points_optimization"
    / "records.json"
)


def load_records() -> dict[int, dict]:
    data = json.loads(RECORDS_PATH.read_text())
    return {int(r["params"]["n"]): r for r in data["records"]}


def make_start(rng: np.random.Generator, n: int, boundary_frac: float) -> np.ndarray:
    pts = rng.random((n, 2))
    if boundary_frac > 0.0:
        for i in range(n):
            if rng.random() < boundary_frac:
                edge = int(rng.integers(4))
                if edge == 0:
                    pts[i, 0] = 0.0
                elif edge == 1:
                    pts[i, 0] = 1.0
                elif edge == 2:
                    pts[i, 1] = 0.0
                else:
                    pts[i, 1] = 1.0
    return pts


def calibrate(ns: list[int], starts: int, seed0: int, rel_tol: float) -> dict:
    records = load_records()
    out: dict[str, dict] = {}
    for n in ns:
        rec = records.get(n)
        target = float(rec["value"]) if rec else None
        status = rec["status"] if rec else "unknown"
        best_area, hits, rounds_sum, t0 = -1.0, 0, 0, time.time()
        for k in range(starts):
            rng = np.random.default_rng(seed0 + k)
            # alternate uniform and boundary-biased starts
            pts = make_start(rng, n, 0.0 if k % 2 == 0 else 0.35)
            res = exact_polish(pts)
            rounds_sum += res.rounds
            if res.area > best_area:
                best_area = res.area
            if target and res.area >= target * (1.0 - rel_tol):
                hits += 1
        dt = time.time() - t0
        out[str(n)] = {
            "target": target,
            "status": status,
            "best": best_area,
            "best_pct_of_target": (100.0 * best_area / target) if target else None,
            "hits": hits,
            "starts": starts,
            "mean_rounds": rounds_sum / starts,
            "secs_per_start": dt / starts,
        }
        tgt = f"{target:.10f}" if target else "?"
        print(
            f"n={n}: target={tgt} ({status}) best={best_area:.10f} "
            f"({out[str(n)]['best_pct_of_target']:.2f}%) hits={hits}/{starts} "
            f"mean_rounds={rounds_sum / starts:.1f} {dt / starts:.2f}s/start",
            flush=True,
        )
    return out


def probe_karpov() -> None:
    for n in (13, 15):
        pts, measured = load_karpov(n)
        t0 = time.time()
        res = exact_polish(pts, max_rounds=6)
        print(
            f"karpov S{n}: measured={measured:.15f} polished={res.area:.15f} "
            f"gain={res.gain:+.3e} rounds={res.rounds} ({time.time() - t0:.1f}s)",
            flush=True,
        )


def probe_state(path: Path, ns: list[int]) -> None:
    state = json.loads(path.read_text())
    records = load_records()
    for n in ns:
        entry = state.get("best", {}).get(str(n))
        if not entry:
            print(f"state has no best for n={n}", flush=True)
            continue
        pts = np.array(entry["points"], dtype=float)
        stored = float(entry["area"])
        revalidated = min_triangle_area(pts)
        res = exact_polish(pts)
        rec = records.get(n)
        target = float(rec["value"]) if rec else float("nan")
        print(
            f"state n={n}: stored={stored:.10f} revalidated={revalidated:.10f} "
            f"polished={res.area:.10f} gain={res.gain:+.3e} rounds={res.rounds} "
            f"| record={target:.10f} remaining_gap={target - res.area:+.6f} "
            f"({100.0 * res.area / target:.2f}% of record)",
            flush=True,
        )


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="Calibrate the exact max-min polish.")
    p.add_argument("--ns", default="5,6,7,8,9")
    p.add_argument("--starts", type=int, default=40)
    p.add_argument("--seed0", type=int, default=1)
    p.add_argument("--rel-tol", type=float, default=1e-6)
    p.add_argument("--probe-karpov", action="store_true")
    p.add_argument("--probe-state", default=None, help="record_run state json path")
    p.add_argument("--probe-ns", default="13,15")
    p.add_argument("--save", default=None, help="write calibration summary JSON here")
    args = p.parse_args(argv if argv is not None else sys.argv[1:])

    if args.probe_karpov:
        probe_karpov()
        return 0
    if args.probe_state:
        probe_state(Path(args.probe_state), [int(x) for x in args.probe_ns.split(",")])
        return 0

    ns = [int(x) for x in args.ns.split(",") if x.strip()]
    out = calibrate(ns, args.starts, args.seed0, args.rel_tol)
    if args.save:
        Path(args.save).write_text(json.dumps(out, indent=2))
        print(f"saved {args.save}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
