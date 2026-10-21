"""Heilbronn benchmark runner.

Compares SA / basin-hopping / DE / CMA-ES / C1 across multiple seeds
per n and reports mean ± 95% confidence interval. The single-seed
runs we used initially could hide noise; CIs make the win/loss
between methods unambiguous.

Each method gets the same total function-evaluation budget
(n_starts × n_steps).

With --candidate, a module exposing
``solve(n: int, seed: int, budget: int) -> (n, 2) array-like`` runs as
method "candidate" (optional ``solve_ablated`` as "candidate_ablated");
its area is ALWAYS recomputed from the emitted coordinates — wrong shape,
out-of-bounds, or non-finite points poison the trial as NaN, and
exceptions are caught. With --json the last stdout line is the autoverify
RESULT_CONTRACT including per_trial traces and — for the candidate —
record_claims with its best validated area per n (compared downstream
against computable_packs/small_points_optimization/records.json). Best
candidate coordinates per n are saved to ``{candidate}.points.json`` so a
record claim is verifiable against the cited sources.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import math
import os
import time
from pathlib import Path

# Cap BLAS/OpenMP threads to 1 BEFORE numpy is imported. The bench is run many
# at a time (autosearch populations, the seeding experiment, the record miner's
# workers); default multi-threaded BLAS then oversubscribes the CPU and thrashes
# — it once made a single n=13 eval look like a 6-hour hang. The matrices here
# are tiny, so one thread per process is faster. Override via the env if needed.
for _v in (
    "OMP_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "MKL_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS",
    "NUMEXPR_NUM_THREADS",
):
    os.environ.setdefault(_v, "1")

import numpy as np  # noqa: E402  (must follow thread-env setup)
from scipy.stats import t as student_t  # noqa: E402

from experiments.heilbronn.baselines import (  # noqa: E402
    sa_halton,
    sa_uniform,
    scipy_basinhopping,
)
from experiments.heilbronn.baselines_cmaes import cmaes_run  # noqa: E402
from experiments.heilbronn.baselines_de import de_run  # noqa: E402
from experiments.heilbronn.c1_smooth import c1_run  # noqa: E402
from experiments.heilbronn.evaluator import min_triangle_area  # noqa: E402


def _c1_adapter(n_points, n_steps, n_starts, seed):
    """Adapter: same signature as the SA methods."""
    return c1_run(n_points, n_steps=n_steps, n_restarts=n_starts, seed=seed)


# Published-SOTA reference (approximate, from Erich Friedman's collection
# at http://www2.stetson.edu/~efriedma/heilbronn/; cite cautiously, exact
# digits vary by source). These are the values we'd LIKE to match.
# Treat as informational only — the apples-to-apples bar is the SA floor.
# STALE for n=7/8 (see computable_packs/small_points_optimization/
# records.json for the current, cited record table).
PUBLISHED_REFERENCE = {
    6: 0.12500,  # = 1/8, proven optimal
    7: 0.08333,  # = 1/12, proven optimal
    8: 0.07286,
    9: 0.05333,
    10: 0.04812,
    12: 0.03224,
    15: 0.02448,
    20: 0.01797,
    25: 0.01163,
}


METHODS = {
    "SA-uniform": sa_uniform,
    "SA-halton": sa_halton,
    "Basin-hop": scipy_basinhopping,
    "DE": de_run,
    "CMA-ES": cmaes_run,
    "C1-smooth": _c1_adapter,
}


def load_candidate(path: str):
    """Load `solve` (and optional `solve_ablated`) from a candidate module."""
    spec = importlib.util.spec_from_file_location("autoverify_candidate", path)
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot load candidate module from {path}")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    solve = getattr(mod, "solve", None)
    if not callable(solve):
        raise ImportError(f"candidate module {path} does not expose solve()")
    ablated = getattr(mod, "solve_ablated", None)
    return solve, (ablated if callable(ablated) else None)


def _validated_pts_area(pts, n_points: int) -> tuple[np.ndarray | None, float]:
    """Recompute the objective from emitted coordinates; NaN on any violation."""
    try:
        arr = np.asarray(pts, dtype=np.float64)
    except Exception:
        return None, float("nan")
    if arr.shape != (n_points, 2) or not np.all(np.isfinite(arr)):
        return None, float("nan")
    if arr.min() < -1e-9 or arr.max() > 1 + 1e-9:
        return None, float("nan")
    arr = np.clip(arr, 0.0, 1.0)
    return arr, float(min_triangle_area(arr))


def _candidate_adapter(solve):
    """Wrap a candidate solve() into the baseline method signature.

    The candidate's reported value is never used: the area is recomputed
    from its coordinates, and exceptions become NaN trials.
    """
    def fn(n_points, n_steps, n_starts, seed):
        try:
            pts = solve(n_points, seed, n_steps * n_starts)
        except Exception:
            return None, float("nan")
        return _validated_pts_area(pts, n_points)
    return fn


def _mean_ci(values: list[float], confidence: float = 0.95) -> tuple[float, float, float]:
    """Return (mean, ci_lower, ci_upper) under a Student-t approximation.

    For n=1 the CI is undefined; return (mean, mean, mean).
    """
    arr = np.asarray(values, dtype=np.float64)
    n = arr.size
    mean = float(arr.mean())
    if n < 2:
        return mean, mean, mean
    sem = float(arr.std(ddof=1)) / math.sqrt(n)
    half = float(student_t.ppf(0.5 + confidence / 2.0, df=n - 1)) * sem
    return mean, mean - half, mean + half


def _run_seeds(fn, n_points: int, n_steps: int, n_starts: int,
               seeds: list[int]) -> tuple[list[float], float, np.ndarray | None, float]:
    """Run `fn` once per seed; return (areas, best, best_pts, total_time_s).

    NaN areas (invalid candidate output) are kept in `areas` so they poison
    means downstream; `best`/`best_pts` track the best FINITE result.
    """
    areas: list[float] = []
    total_time = 0.0
    best = 0.0
    best_pts: np.ndarray | None = None
    for s in seeds:
        t0 = time.time()
        try:
            pts, area = fn(n_points=n_points, n_steps=n_steps, n_starts=n_starts, seed=s)
        except Exception:
            pts, area = None, float("nan")
        total_time += time.time() - t0
        areas.append(float(area))
        if math.isfinite(area) and area > best:
            best = float(area)
            best_pts = np.asarray(pts) if pts is not None else None
    return areas, best, best_pts, total_time


def main() -> int:
    p = argparse.ArgumentParser(description="Heilbronn benchmark with multi-seed CIs")
    p.add_argument("--ns", default="8,10,12", help="comma-separated n values")
    p.add_argument("--seeds", type=int, default=5,
                   help="number of seeds per (method, n); 1 disables CIs")
    p.add_argument("--n-steps", type=int, default=50_000)
    p.add_argument("--n-starts", type=int, default=8)
    p.add_argument("--seed0", type=int, default=42, help="first seed; subsequent are seed0+i")
    p.add_argument("--methods", default=",".join(METHODS.keys()),
                   help="comma-separated method names from METHODS")
    p.add_argument("--candidate", default=None,
                   help="path to a module exposing solve(n, seed, budget) -> (n,2) points")
    p.add_argument("--json", action="store_true",
                   help="emit a RESULT_CONTRACT JSON object as the last stdout line")
    args = p.parse_args()

    ns = [int(s) for s in args.ns.split(",") if s.strip()]
    seeds = list(range(args.seed0, args.seed0 + args.seeds))
    methods = [name.strip() for name in args.methods.split(",") if name.strip()]
    for name in methods:
        if name not in METHODS:
            raise SystemExit(f"unknown method {name!r}; available: {list(METHODS)}")
    runner = {name: METHODS[name] for name in methods}
    if args.candidate:
        solve, ablated = load_candidate(args.candidate)
        runner["candidate"] = _candidate_adapter(solve)
        if ablated is not None:
            runner["candidate_ablated"] = _candidate_adapter(ablated)

    print(f"\nHeilbronn benchmark — n_steps={args.n_steps}, n_starts={args.n_starts}, "
          f"seeds={args.seeds}")
    header = (
        f"{'n':>4s} {'method':18s} {'mean':>10s} "
        f"{'ci_low':>10s} {'ci_high':>10s} {'best':>10s} "
        f"{'pub':>9s} {'gap%':>7s} {'time_s':>8s}"
    )
    print(header)
    print("-" * len(header))
    # method -> "n{n}" -> per-seed areas (None for invalid); candidate best pts per n.
    per_trial: dict[str, dict[str, list[float | None]]] = {m: {} for m in runner}
    best_points: dict[int, dict] = {}
    for n in ns:
        pub = PUBLISHED_REFERENCE.get(n)
        for name, fn in runner.items():
            areas, best, best_pts, elapsed = _run_seeds(fn, n, args.n_steps, args.n_starts, seeds)
            per_trial[name][f"n{n}"] = [a if math.isfinite(a) else None for a in areas]
            if name == "candidate" and best_pts is not None and best > 0:
                best_points[n] = {"area": best, "points": best_pts.tolist()}
            finite = [a for a in areas if math.isfinite(a)]
            mean, lo, hi = _mean_ci(finite if finite else [float("nan")])
            gap = f"{(mean / pub - 1) * 100:+6.2f}%" if pub and finite else "    --"
            pub_str = f"{pub:9.5f}" if pub else "     --"
            print(
                f"{n:>4d} {name:18s} {mean:10.5f} "
                f"{lo:10.5f} {hi:10.5f} {best:10.5f} "
                f"{pub_str} {gap:>7s} {elapsed:8.1f}"
            )
        print()

    if args.json:
        scores: dict[str, float] = {}
        for name, by_n in per_trial.items():
            vals = [a for ks in by_n.values() for a in ks]
            scores[name] = (sum(vals) / len(vals)
                            if vals and all(a is not None for a in vals)
                            else float("nan"))
        sota = "CMA-ES" if "CMA-ES" in runner else \
            next((m for m in methods), "")
        record_claims = [{"params": {"n": n}, "value": info["area"]}
                         for n, info in sorted(best_points.items())]
        if args.candidate and best_points:
            pts_path = Path(f"{args.candidate}.points.json")
            pts_path.write_text(json.dumps(
                {str(n): info for n, info in best_points.items()}, indent=2))
        result = {
            "metric": "objective_value",
            "higher_is_better": True,
            "scores": scores,
            "per_trial": per_trial,
            "ablated_method": ("candidate_ablated"
                               if "candidate_ablated" in runner else ""),
            "composition_method": "candidate" if args.candidate else "",
            "home_field_sota_method": sota,
            "record_claims": record_claims,
            "n_trials": len(ns) * len(seeds),
            "notes": f"min triangle area, unit square; ns={','.join(map(str, ns))}; "
                     f"seeds={seeds[0]}..{seeds[-1]}; budget={args.n_steps}x{args.n_starts}; "
                     f"areas recomputed from emitted coordinates",
        }
        print(json.dumps(result))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
