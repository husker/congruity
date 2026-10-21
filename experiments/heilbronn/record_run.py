"""Heilbronn long-run record miner (candidate-only).

A deterministic candidate (``solve(n, seed, budget)``) is run round-robin over
a set of ``n`` values with a fresh seed each attempt, until a wall-clock
deadline. Every emitted point set is validated exactly as the bench validates
it (``experiments.heilbronn.bench._validated_pts_area`` — shape, bounds,
finiteness, then ``min_triangle_area`` recomputed from coordinates), so a
reported area can never exceed what the coordinates actually achieve.

State is CUMULATIVE and resumable: ``{candidate}.record_run.json`` holds, per
``n``, the best validated area + seed + coordinates seen across ALL invocations
(only ever improving), plus ``next_seed`` so seeds are never reused on resume.
Written after every attempt via tmp+replace, so an interrupt (e.g. the ENOSPC
that killed the original run) loses at most the in-flight attempt.

On a per-``n`` improvement we call
``pipeline.computable.compare_to_record`` against the
``small_points_optimization`` pack. If the new area beats a published
``best_known`` entry beyond tolerance the verdict is ``new-record`` and we
announce it loudly with a HAND-CHECK warning — a flagged candidate is never a
claim until coordinates are checked against the cited sources and a confirm
run relabels the ledger.

Wall-clock gotcha: this candidate takes ~19 min per (n, seed) at budget 80000
(CMA-ES is seconds). Size --hours by measured wall-clock, not eval count.

    uv run --with numpy --with scipy --with cma \
        python -m experiments.heilbronn.record_run \
        --candidate experiments/auto/20260525-042258-C2/search/g2_e1.py \
        --ns 13,15 --budget 80000 --hours 8
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from concurrent.futures import FIRST_COMPLETED, ProcessPoolExecutor, wait
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

# Cap BLAS/OpenMP threads to 1 BEFORE numpy is imported (transitively via the
# bench import below). With one worker per core, multi-threaded BLAS only
# oversubscribes the CPU and thrashes (it was the cause of a "6-hour" attempt
# that was really thread contention, not compute). The matrices here are tiny,
# so single-threaded workers are faster anyway. Override via the env if needed.
for _v in (
    "OMP_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "MKL_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS",
    "NUMEXPR_NUM_THREADS",
):
    os.environ.setdefault(_v, "1")

from experiments.heilbronn.bench import (  # noqa: E402  (must follow thread env)
    _validated_pts_area,
    load_candidate,
)
from pipeline.computable import compare_to_record, load_pack  # noqa: E402


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _default_state_path(candidate: str) -> Path:
    return Path(f"{candidate}.record_run.json")


def _load_state(path: Path, candidate: str, ns: list[int], seed0: int) -> dict:
    """Load cumulative state, or initialize a fresh one.

    Merges nothing destructively: an existing best is preserved as-is and only
    overwritten later when a strictly larger area is found. ``next_seed`` is
    resumed from the file (never below ``seed0``) so seeds are never reused.
    """
    if path.exists():
        try:
            state = json.loads(path.read_text())
        except (OSError, json.JSONDecodeError) as exc:
            raise SystemExit(
                f"state file {path} exists but is unreadable ({exc}); "
                f"move it aside or pass a different --state to start fresh"
            )
        if state.get("candidate") != candidate:
            print(
                f"[warn] state candidate {state.get('candidate')!r} != "
                f"--candidate {candidate!r}; resuming its seeds/bests anyway",
                file=sys.stderr,
            )
        state["next_seed"] = max(int(state.get("next_seed", seed0)), seed0)
        state.setdefault("best", {})
        state.setdefault("total_attempts", 0)
        # Keep the union of historical ns and the ns requested this run.
        state["ns"] = sorted(set(state.get("ns", [])) | set(ns))
        return state
    return {
        "candidate": candidate,
        "pack": None,  # filled in by caller
        "ns": sorted(set(ns)),
        "next_seed": seed0,
        "best": {},
        "total_attempts": 0,
        "created": _now_iso(),
    }


def _save_state(path: Path, state: dict) -> None:
    """Atomic write: an ENOSPC/crash mid-write leaves the prior state intact."""
    state["updated"] = _now_iso()
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(state, indent=2))
    os.replace(tmp, path)


def _announce_record(n: int, area: float, cmp: dict) -> None:
    bar = "=" * 64
    print(f"\n{bar}")
    print(f"  *** POSSIBLE NEW RECORD: n={n}  area={area:.8f} ***")
    print(
        f"  beats {cmp.get('status')} record {cmp.get('record_value')} "
        f"by margin={cmp.get('margin'):+.6g}"
    )
    if cmp.get("source"):
        print(f"  previous: {cmp.get('source')}")
    print("  HAND-CHECK REQUIRED before claiming anything:")
    print("    - verify coords vs Karpov   http://inversed.ru/Ascension.htm")
    print("    - cross-check Erich Friedman's Heilbronn page")
    print("    - run a confirm so the ledger gets the new-record label")
    print(f"  coordinates are in the state file under best['{n}'].points")
    print(f"{bar}\n", flush=True)


def _compare(pack, n: int, area: float) -> dict:
    try:
        return compare_to_record(pack, {"n": n}, area)
    except Exception as exc:  # never let comparison crash the miner
        return {"verdict": "compare-error", "error": str(exc)}


# --- worker side: one candidate load per process, pure compute -----------------

_WORKER_SOLVE: Callable[[int, int, int], Any] | None = None


def _worker_init(candidate_path: str) -> None:
    """Load the candidate once per worker process (spawn-safe)."""
    global _WORKER_SOLVE
    solve, _ = load_candidate(candidate_path)
    _WORKER_SOLVE = solve


def _worker_attempt(n: int, seed: int, budget: int) -> dict[str, Any]:
    """Run + validate one attempt. Never raises: errors come back as data."""
    t0 = time.time()
    try:
        assert _WORKER_SOLVE is not None, "worker not initialized"
        pts = _WORKER_SOLVE(n, seed, budget)
        arr, area = _validated_pts_area(pts, n)
        points = arr.tolist() if arr is not None else None
        err = None
    except Exception as exc:
        area, points, err = float("nan"), None, repr(exc)
    return {
        "n": n,
        "seed": seed,
        "area": area,
        "points": points,
        "duration": time.time() - t0,
        "error": err,
    }


def run(
    candidate: str,
    ns: list[int],
    budget: int,
    hours: float,
    seed0: int,
    state_path: Path,
    pack_name: str,
    max_attempts: int | None,
    workers: int,
) -> int:
    load_candidate(candidate)  # fail fast in the main process too
    pack = load_pack(pack_name)

    state = _load_state(state_path, candidate, ns, seed0)
    state["pack"] = pack_name
    _save_state(state_path, state)  # checkpoint immediately (proves disk is OK)

    deadline = time.time() + hours * 3600.0
    print(
        f"[start] candidate={candidate}\n"
        f"        ns={ns} budget={budget} hours={hours} workers={workers} "
        f"seed0(next_seed)={state['next_seed']}\n"
        f"        pack={pack_name} state={state_path}\n"
        f"        deadline={datetime.fromtimestamp(deadline, timezone.utc).isoformat()}",
        flush=True,
    )

    next_seed = int(state["next_seed"])
    ni = 0
    submitted = 0
    completed = 0
    durations: list[float] = []
    stop_reason: str | None = None

    def can_submit() -> bool:
        nonlocal stop_reason
        if max_attempts is not None and submitted >= max_attempts:
            stop_reason = stop_reason or f"--max-attempts {max_attempts} reached"
            return False
        now = time.time()
        if now >= deadline:
            stop_reason = stop_reason or "wall-clock deadline reached"
            return False
        if durations:
            # One worker per core, so per-attempt wall-clock ~ solo time.
            est = sum(durations) / len(durations)
            if now + est > deadline:
                stop_reason = stop_reason or (
                    f"not enough time for another ~{est / 60:.1f}min attempt"
                )
                return False
        return True

    def record(res: dict) -> None:
        """Fold one completed attempt into cumulative state (main process only)."""
        n, seed, area, dt = res["n"], res["seed"], res["area"], res["duration"]
        valid = res["points"] is not None and area == area  # area==area: not NaN
        key = str(n)
        prev = state["best"].get(key)
        prev_area = prev["area"] if prev else float("-inf")

        if res["error"]:
            print(
                f"[attempt] n={n} seed={seed} ERROR {res['error']} ({dt / 60:.1f}min)",
                file=sys.stderr,
                flush=True,
            )
        if valid and area > prev_area:
            state["best"][key] = {
                "area": area,
                "seed": seed,
                "budget": budget,
                "points": res["points"],
                "found_at": _now_iso(),
            }
            cmp = _compare(pack, n, area)
            state["best"][key]["record_compare"] = cmp
            marker = f"NEW BEST (prev {prev_area:.8f})" if prev else "first"
            print(
                f"[attempt] n={n} seed={seed} area={area:.8f} {marker} "
                f"verdict={cmp.get('verdict')} ({dt / 60:.1f}min)",
                flush=True,
            )
            if cmp.get("verdict") == "new-record":
                _announce_record(n, area, cmp)
        else:
            shown = f"{area:.8f}" if area == area else "INVALID"
            print(
                f"[attempt] n={n} seed={seed} area={shown} "
                f"(best {prev_area:.8f}) ({dt / 60:.1f}min)",
                flush=True,
            )

    ex = ProcessPoolExecutor(
        max_workers=workers, initializer=_worker_init, initargs=(candidate,)
    )
    meta: dict = {}
    inflight: set = set()

    def submit() -> None:
        nonlocal ni, next_seed, submitted
        n = ns[ni % len(ns)]
        ni += 1
        s = next_seed
        next_seed += 1
        submitted += 1
        fut = ex.submit(_worker_attempt, n, s, budget)
        meta[fut] = (n, s)
        inflight.add(fut)

    try:
        while len(inflight) < workers and can_submit():
            submit()
        while inflight:
            done, pending = wait(inflight, return_when=FIRST_COMPLETED)
            inflight = set(pending)
            for fut in done:
                n, seed = meta.pop(fut)
                res: dict[str, Any]
                try:
                    res = fut.result()
                except Exception as exc:  # worker process crashed outright
                    res = {
                        "n": n,
                        "seed": seed,
                        "area": float("nan"),
                        "points": None,
                        "duration": 0.0,
                        "error": f"worker-crash: {exc!r}",
                    }
                completed += 1
                durations.append(float(res["duration"]))
                record(res)
                # next_seed reflects everything SUBMITTED, so a kill never
                # reuses an in-flight seed on resume (a few may be skipped).
                state["next_seed"] = next_seed
                state["total_attempts"] = int(state.get("total_attempts", 0)) + 1
                _save_state(state_path, state)
            while len(inflight) < workers and can_submit():
                submit()
        print(f"[stop] {stop_reason or 'no more work'}", flush=True)
    except KeyboardInterrupt:
        print("[stop] interrupted — saving state, shutting down workers", flush=True)
    finally:
        ex.shutdown(wait=False, cancel_futures=True)
        state["next_seed"] = next_seed
        _save_state(state_path, state)

    print("\n[summary] best validated area per n:")
    for key in sorted(state["best"], key=int):
        b = state["best"][key]
        cmp = b.get("record_compare", {})
        print(
            f"  n={key}: area={b['area']:.8f} seed={b['seed']} "
            f"verdict={cmp.get('verdict')} "
            f"(record {cmp.get('record_value')}, margin {cmp.get('margin')})"
        )
    print(
        f"[summary] completed this run={completed} "
        f"total={state['total_attempts']} next_seed={state['next_seed']}",
        flush=True,
    )
    records = [
        k
        for k, b in state["best"].items()
        if b.get("record_compare", {}).get("verdict") == "new-record"
    ]
    if records:
        print(
            f"[summary] *** {len(records)} flagged new-record candidate(s): "
            f"n={','.join(records)} — HAND-CHECK before claiming ***",
            flush=True,
        )
    return 0


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(
        description="Candidate-only Heilbronn record miner (cumulative, resumable)."
    )
    p.add_argument("--candidate", required=True, help="path to candidate module")
    p.add_argument("--ns", default="13,15", help="comma-separated n values")
    p.add_argument("--budget", type=int, default=80000, help="solve() eval budget")
    p.add_argument("--hours", type=float, default=8.0, help="wall-clock deadline")
    p.add_argument(
        "--workers",
        type=int,
        default=1,
        help="parallel solve workers, one attempt per core; default 1 (serial)",
    )
    p.add_argument(
        "--max-attempts",
        type=int,
        default=None,
        help="cap total attempts this run (for testing); default: time-bounded only",
    )
    p.add_argument(
        "--seed0",
        type=int,
        default=1000,
        help="starting seed when no state file exists (resume overrides this)",
    )
    p.add_argument(
        "--state",
        default=None,
        help="state file path (default: {candidate}.record_run.json)",
    )
    p.add_argument(
        "--pack",
        default="small_points_optimization",
        help="computable pack name for record comparison",
    )
    args = p.parse_args(argv if argv is not None else sys.argv[1:])

    try:
        ns = [int(x) for x in args.ns.split(",") if x.strip()]
    except ValueError:
        raise SystemExit(f"--ns must be comma-separated integers, got {args.ns!r}")
    if not ns:
        raise SystemExit("--ns produced no values")
    if args.hours <= 0:
        raise SystemExit("--hours must be positive")
    if args.workers < 1:
        raise SystemExit("--workers must be >= 1")

    state_path = Path(args.state) if args.state else _default_state_path(args.candidate)

    return run(
        candidate=args.candidate,
        ns=ns,
        budget=args.budget,
        hours=args.hours,
        seed0=args.seed0,
        state_path=state_path,
        pack_name=args.pack,
        max_attempts=args.max_attempts,
        workers=args.workers,
    )


if __name__ == "__main__":
    sys.exit(main())
