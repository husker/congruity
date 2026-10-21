"""Long-run record miner for snake-in-the-box (Tier B).

Mirrors experiments/heilbronn/record_run.py's hardening (all lessons paid
for in production): resumable cumulative state with monotonic per-n bests,
seeds never reused across restarts (persisted next_seed), atomic tmp+replace
state writes after every attempt (an ENOSPC or kill mid-write leaves the
prior state intact), whole-group-safe process-pool workers, and graceful
KeyboardInterrupt shutdown.

Snake-specific additions: the state stores the actual best SEQUENCE per
dimension — a record here is an existence proof, and the certificate is the
snake itself. Every improvement is re-validated in the parent process
(worker results are never trusted) and graded against the pack's record
table (records.json); a `new-record` verdict is announced loudly and the
certificate is also written to its own file.

Usage:
    # Milestone A: can a method recover the proven optimum 50 on Q7?
    uv run python -m experiments.snake_in_box.record_run \
        --method stochastic_beam --dims 7 --attempts 50 --workers 4

    # Record hunt on open territory (best-known bounds start at n=9):
    uv run python -m experiments.snake_in_box.record_run \
        --candidate experiments/auto/20260702-072744-C1/search/g3_e1.py \
        --dims 9 --attempt-budget 5000000 --workers 6 --max-seconds 14400

Stop any time (Ctrl-C / kill): state is saved after every attempt; rerun
the same command to continue with fresh seeds.
"""

from __future__ import annotations

from experiments._harness import cap_threads

cap_threads()  # before any worker imports numpy-adjacent code

import argparse  # noqa: E402
import json  # noqa: E402
import os  # noqa: E402
import sys  # noqa: E402
import time  # noqa: E402
from concurrent.futures import FIRST_COMPLETED, ProcessPoolExecutor, wait  # noqa: E402
from datetime import datetime, timezone  # noqa: E402
from pathlib import Path  # noqa: E402

from experiments.snake_in_box.baselines import SOLVERS, validate_snake  # noqa: E402
from experiments.snake_in_box.bench import load_candidate  # noqa: E402
from pipeline.computable import compare_to_record, load_pack  # noqa: E402

REPO = Path(__file__).resolve().parent.parent.parent
DEFAULT_STATE_DIR = REPO / "data" / "snake_in_box"
DEFAULT_ATTEMPT_BUDGET = 2_000_000


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _load_state(path: Path, solver_id: str, dims: list[int], seed0: int) -> dict:
    if path.exists():
        try:
            state = json.loads(path.read_text())
        except (OSError, json.JSONDecodeError) as exc:
            raise SystemExit(
                f"state file {path} exists but is unreadable ({exc}); "
                f"move it aside or pass a different --state to start fresh")
        if state.get("solver") != solver_id:
            print(f"[warn] state solver {state.get('solver')!r} != {solver_id!r}; "
                  f"resuming its seeds/bests anyway", file=sys.stderr)
        state["next_seed"] = max(int(state.get("next_seed", seed0)), seed0)
        state.setdefault("best", {})
        state.setdefault("total_attempts", 0)
        state["dims"] = sorted(set(state.get("dims", [])) | set(dims))
        return state
    return {
        "solver": solver_id,
        "dims": sorted(set(dims)),
        "next_seed": seed0,
        "best": {},          # str(n) -> {length, seq, seed, at}
        "total_attempts": 0,
        "created": _now_iso(),
    }


def _save_state(path: Path, state: dict) -> None:
    """Atomic write: crash/ENOSPC mid-write leaves the prior state intact."""
    state["updated"] = _now_iso()
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(state, indent=2))
    os.replace(tmp, path)


def _grade(pack, n: int, length: float) -> dict:
    try:
        return compare_to_record(pack, {"n": n}, length) or {"verdict": "no-record"}
    except Exception as e:  # record table problems must not kill a mining run
        return {"verdict": f"record-compare-error: {e}"}


def _announce(n: int, length: int, cmp: dict, cert_path: Path | None) -> None:
    verdict = cmp.get("verdict", "?")
    line = f"  Q{n}: NEW BEST length={length} — {verdict}"
    if cmp.get("record_value") is not None:
        line += f" (record {cmp['record_value']:.0f}, margin {cmp.get('margin', 0):+.0f})"
    print(line, flush=True)
    if verdict == "new-record":
        print("  " + "=" * 66)
        print(f"  *** NEW RECORD CANDIDATE: Q{n} length {length} ***")
        print(f"  *** certificate: {cert_path} ***")
        print("  *** Hand-verify against the published source before ANY claim,")
        print("  *** then re-validate: the sequence is the proof.")
        print("  " + "=" * 66, flush=True)
    elif verdict == "exceeds-proven-optimal-CHECK-OBJECTIVE":
        print("  *** WARNING: exceeded a PROVEN optimum — this is a metric/validator "
              "bug, not a record. Investigate before trusting ANY result. ***",
              flush=True)


# --------------------------------------------------------------------------
# Worker (runs in a subprocess)
# --------------------------------------------------------------------------

_WORKER_SOLVER = None  # set by _worker_init in each subprocess
_WORKER_POLISH = 0     # exact-completion polish depth (0 = off)


def _worker_init(method: str, candidate_path: str, polish: int = 0) -> None:
    global _WORKER_SOLVER, _WORKER_POLISH
    cap_threads()
    _WORKER_POLISH = polish
    if candidate_path:
        _WORKER_SOLVER = load_candidate(candidate_path)
    elif method:
        _WORKER_SOLVER = SOLVERS[method]
    # else: primer mode — attempts carry the primer, no solver needed


def _attempt(n: int, seed: int, budget: int,
             primer_seq: list[int] | None = None) -> tuple[int, int, list[int], float]:
    t0 = time.perf_counter()
    try:
        if primer_seq is not None:
            from experiments.snake_in_box.complete_certificate import primer_attempt
            seq = primer_attempt(n, primer_seq, seed, budget)
        else:
            seq = _WORKER_SOLVER(n, seed, budget)
        seq = [int(v) for v in seq]
    except Exception:
        seq = []
    if _WORKER_POLISH and seq:
        from experiments.snake_in_box.complete_certificate import polish_sequence
        try:
            seq = polish_sequence(n, seq, _WORKER_POLISH)
        except Exception:
            pass  # polish is best-effort; the raw attempt still counts
    return n, seed, seq, time.perf_counter() - t0


# --------------------------------------------------------------------------
# Main loop
# --------------------------------------------------------------------------


def main(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    src = parser.add_mutually_exclusive_group(required=True)
    src.add_argument("--method", choices=sorted(SOLVERS),
                     help="mine with a built-in baseline")
    src.add_argument("--candidate",
                     help="mine with a candidate module (solve(n, seed, budget))")
    src.add_argument("--primer",
                     help="mine by primer playouts from a stored certificate "
                          "(replay a random prefix, force one divergent move, "
                          "randomized capped DFS; every new record becomes "
                          "the next primer — Meta-NRPA harness design)")
    parser.add_argument("--dims", default="7",
                        help="comma-separated dimensions, e.g. 7 or 9,10")
    parser.add_argument("--attempt-budget", type=int,
                        default=DEFAULT_ATTEMPT_BUDGET,
                        help="candidate-extension evaluations per attempt")
    parser.add_argument("--attempts", type=int, default=0,
                        help="stop after N attempts this run (0 = unlimited)")
    parser.add_argument("--max-seconds", type=int, default=0,
                        help="stop submitting new attempts after this long (0 = unlimited)")
    parser.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2) - 2))
    parser.add_argument("--seed0", type=int, default=1)
    parser.add_argument("--state", type=Path, default=None,
                        help="state file (default data/snake_in_box/record_run_<solver>.json)")
    parser.add_argument("--polish", type=int, default=0,
                        help="exact-completion polish: exhaustively complete "
                             "each attempt's prefixes at unwind depths 4..K "
                             "(node-capped) before grading; 0 = off")
    args = parser.parse_args(argv)

    dims = sorted({int(d) for d in args.dims.split(",") if d.strip()})
    primer_seq: list[int] | None = None
    if args.primer:
        if len(dims) != 1:
            parser.error("--primer supports a single --dims value")
        from experiments.snake_in_box.complete_certificate import load_sequence
        _pn, primer_seq = load_sequence(Path(args.primer))
        ok, plen = validate_snake(dims[0], primer_seq)
        if not ok:
            parser.error(f"primer is not a valid Q{dims[0]} snake: {args.primer}")
        print(f"primer: {args.primer} (length {int(plen)} in Q{dims[0]})")
    if args.primer:
        solver_id = f"primer:{Path(args.primer).stem}"
        slug = f"primer_{Path(args.primer).stem}"
    else:
        solver_id = args.candidate or f"builtin:{args.method}"
        slug = (Path(args.candidate).stem if args.candidate else args.method)
    state_path = args.state or (DEFAULT_STATE_DIR / f"record_run_{slug}.json")
    cert_dir = state_path.parent / "certificates"

    pack = load_pack("snake_in_box")
    state = _load_state(state_path, solver_id, dims, args.seed0)
    state["solver"] = solver_id
    _save_state(state_path, state)  # prove disk is writable before working

    print(f"record_run: solver={solver_id} dims={dims} "
          f"attempt_budget={args.attempt_budget} workers={args.workers}")
    print(f"state: {state_path} (next_seed={state['next_seed']}, "
          f"total_attempts={state['total_attempts']})")
    for n in dims:
        b = state["best"].get(str(n))
        cmp = _grade(pack, n, float(b["length"])) if b else None
        print(f"  Q{n}: best so far = "
              f"{b['length'] if b else '-'}"
              + (f" ({cmp.get('verdict')})" if cmp else ""))

    t_start = time.monotonic()
    attempts_this_run = 0
    stop_submitting = False

    def next_job():
        nonlocal attempts_this_run
        n = dims[attempts_this_run % len(dims)]
        seed = state["next_seed"]
        state["next_seed"] += 1
        attempts_this_run += 1
        return n, seed

    def handle(n: int, seed: int, seq: list[int], elapsed: float) -> None:
        state["total_attempts"] += 1
        ok, length = validate_snake(n, seq)  # never trust the worker
        length = int(length) if ok else -1
        prev = state["best"].get(str(n), {}).get("length", -1)
        marker = ""
        if ok and length > prev:
            entry = {"length": length, "seq": seq, "seed": seed, "at": _now_iso()}
            state["best"][str(n)] = entry
            cert_dir.mkdir(parents=True, exist_ok=True)
            cert_path = cert_dir / f"q{n}_len{length}.json"
            cert_path.write_text(json.dumps(
                {"n": n, "length": length, "seq": seq, "seed": seed,
                 "solver": solver_id, "at": entry["at"]}, indent=2))
            cmp = _grade(pack, n, float(length))
            _announce(n, length, cmp, cert_path)
            marker = " *"
        print(f"  attempt {state['total_attempts']:>6} Q{n} seed={seed:<8} "
              f"len={length if ok else 'INVALID':>4} {elapsed:6.1f}s{marker}",
              flush=True)
        _save_state(state_path, state)

    try:
        with ProcessPoolExecutor(
                max_workers=args.workers,
                initializer=_worker_init,
                initargs=(args.method or "", args.candidate or "",
                          args.polish)) as pool:
            pending = set()
            while True:
                timed_out = args.max_seconds and (
                    time.monotonic() - t_start) >= args.max_seconds
                exhausted = args.attempts and attempts_this_run >= args.attempts
                stop_submitting = bool(timed_out or exhausted)
                while (not stop_submitting and len(pending) < args.workers):
                    n, seed = next_job()
                    job_primer = None
                    if primer_seq is not None:
                        # the Meta-NRPA loop: a new record replaces the primer
                        best_entry = state["best"].get(str(n))
                        job_primer = (best_entry["seq"]
                                      if best_entry and len(best_entry["seq"]) > len(primer_seq)
                                      else primer_seq)
                    pending.add(pool.submit(_attempt, n, seed,
                                            args.attempt_budget, job_primer))
                    exhausted = args.attempts and attempts_this_run >= args.attempts
                    stop_submitting = bool(exhausted)
                if not pending:
                    break
                done, pending = wait(pending, return_when=FIRST_COMPLETED)
                for fut in done:
                    handle(*fut.result())
    except KeyboardInterrupt:
        print("\ninterrupted — state saved; rerun the same command to resume",
              flush=True)
        _save_state(state_path, state)
        return 130

    _save_state(state_path, state)
    print("\ndone. bests:")
    for n in dims:
        b = state["best"].get(str(n))
        if b:
            cmp = _grade(pack, n, float(b["length"]))
            print(f"  Q{n}: {b['length']}  ({cmp.get('verdict')}, "
                  f"record {cmp.get('record_value')})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
