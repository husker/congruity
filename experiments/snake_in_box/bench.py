"""Snake-in-the-box bench: longest induced path in the hypercube Q_n.

Every returned sequence is re-validated (validate_snake recomputes the
chord-free condition and the length) — a solver's own claim is never
trusted. Score per method = mean snake length (edges) across
instances x seeds; an exception or invalid snake poisons the method's
score as NaN, which the autoverify grade deterministically rejects.

Usage:
    uv run python -m experiments.snake_in_box.bench --smoke
    uv run python -m experiments.snake_in_box.bench --split train --json \
        --candidate path/to/candidate.py            # autosearch fitness
    uv run python -m experiments.snake_in_box.bench --split holdout --json \
        --candidate path/to/candidate.py            # one-shot confirm

Splits are disjoint in both seeds and budget (holdout adds Q8 and unseen
seed base 101). A candidate module must expose
``solve(n: int, seed: int, budget: int) -> list[int]`` — a vertex sequence
forming a snake, budget counted in candidate-extension evaluations.
"""

from __future__ import annotations

from experiments._harness import cap_threads

# Candidate modules are loaded at runtime and may import numpy; caps must be
# in the environment before that first import.
cap_threads()

import argparse  # noqa: E402
import importlib.util  # noqa: E402
import math  # noqa: E402
import sys  # noqa: E402
import time  # noqa: E402

from experiments._harness import emit_result_contract  # noqa: E402
from experiments.snake_in_box.baselines import SOLVERS, validate_snake  # noqa: E402

SPLITS = {
    # name -> (dims, seed_base, budget)
    "train": ((6, 7), 1, 400_000),
    "holdout": ((6, 7, 8), 101, 1_200_000),
    "smoke": ((5,), 1, 8_000),
}


def validate(instance, solution):
    """Battery/forge-compatible validator: instance is {"n": int}."""
    return validate_snake(int(instance["n"]), solution)


def load_candidate(path: str):
    spec = importlib.util.spec_from_file_location("autoverify_candidate", path)
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot load candidate module from {path}")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    solve = getattr(mod, "solve", None)
    if not callable(solve):
        raise ImportError(f"candidate module {path} does not expose solve()")
    return solve


def main(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--split", choices=sorted(SPLITS), default="train")
    parser.add_argument("--smoke", action="store_true",
                        help="shorthand for --split smoke")
    parser.add_argument("--seeds", type=int, default=3,
                        help="seeds per method per dimension")
    parser.add_argument("--candidate", default=None,
                        help="path to a module exposing solve(n, seed, budget)")
    parser.add_argument("--json", action="store_true",
                        help="emit RESULT_CONTRACT JSON as the last stdout line")
    # Record-mode overrides: run at an arbitrary target dimension/budget
    # (used by autosearch --record-dim via the pack's record_command).
    parser.add_argument("--dims", default=None,
                        help="comma-separated dimension override, e.g. 9")
    parser.add_argument("--budget", type=int, default=None,
                        help="evaluation-budget override")
    parser.add_argument("--seed-base", type=int, default=None,
                        help="first seed override")
    args = parser.parse_args(argv)

    split = "smoke" if args.smoke else args.split
    dims, seed_base, budget = SPLITS[split]
    if args.dims:
        dims = tuple(sorted({int(d) for d in args.dims.split(",") if d.strip()}))
    if args.budget is not None:
        budget = args.budget
    if args.seed_base is not None:
        seed_base = args.seed_base

    solvers = dict(SOLVERS)
    if args.candidate:
        solvers["candidate"] = load_candidate(args.candidate)

    print(f"split={split} dims={dims} seeds={seed_base}..{seed_base + args.seeds - 1} "
          f"budget={budget}")
    print(f"\n{'dim':>4s} {'method':16s} {'seed':>4s} {'length':>6s} {'valid':>5s} {'time_s':>7s}")
    print("-" * 50)

    # method -> dim -> per-seed lengths; None marks exception/invalid snake.
    per_trial: dict[str, dict[str, list[int | None]]] = {
        m: {f"q{n}": [] for n in dims} for m in solvers}
    # Best candidate sequence per dimension: a record is an existence proof,
    # so the contract carries the certificate, not just the number.
    best_seqs: dict[str, list[int]] = {}
    for n in dims:
        for method, solver in solvers.items():
            for seed in range(seed_base, seed_base + args.seeds):
                t0 = time.perf_counter()
                try:
                    seq = solver(n, seed, budget)
                except Exception as e:  # candidate code is untrusted
                    per_trial[method][f"q{n}"].append(None)
                    print(f"{n:4d} {method:16s} {seed:4d} {'-':>6s} {'NO':>5s} "
                          f"{time.perf_counter() - t0:7.2f}  (exception: {e})")
                    continue
                elapsed = time.perf_counter() - t0
                ok, length = validate_snake(n, seq)
                if not ok:
                    per_trial[method][f"q{n}"].append(None)
                    print(f"{n:4d} {method:16s} {seed:4d} {'-':>6s} {'NO':>5s} "
                          f"{elapsed:7.2f}  (invalid snake)")
                    continue
                per_trial[method][f"q{n}"].append(int(length))
                if method == "candidate":
                    prev = best_seqs.get(f"q{n}")
                    if prev is None or length > len(prev) - 1:
                        best_seqs[f"q{n}"] = list(seq)
                print(f"{n:4d} {method:16s} {seed:4d} {int(length):6d} "
                      f"{'yes':>5s} {elapsed:7.2f}")
        print()

    print(f"{'method':18s} {'mean_len':>8s} {'failures':>9s}")
    print("-" * 38)
    scores: dict[str, float] = {}
    for method, by_dim in per_trial.items():
        results = [r for ks in by_dim.values() for r in ks]
        valid = [r for r in results if r is not None]
        n_fail = len(results) - len(valid)
        mean = sum(valid) / len(valid) if valid and n_fail == 0 else math.nan
        scores[method] = mean
        mean_s = f"{mean:8.2f}" if mean == mean else f"{'NaN':>8s}"
        print(f"{method:18s} {mean_s} {n_fail:9d}")

    if args.json:
        emit_result_contract(
            metric="snake_length",
            higher_is_better=True,
            scores=scores,
            # Without --candidate (smoke/baseline runs) the contract still
            # needs a valid method key; autoverify only consumes candidate
            # runs, so pointing at the SOTA method is safe.
            composition_method="candidate" if args.candidate else "stochastic_beam",
            home_field_sota_method="stochastic_beam",
            n_trials=len(dims) * args.seeds,
            notes=f"split={split}; dims={dims}; seeds={seed_base}.."
                  f"{seed_base + args.seeds - 1}; budget={budget}; "
                  "length=edges; invalid snakes poison the method score as NaN",
            per_instance={
                m: {dim: sum(v) / len(v)
                    for dim, ks in by_dim.items()
                    if (v := [k for k in ks if k is not None])}
                for m, by_dim in per_trial.items()
            },
            per_trial=per_trial,
            best_sequences=best_seqs,
            record_claims=[
                {"params": {"n": int(dim[1:])}, "value": float(max(v))}
                for m, by_dim in per_trial.items() if m == "candidate"
                for dim, ks in by_dim.items()
                if (v := [k for k in ks if k is not None])
            ],
        )
        return 0

    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
