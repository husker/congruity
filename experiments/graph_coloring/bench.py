"""Run graph-coloring baselines over deterministic instances; print colors used.

Every assignment is re-validated (conflicts recomputed from the emitted
coloring) before its color count is reported — a solver's own claim is never
trusted.

Usage:
    uv run python -m experiments.graph_coloring.bench
    uv run python -m experiments.graph_coloring.bench \
        --instances gnp60_05,queen8 --seeds 1 --tabu-iters 5000   # smoke
    uv run python -m experiments.graph_coloring.bench \
        --candidate path/to/candidate.py --json                   # autoverify

With --candidate, the module at the given path must expose
``solve(adj: list[set[int]], seed: int, iters: int) -> list[int]`` and is run
as method "candidate" alongside the baselines. With --json, the last stdout
line is a RESULT_CONTRACT JSON object (see pipeline/autoverify.py): mean
colors per method, composition_method="candidate",
home_field_sota_method="tabucol". A candidate exception or invalid coloring
makes its score NaN, which the autoverify grade deterministically rejects.
"""

from __future__ import annotations

from experiments._harness import cap_threads

# Candidate modules are loaded at runtime and often import numpy; the caps
# must be in the environment before that first import (oversubscription made
# a single Heilbronn eval look like a 6-hour hang).
cap_threads()

import argparse  # noqa: E402  (must follow thread-env setup)
import importlib.util  # noqa: E402
import json  # noqa: E402
import math  # noqa: E402
import sys  # noqa: E402
import time  # noqa: E402

from experiments.graph_coloring.baselines import SOLVERS, validate_coloring  # noqa: E402
from experiments.graph_coloring.instances import INSTANCE_GENERATORS  # noqa: E402


def load_candidate(path: str):
    """Load `solve` (and optional `solve_ablated`) from a candidate module file.

    `solve_ablated` is the same method with its key mechanism disabled; when
    present it runs as method "candidate_ablated" so the outcome can report
    how much the mechanism itself contributes.
    """
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


def main(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--instances",
        default=",".join(INSTANCE_GENERATORS),
        help="Comma-separated instance names",
    )
    parser.add_argument("--methods", default=",".join(SOLVERS))
    parser.add_argument("--seeds", type=int, default=3, help="Seeds per stochastic method")
    parser.add_argument("--seed-base", type=int, default=1,
                        help="First solver seed; seeds run [base, base+seeds). "
                             "Holdout confirmation uses a base never seen in search.")
    parser.add_argument("--tabu-iters", type=int, default=100_000)
    parser.add_argument("--candidate", default=None,
                        help="Path to a candidate module exposing solve(adj, seed, iters)")
    parser.add_argument("--json", action="store_true",
                        help="Emit a RESULT_CONTRACT JSON object as the last stdout line")
    args = parser.parse_args(argv)

    names = [s for s in args.instances.split(",") if s]
    methods = [s for s in args.methods.split(",") if s]
    unknown = [s for s in names if s not in INSTANCE_GENERATORS]
    unknown += [s for s in methods if s not in SOLVERS]
    if unknown:
        print(f"unknown instances/methods: {unknown}", file=sys.stderr)
        return 2

    solvers = {m: SOLVERS[m] for m in methods}
    if args.candidate:
        candidate_solve, ablated_solve = load_candidate(args.candidate)
        solvers["candidate"] = lambda adj, seed, iters: candidate_solve(adj, seed, iters)
        if ablated_solve is not None:
            solvers["candidate_ablated"] = (
                lambda adj, seed, iters: ablated_solve(adj, seed, iters))

    print(f"\n{'instance':14s} {'method':9s} {'seed':>4s} {'colors':>6s} {'valid':>5s} {'time_s':>7s}")
    print("-" * 52)
    # method -> instance -> per-seed color counts in seed order; None marks an
    # exception or invalid coloring. This is the primary trace: paired
    # statistics and degenerate (wrapper) detection both depend on it.
    per_trial: dict[str, dict[str, list[int | None]]] = {
        m: {name: [] for name in names} for m in solvers}
    for name in names:
        adj = INSTANCE_GENERATORS[name]()
        for method, solver in solvers.items():
            for seed in range(args.seed_base, args.seed_base + args.seeds):
                t0 = time.time()
                try:
                    colors = solver(adj, seed, args.tabu_iters)
                except Exception as e:  # candidate code is untrusted
                    per_trial[method][name].append(None)
                    print(f"{name:14s} {method:9s} {seed:4d} {'-':>6s} "
                          f"{'NO':>5s} {time.time() - t0:7.2f}  (exception: {e})")
                    continue
                elapsed = time.time() - t0
                ok, n_colors, conflicts = validate_coloring(adj, colors)
                if not ok:
                    per_trial[method][name].append(None)
                    print(f"{name:14s} {method:9s} {seed:4d} {'-':>6s} "
                          f"{'NO':>5s} {elapsed:7.2f}  ({conflicts} conflicts)")
                    continue
                per_trial[method][name].append(n_colors)
                print(f"{name:14s} {method:9s} {seed:4d} {n_colors:6d} {'yes':>5s} {elapsed:7.2f}")
        print()

    print(f"{'method':18s} {'mean':>7s} {'failures':>9s}")
    print("-" * 37)
    failures = 0
    scores: dict[str, float] = {}
    for method, by_inst in per_trial.items():
        results = [r for ks in by_inst.values() for r in ks]
        valid = [r for r in results if r is not None]
        n_fail = len(results) - len(valid)
        failures += n_fail
        mean = sum(valid) / len(valid) if valid and n_fail == 0 else math.nan
        scores[method] = mean
        mean_s = f"{mean:7.2f}" if mean == mean else f"{'NaN':>7s}"
        print(f"{method:18s} {mean_s} {n_fail:9d}")

    if args.json:
        result = {
            "metric": "colors_used",
            "higher_is_better": False,
            "scores": scores,
            "per_instance": {
                m: {inst: sum(v) / len(v)
                    for inst, ks in by_inst.items()
                    if (v := [k for k in ks if k is not None])}
                for m, by_inst in per_trial.items()
            },
            "per_trial": per_trial,
            "ablated_method": ("candidate_ablated"
                               if "candidate_ablated" in solvers else ""),
            "composition_method": "candidate" if args.candidate else "",
            "home_field_sota_method": "tabucol",
            "n_trials": len(names) * args.seeds,
            "notes": f"instances={','.join(names)}; seeds={args.seed_base}.."
                     f"{args.seed_base + args.seeds - 1}; tabu_iters={args.tabu_iters}; "
                     f"invalid colorings poison the method's score as NaN",
        }
        print(json.dumps(result))
        return 0

    if failures:
        print(f"\n{failures} INVALID colorings", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
