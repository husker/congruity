"""Collect solo solver timings + topology features for offline portfolio study.

Solving is the only expensive step (and the only one with any platform
risk), so it runs ONCE here and writes a JSON dataset; the portfolio
policy analysis (`portfolio.py`) reads that dataset and is pure/cheap,
so selection policies can be iterated without re-solving.

Per instance we record: topology features, solo CaDiCaL wall-clock +
outcome, solo DFS wall-clock + outcome, and whether either timed out.
Reuses the verified solvers and feature extractor from `bench.py`.

    uv run --with python-sat --with networkx --with scikit-learn python \
        -m experiments.hamiltonian_regime_classifier.collect \
        --n-per-cell 8 --timeout 5 --out data/hamiltonian/timings.json
"""

from __future__ import annotations

import argparse
import json
import random
import sys
import time
from pathlib import Path

from experiments.hamiltonian_regime_classifier.bench import (
    FULL_FEATURES,
    extract_features,
    generate_corpus,
    solve_hc_dfs,
    solve_hc_sat,
)


def collect(n_per_cell: int, timeout: float, seed: int, out: Path) -> dict:
    rng = random.Random(seed)
    corpus = generate_corpus(rng, n_per_cell)
    print(f"generated {len(corpus)} instances; solving both solvers...", flush=True)

    records: list[dict] = []
    t0 = time.perf_counter()
    sat_timeouts = dfs_timeouts = 0
    for k, inst in enumerate(corpus):
        feats = extract_features(inst.G)
        sat_has, sat_t = solve_hc_sat(inst.G, timeout_s=timeout)
        dfs_has, dfs_t = solve_hc_dfs(inst.G, timeout_s=timeout)
        sat_timeout = sat_has is None
        dfs_timeout = dfs_has is None
        sat_timeouts += int(sat_timeout)
        dfs_timeouts += int(dfs_timeout)
        # On timeout, charge the full budget as the solver's cost (the
        # honest wall-clock a portfolio would pay before giving up).
        records.append(
            {
                "family": inst.family,
                "n": inst.n,
                "p_or_k": inst.p_or_k,
                "seed": inst.seed,
                "features": {f: feats[f] for f in FULL_FEATURES},
                "sat_time": float(sat_t),
                "sat_timeout": bool(sat_timeout),
                "sat_has_hc": None if sat_has is None else bool(sat_has),
                "dfs_time": float(dfs_t),
                "dfs_timeout": bool(dfs_timeout),
                "dfs_has_hc": None if dfs_has is None else bool(dfs_has),
                # agreement flag: when both decided, they must agree (sanity)
                "agree": (
                    None
                    if (sat_has is None or dfs_has is None)
                    else bool(sat_has == dfs_has)
                ),
            }
        )
        if (k + 1) % 25 == 0:
            print(f"  [{k + 1}/{len(corpus)}] {time.perf_counter() - t0:.1f}s", flush=True)

    disagreements = [r for r in records if r["agree"] is False]
    payload = {
        "meta": {
            "n_per_cell": n_per_cell,
            "timeout": timeout,
            "seed": seed,
            "n_instances": len(records),
            "sat_timeouts": sat_timeouts,
            "dfs_timeouts": dfs_timeouts,
            "disagreements": len(disagreements),
            "features": FULL_FEATURES,
            "solve_seconds": time.perf_counter() - t0,
        },
        "records": records,
    }
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2))
    print(
        f"\nsaved {len(records)} records to {out} "
        f"(sat_timeouts={sat_timeouts} dfs_timeouts={dfs_timeouts} "
        f"disagreements={len(disagreements)})",
        flush=True,
    )
    if disagreements:
        print(
            "WARNING: solver disagreements (a correctness bug in an encoding) "
            f"on {len(disagreements)} instances — investigate before trusting timings.",
            flush=True,
        )
    return payload


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-per-cell", type=int, default=8)
    ap.add_argument("--timeout", type=float, default=5.0)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out", default="data/hamiltonian/timings.json")
    args = ap.parse_args(argv if argv is not None else sys.argv[1:])
    collect(args.n_per_cell, args.timeout, args.seed, Path(args.out))
    return 0


if __name__ == "__main__":
    sys.exit(main())
