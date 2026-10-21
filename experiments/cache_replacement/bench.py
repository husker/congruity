"""Replay traces through each policy at multiple cache sizes; print hit rates.

Usage:
    uv run python -m experiments.cache_replacement.bench
"""

from __future__ import annotations

import sys
import time

from experiments.cache_replacement.policies import POLICIES
from experiments.cache_replacement.successor_sieve import SuccessorSIEVE_Default
from experiments.cache_replacement.time_to_next_use import TimeToNextUse_Default
from experiments.cache_replacement.traces import TRACE_GENERATORS

ALL_POLICIES = {
    **POLICIES,
    "Succ-SIEVE": SuccessorSIEVE_Default,
    "TTNU": TimeToNextUse_Default,
}


def hit_rate(policy_cls, capacity: int, trace: list[int]) -> tuple[float, float]:
    pol = policy_cls(capacity)
    hits = 0
    t0 = time.time()
    for k in trace:
        if pol.access(k):
            hits += 1
    elapsed = time.time() - t0
    return hits / len(trace), elapsed


def main(argv: list[str]) -> int:
    # Cache sizes expressed as fractions of unique objects in trace
    fractions = [0.001, 0.01, 0.05, 0.10, 0.25]
    print(f"\n{'trace':12s} {'pol':8s} {'frac':>6s} {'cap':>6s} {'hit_rate':>9s} {'time_s':>7s}")
    print("-" * 60)
    for trace_name, gen in TRACE_GENERATORS.items():
        trace = gen()
        unique = len(set(trace))
        for pol_name, cls in ALL_POLICIES.items():
            for frac in fractions:
                cap = max(1, int(unique * frac))
                hr, elapsed = hit_rate(cls, cap, trace)
                print(
                    f"{trace_name:12s} {pol_name:8s} {frac:6.3f} {cap:6d} "
                    f"{hr:9.4f} {elapsed:7.2f}"
                )
            print()  # blank between policies
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
