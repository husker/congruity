"""Differential-test runner.

Materializes a workload once, runs it through each named policy with the same
capacity, and reports per-policy hit rate. The differential signal — does ARC
beat LRU (or PagedEvictionStub) on LLM-serving-shaped workloads — is in the
hit-rate spread across policies.

Use:
    uv run python -m oracle.runner --n-requests 200 --capacity 500 \\
        --policies lru,arc,paged_stub
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from oracle.policies import POLICIES, CachePolicy
from oracle.workload import WorkloadConfig, generate_workload


def differential_test(
    workload_cfg: WorkloadConfig,
    capacity: int,
    policy_names: list[str],
) -> dict:
    """Run identical events through each named policy. Returns a result dict."""
    events = list(generate_workload(workload_cfg))
    results = []
    for name in policy_names:
        if name not in POLICIES:
            raise ValueError(f"Unknown policy '{name}'. Choices: {sorted(POLICIES)}")
        policy: CachePolicy = POLICIES[name](capacity)
        for ev in events:
            policy.access(ev.key)
        results.append(policy.stats())
    return {
        "workload": vars(workload_cfg),
        "capacity": capacity,
        "n_events": len(events),
        "results": results,
    }


def print_table(result: dict) -> None:
    print(f"\nWorkload: {result['n_events']} events, "
          f"{result['workload']['n_requests']} requests, "
          f"cache capacity={result['capacity']}\n")
    print(f"  {'policy':<18} {'hits':>10} {'misses':>10} {'evicts':>10} {'hit_rate':>10}")
    print(f"  {'-' * 18} {'-' * 10} {'-' * 10} {'-' * 10} {'-' * 10}")
    for r in result["results"]:
        print(f"  {r['policy']:<18} {r['hits']:>10} {r['misses']:>10} "
              f"{r['evictions']:>10} {r['hit_rate']:>10.1%}")
    print()


def main() -> int:
    p = argparse.ArgumentParser(description="Differential cache-policy test on synthetic LLM-serving workload.")
    p.add_argument("--n-requests", type=int, default=200,
                   help="Number of synthetic LLM requests")
    p.add_argument("--capacity", type=int, default=500,
                   help="Cache capacity in blocks")
    p.add_argument("--shared-prefix-blocks", type=int, default=10,
                   help="Blocks shared across all requests (system-prompt prefix caching)")
    p.add_argument("--avg-prompt-blocks", type=int, default=30)
    p.add_argument("--avg-generate-blocks", type=int, default=20)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--policies", default="lru,arc,paged_stub",
                   help="Comma-separated policy names")
    p.add_argument("--output", type=Path, default=None,
                   help="Optional JSON output path")
    args = p.parse_args()

    cfg = WorkloadConfig(
        n_requests=args.n_requests,
        shared_prefix_blocks=args.shared_prefix_blocks,
        avg_prompt_blocks=args.avg_prompt_blocks,
        avg_generate_blocks=args.avg_generate_blocks,
        seed=args.seed,
    )
    result = differential_test(cfg, args.capacity, args.policies.split(","))
    print_table(result)

    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(result, indent=2))
        print(f"Saved: {args.output}\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
