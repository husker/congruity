"""Layer 2 — cross-citation density across topic pairs.

For each (T_i, T_j) pair, count how many of T_i's sampled papers cite T_j's
sampled papers (and vice versa). A pair is Swanson-shape if the cross/within
ratio is below the target threshold (default 10%).

This generalizes scripts/step0_alt_pairs.py from hand-picked pairs to N×N
across a topic list.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from pipeline._openalex import extract_id
from pipeline.corpus import sample_topic
from pipeline.topics import load_or_fetch

DENSITY_CACHE = Path(__file__).resolve().parent.parent / "data" / "pipeline" / "density_matrix.json"
DEFAULT_THRESHOLD = 0.10
DEFAULT_SAMPLE = 50
DEFAULT_TOP_N_TOPICS = 50  # cap input pool to keep API budget bounded


def fetch_all_samples(topic_ids: list[str], sample_size: int) -> dict[str, list[dict]]:
    """Pre-fetch papers + refs for each topic, return {topic_id_short: papers}."""
    out: dict[str, list[dict]] = {}
    for i, tid in enumerate(topic_ids):
        short = extract_id(tid)
        print(f"  [{i+1}/{len(topic_ids)}] sampling topic {short}...",
              end=" ", flush=True)
        papers = sample_topic(tid, cap=sample_size, include_refs=True)
        out[short] = papers
        print(f"{len(papers)} papers")
    return out


def density(side_a: list[dict], side_b: list[dict]) -> dict:
    """Compute cross/within citation density for a pair (A, B)."""
    a_ids = {p["openalex_id"] for p in side_a if p.get("openalex_id")}
    b_ids = {p["openalex_id"] for p in side_b if p.get("openalex_id")}

    a_to_b = a_to_a = 0
    for p in side_a:
        for ref in p.get("referenced_works") or []:
            if ref in b_ids:
                a_to_b += 1
            elif ref in a_ids and ref != p["openalex_id"]:
                a_to_a += 1
    b_to_a = b_to_b = 0
    for p in side_b:
        for ref in p.get("referenced_works") or []:
            if ref in a_ids:
                b_to_a += 1
            elif ref in b_ids and ref != p["openalex_id"]:
                b_to_b += 1

    def r(n: int, d: int) -> float | None:
        return (n / d) if d else None

    return {
        "a_papers": len(a_ids),
        "b_papers": len(b_ids),
        "a_to_b": a_to_b,
        "b_to_a": b_to_a,
        "a_to_a": a_to_a,
        "b_to_b": b_to_b,
        "a_ratio": r(a_to_b, a_to_a),
        "b_ratio": r(b_to_a, b_to_b),
    }


def build_density_matrix(
    samples: dict[str, list[dict]],
    threshold: float = DEFAULT_THRESHOLD,
) -> list[dict]:
    """Compute density for all N*(N-1)/2 pairs. Return list sorted by max_ratio asc."""
    ids = sorted(samples.keys())
    pairs = []
    n = len(ids)
    total = n * (n - 1) // 2
    counter = 0
    for i in range(n):
        for j in range(i + 1, n):
            counter += 1
            if counter % 100 == 0:
                print(f"  density: {counter}/{total} pairs", file=sys.stderr)
            a_id, b_id = ids[i], ids[j]
            d = density(samples[a_id], samples[b_id])
            a_r = d["a_ratio"]
            b_r = d["b_ratio"]
            if a_r is None or b_r is None:
                continue  # skip degenerate cases
            max_r = max(a_r, b_r)
            pairs.append({
                "a": a_id,
                "b": b_id,
                "density": d,
                "max_ratio": max_r,
                "passes_threshold": max_r < threshold,
            })
    pairs.sort(key=lambda p: p["max_ratio"])
    return pairs


def main() -> int:
    p = argparse.ArgumentParser(description="Compute cross-citation density across topic pairs")
    p.add_argument("--sample-size", type=int, default=DEFAULT_SAMPLE)
    p.add_argument("--threshold", type=float, default=DEFAULT_THRESHOLD)
    p.add_argument("--top-n-topics", type=int, default=DEFAULT_TOP_N_TOPICS,
                   help="Cap topic pool size; defaults to top-N by works_count")
    p.add_argument("--output", type=Path, default=DENSITY_CACHE)
    args = p.parse_args()

    topics = load_or_fetch()
    if not topics:
        print("No topics. Run `python -m pipeline.topics` first.", file=sys.stderr)
        return 1

    # Cap by works_count (proxy for "active topic")
    topics_sorted = sorted(topics, key=lambda t: -t.get("works_count", 0))
    selected = topics_sorted[:args.top_n_topics]
    print(f"Selected top-{len(selected)} topics by works_count")

    print(f"\nSampling {args.sample_size} papers per topic (with refs)...")
    samples = fetch_all_samples([t["id"] for t in selected], args.sample_size)

    print(f"\nComputing density matrix ({len(samples)}x{len(samples)} pairs)...")
    pairs = build_density_matrix(samples, threshold=args.threshold)
    passing = [p for p in pairs if p["passes_threshold"]]

    # Attach display names for reporting
    name_by_id = {extract_id(t["id"]): t["display_name"] for t in selected}
    for pp in pairs:
        pp["a_name"] = name_by_id.get(pp["a"], pp["a"])
        pp["b_name"] = name_by_id.get(pp["b"], pp["b"])

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps({
        "threshold": args.threshold,
        "sample_size": args.sample_size,
        "topic_count": len(selected),
        "total_pairs": len(pairs),
        "passing_pairs": len(passing),
        "pairs": pairs,
    }, indent=2))

    print(f"\nTotal pairs computed: {len(pairs)}")
    print(f"Pairs passing threshold (<{args.threshold:.0%}): {len(passing)}")
    print("\nTop 10 most-separated pairs:")
    for pp in pairs[:10]:
        marker = "PASS" if pp["passes_threshold"] else "    "
        print(f"  [{marker}] {pp['max_ratio']:.1%}  {pp['a_name'][:35]:35} ↔ {pp['b_name'][:35]}")
    print(f"\nSaved: {args.output}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
