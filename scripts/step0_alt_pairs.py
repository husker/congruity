"""Step 0 (extended) — Test alternative CS sub-pairs for Swanson-shape.

Followup to step0_crosscitation.py which found online-algorithms↔caching has
20–37% cross/within (FAIL). The reviewer's argument was that online-algorithms-
as-a-field was founded analyzing caching, so co-evolution was structural. The
question now: does Pair A (CS algorithmic equivalence) have ANY clean sub-pair,
or is CS broadly too cross-cited for the Swanson approach?

This script measures 3 additional candidate sub-pairs. If all three fail, the
result is strong evidence to rescope (per § 14 decision rule). If one passes,
that pair becomes the new wedge candidate — pending the same caveats (keyword
search bias, pre-T cutoff sensitivity, etc.) as the first run.

Candidate pairs and hypotheses:
  1. streaming algorithms ↔ database query processing
     Hypothesis: cross-cited. Streaming literature emerged from DB community
     (Cormode/Muthukrishnan) and back-cites it heavily.
  2. PL theory ↔ OS / systems software
     Hypothesis: separated. PL theorists and OS systems people genuinely don't
     read each other much — different conferences, different vocabularies.
  3. formal verification ↔ property-based testing
     Hypothesis: separated. Both formal-flavored but different communities:
     PBT is testing-community; FV is verification-community.
"""

from __future__ import annotations

import os
import json
import sys
import time
from pathlib import Path

import requests

BASE = "https://api.openalex.org/works"
MAILTO = os.environ.get("OPENALEX_MAILTO", "")
PER_PAGE = 100
MAX_PER_QUERY = 200
SLEEP_S = 0.2
REQUEST_TIMEOUT = 60
MAX_RETRIES = 3
TARGET_RATIO = 0.10

CANDIDATES: list[dict] = [
    {
        "name": "streaming algorithms ↔ database query processing",
        "pre_t_year": 2005,
        "side_a_label": "streaming / sketches (theory)",
        "side_a_queries": [
            "streaming algorithm",
            "count-min sketch",
            "heavy hitters streaming",
            "data stream algorithm",
            "frequency moments stream",
        ],
        "side_b_label": "database query processing",
        "side_b_queries": [
            "query optimization database",
            "selectivity estimation",
            "cardinality estimation database",
            "approximate query processing",
            "database histogram",
        ],
    },
    {
        "name": "PL theory ↔ OS / systems software",
        "pre_t_year": 2010,
        "side_a_label": "programming-language theory",
        "side_a_queries": [
            "type system theory",
            "operational semantics",
            "denotational semantics",
            "lambda calculus type",
            "type inference algorithm",
        ],
        "side_b_label": "operating systems / kernel",
        "side_b_queries": [
            "operating system kernel",
            "virtual memory management",
            "file system design",
            "process scheduling kernel",
            "system call performance",
        ],
    },
    {
        "name": "formal verification ↔ property-based testing",
        "pre_t_year": 2015,
        "side_a_label": "formal verification",
        "side_a_queries": [
            "formal verification software",
            "model checking",
            "theorem proving Coq",
            "abstract interpretation",
            "proof assistant Isabelle",
        ],
        "side_b_label": "property-based / random testing",
        "side_b_queries": [
            "property-based testing",
            "QuickCheck random testing",
            "fuzz testing software",
            "random testing software",
            "symbolic execution testing",
        ],
    },
]


def _request_with_retry(params: dict) -> dict | None:
    backoff = 2.0
    for _ in range(MAX_RETRIES):
        try:
            resp = requests.get(BASE, params=params, timeout=REQUEST_TIMEOUT)
            if resp.status_code == 200:
                return resp.json()
            if 500 <= resp.status_code < 600:
                print(f"    HTTP {resp.status_code}, retrying in {backoff:.1f}s",
                      file=sys.stderr)
                time.sleep(backoff)
                backoff *= 2
                continue
            print(f"    HTTP {resp.status_code} — body: {resp.text[:200]}",
                  file=sys.stderr)
            return None
        except (requests.exceptions.Timeout, requests.exceptions.ConnectionError) as e:
            print(f"    network error ({type(e).__name__}), retrying in {backoff:.1f}s",
                  file=sys.stderr)
            time.sleep(backoff)
            backoff *= 2
    print(f"    giving up after {MAX_RETRIES} attempts", file=sys.stderr)
    return None


def fetch_query(query: str, year_max: int, cap: int) -> dict[str, dict]:
    papers: dict[str, dict] = {}
    cursor = "*"
    while len(papers) < cap:
        per_page = min(PER_PAGE, cap - len(papers))
        params = {
            "filter": f"publication_year:<{year_max}",
            "search": query,
            "per-page": per_page,
            "cursor": cursor,
            "select": "id,title,publication_year,referenced_works",
            "mailto": MAILTO,
        }
        body = _request_with_retry(params)
        if body is None:
            break
        results = body.get("results", [])
        if not results:
            break
        for work in results:
            wid = work.get("id")
            if wid and wid not in papers:
                papers[wid] = work
        cursor = body.get("meta", {}).get("next_cursor")
        if not cursor:
            break
        time.sleep(SLEEP_S)
    return papers


def fetch_side(queries: list[str], year_max: int, cap: int) -> dict[str, dict]:
    side: dict[str, dict] = {}
    for q in queries:
        print(f"    query: '{q}' ...", end=" ", flush=True)
        batch = fetch_query(q, year_max, cap)
        before = len(side)
        side.update(batch)
        print(f"+{len(side) - before} new (total {len(side)})")
    return side


def compute_density(a_papers: dict, b_papers: dict) -> dict:
    a_ids = set(a_papers.keys())
    b_ids = set(b_papers.keys())

    a_to_b = a_to_a = 0
    a_with_refs = 0
    for pid, p in a_papers.items():
        refs = p.get("referenced_works") or []
        if refs:
            a_with_refs += 1
        for ref in refs:
            if ref in b_ids:
                a_to_b += 1
            elif ref in a_ids and ref != pid:
                a_to_a += 1

    b_to_a = b_to_b = 0
    b_with_refs = 0
    for pid, p in b_papers.items():
        refs = p.get("referenced_works") or []
        if refs:
            b_with_refs += 1
        for ref in refs:
            if ref in a_ids:
                b_to_a += 1
            elif ref in b_ids and ref != pid:
                b_to_b += 1

    def ratio(num: int, den: int) -> float | None:
        return (num / den) if den else None

    return {
        "a_papers": len(a_ids),
        "b_papers": len(b_ids),
        "a_papers_with_refs": a_with_refs,
        "b_papers_with_refs": b_with_refs,
        "a_to_b": a_to_b,
        "b_to_a": b_to_a,
        "a_to_a": a_to_a,
        "b_to_b": b_to_b,
        "a_cross_over_within": ratio(a_to_b, a_to_a),
        "b_cross_over_within": ratio(b_to_a, b_to_b),
    }


def verdict(density: dict, target: float = TARGET_RATIO) -> str:
    a_r = density["a_cross_over_within"]
    b_r = density["b_cross_over_within"]
    if a_r is None or b_r is None:
        return "INCONCLUSIVE"
    if max(a_r, b_r) < target:
        return f"PASS (both < {target:.0%})"
    if min(a_r, b_r) > target:
        return f"FAIL (both > {target:.0%})"
    return "PARTIAL (asymmetric)"


def fmt_ratio(r: float | None) -> str:
    return f"{r:.1%}" if r is not None else "n/a"


def main() -> int:
    print("Step 0 (extended) — alternative CS sub-pair testing")
    print(f"Target: cross/within < {TARGET_RATIO:.0%} for Swanson-shape.\n")

    all_results: list[dict] = []
    for i, cand in enumerate(CANDIDATES, 1):
        print(f"[{i}/{len(CANDIDATES)}] {cand['name']} (pre-{cand['pre_t_year']})")
        print(f"  fetching side A: {cand['side_a_label']}")
        a_papers = fetch_side(cand["side_a_queries"], cand["pre_t_year"], MAX_PER_QUERY)
        print(f"    side A total: {len(a_papers)}")
        print(f"  fetching side B: {cand['side_b_label']}")
        b_papers = fetch_side(cand["side_b_queries"], cand["pre_t_year"], MAX_PER_QUERY)
        print(f"    side B total: {len(b_papers)}")

        density = compute_density(a_papers, b_papers)
        v = verdict(density)

        result = {
            "name": cand["name"],
            "pre_t_year": cand["pre_t_year"],
            "side_a_label": cand["side_a_label"],
            "side_b_label": cand["side_b_label"],
            "density": density,
            "verdict": v,
        }
        all_results.append(result)

        print(f"  A cross/within: {fmt_ratio(density['a_cross_over_within'])}, "
              f"B cross/within: {fmt_ratio(density['b_cross_over_within'])}")
        print(f"  verdict: {v}\n")

    # Comparison table
    print("=" * 80)
    print("COMPARISON TABLE")
    print("=" * 80)
    header = f"{'Pair':<55} {'A cross':>9} {'B cross':>9} {'Verdict':>10}"
    print(header)
    print("-" * len(header))
    for r in all_results:
        name = r["name"][:53] + ".." if len(r["name"]) > 55 else r["name"]
        a = fmt_ratio(r["density"]["a_cross_over_within"])
        b = fmt_ratio(r["density"]["b_cross_over_within"])
        v = r["verdict"].split(" ")[0]  # just PASS/FAIL/PARTIAL/INCONCLUSIVE
        print(f"{name:<55} {a:>9} {b:>9} {v:>10}")
    print()

    # Reference: include the prior result on online-algorithms↔caching
    print("(Reference: online-algorithms ↔ caching from prior run: 36.7% / 20.1% — FAIL)\n")

    # Save
    out_path = Path(__file__).resolve().parent.parent / "data" / "step0_alt_pairs_result.json"
    out_path.parent.mkdir(exist_ok=True)
    with out_path.open("w") as f:
        json.dump({"target_ratio": TARGET_RATIO, "pairs": all_results}, f, indent=2)
    print(f"Full data: {out_path}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
