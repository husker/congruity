"""Step 0 — Cross-citation density gate for the online-algorithms ↔ caching wedge.

Measures whether the two literatures (theory of online algorithms; systems caching
policies) actually had low cross-citation in the pre-T window, which is the
operational definition of "Swanson-shape" the synthesis doc requires before any
further build work.

Method
------
1. Fetch papers from side A (theory: online algorithms, competitive analysis, paging)
   via OpenAlex title-and-abstract search, restricted to publication_year < PRE_T_YEAR.
2. Fetch papers from side B (systems: cache replacement, page replacement, LRU/LFU,
   buffer pool) under the same year restriction.
3. For each paper, OpenAlex returns `referenced_works` (the papers it cites, by ID).
4. Count cross-side citations (A→B, B→A) and within-side citations (A→A, B→B).
5. Report the cross/within ratio per side; verdict against doc target (<10%).

Caveats (read these before trusting the number)
------------------------------------------------
- Keyword search has false positives: "LRU" matches unrelated work; "online
  algorithm" matches streaming/sequential work. The number is a directional signal
  not a precise measurement.
- OpenAlex pre-1995 coverage is uneven, especially for systems venues. Some papers
  lack `referenced_works` entirely (the field comes back empty). We treat missing
  refs as "no cross-citation observed," which biases the result *toward Swanson-shape*
  (i.e. toward a false PASS). Pessimists should treat the result as an upper bound
  on actual separation, not the true value.
- The pre-T cutoff (1995) is a choice. Online-algorithms-as-a-field was founded by
  Sleator/Tarjan 1985 on paging, so a too-late cutoff captures the bridge papers
  themselves and inflates cross-citation. A too-early cutoff (e.g. 1985) loses
  almost all online-algorithms literature. 1995 sits 10y past founding — long enough
  for both fields to have developed independent vocabulary if they were going to.

Output
------
- Stdout: human-readable summary + verdict.
- data/step0_result.json: full data including all paper IDs and refs (for audit).
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
PRE_T_YEAR = 1995
PER_PAGE = 100       # smaller pages = faster response, less timeout risk
MAX_PER_QUERY = 200  # cap per individual search query
SLEEP_S = 0.2        # ~5 req/sec
REQUEST_TIMEOUT = 60 # generous; OpenAlex can be slow on broad queries
MAX_RETRIES = 3

SIDE_A_QUERIES: list[str] = [
    "competitive ratio",
    "online algorithm",
    "competitive analysis online",
    "k-server problem",
    "paging algorithm competitive",
]

SIDE_B_QUERIES: list[str] = [
    "cache replacement policy",
    "page replacement algorithm",
    "buffer pool",
    "LRU cache",
    "LFU cache",
]


def _request_with_retry(params: dict) -> dict | None:
    """GET with exponential backoff on timeout / 5xx."""
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
    """Fetch up to `cap` papers matching the search query, publication_year < year_max.

    Returns dict keyed by OpenAlex work ID with the slim selected fields.
    """
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
    """Union the per-query results, deduped by work ID."""
    side: dict[str, dict] = {}
    for q in queries:
        print(f"  query: '{q}' ...", end=" ", flush=True)
        batch = fetch_query(q, year_max, cap)
        before = len(side)
        side.update(batch)
        added = len(side) - before
        print(f"+{added} new (total {len(side)})")
    return side


def compute_density(
    a_papers: dict[str, dict],
    b_papers: dict[str, dict],
) -> dict:
    """Count A→B, B→A, A→A, B→B citations."""
    a_ids = set(a_papers.keys())
    b_ids = set(b_papers.keys())

    a_to_b = a_to_a = 0
    a_papers_with_any_refs = 0
    for pid, p in a_papers.items():
        refs = p.get("referenced_works") or []
        if refs:
            a_papers_with_any_refs += 1
        for ref in refs:
            if ref in b_ids:
                a_to_b += 1
            elif ref in a_ids and ref != pid:
                a_to_a += 1

    b_to_a = b_to_b = 0
    b_papers_with_any_refs = 0
    for pid, p in b_papers.items():
        refs = p.get("referenced_works") or []
        if refs:
            b_papers_with_any_refs += 1
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
        "a_papers_with_refs": a_papers_with_any_refs,
        "b_papers_with_refs": b_papers_with_any_refs,
        "a_to_b": a_to_b,
        "b_to_a": b_to_a,
        "a_to_a": a_to_a,
        "b_to_b": b_to_b,
        "a_cross_over_within": ratio(a_to_b, a_to_a),
        "b_cross_over_within": ratio(b_to_a, b_to_b),
    }


def verdict(density: dict, target: float = 0.10) -> str:
    a_r = density["a_cross_over_within"]
    b_r = density["b_cross_over_within"]
    if a_r is None or b_r is None:
        return ("INCONCLUSIVE — at least one side has zero within-side citations; "
                "the keyword filter is likely too narrow, or `referenced_works` is empty.")
    if max(a_r, b_r) < target:
        return (f"PASS — both sides < {target:.0%} cross/within; "
                "Swanson-shape consistent with the doc's wedge.")
    if min(a_r, b_r) > target:
        return (f"FAIL — both sides > {target:.0%} cross/within; "
                "literatures are cross-pollinated. Re-evaluate the wedge.")
    return (f"PARTIAL — one side above {target:.0%}, one below; asymmetric overlap. "
            "Investigate which side is the leaky one and why.")


def main() -> int:
    print(f"Step 0: Cross-citation density gate (pre-{PRE_T_YEAR})")
    print("Wedge: online-algorithms theory ↔ caching/systems policies")
    print()

    print("Fetching side A (online-algorithms theory):")
    a_papers = fetch_side(SIDE_A_QUERIES, PRE_T_YEAR, MAX_PER_QUERY)
    print(f"  total side A: {len(a_papers)} unique papers")
    print()

    print("Fetching side B (caching / systems):")
    b_papers = fetch_side(SIDE_B_QUERIES, PRE_T_YEAR, MAX_PER_QUERY)
    print(f"  total side B: {len(b_papers)} unique papers")
    print()

    density = compute_density(a_papers, b_papers)

    out_dir = Path(__file__).resolve().parent.parent / "data"
    out_dir.mkdir(exist_ok=True)
    out_path = out_dir / "step0_result.json"
    payload = {
        "config": {
            "pre_t_year": PRE_T_YEAR,
            "side_a_queries": SIDE_A_QUERIES,
            "side_b_queries": SIDE_B_QUERIES,
            "max_per_query": MAX_PER_QUERY,
        },
        "density": density,
        "side_a_ids": list(a_papers.keys()),
        "side_b_ids": list(b_papers.keys()),
    }
    with out_path.open("w") as f:
        json.dump(payload, f, indent=2)

    print("=== RESULTS ===")
    print(f"  side A papers: {density['a_papers']} "
          f"({density['a_papers_with_refs']} with refs)")
    print(f"  side B papers: {density['b_papers']} "
          f"({density['b_papers_with_refs']} with refs)")
    print(f"  A→B citations:  {density['a_to_b']}")
    print(f"  B→A citations:  {density['b_to_a']}")
    print(f"  A→A (baseline): {density['a_to_a']}")
    print(f"  B→B (baseline): {density['b_to_b']}")

    def fmt(r: float | None) -> str:
        return f"{r:.1%}" if r is not None else "n/a"

    print()
    print(f"  A cross/within: {fmt(density['a_cross_over_within'])}")
    print(f"  B cross/within: {fmt(density['b_cross_over_within'])}")
    print()
    print(f"VERDICT: {verdict(density)}")
    print()
    print(f"Full data: {out_path}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
