"""Novelty check via Semantic Scholar (OpenAlex reference data missing for these papers).

Same logic as step3_novelty_check.py, different data source. Semantic Scholar
Graph API often has better reference data for recent arXiv preprints than
OpenAlex.
"""

from __future__ import annotations

import json
import re
import sys
import time
from pathlib import Path

import requests

S2_BASE = "https://api.semanticscholar.org/graph/v1"
REQUEST_TIMEOUT = 30
SLEEP_S = 1.5  # S2 free tier is rate-limited; be polite

TARGET_KEY_TITLES = ["Zipage", "IceCache", "PagedEviction"]

CLASSIC_PATTERNS = [
    r"\bbelady\b",
    r"adaptive replacement cache",
    r"\bARC\b.{0,30}(replacement|cache)",
    r"\bLRU[- ]K\b",
    r"\bLRU\b",
    r"\bLFU\b",
    r"\b2Q\b.{0,30}(buffer|page|cache)",
    r"\bLIRS\b",
    r"\bCAR\b.{0,30}(replacement|cache|clock)",
    r"clock.{0,30}replacement",
    r"cache replacement",
    r"page replacement",
    r"buffer pool",
    r"buffer management",
    r"working set",
    r"\bLeCAR\b",
    r"\bHawkeye\b",
    r"reuse distance",
    r"least recently used",
    r"least frequently used",
    r"page fault",
]


def s2_search(title: str) -> str | None:
    """Return Semantic Scholar paperId for a title, or None."""
    try:
        resp = requests.get(
            f"{S2_BASE}/paper/search",
            params={"query": title, "limit": 5, "fields": "paperId,title,year"},
            timeout=REQUEST_TIMEOUT,
        )
        if resp.status_code != 200:
            print(f"  WARN: S2 search HTTP {resp.status_code}: {resp.text[:120]}",
                  file=sys.stderr)
            return None
        data = resp.json()
        for paper in data.get("data", []):
            if paper.get("title") and title.split(":")[0].lower() in paper["title"].lower():
                return paper["paperId"]
        # Fallback: just first
        if data.get("data"):
            return data["data"][0]["paperId"]
        return None
    except requests.exceptions.RequestException as e:
        print(f"  ERR S2 search: {e}", file=sys.stderr)
        return None


def s2_references(paper_id: str) -> list[dict]:
    """Return list of reference dicts {title, year}."""
    refs: list[dict] = []
    offset = 0
    limit = 100
    while True:
        try:
            resp = requests.get(
                f"{S2_BASE}/paper/{paper_id}/references",
                params={"fields": "title,year", "offset": offset, "limit": limit},
                timeout=REQUEST_TIMEOUT,
            )
            if resp.status_code != 200:
                print(f"    WARN: S2 refs HTTP {resp.status_code}: {resp.text[:120]}",
                      file=sys.stderr)
                break
            data = resp.json()
            chunk = data.get("data", [])
            if not chunk:
                break
            for entry in chunk:
                cited = entry.get("citedPaper") or {}
                if cited.get("title"):
                    refs.append({
                        "title": cited["title"],
                        "year": cited.get("year"),
                    })
            offset += len(chunk)
            if offset >= data.get("total", 0) or len(chunk) < limit:
                break
            time.sleep(SLEEP_S)
        except requests.exceptions.RequestException as e:
            print(f"    ERR S2 refs: {e}", file=sys.stderr)
            break
    return refs


def match_classic(title: str) -> list[str]:
    hits = []
    for pat in CLASSIC_PATTERNS:
        if re.search(pat, title or "", re.IGNORECASE):
            hits.append(pat)
    return hits


def main() -> int:
    corp_path = Path(__file__).resolve().parent.parent / "data" / "step3_experiment" / "corpus.json"
    with corp_path.open() as f:
        corpus = json.load(f)

    targets = []
    seen_ids: set[str] = set()
    for p in corpus["side_a"]:
        for needle in TARGET_KEY_TITLES:
            if needle.lower() in p["title"].lower():
                if p["openalex_id"] not in seen_ids:
                    seen_ids.add(p["openalex_id"])
                    targets.append(p)
                break

    print(f"Targets ({len(targets)}):")
    for t in targets:
        print(f"  - {t['title']}")
    print()

    results = []
    for paper in targets:
        print(f"--- {paper['title']} ---")
        pid = s2_search(paper["title"])
        if not pid:
            print("  S2 search: no match")
            continue
        print(f"  S2 paperId: {pid}")
        time.sleep(SLEEP_S)
        refs = s2_references(pid)
        print(f"  references via S2: {len(refs)}")
        if not refs:
            continue

        classic = []
        for r in refs:
            hits = match_classic(r["title"])
            if hits:
                classic.append({**r, "patterns": hits})

        print(f"  classical cache-replacement hits: {len(classic)}/{len(refs)}")
        for c in classic[:20]:
            print(f"    [{c.get('year')}] {c['title'][:80]}")
            print(f"      ↳ {', '.join(c['patterns'][:3])}")
        if len(classic) > 20:
            print(f"    ... and {len(classic) - 20} more")
        print()

        results.append({
            "paper": paper["title"],
            "s2_id": pid,
            "total_refs": len(refs),
            "classic_refs": len(classic),
            "classic_detail": classic,
        })
        time.sleep(SLEEP_S)

    out_path = corp_path.parent / "novelty_check_s2.json"
    with out_path.open("w") as f:
        json.dump(results, f, indent=2)

    print("=" * 70)
    print("VERDICT")
    print("=" * 70)
    for r in results:
        pct = (r["classic_refs"] / r["total_refs"]) * 100 if r["total_refs"] else 0
        print(f"  {r['paper'][:55]:55} → {r['classic_refs']}/{r['total_refs']} ({pct:.0f}%)")

    total_c = sum(r["classic_refs"] for r in results)
    total_r = sum(r["total_refs"] for r in results)
    if total_r:
        print(f"\n  OVERALL: {total_c}/{total_r} ({100 * total_c / total_r:.1f}%) of refs are classical cache-replacement work.")
    print()
    print("Interpretation:")
    print("  >15% : bridge is folklore — the community knows.")
    print("  5-15%: partial recognition.")
    print("  <5%  : bridge plausibly novel and worth pursuing.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
