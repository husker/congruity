"""Novelty check on the KV-cache ↔ cache-replacement bridge.

For each of the 3 Corpus A papers Codex cited (Zipage, IceCache, PagedEviction),
fetch its `referenced_works` and check how many references point at classical
cache-replacement literature.

If most or all already cite Belady / ARC / LRU-K / LIRS / 2Q / CAR / "cache
replacement", the bridge is folklore — the LLM-serving community already knows
they're reinventing a classical area. If almost none do, the bridge is
plausibly novel and worth chasing.
"""

from __future__ import annotations

import os
import json
import re
import sys
import time
from pathlib import Path

import requests

BASE = "https://api.openalex.org/works"
MAILTO = os.environ.get("OPENALEX_MAILTO", "")
REQUEST_TIMEOUT = 60
SLEEP_S = 0.25
BATCH = 50  # OpenAlex IDs per batched lookup

TARGET_KEY_TITLES = ["Zipage", "IceCache", "PagedEviction"]

# Title patterns that signal classical cache-replacement work
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
]


def extract_id(openalex_url: str) -> str:
    return openalex_url.rstrip("/").split("/")[-1]


def fetch_work_refs(work_id: str) -> dict | None:
    try:
        resp = requests.get(f"{BASE}/{work_id}",
                            params={"mailto": MAILTO,
                                    "select": "id,title,referenced_works"},
                            timeout=REQUEST_TIMEOUT)
        if resp.status_code != 200:
            print(f"  WARN: HTTP {resp.status_code} for {work_id}", file=sys.stderr)
            return None
        return resp.json()
    except (requests.exceptions.Timeout, requests.exceptions.ConnectionError) as e:
        print(f"  ERR {type(e).__name__}", file=sys.stderr)
        return None


def fetch_titles_for_ids(ids: list[str]) -> dict[str, dict]:
    out: dict[str, dict] = {}
    for i in range(0, len(ids), BATCH):
        batch = ids[i:i + BATCH]
        filter_arg = "openalex:" + "|".join(extract_id(r) for r in batch)
        params = {
            "filter": filter_arg,
            "per-page": BATCH,
            "select": "id,title,publication_year",
            "mailto": MAILTO,
        }
        try:
            resp = requests.get(BASE, params=params, timeout=REQUEST_TIMEOUT)
            if resp.status_code != 200:
                print(f"    WARN: HTTP {resp.status_code} on batch", file=sys.stderr)
                time.sleep(SLEEP_S)
                continue
            for work in resp.json().get("results", []):
                out[work["id"]] = work
        except (requests.exceptions.Timeout, requests.exceptions.ConnectionError) as e:
            print(f"    ERR {type(e).__name__}", file=sys.stderr)
        time.sleep(SLEEP_S)
    return out


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
    for p in corpus["side_a"]:
        for needle in TARGET_KEY_TITLES:
            if needle.lower() in p["title"].lower():
                targets.append(p)
                break
    # Dedupe (Zipage appeared twice in corpus)
    seen_ids: set[str] = set()
    unique_targets = []
    for t in targets:
        if t["openalex_id"] not in seen_ids:
            seen_ids.add(t["openalex_id"])
            unique_targets.append(t)
    targets = unique_targets

    print(f"Checking {len(targets)} target papers:")
    for t in targets:
        print(f"  - {t['title']}")
    print()

    overall_results = []
    for paper in targets:
        wid = extract_id(paper["openalex_id"])
        print(f"--- {paper['title']} ({paper['year']}) ---")
        full = fetch_work_refs(wid)
        if full is None:
            continue
        refs: list[str] = full.get("referenced_works") or []
        print(f"  referenced_works: {len(refs)}")
        if not refs:
            print("  (no references in OpenAlex metadata; skipping)")
            continue

        ref_meta = fetch_titles_for_ids(refs)
        print(f"  resolved titles: {len(ref_meta)} / {len(refs)}")

        classic_refs = []
        for rid, rmeta in ref_meta.items():
            title = rmeta.get("title", "") or ""
            hits = match_classic(title)
            if hits:
                classic_refs.append({
                    "title": title,
                    "year": rmeta.get("publication_year"),
                    "patterns": hits,
                })

        print(f"  references matching classical cache-replacement patterns: {len(classic_refs)}")
        for c in classic_refs[:15]:
            print(f"    [{c['year']}] {c['title'][:80]}")
            print(f"      ↳ matched: {', '.join(c['patterns'])}")
        if len(classic_refs) > 15:
            print(f"    ... and {len(classic_refs) - 15} more")
        print()

        overall_results.append({
            "paper_title": paper["title"],
            "paper_year": paper["year"],
            "openalex_id": paper["openalex_id"],
            "total_refs": len(refs),
            "refs_with_metadata": len(ref_meta),
            "classic_refs_count": len(classic_refs),
            "classic_refs": classic_refs,
        })

    # Save
    out_path = corp_path.parent / "novelty_check.json"
    with out_path.open("w") as f:
        json.dump(overall_results, f, indent=2)

    # Verdict
    print("=" * 70)
    print("NOVELTY VERDICT")
    print("=" * 70)
    for r in overall_results:
        if r["total_refs"] == 0:
            verdict = "N/A (no refs)"
        else:
            pct = (r["classic_refs_count"] / r["total_refs"]) * 100
            verdict = f"{r['classic_refs_count']}/{r['total_refs']} refs ({pct:.0f}%) cite classical cache-replacement"
        print(f"  {r['paper_title'][:55]:55} → {verdict}")

    total_classic = sum(r["classic_refs_count"] for r in overall_results)
    total_refs = sum(r["total_refs"] for r in overall_results)
    print()
    if total_refs:
        pct = (total_classic / total_refs) * 100
        print(f"  OVERALL: {total_classic}/{total_refs} ({pct:.1f}%) of references are classical cache-replacement work.")
    print()
    print("Interpretation:")
    print("  - >15% : bridge is folklore. The community knows.")
    print("  - 5-15%: partial recognition. Some papers cite, some don't.")
    print("  - <5%  : bridge is likely novel and worth pursuing.")

    return 0


if __name__ == "__main__":
    sys.exit(main())
