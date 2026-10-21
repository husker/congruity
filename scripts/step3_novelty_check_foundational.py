"""Novelty check on foundational + recent KV-cache papers.

The very-recent 2025-2026 papers aren't indexed yet by Semantic Scholar.
Pivot: check the foundational KV-cache papers that are well-indexed
(PagedAttention/vLLM 2023, vAttention 2025). The novelty question is
population-level: does the LLM-serving literature cite classical cache-
replacement work, or has the field been blind to it?

Codex training-data contamination on PagedAttention specifically isn't the
question here — we're asking whether the cited references of the
LLM-serving community include classical cache-replacement work.
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
SLEEP_S = 3.5  # be conservative with S2 free tier

TARGETS = [
    # title, expected_first_author_keyword (for disambiguation)
    ("Efficient Memory Management for Large Language Model Serving with PagedAttention", "Kwon"),
    ("vAttention: Dynamic Memory Management for Serving LLMs without PagedAttention", "Prabhu"),
    ("SGLang: Efficient Execution of Structured Language Model Programs", "Zheng"),
    ("FlashInfer: Efficient and Customizable Attention Engine for LLM Inference Serving", "Ye"),
]

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
    r"virtual memory",
    r"demand paging",
]


def s2_search_with_retry(title: str, max_retries: int = 4) -> str | None:
    backoff = 5.0
    for attempt in range(max_retries):
        try:
            resp = requests.get(
                f"{S2_BASE}/paper/search",
                params={"query": title, "limit": 5, "fields": "paperId,title,year,authors"},
                timeout=REQUEST_TIMEOUT,
            )
            if resp.status_code == 429:
                print(f"  rate-limited (attempt {attempt + 1}), sleeping {backoff}s", file=sys.stderr)
                time.sleep(backoff)
                backoff *= 1.8
                continue
            if resp.status_code != 200:
                print(f"  HTTP {resp.status_code}: {resp.text[:150]}", file=sys.stderr)
                return None
            data = resp.json()
            for paper in data.get("data", []):
                if title.split(":")[0].lower() in (paper.get("title") or "").lower():
                    return paper["paperId"]
            if data.get("data"):
                return data["data"][0]["paperId"]
            return None
        except requests.exceptions.RequestException as e:
            print(f"  network err: {e}", file=sys.stderr)
            time.sleep(backoff)
            backoff *= 1.8
    return None


def s2_references_with_retry(paper_id: str, max_retries: int = 4) -> list[dict]:
    refs: list[dict] = []
    offset = 0
    backoff = 5.0
    while True:
        attempt = 0
        while attempt < max_retries:
            try:
                resp = requests.get(
                    f"{S2_BASE}/paper/{paper_id}/references",
                    params={"fields": "title,year,authors", "offset": offset, "limit": 100},
                    timeout=REQUEST_TIMEOUT,
                )
                if resp.status_code == 429:
                    print(f"    rate-limited, sleep {backoff}s", file=sys.stderr)
                    time.sleep(backoff)
                    backoff *= 1.8
                    attempt += 1
                    continue
                if resp.status_code != 200:
                    print(f"    HTTP {resp.status_code}", file=sys.stderr)
                    return refs
                data = resp.json()
                chunk = data.get("data", [])
                for entry in chunk:
                    cited = entry.get("citedPaper") or {}
                    if cited.get("title"):
                        refs.append({
                            "title": cited["title"],
                            "year": cited.get("year"),
                        })
                offset += len(chunk)
                if len(chunk) < 100:
                    return refs
                time.sleep(SLEEP_S)
                break
            except requests.exceptions.RequestException as e:
                print(f"    err: {e}", file=sys.stderr)
                time.sleep(backoff)
                backoff *= 1.8
                attempt += 1
        if attempt >= max_retries:
            return refs


def match_classic(title: str) -> list[str]:
    return [pat for pat in CLASSIC_PATTERNS if re.search(pat, title or "", re.IGNORECASE)]


def main() -> int:
    results = []
    print(f"Checking {len(TARGETS)} foundational/recent KV-cache papers via Semantic Scholar:\n")

    # Warm up — initial pause to recover from previous rate-limit window
    print("Initial 10s pause to clear any rate-limit state...")
    time.sleep(10)

    for title, _author in TARGETS:
        print(f"--- {title[:70]} ---")
        pid = s2_search_with_retry(title)
        if not pid:
            print("  no S2 match\n")
            continue
        print(f"  S2 paperId: {pid}")
        time.sleep(SLEEP_S)
        refs = s2_references_with_retry(pid)
        print(f"  references: {len(refs)}")
        if not refs:
            print()
            continue

        classic = [{**r, "patterns": match_classic(r["title"])} for r in refs
                   if match_classic(r["title"])]
        print(f"  classical cache-replacement matches: {len(classic)}/{len(refs)}")
        for c in classic[:15]:
            print(f"    [{c.get('year')}] {c['title'][:80]}")
        if len(classic) > 15:
            print(f"    ... and {len(classic) - 15} more")
        print()

        results.append({
            "paper": title,
            "s2_id": pid,
            "total_refs": len(refs),
            "classic_refs": len(classic),
            "classic_detail": classic,
        })
        time.sleep(SLEEP_S)

    out_path = Path("data/step3_experiment/novelty_check_foundational.json")
    out_path.write_text(json.dumps(results, indent=2))

    print("=" * 70)
    print("VERDICT — Does the LLM-serving literature cite classical cache work?")
    print("=" * 70)
    if not results:
        print("  No usable results — Semantic Scholar unavailable.")
        return 1
    for r in results:
        pct = (r["classic_refs"] / r["total_refs"]) * 100 if r["total_refs"] else 0
        print(f"  {r['paper'][:55]:55} → {r['classic_refs']}/{r['total_refs']} ({pct:.0f}%)")
    total_c = sum(r["classic_refs"] for r in results)
    total_r = sum(r["total_refs"] for r in results)
    if total_r:
        print(f"\n  OVERALL: {total_c}/{total_r} ({100 * total_c / total_r:.1f}%) "
              f"of refs are classical cache-replacement work.")
    print()
    print("Interpretation:")
    print("  >15% : the community is well aware of the connection.")
    print("  5-15%: partial — some papers cite, some don't.")
    print("  <5%  : the community has been broadly blind to it.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
