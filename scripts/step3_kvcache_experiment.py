"""Step 3 — KV-cache management ↔ database buffer-pool experiment.

Tests Codex on cross-subfield bridge-finding using papers from 2025–2026 only,
past Codex's training cutoff. If Codex still surfaces a bridge, it cannot be
explained by training-data memorization (since it never read these papers).
The harder question — is the proposed bridge real or hallucinated — needs a
domain expert or the differential-test oracle to answer.

Method
------
1. Fetch ~5 papers per side from OpenAlex, restricted to publication_year >= 2025.
2. Side A: KV-cache / attention-cache management for LLM serving.
3. Side B: cache replacement / buffer-pool work in databases and systems.
4. Build a single prompt (no redaction needed; post-cutoff content is the
   ablation).
5. Run Codex on the prompt; save output.
6. Run the existing citation checker against the corpus.
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
PER_PAGE = 25
MAX_PER_QUERY = 40
SLEEP_S = 0.25
REQUEST_TIMEOUT = 60
YEAR_MIN = 2025  # post-Codex training cutoff

SIDE_A_QUERIES = [
    "PagedAttention KV cache",
    "KV cache eviction LLM",
    "prefix caching LLM serving",
    "attention sink long context",
    "vLLM KV cache memory",
]

SIDE_A_LABEL = "LLM serving — KV-cache management"

SIDE_B_QUERIES = [
    "learned cache replacement policy",
    "buffer pool replacement database",
    "adaptive cache replacement",
    "tiered storage cache management",
    "page replacement machine learning",
]

SIDE_B_LABEL = "databases / systems — cache replacement"


def abstract_from_inverted(idx: dict | None) -> str:
    if not idx:
        return ""
    positions: dict[int, str] = {}
    for word, poses in idx.items():
        for p in poses:
            positions[p] = word
    if not positions:
        return ""
    return " ".join(positions[i] for i in sorted(positions.keys()))


def fetch_by_keyword(query: str, year_min: int, cap: int) -> list[dict]:
    """Fetch papers matching the keyword query, publication_year >= year_min.
    Returns a list of works with abstracts."""
    papers: list[dict] = []
    cursor = "*"
    while len(papers) < cap:
        per_page = min(PER_PAGE, cap - len(papers))
        params = {
            "filter": f"publication_year:>{year_min - 1}",
            "search": query,
            "per-page": per_page,
            "cursor": cursor,
            "select": "id,title,publication_year,authorships,abstract_inverted_index",
            "mailto": MAILTO,
        }
        try:
            resp = requests.get(BASE, params=params, timeout=REQUEST_TIMEOUT)
            if resp.status_code != 200:
                print(f"  WARN: HTTP {resp.status_code}", file=sys.stderr)
                break
            body = resp.json()
        except (requests.exceptions.Timeout, requests.exceptions.ConnectionError) as e:
            print(f"  ERR {type(e).__name__}", file=sys.stderr)
            break

        results = body.get("results", [])
        if not results:
            break

        for work in results:
            abstract = abstract_from_inverted(work.get("abstract_inverted_index"))
            if not abstract or len(abstract.split()) < 30:
                continue  # skip short/missing abstracts
            authors = [a["author"]["display_name"]
                       for a in work.get("authorships", [])
                       if a.get("author")]
            papers.append({
                "openalex_id": work.get("id"),
                "title": work.get("title"),
                "year": work.get("publication_year"),
                "authors": authors[:5],
                "abstract": abstract,
                "matched_query": query,
            })

        cursor = body.get("meta", {}).get("next_cursor")
        if not cursor:
            break
        time.sleep(SLEEP_S)
    return papers


def is_topically_relevant(paper: dict, keywords: list[str]) -> bool:
    """Quick check: does the abstract contain at least one of the topic keywords?
    Defends against off-topic results."""
    text = (paper["title"] + " " + paper["abstract"]).lower()
    return any(kw.lower() in text for kw in keywords)


def dedupe_keep_best(papers: list[dict]) -> list[dict]:
    seen = {}
    for p in papers:
        wid = p["openalex_id"]
        if wid and wid not in seen:
            seen[wid] = p
    return list(seen.values())


def fetch_side(queries: list[str], topic_keywords: list[str], cap_per_query: int = MAX_PER_QUERY) -> list[dict]:
    all_papers: list[dict] = []
    for q in queries:
        print(f"  query: '{q}' ...", end=" ", flush=True)
        batch = fetch_by_keyword(q, YEAR_MIN, cap_per_query)
        print(f"+{len(batch)}")
        all_papers.extend(batch)
    deduped = dedupe_keep_best(all_papers)
    relevant = [p for p in deduped if is_topically_relevant(p, topic_keywords)]
    print(f"  deduped: {len(deduped)} unique; topic-filtered: {len(relevant)}")
    return relevant


def build_prompt(side_a: list[dict], side_b: list[dict]) -> str:
    lines = [
        "You are reading abstracts from two corpora of computer science papers, "
        "all published in 2025 or 2026. Each corpus is drawn from a distinct "
        "research community with different conferences, vocabulary, and citation networks.",
        "",
        "Your task: identify any *operational equivalence* between concepts in "
        "Corpus A and concepts in Corpus B. By \"operational equivalence\" I mean "
        "a structural correspondence such that a mechanism developed for one "
        "corpus could be applied to solve problems in the other corpus — even if "
        "the authors in neither corpus state this.",
        "",
        "Output:",
        "  1. Equivalence claim (one sentence).",
        "  2. Supporting papers (cite by title and year from the corpus only).",
        "  3. Mechanism transferred — what would actually be reused.",
        "  4. Why the original authors likely did not state this equivalence.",
        "  5. Confidence rating 1–10 with a one-sentence reason.",
        "",
        "If you cannot find a substantive equivalence, say so explicitly. Do not "
        "fabricate citations. Cite ONLY papers present in the corpus below.",
        "",
        f"=========== Corpus A — {len(side_a)} papers ===========",
        "",
    ]
    for i, p in enumerate(side_a, 1):
        lines.append(f"--- Paper A{i} ({p['year']}): {p['title']} ---")
        if p.get("authors"):
            lines.append(f"Authors: {', '.join(p['authors'])}")
        lines.append("")
        lines.append(p["abstract"])
        lines.append("")
    lines.append(f"=========== Corpus B — {len(side_b)} papers ===========")
    lines.append("")
    for i, p in enumerate(side_b, 1):
        lines.append(f"--- Paper B{i} ({p['year']}): {p['title']} ---")
        if p.get("authors"):
            lines.append(f"Authors: {', '.join(p['authors'])}")
        lines.append("")
        lines.append(p["abstract"])
        lines.append("")
    return "\n".join(lines)


def main() -> int:
    out_dir = Path(__file__).resolve().parent.parent / "data" / "step3_experiment"
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"Fetching side A: {SIDE_A_LABEL} (year >= {YEAR_MIN})")
    side_a = fetch_side(
        SIDE_A_QUERIES,
        topic_keywords=["kv cache", "kv-cache", "attention", "llm", "transformer", "pagedattention", "vllm"],
    )

    print(f"\nFetching side B: {SIDE_B_LABEL} (year >= {YEAR_MIN})")
    side_b = fetch_side(
        SIDE_B_QUERIES,
        topic_keywords=["cache replacement", "buffer pool", "page replacement", "eviction", "lru", "buffer"],
    )

    # Cap each side at 8 to keep prompt manageable
    side_a = side_a[:8]
    side_b = side_b[:8]

    print(f"\nFinal corpus: A={len(side_a)}, B={len(side_b)}")
    if len(side_a) < 3 or len(side_b) < 3:
        print("Insufficient corpus.", file=sys.stderr)
        # Still save what we have for inspection
        with (out_dir / "corpus.json").open("w") as f:
            json.dump({"side_a": side_a, "side_b": side_b}, f, indent=2)
        return 1

    # Save corpus
    with (out_dir / "corpus.json").open("w") as f:
        json.dump({
            "side_a_label": SIDE_A_LABEL,
            "side_b_label": SIDE_B_LABEL,
            "side_a_queries": SIDE_A_QUERIES,
            "side_b_queries": SIDE_B_QUERIES,
            "year_min": YEAR_MIN,
            "side_a": side_a,
            "side_b": side_b,
        }, f, indent=2)

    prompt = build_prompt(side_a, side_b)
    (out_dir / "prompt.txt").write_text(prompt)

    print(f"\nCorpus saved: {out_dir / 'corpus.json'}")
    print(f"Prompt saved: {out_dir / 'prompt.txt'} ({len(prompt)} chars)")
    print("\nPapers in corpus (verify topic-relevance by eye before running codex):")
    print(f"\n--- Side A ({SIDE_A_LABEL}) ---")
    for i, p in enumerate(side_a, 1):
        print(f"  A{i} ({p['year']}) {p['title'][:80]}")
    print(f"\n--- Side B ({SIDE_B_LABEL}) ---")
    for i, p in enumerate(side_b, 1):
        print(f"  B{i} ({p['year']}) {p['title'][:80]}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
