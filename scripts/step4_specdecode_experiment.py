"""Step 4 — Speculative decoding ↔ CPU branch prediction experiment.

Tests whether the result from step 3 (KV-cache bridge) generalizes.
Same architecture: post-cutoff Side A vs. broader Side B (branch prediction
has decades of literature; we mix recent and classical for a usable corpus).

Hypothesis: speculative decoding for LLMs (2023+) and CPU branch prediction
(1980s–2025) are operationally equivalent — predict-then-verify-with-rollback,
parallel speculative execution, confidence calibration, accept rate as
mispredict rate's mirror. The LLM community plausibly hasn't engaged deeply
with the branch-prediction literature.
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

SIDE_A_LABEL = "LLM speculative decoding"
SIDE_A_YEAR_MIN = 2025  # post-Codex cutoff
SIDE_A_QUERIES = [
    "speculative decoding LLM",
    "draft model speculative sampling",
    "lookahead decoding language model",
    "speculative sampling acceptance",
    "EAGLE speculative decoding",
]
SIDE_A_TOPIC_KEYWORDS = ["speculative", "draft", "verify", "accept", "llm", "transformer", "language model"]

SIDE_B_LABEL = "CPU branch prediction"
SIDE_B_YEAR_MIN = 2010  # broader window; branch prediction is mature
SIDE_B_QUERIES = [
    "branch predictor accuracy",
    "TAGE branch predictor",
    "perceptron branch prediction",
    "neural branch prediction",
    "branch misprediction recovery",
]
SIDE_B_TOPIC_KEYWORDS = ["branch prediction", "branch predictor", "mispredict", "perceptron", "tage", "speculative execution"]


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
                continue
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
    text = (paper["title"] + " " + paper["abstract"]).lower()
    return any(kw.lower() in text for kw in keywords)


def fetch_side(queries: list[str], year_min: int, topic_keywords: list[str]) -> list[dict]:
    all_papers: list[dict] = []
    seen = set()
    for q in queries:
        print(f"  query: '{q}' (year >= {year_min}) ...", end=" ", flush=True)
        batch = fetch_by_keyword(q, year_min, MAX_PER_QUERY)
        before = len(all_papers)
        for p in batch:
            wid = p["openalex_id"]
            if wid and wid not in seen:
                seen.add(wid)
                all_papers.append(p)
        print(f"+{len(all_papers) - before}")
    relevant = [p for p in all_papers if is_topically_relevant(p, topic_keywords)]
    print(f"  filtered: {len(relevant)} / {len(all_papers)} topically relevant")
    return relevant


def build_prompt(side_a: list[dict], side_b: list[dict]) -> str:
    lines = [
        "You are reading abstracts from two corpora of computer science papers. "
        "Corpus A is published in 2025 or 2026; Corpus B is published 2010 or "
        "later. Each corpus is from a distinct research community with different "
        "conferences, vocabulary, and citation networks.",
        "",
        "Your task: identify any *operational equivalence* between concepts in "
        "Corpus A and concepts in Corpus B. By \"operational equivalence\" I "
        "mean a structural correspondence such that a mechanism developed for "
        "one corpus could be applied to solve problems in the other corpus — "
        "even if the authors in neither corpus state this.",
        "",
        "Output:",
        "  1. Equivalence claim (one sentence).",
        "  2. Supporting papers (cite by title and year from the corpus only).",
        "  3. Mechanism transferred — what would actually be reused.",
        "  4. Why the original authors likely did not state this equivalence.",
        "  5. Confidence rating 1–10 with a one-sentence reason.",
        "",
        "Do not fabricate citations. Cite ONLY papers present in the corpus below.",
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
    out_dir = Path(__file__).resolve().parent.parent / "data" / "step4_experiment"
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"Fetching side A: {SIDE_A_LABEL} (year >= {SIDE_A_YEAR_MIN})")
    side_a = fetch_side(SIDE_A_QUERIES, SIDE_A_YEAR_MIN, SIDE_A_TOPIC_KEYWORDS)[:8]

    print(f"\nFetching side B: {SIDE_B_LABEL} (year >= {SIDE_B_YEAR_MIN})")
    side_b = fetch_side(SIDE_B_QUERIES, SIDE_B_YEAR_MIN, SIDE_B_TOPIC_KEYWORDS)[:8]

    print(f"\nFinal: A={len(side_a)}, B={len(side_b)}")
    if len(side_a) < 3 or len(side_b) < 3:
        print("Insufficient corpus.", file=sys.stderr)
        with (out_dir / "corpus.json").open("w") as f:
            json.dump({"side_a": side_a, "side_b": side_b}, f, indent=2)
        return 1

    with (out_dir / "corpus.json").open("w") as f:
        json.dump({
            "side_a_label": SIDE_A_LABEL,
            "side_b_label": SIDE_B_LABEL,
            "side_a_queries": SIDE_A_QUERIES,
            "side_b_queries": SIDE_B_QUERIES,
            "side_a_year_min": SIDE_A_YEAR_MIN,
            "side_b_year_min": SIDE_B_YEAR_MIN,
            "side_a": side_a,
            "side_b": side_b,
        }, f, indent=2)

    prompt = build_prompt(side_a, side_b)
    (out_dir / "prompt.txt").write_text(prompt)

    print(f"\nSaved: {out_dir / 'corpus.json'} ({len(prompt)} chars prompt)")
    print(f"\nSide A ({SIDE_A_LABEL}):")
    for i, p in enumerate(side_a, 1):
        print(f"  A{i} ({p['year']}) {p['title'][:80]}")
    print(f"\nSide B ({SIDE_B_LABEL}):")
    for i, p in enumerate(side_b, 1):
        print(f"  B{i} ({p['year']}) {p['title'][:80]}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
