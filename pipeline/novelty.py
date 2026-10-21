"""Post-L3 novelty check.

Classifies each composition as known / adjacent-known / not-found / unverified
by:
  - searching S2 (Semantic Scholar) + OpenAlex titles for compound queries
    built from the claim's key phrases
  - (not yet implemented) measuring co-citation density across the
    constituent papers — requires corpus-level referenced_works, which this
    module doesn't have access to; _cocitation_density returns 0.0.

The S2 + OpenAlex helpers are stubbable so tests can pin the classification
logic without hitting the network. The default `with_search=False`
short-circuits to `unverified` so compose runs stay cheap and offline-capable
unless the user explicitly opts in (`--with-novelty`).

Similarity is deliberately cheap and deterministic: the fraction of query
tokens present in the hit's title. Compound 2-word queries mean a "strong
hit" (>= KNOWN_SIM_THRESHOLD) is a title containing essentially every query
term — a real overlap signal, not a single shared word.

Classification thresholds:
  known          : >= KNOWN_HIT_THRESHOLD strong hits
  adjacent-known : 1-2 strong hits OR co-citation density >= ADJACENT_COCITE_THRESHOLD
  not-found      : zero strong hits and low density
  unverified     : search disabled or APIs failed
"""

from __future__ import annotations

import json
import re
import time
from collections import Counter
from pathlib import Path
from typing import Iterable

import requests


KNOWN_HIT_THRESHOLD = 3
KNOWN_SIM_THRESHOLD = 0.85
ADJACENT_COCITE_THRESHOLD = 0.3

S2_SEARCH_URL = "https://api.semanticscholar.org/graph/v1/paper/search"
SEARCH_LIMIT = 10
SEARCH_RETRIES = 2
# Both APIs are shared-pool rate-limited; stay polite (>= 1s between calls).
SEARCH_THROTTLE_S = 1.1


def _extract_key_phrases(composition: dict, state: dict) -> list[str]:
    """Cheap deterministic fallback: pick the most-frequent long words from
    the claim and problem. A future revision can route this through codex
    for higher-quality phrase extraction."""
    text = " ".join([
        composition.get("synthesis_claim", "") or "",
        state.get("problem", "") or "",
    ]).lower()
    words = [w.strip(".,;:") for w in text.split() if len(w) > 4]
    counts = Counter(words)
    return [w for w, _ in counts.most_common(6)]


def _title_similarity(query: str, title: str) -> float:
    """Fraction of query content-tokens (len > 3) present in the title."""
    q = {w for w in re.findall(r"[a-z0-9]+", query.lower()) if len(w) > 3}
    if not q:
        return 0.0
    t = set(re.findall(r"[a-z0-9]+", title.lower()))
    return len(q & t) / len(q)


def _s2_title_search(query: str) -> list[dict]:
    """Search Semantic Scholar paper titles. Monkeypatched in tests.

    Returns [{"title", "year", "similarity"}]; empty list on any persistent
    failure (the caller treats empty as no-signal, not as evidence of
    novelty — hence the separate `unverified` class when search is off).
    """
    backoff = 2.0
    for _ in range(SEARCH_RETRIES + 1):
        try:
            time.sleep(SEARCH_THROTTLE_S)  # shared-pool API; stay polite
            resp = requests.get(
                S2_SEARCH_URL,
                params={"query": query, "fields": "title,year",
                        "limit": SEARCH_LIMIT},
                timeout=30,
            )
            if resp.status_code == 200:
                data = resp.json().get("data") or []
                return [
                    {"title": d.get("title") or "",
                     "year": d.get("year"),
                     "similarity": _title_similarity(query, d.get("title") or "")}
                    for d in data
                ]
            if resp.status_code != 429:
                return []
        except requests.RequestException:
            pass
        time.sleep(backoff)
        backoff *= 2
    return []


def _openalex_title_search(query: str) -> list[dict]:
    """Search OpenAlex works by title/fulltext relevance. Monkeypatched in
    tests. Same return shape as _s2_title_search."""
    from pipeline._openalex import request_with_retry

    time.sleep(SEARCH_THROTTLE_S)  # stay polite alongside the S2 calls
    data = request_with_retry("/works", {"search": query,
                                         "per-page": SEARCH_LIMIT})
    if not data:
        return []
    return [
        {"title": w.get("title") or "",
         "year": w.get("publication_year"),
         "openalex_id": w.get("id"),
         "similarity": _title_similarity(query, w.get("title") or "")}
        for w in (data.get("results") or [])
    ]


def _cocitation_density(papers: Iterable[dict]) -> float:
    """NOT IMPLEMENTED — always 0.0.

    A real implementation needs each constituent paper's referenced_works
    from OpenAlex, which requires corpus-level IDs this module doesn't
    receive. Tests monkeypatch this to exercise the adjacent-known branch;
    in production only title-search hits drive the classification.
    """
    _ = papers
    return 0.0


def check_novelty(
    composition: dict,
    state: dict,
    run_dir: Path,
    with_search: bool = False,
) -> dict:
    """Run the novelty check for one composition.

    Returns a NoveltyReport-shaped dict and writes a JSON copy to
    <run_dir>/novelty/<cid>/report.json for audit.
    """
    cid = composition.get("id", "unknown")

    if not with_search:
        out = {
            "classification": "unverified",
            "key_phrases": [],
            "hits": [],
            "cocite_density": 0.0,
            "queries_issued": [],
            "note": "search disabled (with_search=False)",
        }
        _persist(out, run_dir, cid)
        return out

    key_phrases = _extract_key_phrases(composition, state)
    queries_issued: list[str] = []
    hits: list[dict] = []

    # Compound 2-word queries: with token-containment similarity, a strong
    # hit on a 2-word query means the title carries both key terms — a real
    # overlap signal. Single-word queries would make every title containing
    # one common word a "strong hit".
    import itertools
    top = key_phrases[:3]
    queries = [" ".join(pair) for pair in itertools.combinations(top, 2)] or top

    for phrase in queries[:3]:
        queries_issued.append(f"s2:{phrase}")
        for h in _s2_title_search(phrase):
            hits.append({**h, "source": "s2"})
        queries_issued.append(f"openalex:{phrase}")
        for h in _openalex_title_search(phrase):
            hits.append({**h, "source": "openalex"})

    density = _cocitation_density(composition.get("papers", []))

    strong_hits = [h for h in hits if h.get("similarity", 0.0) >= KNOWN_SIM_THRESHOLD]
    if len(strong_hits) >= KNOWN_HIT_THRESHOLD:
        classification = "known"
    elif strong_hits or density >= ADJACENT_COCITE_THRESHOLD:
        classification = "adjacent-known"
    else:
        classification = "not-found"

    out = {
        "classification": classification,
        "key_phrases": key_phrases,
        "hits": hits,
        "cocite_density": density,
        "queries_issued": queries_issued,
        "note": "",
    }
    _persist(out, run_dir, cid)
    return out


def _persist(report: dict, run_dir: Path, cid: str) -> None:
    artifact_dir = run_dir / "novelty" / cid
    artifact_dir.mkdir(parents=True, exist_ok=True)
    (artifact_dir / "report.json").write_text(json.dumps(report, indent=2))
