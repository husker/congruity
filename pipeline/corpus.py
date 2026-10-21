"""Shared corpus-building utilities for pair_filter.py and discover.py.

Given a topic ID + year range, fetch a sample of papers (with references for
pair filtering, or with abstracts for discovery). Cached per-topic to avoid
redundant API calls across pipeline stages.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

from pipeline._openalex import (
    DEFAULT_SLEEP,
    abstract_from_inverted,
    extract_id,
    paginate_works,
)

CACHE_ROOT = Path(__file__).resolve().parent.parent / "data" / "pipeline" / "topic_samples"


def _paper_min(work: dict, include_refs: bool, include_abstract: bool) -> dict:
    out = {
        "openalex_id": work.get("id"),
        "title": work.get("title"),
        "year": work.get("publication_year"),
    }
    if include_abstract:
        out["abstract"] = abstract_from_inverted(work.get("abstract_inverted_index"))
        out["authors"] = [
            a["author"]["display_name"]
            for a in work.get("authorships", [])
            if a.get("author")
        ][:5]
    if include_refs:
        out["referenced_works"] = work.get("referenced_works") or []
    return out


def sample_topic(
    topic_id: str,
    year_min: int | None = None,
    year_max: int | None = None,
    cap: int = 50,
    include_refs: bool = True,
    include_abstract: bool = False,
    use_cache: bool = True,
) -> list[dict]:
    """Fetch a sample of papers in a topic, with optional year filter.

    Cache key includes parameters that affect content. Cached file lives at
    `data/pipeline/topic_samples/{topic_id}_{year_min}_{year_max}_{cap}_{r}{a}.json`.
    """
    short = extract_id(topic_id)
    parts = [short, str(year_min or "0"), str(year_max or "0"), str(cap),
             "r" if include_refs else "x", "a" if include_abstract else "x"]
    cache_path = CACHE_ROOT / ("_".join(parts) + ".json")
    if use_cache and cache_path.exists():
        return json.loads(cache_path.read_text())

    filter_parts = [f"primary_topic.id:{short}"]
    if year_min is not None:
        filter_parts.append(f"publication_year:>{year_min - 1}")
    if year_max is not None:
        filter_parts.append(f"publication_year:<{year_max + 1}")

    select = "id,title,publication_year"
    if include_abstract:
        select += ",authorships,abstract_inverted_index"
    if include_refs:
        select += ",referenced_works"

    works = paginate_works(
        params={"filter": ",".join(filter_parts)},
        cap=cap,
        per_page=min(50, cap),
        select=select,
        sleep_s=DEFAULT_SLEEP,
    )

    papers = [_paper_min(w, include_refs, include_abstract) for w in works]
    # Filter out missing essentials
    if include_abstract:
        papers = [p for p in papers if p.get("abstract") and len(p["abstract"].split()) >= 30]
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    cache_path.write_text(json.dumps(papers, indent=2))
    return papers


def search_works(
    query: str,
    year_min: int | None = None,
    cap: int = 50,
    include_abstract: bool = True,
    use_cache: bool = True,
) -> list[dict]:
    """Fetch papers matching a full-text search query, relevance-ranked.

    Paper-direct alternative to :func:`sample_topic` for problems narrower
    than any OpenAlex topic (e.g. graph coloring, merge sort), where random
    within-topic sampling yields almost no relevant papers. OpenAlex orders
    ``search=`` results by relevance score, so the top of the result list is
    dense in on-problem papers.

    Cached at ``data/pipeline/topic_samples/q-{slug}_{year_min}_{cap}_{a}.json``.
    """
    slug = re.sub(r"[^a-z0-9]+", "-", query.lower()).strip("-")[:60]
    parts = [f"q-{slug}", str(year_min or "0"), str(cap),
             "a" if include_abstract else "x"]
    cache_path = CACHE_ROOT / ("_".join(parts) + ".json")
    if use_cache and cache_path.exists():
        return json.loads(cache_path.read_text())

    params: dict = {"search": query}
    if year_min is not None:
        params["filter"] = f"publication_year:>{year_min - 1}"

    select = "id,title,publication_year"
    if include_abstract:
        select += ",authorships,abstract_inverted_index"

    works = paginate_works(
        params=params,
        cap=cap,
        per_page=min(50, cap),
        select=select,
        sleep_s=DEFAULT_SLEEP,
    )

    papers = [_paper_min(w, include_refs=False, include_abstract=include_abstract)
              for w in works]
    if include_abstract:
        papers = [p for p in papers if p.get("abstract") and len(p["abstract"].split()) >= 30]
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    cache_path.write_text(json.dumps(papers, indent=2))
    return papers
