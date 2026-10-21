"""Layer 1 — fetch and cache OpenAlex topics.

OpenAlex has ~4400 auto-clustered Topics organized under Domain > Field > Subfield > Topic.
We support three modes for picking a topic pool:

1. **By field**: fetch all topics in a named OpenAlex field (e.g. "Computer Science",
   "Environmental Engineering").
2. **By keyword search**: fetch topics that match a set of keywords (e.g. "carbon
   capture", "CO2 sequestration"). Useful when the area of interest spans multiple
   OpenAlex fields.
3. **From a saved topic list**: load a hand-curated list of OpenAlex topic IDs from
   a JSON file at `data/pipeline/topic_lists/{name}.json`.

CLI:
    uv run python -m pipeline.topics fetch --field "Computer Science"
    uv run python -m pipeline.topics fetch --field "Environmental Engineering" --cache topics_env.json
    uv run python -m pipeline.topics search --keywords "carbon capture,sorbent,direct air capture" \\
        --save data/pipeline/topic_lists/carbon_capture.json
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

from pipeline._openalex import request_with_retry, DEFAULT_SLEEP

DATA_ROOT = Path(__file__).resolve().parent.parent / "data" / "pipeline"
CACHE_PATH = DATA_ROOT / "topics_cs.json"
TOPIC_LISTS = DATA_ROOT / "topic_lists"


def find_field_id(name: str = "Computer Science") -> str | None:
    """Return the OpenAlex field ID for the named field, or None."""
    body = request_with_retry("/fields", {"search": name, "per-page": 10})
    if not body:
        return None
    for f in body.get("results", []):
        if f.get("display_name", "").lower() == name.lower():
            return f["id"]
    # Fallback: first result
    results = body.get("results", [])
    return results[0]["id"] if results else None


def fetch_field_topics(field_id: str, sleep_s: float = DEFAULT_SLEEP) -> list[dict]:
    """Cursor-paginate /topics filtered to a field, return slim metadata."""
    short = field_id.rstrip("/").split("/")[-1]
    topics: list[dict] = []
    cursor = "*"
    while True:
        body = request_with_retry("/topics", {
            "filter": f"field.id:{short}",
            "per-page": 200,
            "cursor": cursor,
            "select": "id,display_name,description,works_count,cited_by_count,subfield",
        })
        if body is None:
            break
        results = body.get("results", [])
        if not results:
            break
        topics.extend(results)
        cursor = body.get("meta", {}).get("next_cursor")
        if not cursor:
            break
        time.sleep(sleep_s)
    return topics


def load_or_fetch(force: bool = False, cache_path: Path = CACHE_PATH,
                  field: str = "Computer Science") -> list[dict]:
    """Return cached topics for a field, or fetch + cache."""
    if cache_path.exists() and not force:
        return json.loads(cache_path.read_text())
    field_id = find_field_id(field)
    if not field_id:
        print(f"ERROR: could not find field '{field}' in OpenAlex", file=sys.stderr)
        return []
    topics = fetch_field_topics(field_id)
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    cache_path.write_text(json.dumps(topics, indent=2))
    return topics


def search_topics(keyword: str, cap: int = 20) -> list[dict]:
    """Search OpenAlex /topics for a keyword. Returns slim metadata.

    Uses the search endpoint over topic display_names + descriptions.
    """
    body = request_with_retry("/topics", {
        "search": keyword,
        "per-page": cap,
        "select": "id,display_name,description,works_count,cited_by_count,subfield,field",
    })
    if body is None:
        return []
    return body.get("results", [])


def harvest_topics_by_keywords(keywords: list[str],
                               per_keyword_cap: int = 20,
                               min_works: int = 100) -> list[dict]:
    """For each keyword, search OpenAlex topics. Union and dedupe by ID.
    Filter to topics with ≥min_works works (excludes obscure topics)."""
    seen: dict[str, dict] = {}
    for kw in keywords:
        print(f"  searching '{kw}'...", end=" ", flush=True)
        results = search_topics(kw, cap=per_keyword_cap)
        added = 0
        for t in results:
            tid = t.get("id")
            if not tid or tid in seen:
                continue
            if t.get("works_count", 0) < min_works:
                continue
            seen[tid] = t
            added += 1
        print(f"+{added} (cumulative {len(seen)})")
        time.sleep(DEFAULT_SLEEP)
    return list(seen.values())


def harvest_topics_from_papers(queries: list[str],
                               papers_per_query: int = 50,
                               year_min: int | None = None,
                               min_works: int = 100) -> list[dict]:
    """For each query, search OpenAlex /works, extract primary_topic IDs from
    the returned papers, then re-fetch full topic metadata. This finds topics
    that LITERATURE actually associates with the query — much richer than
    searching the topic database directly. Cross-disciplinary by construction.
    """
    from pipeline._openalex import paginate_works

    topic_ids: dict[str, dict] = {}  # short_id -> partial metadata from paper
    for q in queries:
        print(f"  query '{q}': fetching papers...", end=" ", flush=True)
        filter_parts = []
        if year_min is not None:
            filter_parts.append(f"publication_year:>{year_min - 1}")
        params = {"search": q}
        if filter_parts:
            params["filter"] = ",".join(filter_parts)
        works = paginate_works(
            params=params,
            cap=papers_per_query,
            per_page=50,
            select="id,primary_topic",
        )
        new = 0
        for w in works:
            pt = w.get("primary_topic")
            if not pt or not pt.get("id"):
                continue
            short = pt["id"].rstrip("/").split("/")[-1]
            if short not in topic_ids:
                topic_ids[short] = pt
                new += 1
        print(f"+{new} new topic-ids (cumulative {len(topic_ids)})")
        time.sleep(DEFAULT_SLEEP)

    # Re-fetch full topic metadata in batches (so we get works_count etc.)
    print(f"\n  re-fetching full metadata for {len(topic_ids)} topics...")
    full_topics: list[dict] = []
    short_ids = list(topic_ids.keys())
    for i in range(0, len(short_ids), 50):
        chunk = short_ids[i:i + 50]
        filter_arg = "openalex:" + "|".join(chunk)
        body = request_with_retry("/topics", {
            "filter": filter_arg,
            "per-page": 50,
            "select": "id,display_name,description,works_count,cited_by_count,subfield,field",
        })
        if body is None:
            continue
        for t in body.get("results", []):
            if t.get("works_count", 0) >= min_works:
                full_topics.append(t)
        time.sleep(DEFAULT_SLEEP)

    return full_topics


def load_topic_list(path: Path) -> list[dict]:
    """Load a saved topic list (JSON array of topic dicts)."""
    return json.loads(path.read_text())


def save_topic_list(topics: list[dict], path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(topics, indent=2))
    return path


def main() -> int:
    p = argparse.ArgumentParser(description="OpenAlex topic fetcher")
    sub = p.add_subparsers(dest="cmd", required=True)

    pf = sub.add_parser("fetch", help="Fetch all topics in a named field")
    pf.add_argument("--field", default="Computer Science")
    pf.add_argument("--cache", type=Path, default=None,
                    help="Cache path (default: data/pipeline/topics_{field_slug}.json)")
    pf.add_argument("--force", action="store_true")

    ps = sub.add_parser("search", help="Search topics by keyword(s) and save as a topic list")
    ps.add_argument("--keywords", required=True,
                    help="Comma-separated keywords (e.g. 'carbon capture,sorbent,direct air capture')")
    ps.add_argument("--per-keyword-cap", type=int, default=20)
    ps.add_argument("--min-works", type=int, default=100)
    ps.add_argument("--save", type=Path, required=True,
                    help="Output path under data/pipeline/topic_lists/")

    ph = sub.add_parser("harvest",
                        help="Harvest topics from papers matching queries — much richer "
                             "cross-disciplinary coverage than `search`")
    ph.add_argument("--queries", required=True,
                    help="Comma-separated queries; topics are pulled from primary_topic of "
                         "matching papers")
    ph.add_argument("--papers-per-query", type=int, default=50)
    ph.add_argument("--year-min", type=int, default=None,
                    help="Restrict to papers published in/after this year")
    ph.add_argument("--min-works", type=int, default=100)
    ph.add_argument("--save", type=Path, required=True)

    args = p.parse_args()

    if args.cmd == "fetch":
        cache = args.cache
        if cache is None:
            slug = args.field.lower().replace(" ", "_")
            cache = DATA_ROOT / f"topics_{slug}.json"
        topics = load_or_fetch(force=args.force, cache_path=cache, field=args.field)
        print(f"{len(topics)} topics in '{args.field}'. Cache: {cache}")
        if topics:
            print("\nSample topics:")
            for t in sorted(topics, key=lambda x: -x.get("works_count", 0))[:10]:
                wc = t.get("works_count", 0)
                print(f"  [{wc:>8}] {t['display_name']}")
    elif args.cmd == "search":
        keywords = [k.strip() for k in args.keywords.split(",") if k.strip()]
        print(f"Searching {len(keywords)} keyword(s)...")
        topics = harvest_topics_by_keywords(
            keywords,
            per_keyword_cap=args.per_keyword_cap,
            min_works=args.min_works,
        )
        save_topic_list(topics, args.save)
        print(f"\n{len(topics)} unique topics saved to {args.save}")
        if topics:
            print("\nTop 10 by works_count:")
            for t in sorted(topics, key=lambda x: -x.get("works_count", 0))[:10]:
                field = (t.get("field") or {}).get("display_name", "?")
                print(f"  [{t.get('works_count', 0):>8}] {t['display_name']:55} ({field})")
    elif args.cmd == "harvest":
        queries = [q.strip() for q in args.queries.split(",") if q.strip()]
        print(f"Harvesting topics from papers matching {len(queries)} queries...")
        topics = harvest_topics_from_papers(
            queries,
            papers_per_query=args.papers_per_query,
            year_min=args.year_min,
            min_works=args.min_works,
        )
        save_topic_list(topics, args.save)
        print(f"\n{len(topics)} unique topics saved to {args.save}")
        if topics:
            # Show distribution by field
            from collections import Counter
            field_counts = Counter(
                (t.get("field") or {}).get("display_name", "?") for t in topics
            )
            print("\nField distribution:")
            for field, count in field_counts.most_common():
                print(f"  {count:>3} topics in {field}")
            print("\nTop 15 by works_count:")
            for t in sorted(topics, key=lambda x: -x.get("works_count", 0))[:15]:
                field = (t.get("field") or {}).get("display_name", "?")
                print(f"  [{t.get('works_count', 0):>8}] {t['display_name']:55} ({field})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
