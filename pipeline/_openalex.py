"""Shared OpenAlex helpers used across pipeline layers.

Single canonical home for HTTP plumbing, retry logic, and abstract reconstruction.
Refactored from the original scripts/step*.py modules.
"""

from __future__ import annotations

import os
import sys
import time

import requests

OPENALEX = "https://api.openalex.org"


def _mailto() -> str:
    """Contact address for the OpenAlex polite pool (faster rate limits).

    Read from OPENALEX_MAILTO at call time so `load_env()` ordering doesn't
    matter. Empty means anonymous pool — requests still work, just slower.
    """
    return os.environ.get("OPENALEX_MAILTO", "")
DEFAULT_TIMEOUT = 60
DEFAULT_SLEEP = 0.2
MAX_RETRIES = 3


def abstract_from_inverted(idx: dict | None) -> str:
    """Reconstruct linear abstract from OpenAlex's inverted index."""
    if not idx:
        return ""
    positions: dict[int, str] = {}
    for word, poses in idx.items():
        for p in poses:
            positions[p] = word
    if not positions:
        return ""
    return " ".join(positions[i] for i in sorted(positions.keys()))


def request_with_retry(
    path: str,
    params: dict,
    timeout: int = DEFAULT_TIMEOUT,
) -> dict | None:
    """GET with exponential backoff on timeout / 5xx. Returns parsed JSON or None."""
    url = f"{OPENALEX}{path}" if path.startswith("/") else path
    params = dict(params)
    if _mailto():
        params["mailto"] = _mailto()
    # Free API key (https://openalex.org/rest-api): anonymous /works?search=
    # gets 503-throttled under load; a key rides through it.
    api_key = os.environ.get("OPENALEX_API_KEY", "")
    if api_key:
        params["api_key"] = api_key
    backoff = 2.0
    for _ in range(MAX_RETRIES):
        try:
            resp = requests.get(url, params=params, timeout=timeout)
            if resp.status_code == 200:
                return resp.json()
            if resp.status_code == 429 or 500 <= resp.status_code < 600:
                # 429: honor Retry-After when present; otherwise back off.
                # Treating 429 as fatal used to silently truncate the corpus.
                wait = backoff
                retry_after = resp.headers.get("Retry-After", "")
                if retry_after.isdigit():
                    wait = max(wait, float(retry_after))
                print(f"    HTTP {resp.status_code} on {path}, retry in {wait:.1f}s",
                      file=sys.stderr)
                time.sleep(wait)
                backoff *= 2
                continue
            print(f"    HTTP {resp.status_code} on {path}: {resp.text[:200]}",
                  file=sys.stderr)
            return None
        except (requests.exceptions.Timeout, requests.exceptions.ConnectionError) as e:
            print(f"    network err ({type(e).__name__}) on {path}, retry in {backoff:.1f}s",
                  file=sys.stderr)
            time.sleep(backoff)
            backoff *= 2
    print(f"    giving up on {path}", file=sys.stderr)
    return None


def paginate_works(
    params: dict,
    cap: int = 200,
    per_page: int = 50,
    select: str = "id,title,publication_year,authorships,abstract_inverted_index",
    sleep_s: float = DEFAULT_SLEEP,
) -> list[dict]:
    """Cursor-paginate /works. Returns a list of work dicts (no abstract reconstruction)."""
    out: list[dict] = []
    cursor = "*"
    while len(out) < cap:
        page = min(per_page, cap - len(out))
        body = request_with_retry("/works", {
            **params,
            "per-page": page,
            "cursor": cursor,
            "select": select,
        })
        if body is None:
            break
        results = body.get("results", [])
        if not results:
            break
        out.extend(results)
        cursor = body.get("meta", {}).get("next_cursor")
        if not cursor:
            break
        time.sleep(sleep_s)
    return out


def extract_id(url_or_id: str) -> str:
    """OpenAlex IDs come as full URLs (https://openalex.org/W...). Return just the W-id."""
    return url_or_id.rstrip("/").split("/")[-1]
