"""Stable paper identifiers.

paper_uid(paper) returns a 16-hex-char SHA-256 prefix derived from the
paper's most stable identifier. Used to cross-reference saved corpus
entries across re-indexing (P1..PN may change between runs; paper_uid
does not).
"""

from __future__ import annotations

import hashlib


def paper_uid(paper: dict) -> str:
    """Return a 16-hex-char stable identifier for a paper.

    Preference order:
        1. openalex_id (most authoritative when present)
        2. doi
        3. "title:" + title + ":" + year
    """
    key: str
    oid = paper.get("openalex_id")
    if oid:
        key = str(oid)
    elif paper.get("doi"):
        key = "doi:" + str(paper["doi"])
    else:
        title = str(paper.get("title", ""))
        year = str(paper.get("year", ""))
        key = f"title:{title}:{year}"
    return hashlib.sha256(key.encode("utf-8")).hexdigest()[:16]
