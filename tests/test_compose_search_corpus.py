"""Tests for the paper-direct (--paper-search) corpus mode."""

from __future__ import annotations

import pytest


def _fake_papers(query: str, n: int) -> list[dict]:
    return [
        {
            "openalex_id": f"https://openalex.org/W{abs(hash((query, i))) % 10**8}",
            "title": f"{query} paper {i}",
            "year": 2025,
            "abstract": "word " * 40,
            "authors": ["A"],
        }
        for i in range(n)
    ]


def test_assemble_corpus_from_search_tags_and_dedupes(monkeypatch):
    import pipeline.compose as compose

    calls: list[str] = []

    def fake_search(query, year_min=None, cap=50, include_abstract=True):
        calls.append(query)
        papers = _fake_papers("shared", 2) + _fake_papers(query, 3)
        return papers[:cap]

    monkeypatch.setattr(compose, "search_works", fake_search)
    corpus = compose.assemble_corpus_from_search(
        ["graph coloring", "tabu search"], papers_per_query=10, max_papers=50
    )

    assert calls == ["graph coloring", "tabu search"]
    # 2 shared papers must appear once; 3 unique per query.
    assert len(corpus) == 2 + 3 + 3
    ids = [p["openalex_id"] for p in corpus]
    assert len(ids) == len(set(ids))
    assert corpus[0]["source_topic"] == "search: graph coloring"


def test_assemble_corpus_from_search_respects_max_papers(monkeypatch):
    import pipeline.compose as compose

    monkeypatch.setattr(
        compose, "search_works",
        lambda query, **kw: _fake_papers(query, 10),
    )
    corpus = compose.assemble_corpus_from_search(
        ["a", "b", "c"], papers_per_query=10, max_papers=12
    )
    assert len(corpus) == 12


def test_compose_one_requires_exactly_one_corpus_source():
    from pipeline.compose import compose_one

    with pytest.raises(ValueError, match="exactly one"):
        compose_one(topic_list_path=None, problem="p", search_queries=None)
