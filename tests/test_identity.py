"""Tests for stable paper identifier generation."""

from __future__ import annotations

from pipeline.identity import paper_uid


def test_paper_uid_is_stable_for_same_openalex_id():
    a = {"openalex_id": "https://openalex.org/W123", "title": "X", "year": 2024}
    b = {"openalex_id": "https://openalex.org/W123", "title": "DIFFERENT", "year": 1999}
    # Different metadata but same openalex_id -> same uid (openalex wins)
    assert paper_uid(a) == paper_uid(b)


def test_paper_uid_differs_for_different_openalex_id():
    a = {"openalex_id": "https://openalex.org/W123", "title": "X", "year": 2024}
    b = {"openalex_id": "https://openalex.org/W456", "title": "X", "year": 2024}
    assert paper_uid(a) != paper_uid(b)


def test_paper_uid_falls_back_to_title_year_when_no_openalex():
    a = {"openalex_id": None, "title": "An attention mechanism", "year": 2023}
    b = {"title": "An attention mechanism", "year": 2023}  # missing key OK
    assert paper_uid(a) == paper_uid(b)


def test_paper_uid_distinguishes_title_year_pairs():
    a = {"title": "A", "year": 2023}
    b = {"title": "A", "year": 2024}
    assert paper_uid(a) != paper_uid(b)


def test_paper_uid_is_16_hex_chars():
    uid = paper_uid({"openalex_id": "https://openalex.org/W1"})
    assert len(uid) == 16
    assert all(c in "0123456789abcdef" for c in uid)
