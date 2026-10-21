"""Tests for the post-L3 novelty check stage."""

from __future__ import annotations

import json

import pytest


def test_novelty_report_classification_literal():
    from pydantic import ValidationError
    from pipeline.schemas import NoveltyReport
    with pytest.raises(ValidationError):
        NoveltyReport(classification="dunno")


def test_novelty_report_accepts_all_documented_classifications():
    from pipeline.schemas import NoveltyReport
    for c in ["known", "adjacent-known", "not-found", "unverified"]:
        NoveltyReport(classification=c)


def test_classify_known_with_stubbed_hits(monkeypatch, tmp_path):
    from pipeline import novelty

    def fake_s2(query):
        return [
            {"title": "PCA-banded PQ outperforms PQ", "url": "u1", "similarity": 0.95},
            {"title": "Almost the same paper", "url": "u2", "similarity": 0.91},
            {"title": "Variant 3", "url": "u3", "similarity": 0.88},
        ]

    monkeypatch.setattr(novelty, "_s2_title_search", fake_s2)
    monkeypatch.setattr(novelty, "_openalex_title_search", lambda q: [])
    monkeypatch.setattr(novelty, "_cocitation_density", lambda papers: 0.5)

    composition = {"id": "C1", "synthesis_claim": "PCA-banded PQ embedding quantization"}
    state = {"problem": "compress sentence embeddings to under 64 bits"}
    out = novelty.check_novelty(composition, state, run_dir=tmp_path,
                                with_search=True)
    assert out["classification"] == "known"
    assert len(out["hits"]) >= 3
    # Artifact persisted
    assert (tmp_path / "novelty" / "C1" / "report.json").exists()


def test_classify_adjacent_known_via_cocite_density(monkeypatch, tmp_path):
    from pipeline import novelty
    monkeypatch.setattr(novelty, "_s2_title_search", lambda q: [])
    monkeypatch.setattr(novelty, "_openalex_title_search", lambda q: [])
    monkeypatch.setattr(novelty, "_cocitation_density", lambda papers: 0.4)
    composition = {"id": "C1", "synthesis_claim": "x"}
    state = {"problem": "x"}
    out = novelty.check_novelty(composition, state, run_dir=tmp_path,
                                with_search=True)
    assert out["classification"] == "adjacent-known"


def test_classify_not_found_with_no_hits_and_low_density(monkeypatch, tmp_path):
    from pipeline import novelty
    monkeypatch.setattr(novelty, "_s2_title_search", lambda q: [])
    monkeypatch.setattr(novelty, "_openalex_title_search", lambda q: [])
    monkeypatch.setattr(novelty, "_cocitation_density", lambda papers: 0.0)
    composition = {"id": "C1", "synthesis_claim": "x"}
    state = {"problem": "x"}
    out = novelty.check_novelty(composition, state, run_dir=tmp_path,
                                with_search=True)
    assert out["classification"] == "not-found"


def test_classify_unverified_when_search_disabled(tmp_path):
    from pipeline import novelty
    composition = {"id": "C1", "synthesis_claim": "x"}
    state = {"problem": "x"}
    out = novelty.check_novelty(composition, state, run_dir=tmp_path,
                                with_search=False)
    assert out["classification"] == "unverified"
    assert "search disabled" in out["note"]
    # Artifact still persisted so audit can see the skip.
    saved = json.loads((tmp_path / "novelty" / "C1" / "report.json").read_text())
    assert saved["classification"] == "unverified"


# ---------------------------------------------------------------------------
# Production search primitives (implemented 2026-07-01; previously stubs)
# ---------------------------------------------------------------------------


def test_title_similarity_containment():
    from pipeline.novelty import _title_similarity
    assert _title_similarity("hawkes spillover", "A Hawkes model of spillover effects") == 1.0
    assert _title_similarity("hawkes spillover", "A Hawkes process primer") == 0.5
    assert _title_similarity("hawkes spillover", "Unrelated title") == 0.0
    assert _title_similarity("", "anything") == 0.0


def test_s2_title_search_parses_payload(monkeypatch):
    import pipeline.novelty as novelty

    class _Resp:
        status_code = 200

        @staticmethod
        def json():
            return {"data": [{"title": "Hawkes spillover networks", "year": 2024}]}

    monkeypatch.setattr(novelty.time, "sleep", lambda s: None)
    monkeypatch.setattr(novelty.requests, "get", lambda *a, **k: _Resp())
    hits = novelty._s2_title_search("hawkes spillover")
    assert len(hits) == 1
    assert hits[0]["similarity"] == 1.0
    assert hits[0]["year"] == 2024


def test_s2_title_search_gives_up_after_429(monkeypatch):
    import pipeline.novelty as novelty

    class _Resp:
        status_code = 429

    monkeypatch.setattr(novelty.time, "sleep", lambda s: None)
    monkeypatch.setattr(novelty.requests, "get", lambda *a, **k: _Resp())
    assert novelty._s2_title_search("q") == []


def test_openalex_title_search_maps_results(monkeypatch):
    import pipeline._openalex as oa
    import pipeline.novelty as novelty

    monkeypatch.setattr(novelty.time, "sleep", lambda s: None)
    monkeypatch.setattr(oa, "request_with_retry", lambda path, params: {
        "results": [{"title": "Hawkes spillover atlas",
                     "publication_year": 2025,
                     "id": "https://openalex.org/W1"}]})
    hits = novelty._openalex_title_search("hawkes spillover")
    assert hits == [{"title": "Hawkes spillover atlas", "year": 2025,
                     "openalex_id": "https://openalex.org/W1",
                     "similarity": 1.0}]
