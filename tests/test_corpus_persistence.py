"""Tests for corpus_raw.json + corpus_validated.json dual-write."""

from __future__ import annotations

import json
from pathlib import Path


from pipeline.compose import _persist_corpus_artifacts


def _make_paper(i: int, openalex_id: str | None = None) -> dict:
    return {
        "paper_id": f"P{i}",
        "openalex_id": openalex_id or f"https://openalex.org/W{i:04d}",
        "title": f"Paper {i}",
        "year": 2024,
        "abstract": "x" * 200,
        "source_topic": "test",
    }


def test_persist_corpus_artifacts_writes_raw_and_validated(tmp_path: Path):
    raw = [_make_paper(i) for i in range(1, 4)]
    dropped_ids = {"P2"}
    survivors = [p for p in raw if p["paper_id"] not in dropped_ids]
    _persist_corpus_artifacts(tmp_path, raw_corpus=raw, validated_corpus=survivors)

    raw_path = tmp_path / "corpus_raw.json"
    val_path = tmp_path / "corpus_validated.json"
    legacy_path = tmp_path / "corpus.json"

    assert raw_path.exists()
    assert val_path.exists()
    assert legacy_path.exists()

    raw_data = json.loads(raw_path.read_text())
    val_data = json.loads(val_path.read_text())
    legacy_data = json.loads(legacy_path.read_text())

    assert len(raw_data) == 3
    assert len(val_data) == 2
    assert legacy_data == val_data  # legacy mirrors validated

    # paper_uid is present on every paper
    assert all("paper_uid" in p for p in raw_data)
    assert all("paper_uid" in p for p in val_data)
    # paper_uid is stable across raw -> validated mapping (same paper, same uid)
    raw_uids_by_oid = {p["openalex_id"]: p["paper_uid"] for p in raw_data}
    for p in val_data:
        assert p["paper_uid"] == raw_uids_by_oid[p["openalex_id"]]
