"""Tests for the domain-pack loader + apply CLI."""

from __future__ import annotations


def test_list_packs_includes_cs_algorithms():
    from pipeline.domain_pack import list_packs
    names = list_packs()
    assert "cs_algorithms" in names


def test_load_cs_algorithms_pack_has_required_keys():
    from pipeline.domain_pack import load_pack
    pack = load_pack("cs_algorithms")
    assert pack["topic_queries"] is not None
    assert isinstance(pack["topic_queries"], list)
    assert isinstance(pack["known_baselines"], dict)
    assert "embedding_compression" in pack["known_baselines"]
    assert any("PQ" in b for b in pack["known_baselines"]["embedding_compression"])


def test_load_pack_raises_for_unknown_name():
    import pytest
    from pipeline.domain_pack import load_pack
    with pytest.raises(FileNotFoundError):
        load_pack("nonexistent_pack")


def test_apply_dry_run_returns_structured_summary():
    from pipeline.domain_pack import apply_pack
    out = apply_pack(
        "cs_algorithms",
        problem="compress sentence embeddings to under 64 bits",
        dry_run=True,
    )
    assert out["domain"] == "cs_algorithms"
    assert out["dry_run"] is True
    assert "embedding_compression" in out["known_baselines_hint"]
    assert "embedding_compression" in out["evaluation_datasets_hint"]
    assert isinstance(out["topic_queries"], list)
    assert out["gotchas_preview"]


def test_main_list_subcommand_prints_pack(capsys):
    from pipeline.domain_pack import main
    rc = main(["list"])
    captured = capsys.readouterr()
    assert rc == 0
    assert "cs_algorithms" in captured.out


def test_main_apply_subcommand_emits_json(capsys):
    import json as _json
    from pipeline.domain_pack import main
    rc = main(["apply", "cs_algorithms", "--problem", "x"])
    captured = capsys.readouterr()
    assert rc == 0
    parsed = _json.loads(captured.out)
    assert parsed["domain"] == "cs_algorithms"
