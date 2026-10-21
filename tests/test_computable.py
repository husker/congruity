"""Tests for computable problem pack loading, matching, and preflight."""

from __future__ import annotations

import json


def test_list_packs_includes_seed_packs():
    from pipeline.computable import list_packs

    names = list_packs()

    assert names == [
        "cache_replacement",
        "graph_coloring",
        "short_string_compression",
        "small_points_optimization",
        "snake_in_box",
    ]


def test_load_short_string_pack_has_executable_contract():
    from pipeline.computable import load_pack

    pack = load_pack("short_string_compression")

    assert pack.pack.id == "short_string_compression"
    assert pack.pack.problem_shape.verification_kind == "roundtrip"
    assert pack.evaluator.metric_primary == "compression_ratio"
    assert pack.evaluator.higher_is_better is False
    assert any(b.level == "strong_reproducible" for b in pack.baselines.baselines)
    assert any(d.id == "real_http_headers" for d in pack.datasets.datasets)
    assert pack.gotchas


def test_pack_to_dict_is_json_serializable():
    from pipeline.computable import load_pack, pack_to_dict

    pack = load_pack("cache_replacement")
    data = pack_to_dict(pack)

    assert data["pack"]["id"] == "cache_replacement"
    assert data["evaluator"]["metric_primary"] == "hit_rate"
    assert data["gotchas"]
    json.dumps(data)


def test_match_problem_ranks_short_string_first():
    from pipeline.computable import match_problem

    matches = match_problem("compress short HTTP headers better than HPACK")

    assert matches[0]["pack_id"] == "short_string_compression"
    assert matches[0]["score"] > 0


def test_match_problem_ranks_cache_first():
    from pipeline.computable import match_problem

    matches = match_problem("design a cache eviction policy for scan workloads")

    assert matches[0]["pack_id"] == "cache_replacement"
    assert matches[0]["score"] > 0


def test_preflight_seed_packs_are_ai_verifiable():
    from pipeline.computable import preflight_pack

    for name in [
        "short_string_compression",
        "small_points_optimization",
        "cache_replacement",
        "graph_coloring",
    ]:
        result = preflight_pack(name)
        assert result.computable is True
        assert result.ai_verifiable is True
        assert result.missing == []
        assert "strong_reproducible" in result.bar_levels_available
        assert result.smoke_command
        assert result.recommended_compose_constraints


def test_records_load_and_compare():
    from pipeline.computable import compare_to_record, load_pack, preflight_pack

    pack = load_pack("small_points_optimization")
    assert pack.records is not None
    assert preflight_pack("small_points_optimization").record_ready is True
    # graph_coloring has no records file
    assert preflight_pack("graph_coloring").record_ready is False

    # beat the n=11 best-known (1/27) -> new record
    cmp = compare_to_record(pack, {"n": 11}, 0.0380)
    assert cmp["verdict"] == "new-record"
    assert cmp["record_id"] == "heilbronn_square_n11"

    # below the record
    assert compare_to_record(pack, {"n": 11}, 0.0360)["verdict"] == "below-record"
    # within tolerance of the record
    assert compare_to_record(pack, {"n": 11}, 1 / 27)["verdict"] == "matches-record"
    # "beating" a proven optimum flags an objective mismatch
    assert compare_to_record(pack, {"n": 6}, 0.13)["verdict"] == \
        "exceeds-proven-optimal-CHECK-OBJECTIVE"
    # unknown params
    assert compare_to_record(pack, {"n": 99}, 0.5)["verdict"] == "no-record"


def test_apply_record_claims_upgrades_and_quarantines():
    from datetime import datetime, timezone

    from pipeline.autoverify import apply_record_claims, build_outcome
    from pipeline.computable import load_pack

    pack = load_pack("small_points_optimization")
    comp = {"id": "C1", "synthesis_claim": "x", "papers": []}
    base = {"metric": "objective_value", "higher_is_better": True,
            "scores": {"candidate": 0.038, "sota": 0.036},
            "composition_method": "candidate", "home_field_sota_method": "sota",
            "n_trials": 3, "notes": ""}
    now = datetime(2026, 6, 11, tzinfo=timezone.utc)

    # beats the 1972 n=11 best-known -> new-record
    r = dict(base, record_claims=[{"params": {"n": 11}, "value": 0.0380}])
    out = build_outcome("r", "C1", comp, {}, r, now)
    apply_record_claims(out, r, pack)
    assert out.grade == "new-record" and out.beats_home_field is True
    assert "new-record" in out.notes

    # "beats" the proven n=6 optimum -> invalid + quarantined
    r = dict(base, record_claims=[{"params": {"n": 6}, "value": 0.13}])
    out = build_outcome("r", "C1", comp, {}, r, now)
    apply_record_claims(out, r, pack)
    assert out.valid is False and out.grade == "invalid"
    assert out.confidence == "quarantined"


def test_pack_context_renders_failure_modes():
    from pipeline.compose import _pack_context

    ctx = _pack_context("graph_coloring")
    assert "KNOWN FAILURE MODES" in ctx
    assert "TabuCol" in ctx          # failure modes mention the SOTA
    assert "gotchas" in ctx.lower()


def test_composition_schema_accepts_mechanism_map():
    from pipeline.schemas import Composition

    c = Composition.model_validate({
        "id": "C1", "synthesis_claim": "x",
        "papers": [],
        "mechanism_map": {
            "transferring_structure": "cluster moves on Potts configurations",
            "sota_failure_mode_addressed": "tuned on one family, regresses on queen",
        },
    })
    assert c.mechanism_map is not None
    assert "Potts" in c.mechanism_map.transferring_structure
    # and still parses without one (older runs)
    c2 = Composition.model_validate({"id": "C2", "synthesis_claim": "y", "papers": []})
    assert c2.mechanism_map is None


def test_preflight_candidate_ready():
    from pipeline.computable import preflight_pack

    # graph_coloring declares the candidate interface for pack-bench autoverify
    pre = preflight_pack("graph_coloring")
    assert pre.candidate_ready is True
    # and the train/holdout commands for autosearch
    assert pre.search_ready is True
    # the older packs don't (yet)
    old = preflight_pack("cache_replacement")
    assert old.candidate_ready is False
    assert old.search_ready is False


def test_preflight_reports_missing_files(tmp_path):
    from pipeline.computable import preflight_pack

    d = tmp_path / "broken_pack"
    d.mkdir()
    (d / "pack.json").write_text(json.dumps({
        "id": "broken_pack",
        "display_name": "Broken Pack",
        "description": "Missing executable files",
        "problem_shape": {
            "objective": "x",
            "input_representation": "x",
            "output_representation": "x",
            "constraints": ["x"],
            "failure_modes": ["x"],
            "verification_kind": "benchmark",
        },
        "recommended_problem_template": "x",
        "compose_query_hints": ["x"],
    }))

    result = preflight_pack("broken_pack", root=tmp_path)

    assert result.computable is False
    assert result.ai_verifiable is False
    assert "evaluator.json" in result.missing
    assert "baselines.json" in result.missing
    assert "datasets.json" in result.missing
    assert "sota.json" in result.missing


def test_main_list_subcommand_prints_seed_pack(capsys):
    from pipeline.computable import main

    rc = main(["list"])
    captured = capsys.readouterr()

    assert rc == 0
    assert "short_string_compression" in captured.out
    assert "cache_replacement" in captured.out


def test_main_inspect_subcommand_emits_json(capsys):
    from pipeline.computable import main

    rc = main(["inspect", "short_string_compression"])
    captured = capsys.readouterr()

    assert rc == 0
    parsed = json.loads(captured.out)
    assert parsed["pack"]["id"] == "short_string_compression"
    assert parsed["evaluator"]["validator"] == "lossless_roundtrip"


def test_main_match_subcommand_emits_ranked_json(capsys):
    from pipeline.computable import main

    rc = main(["match", "--problem", "maximize points in a square by physics relaxation"])
    captured = capsys.readouterr()

    assert rc == 0
    parsed = json.loads(captured.out)
    assert parsed[0]["pack_id"] == "small_points_optimization"


def test_main_preflight_subcommand_emits_json(capsys):
    from pipeline.computable import main

    rc = main(["preflight", "cache_replacement"])
    captured = capsys.readouterr()

    assert rc == 0
    parsed = json.loads(captured.out)
    assert parsed["pack_id"] == "cache_replacement"
    assert parsed["ai_verifiable"] is True
