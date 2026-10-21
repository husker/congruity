"""Tests for the LLM I/O wrapper layer."""

from __future__ import annotations

import json


from pipeline.llm_io import ParseResult, parse_or_repair
from pipeline.schemas import DimensionsOutput


def _good_dim_json() -> str:
    return json.dumps({
        "dimensions": [
            {"name": "x", "description": "y", "example_question": "z"},
        ]
    })


def test_parse_or_repair_happy_path():
    raw = "```json\n" + _good_dim_json() + "\n```"
    result = parse_or_repair(raw, DimensionsOutput, repair_call=None)
    assert isinstance(result, ParseResult)
    assert result.status == "ok"
    assert result.value.dimensions[0].name == "x"
    assert result.errors == []


def test_parse_or_repair_retries_once_on_invalid_json():
    calls: list[str] = []

    def fake_repair(repair_prompt: str) -> tuple[bool, str, str]:
        calls.append(repair_prompt)
        return (True, _good_dim_json(), "")

    bad_raw = "{ not json at all"
    result = parse_or_repair(bad_raw, DimensionsOutput, repair_call=fake_repair)
    assert result.status == "repaired"
    assert result.value.dimensions[0].name == "x"
    assert len(calls) == 1


def test_parse_or_repair_gives_up_after_one_failed_repair():
    def always_bad(_repair_prompt: str) -> tuple[bool, str, str]:
        return (True, "still not json", "")

    result = parse_or_repair("{ broken", DimensionsOutput, repair_call=always_bad)
    assert result.status == "failed"
    assert result.value is None
    assert len(result.errors) >= 1


def test_parse_or_repair_returns_failed_when_no_repair_caller_and_bad_input():
    result = parse_or_repair("not json", DimensionsOutput, repair_call=None)
    assert result.status == "failed"
    assert result.value is None


def test_run_codex_with_artifacts_persists_full_call(tmp_path, monkeypatch):
    from pipeline import llm_io

    # Stub the underlying codex invocation so the test never spawns a subprocess.
    def fake_invoker(prompt: str, reasoning: str) -> tuple[bool, str, str]:
        return True, json.dumps({"hello": "world"}), ""

    monkeypatch.setattr(llm_io, "_invoke_codex", fake_invoker)

    ok, stdout, stderr, artifact_dir = llm_io.run_codex_with_artifacts(
        stage="dimensions",
        prompt="some prompt text",
        run_dir=tmp_path,
        reasoning="high",
    )

    assert ok
    assert "hello" in stdout
    assert artifact_dir.is_dir()
    assert (artifact_dir / "prompt.txt").read_text() == "some prompt text"
    assert "hello" in (artifact_dir / "stdout.txt").read_text()
    meta = json.loads((artifact_dir / "meta.json").read_text())
    assert meta["stage"] == "dimensions"
    assert meta["reasoning"] == "high"
    assert meta["exit_code"] == 0
    assert "duration_s" in meta


def test_parse_or_repair_stamps_meta_on_successful_repair(tmp_path):
    """When repair_call returns the 4-tuple (ok, stdout, stderr, artifact_dir),
    parse_or_repair updates the repair artifact's meta.json so the audit
    trail records the repair outcome instead of leaving it at 'pending'."""

    # Seed an artifact dir as if run_codex_with_artifacts had just written it.
    repair_dir = tmp_path / "01"
    repair_dir.mkdir()
    (repair_dir / "meta.json").write_text(json.dumps({
        "stage": "dimensions_repair", "exit_code": 0,
        "parse_status": "pending",
    }))

    def fake_repair(_prompt: str) -> tuple:
        return (True, _good_dim_json(), "", repair_dir)

    result = parse_or_repair("{ broken", DimensionsOutput, repair_call=fake_repair)
    assert result.status == "repaired"
    meta = json.loads((repair_dir / "meta.json").read_text())
    assert meta["parse_status"] == "repair_ok"


def test_parse_or_repair_stamps_meta_on_failed_repair(tmp_path):

    repair_dir = tmp_path / "01"
    repair_dir.mkdir()
    (repair_dir / "meta.json").write_text(json.dumps({
        "stage": "dimensions_repair", "exit_code": 0,
        "parse_status": "pending",
    }))

    def still_bad(_prompt: str) -> tuple:
        return (True, "still not json", "", repair_dir)

    result = parse_or_repair("{ broken", DimensionsOutput, repair_call=still_bad)
    assert result.status == "failed"
    meta = json.loads((repair_dir / "meta.json").read_text())
    assert meta["parse_status"] == "repair_failed"
    assert "validation_errors" in meta


def test_parse_or_repair_back_compat_3_tuple_repair_call():
    """The old (ok, stdout, stderr) repair-call shape must keep working
    so callers can be migrated incrementally without breaking tests."""
    def fake_repair(_prompt: str) -> tuple[bool, str, str]:
        return (True, _good_dim_json(), "")

    result = parse_or_repair("{ broken", DimensionsOutput, repair_call=fake_repair)
    assert result.status == "repaired"


def test_run_codex_with_artifacts_sequences_multiple_calls(tmp_path, monkeypatch):
    from pipeline import llm_io

    monkeypatch.setattr(llm_io, "_invoke_codex",
                        lambda prompt, reasoning: (True, "{}", ""))

    dirs = []
    for _ in range(3):
        _, _, _, d = llm_io.run_codex_with_artifacts(
            stage="dimensions",
            prompt="x",
            run_dir=tmp_path,
            reasoning="high",
        )
        dirs.append(d.name)

    # Three calls under the same stage get sequential zero-padded names
    assert dirs == ["01", "02", "03"]
