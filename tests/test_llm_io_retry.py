"""Tests for _invoke_codex retry behavior (transient failures only)."""

from __future__ import annotations

import subprocess

import pipeline.llm_io as llm_io


class _Result:
    def __init__(self, returncode, stdout="", stderr=""):
        self.returncode = returncode
        self.stdout = stdout
        self.stderr = stderr


def _capture_sleep(monkeypatch):
    sleeps: list[float] = []
    monkeypatch.setattr(llm_io.time, "sleep", sleeps.append)
    return sleeps


def test_retries_transient_nonzero_exit(monkeypatch):
    sleeps = _capture_sleep(monkeypatch)
    results = [_Result(1, stderr="stream error"), _Result(0, stdout="ok")]
    calls = []

    def fake_run(cmd, **kwargs):
        calls.append(cmd)
        return results.pop(0)

    monkeypatch.setattr(llm_io.subprocess, "run", fake_run)
    ok, out, err = llm_io._invoke_codex("prompt", "high")
    assert ok is True
    assert out == "ok"
    assert len(calls) == 2
    assert len(sleeps) == 1


def test_persistent_failure_returns_last_result(monkeypatch):
    _capture_sleep(monkeypatch)
    monkeypatch.setattr(
        llm_io.subprocess, "run", lambda cmd, **k: _Result(1, stderr="boom")
    )
    ok, out, err = llm_io._invoke_codex("prompt", "high")
    assert ok is False
    assert err == "boom"


def test_timeout_is_not_retried(monkeypatch):
    sleeps = _capture_sleep(monkeypatch)
    calls = []

    def fake_run(cmd, **kwargs):
        calls.append(cmd)
        raise subprocess.TimeoutExpired(cmd="codex", timeout=42)

    monkeypatch.setattr(llm_io.subprocess, "run", fake_run)
    ok, out, err = llm_io._invoke_codex("prompt", "high", timeout_s=42)
    assert ok is False
    assert "timeout" in err
    assert len(calls) == 1  # budget-controlled by callers; never doubled
    assert sleeps == []


def test_missing_binary_is_not_retried(monkeypatch):
    sleeps = _capture_sleep(monkeypatch)
    calls = []

    def fake_run(cmd, **kwargs):
        calls.append(cmd)
        raise FileNotFoundError("codex")

    monkeypatch.setattr(llm_io.subprocess, "run", fake_run)
    ok, out, err = llm_io._invoke_codex("prompt", "high")
    assert ok is False
    assert "not found" in err
    assert len(calls) == 1
    assert sleeps == []
