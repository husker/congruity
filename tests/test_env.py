"""Tests for the .env loader."""

from __future__ import annotations

import os


def test_load_env_returns_none_for_missing_file(tmp_path, monkeypatch):
    from pipeline import env
    monkeypatch.setattr(env, "ENV_PATH", tmp_path / ".env")
    assert env.load_env() is None


def test_load_env_populates_missing_vars(tmp_path, monkeypatch):
    from pipeline import env
    f = tmp_path / ".env"
    f.write_text("ANTHROPIC_API_KEY=sk-ant-test123\nOTHER_VAR=hello\n")
    monkeypatch.setattr(env, "ENV_PATH", f)
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    monkeypatch.delenv("OTHER_VAR", raising=False)
    loaded = env.load_env()
    assert loaded == f
    assert os.environ.get("ANTHROPIC_API_KEY") == "sk-ant-test123"
    assert os.environ.get("OTHER_VAR") == "hello"


def test_load_env_does_not_override_existing_by_default(tmp_path, monkeypatch):
    from pipeline import env
    f = tmp_path / ".env"
    f.write_text("ANTHROPIC_API_KEY=from-file\n")
    monkeypatch.setattr(env, "ENV_PATH", f)
    monkeypatch.setenv("ANTHROPIC_API_KEY", "from-shell")
    env.load_env()
    assert os.environ.get("ANTHROPIC_API_KEY") == "from-shell"


def test_load_env_with_explicit_override(tmp_path, monkeypatch):
    from pipeline import env
    f = tmp_path / ".env"
    f.write_text("ANTHROPIC_API_KEY=from-file\n")
    monkeypatch.setattr(env, "ENV_PATH", f)
    monkeypatch.setenv("ANTHROPIC_API_KEY", "from-shell")
    env.load_env(override=True)
    assert os.environ.get("ANTHROPIC_API_KEY") == "from-file"


def test_load_env_explicit_path(tmp_path, monkeypatch):
    from pipeline import env
    f = tmp_path / "custom.env"
    f.write_text("MY_KEY=custom-value\n")
    monkeypatch.delenv("MY_KEY", raising=False)
    loaded = env.load_env(path=f)
    assert loaded == f
    assert os.environ.get("MY_KEY") == "custom-value"
