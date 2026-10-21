"""Shared test fixtures.

The default ``resolve_mode("auto")`` in ``pipeline.relevance`` will pick the
``codex`` backend if the ``codex`` binary is on PATH. That's correct for
end-users but breaks tests that assert "no auth → relevance skipped"
semantics — the suite would try to invoke real codex and either hang or
hit the network. This fixture hides the codex binary from the path
resolver by default; individual tests that want to exercise the codex
backend re-install shutil.which via their own monkeypatch.
"""

from __future__ import annotations

import shutil

import pytest


@pytest.fixture(autouse=True)
def _hide_codex_from_resolve(monkeypatch):
    real_which = shutil.which

    def fake_which(name):
        if name == "codex":
            return None
        return real_which(name)

    monkeypatch.setattr(shutil, "which", fake_which)
    yield
