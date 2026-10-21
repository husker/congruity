"""Tests for pipeline._openalex — retry/backoff behavior and mailto injection."""

from __future__ import annotations

import pipeline._openalex as oa


class _Resp:
    def __init__(self, status_code, payload=None, headers=None, text=""):
        self.status_code = status_code
        self._payload = payload or {}
        self.headers = headers or {}
        self.text = text

    def json(self):
        return self._payload


def _no_sleep(monkeypatch):
    sleeps: list[float] = []
    monkeypatch.setattr(oa.time, "sleep", sleeps.append)
    return sleeps


def test_retries_429_then_succeeds(monkeypatch):
    sleeps = _no_sleep(monkeypatch)
    responses = [_Resp(429), _Resp(200, {"ok": True})]
    calls = []

    def fake_get(url, params, timeout):
        calls.append(url)
        return responses.pop(0)

    monkeypatch.setattr(oa.requests, "get", fake_get)
    out = oa.request_with_retry("/works", {})
    assert out == {"ok": True}
    assert len(calls) == 2
    assert len(sleeps) == 1


def test_429_honors_retry_after_header(monkeypatch):
    sleeps = _no_sleep(monkeypatch)
    responses = [_Resp(429, headers={"Retry-After": "7"}), _Resp(200, {"ok": 1})]
    monkeypatch.setattr(oa.requests, "get", lambda *a, **k: responses.pop(0))
    out = oa.request_with_retry("/works", {})
    assert out == {"ok": 1}
    assert sleeps == [7.0]


def test_retries_5xx_with_backoff(monkeypatch):
    sleeps = _no_sleep(monkeypatch)
    responses = [_Resp(503), _Resp(502), _Resp(200, {"ok": 2})]
    monkeypatch.setattr(oa.requests, "get", lambda *a, **k: responses.pop(0))
    out = oa.request_with_retry("/works", {})
    assert out == {"ok": 2}
    assert sleeps == [2.0, 4.0]  # exponential backoff


def test_gives_up_after_max_retries(monkeypatch):
    _no_sleep(monkeypatch)
    monkeypatch.setattr(oa.requests, "get", lambda *a, **k: _Resp(429))
    assert oa.request_with_retry("/works", {}) is None


def test_4xx_other_than_429_is_fatal(monkeypatch):
    _no_sleep(monkeypatch)
    calls = []

    def fake_get(url, params, timeout):
        calls.append(url)
        return _Resp(404, text="nope")

    monkeypatch.setattr(oa.requests, "get", fake_get)
    assert oa.request_with_retry("/works", {}) is None
    assert len(calls) == 1  # no retry on 404


def test_mailto_injected_from_env(monkeypatch):
    monkeypatch.setenv("OPENALEX_MAILTO", "polite@example.org")
    seen = {}

    def fake_get(url, params, timeout):
        seen.update(params)
        return _Resp(200, {"ok": 3})

    monkeypatch.setattr(oa.requests, "get", fake_get)
    oa.request_with_retry("/works", {"q": "x"})
    assert seen["mailto"] == "polite@example.org"
    assert seen["q"] == "x"


def test_mailto_omitted_when_env_empty(monkeypatch):
    monkeypatch.delenv("OPENALEX_MAILTO", raising=False)
    seen = {}

    def fake_get(url, params, timeout):
        seen.update(params)
        return _Resp(200, {})

    monkeypatch.setattr(oa.requests, "get", fake_get)
    oa.request_with_retry("/works", {"q": "x"})
    assert "mailto" not in seen


def test_abstract_from_inverted_roundtrip():
    idx = {"world": [1], "hello": [0], "again": [2]}
    assert oa.abstract_from_inverted(idx) == "hello world again"
    assert oa.abstract_from_inverted(None) == ""
    assert oa.abstract_from_inverted({}) == ""


def test_api_key_injected_from_env(monkeypatch):
    monkeypatch.setenv("OPENALEX_API_KEY", "key123")
    seen = {}

    def fake_get(url, params, timeout):
        seen.update(params)
        return _Resp(200, {})

    monkeypatch.setattr(oa.requests, "get", fake_get)
    oa.request_with_retry("/works", {"q": "x"})
    assert seen["api_key"] == "key123"


def test_api_key_omitted_when_env_empty(monkeypatch):
    monkeypatch.delenv("OPENALEX_API_KEY", raising=False)
    seen = {}

    def fake_get(url, params, timeout):
        seen.update(params)
        return _Resp(200, {})

    monkeypatch.setattr(oa.requests, "get", fake_get)
    oa.request_with_retry("/works", {"q": "x"})
    assert "api_key" not in seen
