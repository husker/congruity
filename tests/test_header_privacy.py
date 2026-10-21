"""Publication hygiene for the optional HTTP-header benchmark corpus."""
from types import SimpleNamespace

import requests

from experiments.c1_short_string import fetch_real_headers as fetch


def test_capture_excludes_cookies_and_custom_session_headers(monkeypatch):
    response = SimpleNamespace(
        raw=SimpleNamespace(headers={
            "Content-Type": "text/html",
            "cAcHe-CoNtRoL": "max-age=60",
            "Set-Cookie": "session=private-fixture",
            "Authorization": "Bearer private-fixture",
            "X-Vendor-Session-ID": "private-fixture",
            "Location": "https://example.org/?code=private-fixture",
        }),
        url="https://example.org/result?token=private-fixture#secret",
        status_code=200,
    )
    monkeypatch.setattr(fetch.requests, "get", lambda *a, **kw: response)
    result = fetch.fetch_one("https://user:private-fixture@example.org/?key=private-fixture")
    assert result["headers"] == [("Content-Type", "text/html"), ("cAcHe-CoNtRoL", "max-age=60")]
    assert result["url"] == "https://example.org/"
    assert result["final_url"] == "https://example.org/result"
    assert "private-fixture" not in str(result)


def test_request_error_does_not_persist_exception_text(monkeypatch):
    def fail(*args, **kwargs):
        raise requests.ConnectionError("proxy URL contains private-fixture")

    monkeypatch.setattr(fetch.requests, "get", fail)
    assert fetch.fetch_one("https://example.org/?token=private-fixture") == {
        "url": "https://example.org/", "error": "ConnectionError",
    }
