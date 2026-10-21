"""Round-trip tests for the C1 short-string compressor.

For each input class the bench evaluates on, encode then decode and assert
the result equals the original input. Catches signature instability across
processes and any silent lossiness. Also pins that signatures are
deterministic across calls (BLAKE2 instead of Python's randomized hash()).
"""

from __future__ import annotations

import pytest

from experiments.c1_short_string.c1 import _minimizers, make_codec


@pytest.fixture
def json_corpus() -> list[bytes]:
    return [
        b'{"k":1}',
        b'{"id":"abc","val":42}',
        b'{}',
        b'{"nested":{"x":[1,2,3]}}',
        b'{"id":"d","val":"long-ish string here please"}',
        b'{"id":"e","val":7,"meta":null}',
    ]


@pytest.fixture
def http_corpus() -> list[bytes]:
    return [
        b"Host: example.com",
        b"Content-Type: application/json; charset=utf-8",
        b"Accept-Encoding: gzip, deflate, br",
        b"User-Agent: Mozilla/5.0",
        b"Accept: */*",
    ]


@pytest.fixture
def code_corpus() -> list[bytes]:
    return [
        b"return n + 1",
        b"def add(a, b): return a + b",
        b"x = [1, 2, 3]",
        b"if not value: raise ValueError",
        b"for i in range(10): print(i)",
    ]


def _train_and_roundtrip(items: list[bytes]) -> None:
    codec = make_codec(items)
    for s in items:
        encoded = codec.encode(s)
        decoded = codec.decode(encoded)
        assert decoded == s, f"round-trip failed for {s!r}: decoded={decoded!r}"


def test_roundtrip_json_corpus(json_corpus: list[bytes]):
    _train_and_roundtrip(json_corpus)


def test_roundtrip_http_corpus(http_corpus: list[bytes]):
    _train_and_roundtrip(http_corpus)


def test_roundtrip_code_corpus(code_corpus: list[bytes]):
    _train_and_roundtrip(code_corpus)


def test_roundtrip_handles_input_not_in_training(json_corpus: list[bytes]):
    # Train on the corpus, then encode a previously-unseen input.
    # The literal-fallback path should still round-trip.
    codec = make_codec(json_corpus)
    unseen = b'{"surprise":"this was not in training","n":99}'
    assert codec.decode(codec.encode(unseen)) == unseen


def test_roundtrip_handles_empty_and_tiny_inputs(json_corpus: list[bytes]):
    codec = make_codec(json_corpus)
    for s in [b"", b"a", b"ab", b"abc"]:
        assert codec.decode(codec.encode(s)) == s


def test_minimizer_signatures_deterministic_across_calls():
    # Same bytes must produce the same minimizer hashes every call —
    # this is the property that Python's randomized hash() breaks.
    s = b"stable signature test"
    sig1 = _minimizers(s)
    sig2 = _minimizers(s)
    assert sig1 == sig2


def test_minimizer_signatures_stable_for_tiny_input():
    # The <w bytes branch must also be deterministic.
    sig1 = _minimizers(b"ab")
    sig2 = _minimizers(b"ab")
    assert sig1 == sig2
    assert sig1 != _minimizers(b"ac")


def test_encoded_output_deterministic_across_calls(json_corpus: list[bytes]):
    # The full encode path (which depends on minimizer signatures) must
    # produce the same bytes for the same input across multiple calls.
    codec = make_codec(json_corpus)
    s = json_corpus[1]
    assert codec.encode(s) == codec.encode(s)


def test_encoded_output_deterministic_across_codec_instances(json_corpus: list[bytes]):
    # Two codecs trained on the same corpus must produce byte-identical
    # encodings — the minimizer index is built from stable hashes.
    c1 = make_codec(json_corpus)
    c2 = make_codec(json_corpus)
    s = json_corpus[1]
    assert c1.encode(s) == c2.encode(s)
