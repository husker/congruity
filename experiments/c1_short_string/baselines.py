"""Compression baselines for short-string C1 verification.

Methods:
- raw: no compression (control)
- gzip: stdlib zlib at level 9
- brotli: level 11 + optional shared dictionary
- zstd: level 19 + optional trained dictionary

Compression-ratio accounting: shared dictionary or trained-dictionary bytes
are NOT charged per record. This matches how Brotli's shared static dict
and Zstd's trained dict are deployed (amortised across millions of msgs).
The C1 prototype uses the same accounting rule for its family-table.
"""

from __future__ import annotations

import gzip
import zlib
from dataclasses import dataclass
from typing import Callable


@dataclass
class Method:
    name: str
    encode: Callable[[bytes], bytes]
    train: Callable[[list[bytes]], None] | None = None  # optional training step


def raw_method() -> Method:
    return Method(name="raw", encode=lambda b: b)


def gzip_method() -> Method:
    return Method(
        name="gzip-9",
        encode=lambda b: gzip.compress(b, compresslevel=9, mtime=0),
    )


def zlib_method() -> Method:
    return Method(
        name="zlib-9",
        encode=lambda b: zlib.compress(b, level=9),
    )


def brotli_method(quality: int = 11) -> Method:
    import brotli
    return Method(
        name=f"brotli-{quality}",
        encode=lambda b: brotli.compress(b, quality=quality, mode=brotli.MODE_GENERIC),
    )


def zstd_method(level: int = 19) -> Method:
    import zstandard
    cctx = zstandard.ZstdCompressor(level=level)
    return Method(
        name=f"zstd-{level}",
        encode=lambda b: cctx.compress(b),
    )


def zstd_dict_method(training: list[bytes], level: int = 19, dict_size: int = 1 << 15) -> Method:
    """Zstd trained-dictionary method.

    train_dictionary requires at least a few KB of training data; we sample.
    """
    import zstandard
    sample = list(training)
    # zstd dict trainer wants a list of bytes samples; min ~7 samples
    if len(sample) < 8:
        sample = (sample * 8)[:32]
    zdict = zstandard.train_dictionary(dict_size, sample)
    cctx = zstandard.ZstdCompressor(level=level, dict_data=zdict)
    return Method(
        name=f"zstd-{level}+dict",
        encode=lambda b: cctx.compress(b),
    )


def hpack_per_line_method() -> Method:
    """HPACK encoding a single 'Name: Value' line as a 1-element header block.

    Stateless per-record: a fresh Encoder per line, so only HPACK's static
    table contributes. This is the apples-to-apples comparison against
    other stateless per-line methods (Brotli, Zstd, Zstd+dict, C1).

    Lines that don't parse as 'Name: Value' are encoded as a literal header
    `x-raw: <line>` so HPACK has something to encode.
    """
    import hpack

    def enc(b: bytes) -> bytes:
        s = b.decode("utf-8", errors="replace")
        if ": " in s:
            name, value = s.split(": ", 1)
        else:
            name, value = "x-raw", s
        e = hpack.Encoder()
        return e.encode([(name.lower(), value)])

    return Method(name="hpack-static", encode=enc)


def all_methods(training: list[bytes]) -> list[Method]:
    return [
        raw_method(),
        gzip_method(),
        zlib_method(),
        brotli_method(quality=11),
        zstd_method(level=19),
        zstd_dict_method(training, level=19),
    ]


def hpack_stateful_per_line_method() -> Method:
    """HPACK with one Encoder shared across all records (dynamic table grows).

    This is how HPACK actually deploys in HTTP/2: a single connection encoder
    that accumulates state. Across many headers, the dynamic table makes
    repeats nearly free. This is HPACK's strongest deployment mode.

    Note: this method is order-sensitive. The test set must be encoded in a
    consistent order; we accept whatever the caller passes.
    """
    import hpack

    encoder = hpack.Encoder()

    def enc(b: bytes) -> bytes:
        s = b.decode("utf-8", errors="replace")
        if ": " in s:
            name, value = s.split(": ", 1)
        else:
            name, value = "x-raw", s
        return encoder.encode([(name.lower(), value)])

    return Method(name="hpack-stateful", encode=enc)


def all_methods_with_hpack(training: list[bytes]) -> list[Method]:
    return all_methods(training) + [hpack_per_line_method(), hpack_stateful_per_line_method()]
