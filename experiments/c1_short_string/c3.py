"""C3 prototype: fingerprint-routed static micro-LM + arithmetic coding.

Faithful minimal implementation of the C3 composition from
findings/compose-20260520-053938.md (run 20260521-023514 ... actually
20260520-053938), which the L1-L4 gate REJECTED with L3=invalid. We build it
to test gate precision (does a rejected composition actually fail?).

  C3 claim: "two-stage short-input compressor: use tiny fingerprints to pick a
  few candidate learned-token dictionaries, then use a precomputed micro
  language model or probability table over those tokens, laid out for SIMD
  batch decode. Bridges genomic seed-and-chain lookup with language-model
  entropy coding while avoiding online adaptation warmup."

The operative, ratio-relevant ideas (SIMD/latency aren't testable in Python):
  1. STATIC model trained offline → no per-message adaptation warmup. This is
     the wedge for SHORT inputs, where gzip/zstd/brotli pay warmup cost.
  2. A micro language model = a low-order byte context model with backoff.
  3. Fingerprint routing = pick among a few per-family context models by a
     cheap fingerprint of the input; a 1-byte selector is stored in the stream.

We measure compression RATIO with the harness's amortized-model accounting
(the trained tables are deployed once, not charged per record), identical to
how Brotli-dict / Zstd-dict / C1 are scored.

Entropy coder: a 32-bit carryless range coder (Subbotin style). Losslessness is
guaranteed by add-one smoothing on a blended order-2/1/0 model (every one of the
256 byte values always has nonzero probability) plus a stored length prefix. A
round-trip assertion over the test corpus is run in the bench.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field

# ---------------------------------------------------------------------------
# Range coder (32-bit, carryless; TOP/BOT renormalization)
# ---------------------------------------------------------------------------
_TOP = 1 << 24
_BOT = 1 << 16
_MASK = (1 << 32) - 1


class RangeEncoder:
    def __init__(self) -> None:
        self.low = 0
        self.range = _MASK
        self.out = bytearray()

    def encode(self, cum: int, freq: int, total: int) -> None:
        # Narrow [low, low+range) to the symbol's sub-interval.
        r = self.range // total
        self.low = (self.low + r * cum) & _MASK
        self.range = r * freq
        # Renormalize.
        while True:
            if (self.low ^ (self.low + self.range)) < _TOP:
                pass
            elif self.range < _BOT:
                self.range = (-self.low) & (_BOT - 1)
            else:
                break
            self.out.append((self.low >> 24) & 0xFF)
            self.low = (self.low << 8) & _MASK
            self.range = (self.range << 8) & _MASK

    def finish(self) -> bytes:
        for _ in range(4):
            self.out.append((self.low >> 24) & 0xFF)
            self.low = (self.low << 8) & _MASK
        return bytes(self.out)


class RangeDecoder:
    def __init__(self, data: bytes) -> None:
        self.data = data
        self.pos = 0
        self.low = 0
        self.range = _MASK
        self.code = 0
        for _ in range(4):
            self.code = ((self.code << 8) | self._byte()) & _MASK

    def _byte(self) -> int:
        if self.pos < len(self.data):
            b = self.data[self.pos]
            self.pos += 1
            return b
        return 0

    def get_freq(self, total: int) -> int:
        self._r = self.range // total
        v = (self.code - self.low) // self._r
        return v if v < total else total - 1

    def decode(self, cum: int, freq: int, total: int) -> None:
        r = self._r
        self.low = (self.low + r * cum) & _MASK
        self.range = r * freq
        while True:
            if (self.low ^ (self.low + self.range)) < _TOP:
                pass
            elif self.range < _BOT:
                self.range = (-self.low) & (_BOT - 1)
            else:
                break
            self.code = ((self.code << 8) | self._byte()) & _MASK
            self.low = (self.low << 8) & _MASK
            self.range = (self.range << 8) & _MASK


# ---------------------------------------------------------------------------
# Static blended order-2/1/0 byte model with add-one smoothing
# ---------------------------------------------------------------------------
# Weights favor higher orders; the trailing +1 floor guarantees every symbol is
# representable (losslessness) and provides backoff to lower orders.
_W2, _W1, _W0 = 48, 12, 3
_BOS = 0  # byte-of-stream start context padding
# Range coder precision: cumulative total must stay below ~2^16 so that
# range // total never underflows. We rescale frequency tables to this cap
# while keeping every symbol >= 1 (preserves losslessness).
_MAX_TOTAL = 1 << 15


@dataclass
class _Model:
    c0: list[int] = field(default_factory=lambda: [0] * 256)
    c1: dict[int, list[int]] = field(default_factory=lambda: defaultdict(lambda: [0] * 256))
    c2: dict[int, list[int]] = field(default_factory=lambda: defaultdict(lambda: [0] * 256))
    n: int = 0
    # Cache of precomputed (freqs, cum) per context int — the static model's
    # "precomputed probability tables" (C3's stated mechanism). Built lazily.
    _cache: dict[int, tuple[list[int], list[int]]] = field(default_factory=dict)

    def train_record(self, rec: bytes) -> None:
        p1 = _BOS
        p2 = _BOS
        for byte in rec:
            self.c0[byte] += 1
            self.c1[p1][byte] += 1
            self.c2[(p1 << 8) | p2][byte] += 1
            p2 = p1
            p1 = byte
        self.n += 1

    def freq_table(self, p1: int, p2: int) -> tuple[list[int], list[int]]:
        """Return (freqs[256], cumulative[257]) for the current context.

        Cached per context: the model is static after training, so a context's
        table never changes once computed.
        """
        ctx = (p1 << 8) | p2
        hit = self._cache.get(ctx)
        if hit is not None:
            return hit
        c2 = self.c2.get(ctx)
        c1 = self.c1.get(p1)
        c0 = self.c0
        freqs = [0] * 256
        total = 0
        for s in range(256):
            f = _W0 * c0[s] + 1
            if c1 is not None:
                f += _W1 * c1[s]
            if c2 is not None:
                f += _W2 * c2[s]
            freqs[s] = f
            total += f
        # Rescale to keep total < _MAX_TOTAL (range-coder precision), floor 1.
        if total > _MAX_TOTAL:
            scale = total / _MAX_TOTAL
            for s in range(256):
                freqs[s] = max(1, int(freqs[s] / scale))
        cum = [0] * 257
        acc = 0
        for s in range(256):
            cum[s] = acc
            acc += freqs[s]
        cum[256] = acc
        self._cache[ctx] = (freqs, cum)
        return freqs, cum


def _varint_enc(n: int) -> bytes:
    out = bytearray()
    while True:
        if n < 0x80:
            out.append(n)
            return bytes(out)
        out.append((n & 0x7F) | 0x80)
        n >>= 7


def _varint_dec(data: bytes, off: int) -> tuple[int, int]:
    n = shift = 0
    while True:
        b = data[off]
        off += 1
        n |= (b & 0x7F) << shift
        if not (b & 0x80):
            return n, off
        shift += 7


def _fingerprint(rec: bytes) -> bytes:
    """Cheap routing key: header field-name (before ':') lowercased, else b''.

    This is the 'tiny fingerprint' that selects a candidate family model. For
    non-header inputs it falls back to the empty key → global model.
    """
    i = rec.find(b":")
    if 0 < i <= 40:
        return rec[:i].lower()
    return b""


@dataclass
class C3Codec:
    """Fingerprint-routed static micro-LM entropy coder."""

    models: list[_Model]               # models[0] is the global fallback
    family_id: dict[bytes, int]        # fingerprint -> model index (>=1)

    @classmethod
    def train(cls, training: list[bytes], max_families: int = 32, min_family: int = 40) -> "C3Codec":
        # Bucket training records by fingerprint.
        buckets: dict[bytes, list[bytes]] = defaultdict(list)
        for rec in training:
            buckets[_fingerprint(rec)].append(rec)
        # Keep the most populous families (bounded model budget).
        ranked = sorted(buckets.items(), key=lambda kv: -len(kv[1]))
        family_id: dict[bytes, int] = {}
        models: list[_Model] = [_Model()]  # global model (index 0)
        for fp, recs in ranked:
            if fp == b"" or len(recs) < min_family or len(models) > max_families:
                continue
            m = _Model()
            for r in recs:
                m.train_record(r)
            family_id[fp] = len(models)
            models.append(m)
        # The global model is trained on everything (backstop for all inputs).
        for rec in training:
            models[0].train_record(rec)
        return cls(models=models, family_id=family_id)

    def _select(self, rec: bytes) -> int:
        return self.family_id.get(_fingerprint(rec), 0)

    def encode(self, b: bytes) -> bytes:
        mid = self._select(b)
        model = self.models[mid]
        enc = RangeEncoder()
        p1 = _BOS
        p2 = _BOS
        for byte in b:
            freqs, cum = model.freq_table(p1, p2)
            enc.encode(cum[byte], freqs[byte], cum[256])
            p2 = p1
            p1 = byte
        body = enc.finish()
        # Header: model id + record length (both amortized-free table aside).
        return _varint_enc(mid) + _varint_enc(len(b)) + body

    def decode(self, enc: bytes) -> bytes:
        mid, off = _varint_dec(enc, 0)
        n, off = _varint_dec(enc, off)
        model = self.models[mid]
        dec = RangeDecoder(enc[off:])
        out = bytearray()
        p1 = _BOS
        p2 = _BOS
        for _ in range(n):
            freqs, cum = model.freq_table(p1, p2)
            total = cum[256]
            v = dec.get_freq(total)
            # Locate symbol whose cumulative interval contains v.
            lo, hi = 0, 256
            while lo + 1 < hi:
                mid_s = (lo + hi) // 2
                if cum[mid_s] <= v:
                    lo = mid_s
                else:
                    hi = mid_s
            sym = lo
            dec.decode(cum[sym], freqs[sym], total)
            out.append(sym)
            p2 = p1
            p1 = sym
        return bytes(out)


def make_codec(training: list[bytes]) -> C3Codec:
    return C3Codec.train(training)
