"""C1 prototype: minimizer-fingerprint -> template lookup -> sparse residual coding.

Faithful minimal implementation of the C1 composition surfaced in
findings/compose-20260520-053938.md:

  P8 (minimap2-style minimizer lookup)
  + P17 (hierarchical family backoff — simplified to flat templates here)
  + P27 (sparse residual coding)

Goal: round-trip lossless compression of arbitrary 8-256 byte inputs whose
total bytes (template_id + residual ops) compete with Brotli+shared-dict
and Zstd+trained-dict.

Pure-Python implementation; we measure compression RATIO, not latency.
The L3 adversary's latency objection is not testable from Python anyway.

Bitstream format (compact varints, byte-aligned for simplicity):

  byte 0 header:
    0x00 = LITERAL fallback
    0x01 = TEMPLATE_HIT (same length as template)
    0x02 = TEMPLATE_HIT_LEN (template + length diff)

  LITERAL:        0x00 | <raw_input_bytes>
  TEMPLATE_HIT:   0x01 | varint(tid) | varint(n_ops) | [op]*n_ops
  TEMPLATE_HIT_LEN: 0x02 | varint(tid) | varint(out_len) | varint(n_ops) | [op]*n_ops

  op = varint(delta_pos) | u8(new_byte)

This is byte-aligned (not bit-packed). A production version would use ANS/Huffman.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from hashlib import blake2b


def _stable_hash(b: bytes) -> int:
    """Deterministic 64-bit hash for minimizer signatures.

    Python's builtin hash() is randomized per process (PYTHONHASHSEED),
    so reusing it across runs (or comparing encoded outputs from different
    processes) would silently break the minimizer index. BLAKE2 is fast,
    keyless, and stable.
    """
    return int.from_bytes(blake2b(b, digest_size=8).digest(), "big")


def _varint_enc(n: int) -> bytes:
    if n < 0:
        raise ValueError("negative varint")
    out = bytearray()
    while True:
        if n < 0x80:
            out.append(n)
            return bytes(out)
        out.append((n & 0x7F) | 0x80)
        n >>= 7


def _varint_dec(buf: memoryview, off: int) -> tuple[int, int]:
    n = 0
    shift = 0
    while True:
        b = buf[off]
        off += 1
        n |= (b & 0x7F) << shift
        if not (b & 0x80):
            return n, off
        shift += 7


def _minimizers(b: bytes, w: int = 4) -> list[int]:
    """All w-gram hashes — used as a candidate-template signature.

    Tiny inputs (<w bytes) yield a single one-shot signature based on the
    whole input (zero-padded).
    """
    if len(b) < w:
        return [_stable_hash(b)]
    out = []
    for i in range(len(b) - w + 1):
        out.append(_stable_hash(b[i:i+w]))
    return out


@dataclass
class C1Codec:
    templates: list[bytes]
    # minimizer-hash -> list of template indices that contain it
    minimizer_index: dict[int, list[int]]
    fallback_count: int = 0
    hit_count: int = 0

    @classmethod
    def train(cls, training: list[bytes], max_templates: int = 1024, w: int = 4) -> "C1Codec":
        """Pick the K most-frequent training records as flat templates.

        A real Pfam-style hierarchy would cluster these; we use frequency
        ranking as a simple proxy.
        """
        c = Counter(training)
        templates = [t for t, _ in c.most_common(max_templates)]
        # Build minimizer index: hash -> [template_id...]
        idx: dict[int, list[int]] = {}
        for tid, t in enumerate(templates):
            for h in _minimizers(t, w=w):
                idx.setdefault(h, []).append(tid)
        return cls(templates=templates, minimizer_index=idx)

    def _best_template(self, b: bytes, w: int = 4) -> tuple[int, int] | None:
        """Score candidate templates by # of shared minimizer hits; return (tid, n_ops).

        We pick the template that minimises the encoded residual size:
        approx = abs(len_diff) + (n_byte_substitutions).
        We try at most TOP_K candidates; minimizer hits give the candidate set.
        """
        # Gather candidates from minimizer index
        sig = _minimizers(b, w=w)
        votes: Counter[int] = Counter()
        for h in sig:
            for tid in self.minimizer_index.get(h, ()):
                votes[tid] += 1
        if not votes:
            return None
        # Score top candidates by actual residual size
        TOP_K = 8
        best_tid = -1
        best_cost = float("inf")
        for tid, _v in votes.most_common(TOP_K):
            t = self.templates[tid]
            cost = self._residual_cost(b, t)
            if cost < best_cost:
                best_cost = cost
                best_tid = tid
        if best_tid < 0:
            return None
        return best_tid, int(best_cost)

    @staticmethod
    def _residual_cost(b: bytes, template: bytes) -> int:
        """Number of substitution ops + varint cost for length diff.

        Approximates encoded residual size: ops_count * 2 bytes (delta + value)
        + len-diff overhead. Real encoder is exact; this is the planner's score.
        """
        diff_byte_count = 0
        L = min(len(b), len(template))
        for i in range(L):
            if b[i] != template[i]:
                diff_byte_count += 1
        diff_byte_count += abs(len(b) - len(template))
        return diff_byte_count * 2  # rough cost per op (delta varint + byte)

    def encode(self, b: bytes) -> bytes:
        """Encode `b`. Always lossless (literal fallback if no template helps)."""
        best = self._best_template(b)
        if best is None:
            self.fallback_count += 1
            return self._encode_literal(b)
        tid, _ = best
        template = self.templates[tid]
        same_len_payload = self._try_encode_same_len(tid, b, template)
        diff_len_payload = self._encode_diff_len(tid, b)
        literal = self._encode_literal(b)
        candidates: list[bytes] = [literal, diff_len_payload]
        if same_len_payload is not None:
            candidates.append(same_len_payload)
        best_enc = candidates[0]
        for cand in candidates[1:]:
            if len(cand) < len(best_enc):
                best_enc = cand
        if best_enc is literal:
            self.fallback_count += 1
        else:
            self.hit_count += 1
        return best_enc

    @staticmethod
    def _encode_literal(b: bytes) -> bytes:
        return b"\x00" + b

    def _try_encode_same_len(self, tid: int, b: bytes, template: bytes) -> bytes | None:
        if len(b) != len(template):
            return None
        ops = bytearray()
        n_ops = 0
        last_pos = 0
        for i in range(len(b)):
            if b[i] != template[i]:
                delta = i - last_pos
                ops += _varint_enc(delta)
                ops.append(b[i])
                last_pos = i + 1
                n_ops += 1
        return b"\x01" + _varint_enc(tid) + _varint_enc(n_ops) + bytes(ops)

    def _encode_diff_len(self, tid: int, b: bytes) -> bytes:
        """For length mismatch: emit the input bytes after a header.

        A real implementation would use a proper edit script (insert/delete).
        We approximate: emit varint(tid) + varint(out_len) + raw input.
        This will rarely beat literal — and that's the point: the prototype
        falls back to literal when len differs.
        """
        return b"\x02" + _varint_enc(tid) + _varint_enc(len(b)) + b

    def decode(self, enc: bytes) -> bytes:
        m = memoryview(enc)
        header = m[0]
        if header == 0x00:
            return bytes(m[1:])
        if header == 0x01:
            tid, off = _varint_dec(m, 1)
            n_ops, off = _varint_dec(m, off)
            template = bytearray(self.templates[tid])
            pos = 0
            for _ in range(n_ops):
                delta, off = _varint_dec(m, off)
                pos += delta
                template[pos] = m[off]
                off += 1
                pos += 1
            return bytes(template)
        if header == 0x02:
            _tid, off = _varint_dec(m, 1)
            out_len, off = _varint_dec(m, off)
            return bytes(m[off:off + out_len])
        raise ValueError(f"unknown header {header}")


def make_codec(training: list[bytes], max_templates: int = 1024, w: int = 4) -> C1Codec:
    return C1Codec.train(training, max_templates=max_templates, w=w)


def encode_records(codec: C1Codec, records: list[bytes]) -> list[bytes]:
    return [codec.encode(r) for r in records]


if __name__ == "__main__":
    # Quick smoke test
    import sys
    sys.path.insert(0, ".")
    from experiments.c1_short_string.corpora import http_headers
    recs = http_headers(n=500)
    half = len(recs) // 2
    train, test = recs[:half], recs[half:]
    codec = make_codec(train)
    print(f"templates: {len(codec.templates)}")
    encoded = encode_records(codec, test)
    raw_sz = sum(len(r) for r in test)
    enc_sz = sum(len(e) for e in encoded)
    print(f"raw bytes: {raw_sz}")
    print(f"enc bytes: {enc_sz}")
    print(f"ratio: {enc_sz/raw_sz:.3f}  (lower is better)")
    print(f"hits/fallbacks: {codec.hit_count} / {codec.fallback_count}")

    # Round-trip check
    decoded = [codec.decode(e) for e in encoded]
    assert decoded == test, "C1 round-trip failed!"
    print("round-trip: OK")
