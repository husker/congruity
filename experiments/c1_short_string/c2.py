"""C2 prototype: learned BPE tokens + per-domain static token model + range coding.

Faithful minimal implementation of C2 from run 20260520-053938, which the gate
marked **purely_testable=False (1 FUNDAMENTAL objection)** — we build it to test
whether that `fund>0` flag actually predicts failure (gate-precision series).

  C2 claim: "micro-domain entropy coder: tokenize short records into learned
  sub-byte or multi-byte units, select a tiny domain prefix/codebook for headers
  vs JSON vs identifiers, and decode many symbols through table/vector operations
  rather than branch-heavy adaptive modeling."

L4's single FUNDAMENTAL objection was the pigeonhole bound: "a compressor cannot
make every arbitrary 8-256 byte string shorter… if any inputs shrink, some must
expand." That is mathematically true — so this experiment measures TWO things:
  (1) the universal claim, via a random-bytes corpus (expect expansion → objection
      holds, exactly as it holds for gzip/brotli/zstd too); and
  (2) the practical outcome metric — does it beat baselines on TYPICAL corpora?

Mechanism (distinct from c3.py's pure byte context model):
  - Learned BPE vocabulary over training bytes (all 256 single bytes are always
    in-vocab → arbitrary input is representable → lossless, with byte tokens as
    the literal/escape path the objections asked for).
  - Greedy longest-match tokenization at encode time.
  - Per-domain (fingerprint-routed) static order-1 token model, range-coded.

Amortized accounting: BPE merges + token tables are deployed once (not charged
per record), same rule as Brotli-dict / Zstd-dict / C1 / C3. Per-record model-id
and byte-length varints ARE charged.
"""

from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import dataclass, field

from experiments.c1_short_string.c3 import (
    RangeDecoder,
    RangeEncoder,
    _fingerprint,
    _varint_dec,
    _varint_enc,
)

_MAX_TOTAL = 1 << 15
_W1, _W0 = 16, 3  # order-1 token context weight, order-0 backoff weight


def _train_bpe(records: list[bytes], max_merges: int = 256) -> list[tuple[int, int]]:
    """Greedy BPE: learn the most-frequent adjacent token merges.

    Returns an ordered list of merge rules (a, b) -> new_id (256 + rank).
    All 256 raw bytes are implicit base tokens (ids 0..255).
    """
    seqs = [list(r) for r in records if r]
    merges: list[tuple[int, int]] = []
    for rank in range(max_merges):
        pair_counts: Counter[tuple[int, int]] = Counter()
        for seq in seqs:
            for i in range(len(seq) - 1):
                pair_counts[(seq[i], seq[i + 1])] += 1
        if not pair_counts:
            break
        (a, b), cnt = pair_counts.most_common(1)[0]
        if cnt < 2:
            break
        new_id = 256 + rank
        merges.append((a, b))
        # Apply the merge across all sequences.
        for seq in seqs:
            i = 0
            out = []
            while i < len(seq):
                if i + 1 < len(seq) and seq[i] == a and seq[i + 1] == b:
                    out.append(new_id)
                    i += 2
                else:
                    out.append(seq[i])
                    i += 1
            seq[:] = out
    return merges


def _apply_bpe(rec: bytes, merge_rank: dict[tuple[int, int], int]) -> list[int]:
    """Tokenize a record by repeatedly applying the lowest-rank applicable merge."""
    seq = list(rec)
    if not seq:
        return seq
    while True:
        best_rank = None
        best_pos = -1
        for i in range(len(seq) - 1):
            r = merge_rank.get((seq[i], seq[i + 1]))
            if r is not None and (best_rank is None or r < best_rank):
                best_rank = r
                best_pos = i
        if best_rank is None:
            return seq
        seq[best_pos:best_pos + 2] = [256 + best_rank]


@dataclass
class _TokModel:
    vocab_size: int
    c0: dict[int, int] = field(default_factory=lambda: defaultdict(int))
    c1: dict[int, dict[int, int]] = field(default_factory=lambda: defaultdict(lambda: defaultdict(int)))
    _cache: dict[int, tuple[list[int], list[int]]] = field(default_factory=dict)

    def train_seq(self, toks: list[int]) -> None:
        prev = 0  # BOS
        for t in toks:
            self.c0[t] += 1
            self.c1[prev][t] += 1
            prev = t

    def freq_table(self, prev: int) -> tuple[list[int], list[int]]:
        hit = self._cache.get(prev)
        if hit is not None:
            return hit
        V = self.vocab_size
        c1 = self.c1.get(prev)
        freqs = [0] * V
        total = 0
        for s in range(V):
            f = _W0 * self.c0.get(s, 0) + 1
            if c1 is not None:
                f += _W1 * c1.get(s, 0)
            freqs[s] = f
            total += f
        if total > _MAX_TOTAL:
            scale = total / _MAX_TOTAL
            for s in range(V):
                freqs[s] = max(1, int(freqs[s] / scale))
        cum = [0] * (V + 1)
        acc = 0
        for s in range(V):
            cum[s] = acc
            acc += freqs[s]
        cum[V] = acc
        self._cache[prev] = (freqs, cum)
        return freqs, cum


@dataclass
class C2Codec:
    merges: list[tuple[int, int]]
    merge_rank: dict[tuple[int, int], int]
    vocab_size: int
    models: list[_TokModel]
    family_id: dict[bytes, int]

    @classmethod
    def train(cls, training: list[bytes], max_merges: int = 256,
              max_families: int = 32, min_family: int = 40) -> "C2Codec":
        merges = _train_bpe(training, max_merges=max_merges)
        merge_rank = {pair: i for i, pair in enumerate(merges)}
        vocab_size = 256 + len(merges)

        buckets: dict[bytes, list[bytes]] = defaultdict(list)
        for rec in training:
            buckets[_fingerprint(rec)].append(rec)
        ranked = sorted(buckets.items(), key=lambda kv: -len(kv[1]))
        family_id: dict[bytes, int] = {}
        models: list[_TokModel] = [_TokModel(vocab_size)]  # global
        for fp, recs in ranked:
            if fp == b"" or len(recs) < min_family or len(models) > max_families:
                continue
            m = _TokModel(vocab_size)
            for r in recs:
                m.train_seq(_apply_bpe(r, merge_rank))
            family_id[fp] = len(models)
            models.append(m)
        for rec in training:
            models[0].train_seq(_apply_bpe(rec, merge_rank))
        return cls(merges=merges, merge_rank=merge_rank, vocab_size=vocab_size,
                   models=models, family_id=family_id)

    def encode(self, b: bytes) -> bytes:
        mid = self.family_id.get(_fingerprint(b), 0)
        model = self.models[mid]
        toks = _apply_bpe(b, self.merge_rank)
        enc = RangeEncoder()
        prev = 0
        for t in toks:
            freqs, cum = model.freq_table(prev)
            enc.encode(cum[t], freqs[t], cum[self.vocab_size])
            prev = t
        return _varint_enc(mid) + _varint_enc(len(b)) + _varint_enc(len(toks)) + enc.finish()

    def decode(self, data: bytes) -> bytes:
        mid, off = _varint_dec(data, 0)
        _nbytes, off = _varint_dec(data, off)
        ntoks, off = _varint_dec(data, off)
        model = self.models[mid]
        dec = RangeDecoder(data[off:])
        # Reverse map token id -> byte sequence (expand merges).
        out = bytearray()
        prev = 0
        for _ in range(ntoks):
            freqs, cum = model.freq_table(prev)
            total = cum[self.vocab_size]
            v = dec.get_freq(total)
            lo, hi = 0, self.vocab_size
            while lo + 1 < hi:
                m = (lo + hi) // 2
                if cum[m] <= v:
                    lo = m
                else:
                    hi = m
            tok = lo
            dec.decode(cum[tok], freqs[tok], total)
            out += self._expand(tok)
            prev = tok
        return bytes(out)

    def _expand(self, tok: int) -> bytes:
        if tok < 256:
            return bytes([tok])
        a, b = self.merges[tok - 256]
        return self._expand(a) + self._expand(b)


def make_codec(training: list[bytes]) -> C2Codec:
    return C2Codec.train(training)
