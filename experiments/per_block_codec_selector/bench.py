"""C1 oracle verification — compose run 20260523-180439.

C1's recipe:
    P172's low-cost algorithm-selection idea + P206's drift monitoring +
    P135's codec-comparison framing yields a streaming lossless
    compressor that samples cheap block features, chooses among
    gzip/brotli/zstd/lz4 per block, and monitors drift.

Stop rule (cheapest falsification):
    First run an oracle that compresses each block with every candidate
    codec and picks the smallest output including metadata; if this
    oracle fails to beat the best fixed whole-file baseline by 3%,
    abandon.

The oracle is the structural upper bound for any learned codec
selector. A learned selector can only do worse — its prediction is
imperfect — so if the oracle gap is < 3%, no learner can deliver
the composition's claimed advantage.

Setup. Synthesise a HETEROGENEOUS corpus (text, code, JSON, binary
noise, repeated runs, smoothly varying integers) — exactly the regime
where per-block selection should help — then:

  1. Block the corpus at fixed size (default 64 KiB).
  2. For each block, compress with each codec at a fixed level and
     record the compressed length.
  3. Oracle = sum_b min_c (compressed_size(block_b, codec_c)) +
              n_blocks * METADATA_BYTES
     (we pay metadata for codec id + length prefix per block).
  4. Each fixed baseline = compressed_size(full_corpus, codec_c).
  5. Verdict: oracle savings vs best fixed baseline >= 3%?

Codecs tested:
  - zstd level 3 (good ratio at fast speed)
  - brotli quality 5 (text-heavy default)
  - gzip level 6 (canonical baseline)
  - lz4 default (fastest, weak ratio)

Run:
    uv run --with zstandard --with brotli --with lz4 \
        python -m experiments.per_block_codec_selector.bench
"""

from __future__ import annotations

import argparse
import gzip
import hashlib
import io
import json
import random
import string
import sys

import brotli
import lz4.frame
import zstandard


METADATA_BYTES_PER_BLOCK = 5  # 1 byte codec id + 4 byte length prefix


# ---------------------------------------------------------------------------
# Codecs (compress only — exact round-trip is asserted by the libraries)
# ---------------------------------------------------------------------------


def c_zstd_3(data: bytes) -> int:
    return len(zstandard.ZstdCompressor(level=3).compress(data))


def c_brotli_5(data: bytes) -> int:
    return len(brotli.compress(data, quality=5))


def c_gzip_6(data: bytes) -> int:
    buf = io.BytesIO()
    with gzip.GzipFile(fileobj=buf, mode="wb", compresslevel=6, mtime=0) as f:
        f.write(data)
    return len(buf.getvalue())


def c_lz4(data: bytes) -> int:
    return len(lz4.frame.compress(data))


def c_zstd_22(data: bytes) -> int:
    return len(zstandard.ZstdCompressor(level=22).compress(data))


def c_brotli_11(data: bytes) -> int:
    return len(brotli.compress(data, quality=11))


def c_gzip_9(data: bytes) -> int:
    buf = io.BytesIO()
    with gzip.GzipFile(fileobj=buf, mode="wb", compresslevel=9, mtime=0) as f:
        f.write(data)
    return len(buf.getvalue())


# Cheap codecs the oracle is allowed to pick from per block (these match
# the recipe's "samples cheap block features, chooses among gzip/brotli/
# zstd/lz4 per block" language — fast levels suitable for streaming).
CHEAP_CODECS = {
    "zstd_3": c_zstd_3,
    "brotli_5": c_brotli_5,
    "gzip_6": c_gzip_6,
    "lz4": c_lz4,
}

# Expensive whole-file fixed baselines. We test the oracle against these
# too — if cheap per-block selection cannot beat expensive fixed
# whole-file, the case for per-block selection collapses entirely.
EXPENSIVE_FIXED = {
    "zstd_22": c_zstd_22,
    "brotli_11": c_brotli_11,
    "gzip_9": c_gzip_9,
}

CODECS = CHEAP_CODECS  # alias preserved for callers


# ---------------------------------------------------------------------------
# Heterogeneous synthetic corpus
# ---------------------------------------------------------------------------


def english_like(n_bytes: int, rng: random.Random) -> bytes:
    """English-shaped tokens drawn from a small dictionary; reasonable
    compression target for brotli/gzip (text models)."""
    words = (
        "the quick brown fox jumps over the lazy dog and runs through "
        "the forest while watching birds sing in the trees above the "
        "river that flows slowly toward the ocean where waves crash on "
        "the rocky shore as the sun sets behind the mountains casting "
        "long shadows across the green valley below where farmers tend "
        "their fields and shepherds guide their flocks home for the "
        "night while smoke rises from the chimneys of small cottages "
        "and the smell of bread baking fills the cool evening air"
    ).split()
    out: list[str] = []
    n = 0
    while n < n_bytes:
        w = rng.choice(words)
        out.append(w)
        n += len(w) + 1
    text = " ".join(out)[:n_bytes].encode()
    return text


def code_like(n_bytes: int, rng: random.Random) -> bytes:
    """Python-shaped tokens; heavily repetitive keyword/operator structure
    favours dictionary-based codecs."""
    kw = ["def", "if", "for", "while", "return", "import", "class",
          "True", "False", "None", "self", "in", "and", "or", "not"]
    ops = [" = ", " + ", " * ", " - ", " == ", " != ", " < ", " > ",
           " <= ", " >= ", "(", ")", "[", "]", "{", "}", ", ", ": "]
    names = ["x", "y", "z", "i", "j", "k", "n", "result", "value",
             "data", "item", "obj", "key", "tmp", "out"]
    out: list[str] = []
    n = 0
    while n < n_bytes:
        line: list[str] = ["    " * rng.randint(0, 3)]
        for _ in range(rng.randint(1, 4)):
            line.append(rng.choice(kw + names))
            line.append(rng.choice(ops))
        line.append("\n")
        s = "".join(line)
        out.append(s)
        n += len(s)
    return "".join(out)[:n_bytes].encode()


def json_like(n_bytes: int, rng: random.Random) -> bytes:
    """JSON records with a small key vocabulary — structured, repeated."""
    keys = ["id", "name", "value", "ts", "type", "flag", "count",
            "user", "action", "version", "tag", "meta", "size"]
    out: list[str] = []
    n = 0
    while n < n_bytes:
        rec = {
            rng.choice(keys): rng.randint(0, 999999),
            rng.choice(keys): "".join(rng.choices(string.ascii_lowercase, k=8)),
            rng.choice(keys): rng.choice([True, False, None]),
            rng.choice(keys): [rng.randint(0, 999) for _ in range(rng.randint(0, 5))],
        }
        s = json.dumps(rec) + "\n"
        out.append(s)
        n += len(s)
    return "".join(out)[:n_bytes].encode()


def binary_random(n_bytes: int, rng: random.Random) -> bytes:
    """Uniform random bytes — incompressible regardless of codec."""
    return rng.randbytes(n_bytes)


def repeated_runs(n_bytes: int, rng: random.Random) -> bytes:
    """Mostly-repeated bytes with occasional changes — extreme RLE/LZ
    target."""
    out = bytearray()
    while len(out) < n_bytes:
        run = rng.randint(50, 500)
        b = rng.randint(0, 255)
        out.extend(bytes([b]) * run)
    return bytes(out[:n_bytes])


def delta_integers(n_bytes: int, rng: random.Random) -> bytes:
    """Sequence of int64 values that slowly drift — favours delta+RLE
    codecs and dictionary compression of the leading bytes."""
    n_ints = n_bytes // 8
    out = bytearray()
    val = rng.randint(0, 2**16)
    for _ in range(n_ints):
        val += rng.randint(-50, 50)
        out.extend(val.to_bytes(8, "little", signed=True))
    return bytes(out)


def hash_blob(n_bytes: int, rng: random.Random) -> bytes:
    """SHA256-chained pseudo-random bytes — like binary_random but
    deterministic-from-seed and structurally identical to opaque
    cryptographic output (incompressible)."""
    seed = rng.getrandbits(256).to_bytes(32, "big")
    out = bytearray()
    cur = seed
    while len(out) < n_bytes:
        cur = hashlib.sha256(cur).digest()
        out.extend(cur)
    return bytes(out[:n_bytes])


GENERATORS = {
    "english": english_like,
    "code": code_like,
    "json": json_like,
    "random": binary_random,
    "repeated": repeated_runs,
    "delta_ints": delta_integers,
    "hashblob": hash_blob,
}


def build_corpus(
    n_blocks: int, block_size: int, mix: list[str], seed: int,
) -> tuple[bytes, list[str]]:
    """Build a heterogeneous corpus where each block is drawn from one
    of the named generators (cycled through `mix`)."""
    rng = random.Random(seed)
    out = bytearray()
    labels: list[str] = []
    for i in range(n_blocks):
        kind = mix[i % len(mix)]
        block = GENERATORS[kind](block_size, rng)
        out.extend(block)
        labels.append(kind)
    return bytes(out), labels


# ---------------------------------------------------------------------------
# Bench
# ---------------------------------------------------------------------------


def measure(corpus: bytes, block_size: int) -> dict:
    n = len(corpus)
    n_blocks = (n + block_size - 1) // block_size

    # Per-codec whole-file baselines, cheap and expensive.
    fixed_cheap = {name: codec(corpus) for name, codec in CHEAP_CODECS.items()}
    fixed_expensive = {name: codec(corpus) for name, codec in EXPENSIVE_FIXED.items()}

    # Per-block table for cheap codecs (oracle picks from these).
    per_block: list[dict[str, int]] = []
    per_codec_block_total = {name: 0 for name in CHEAP_CODECS}
    for i in range(n_blocks):
        block = corpus[i * block_size : (i + 1) * block_size]
        sizes = {name: codec(block) for name, codec in CHEAP_CODECS.items()}
        per_block.append(sizes)
        for name, s in sizes.items():
            per_codec_block_total[name] += s

    # Oracle = sum of min per block + metadata for each block.
    oracle_payload = sum(min(b.values()) for b in per_block)
    oracle_total = oracle_payload + n_blocks * METADATA_BYTES_PER_BLOCK

    # Blocked fixed baselines (same cheap codec applied per block + metadata).
    blocked_fixed = {
        name: per_codec_block_total[name] + n_blocks * METADATA_BYTES_PER_BLOCK
        for name in CHEAP_CODECS
    }

    return {
        "raw_bytes": n,
        "n_blocks": n_blocks,
        "fixed_whole_file_cheap": fixed_cheap,
        "fixed_whole_file_expensive": fixed_expensive,
        "blocked_fixed": blocked_fixed,
        "oracle_payload": oracle_payload,
        "oracle_overhead": n_blocks * METADATA_BYTES_PER_BLOCK,
        "oracle_total": oracle_total,
        "per_block": per_block,
    }


def best_codec_distribution(per_block: list[dict[str, int]]) -> dict[str, int]:
    counts: dict[str, int] = {name: 0 for name in CHEAP_CODECS}
    for b in per_block:
        winner = min(b.items(), key=lambda kv: kv[1])[0]
        counts[winner] += 1
    return counts


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--block-size", type=int, default=64 * 1024,
                    help="Block size in bytes (default 64 KiB)")
    ap.add_argument("--blocks-per-type", type=int, default=8,
                    help="Number of blocks per generator type")
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--types", type=str, nargs="+",
                    default=["english", "code", "json", "random",
                             "repeated", "delta_ints", "hashblob"])
    args = ap.parse_args()

    print("\n=== C1 per-block codec selection oracle verification ===")
    print(f"block size: {args.block_size} B, types: {args.types}")
    print(
        f"blocks per type: {args.blocks_per_type}  "
        f"(total: {args.blocks_per_type * len(args.types)} blocks per seed)"
    )
    print(
        "stop rule: oracle savings vs best fixed whole-file baseline >= 3%"
    )

    rows = []
    for seed in range(args.seeds):
        # Interleave block types so adjacent blocks change kind — exactly
        # the regime where per-block selection should win.
        mix = list(args.types) * args.blocks_per_type
        corpus, labels = build_corpus(
            n_blocks=len(mix), block_size=args.block_size, mix=mix, seed=seed,
        )
        r = measure(corpus, args.block_size)
        rows.append(r)
        print(
            f"\nseed {seed}: raw={r['raw_bytes']:,} B in {r['n_blocks']} blocks"
        )
        print("  fixed whole-file (cheap levels — same as oracle pool):")
        for name, b in r["fixed_whole_file_cheap"].items():
            print(f"    {name:>10}: {b:>10,} B  ratio={b / r['raw_bytes']:.4f}")
        print("  fixed whole-file (expensive levels):")
        for name, b in r["fixed_whole_file_expensive"].items():
            print(f"    {name:>10}: {b:>10,} B  ratio={b / r['raw_bytes']:.4f}")
        print(f"  blocked fixed (with {METADATA_BYTES_PER_BLOCK}B/block overhead):")
        for name, b in r["blocked_fixed"].items():
            print(f"    {name:>10}: {b:>10,} B  ratio={b / r['raw_bytes']:.4f}")
        print(
            f"  ORACLE total:  {r['oracle_total']:>10,} B  "
            f"ratio={r['oracle_total'] / r['raw_bytes']:.4f}  "
            f"(payload {r['oracle_payload']:,} + overhead {r['oracle_overhead']:,})"
        )
        bcd = best_codec_distribution(r["per_block"])
        print(f"  best-codec distribution over blocks: {bcd}")

    # Aggregate over seeds
    print(f"\n=== aggregate (mean over {args.seeds} seeds) ===")
    mean_raw = sum(r["raw_bytes"] for r in rows) / len(rows)
    fixed_cheap_mean = {
        name: sum(r["fixed_whole_file_cheap"][name] for r in rows) / len(rows)
        for name in CHEAP_CODECS
    }
    fixed_expensive_mean = {
        name: sum(r["fixed_whole_file_expensive"][name] for r in rows) / len(rows)
        for name in EXPENSIVE_FIXED
    }
    oracle_mean = sum(r["oracle_total"] for r in rows) / len(rows)

    best_cheap_name = min(fixed_cheap_mean, key=fixed_cheap_mean.__getitem__)
    best_cheap_bytes = fixed_cheap_mean[best_cheap_name]
    best_expensive_name = min(fixed_expensive_mean, key=fixed_expensive_mean.__getitem__)
    best_expensive_bytes = fixed_expensive_mean[best_expensive_name]

    oracle_vs_cheap = (best_cheap_bytes - oracle_mean) / best_cheap_bytes * 100.0
    oracle_vs_expensive = (best_expensive_bytes - oracle_mean) / best_expensive_bytes * 100.0

    print(
        f"  best fixed whole-file (cheap):     {best_cheap_name} = "
        f"{best_cheap_bytes:,.0f} B  ratio {best_cheap_bytes / mean_raw:.4f}"
    )
    print(
        f"  best fixed whole-file (expensive): {best_expensive_name} = "
        f"{best_expensive_bytes:,.0f} B  ratio {best_expensive_bytes / mean_raw:.4f}"
    )
    print(
        f"  oracle (per-block min + metadata): {oracle_mean:,.0f} B  "
        f"ratio {oracle_mean / mean_raw:.4f}"
    )
    print(
        f"  oracle savings vs best fixed CHEAP whole-file:     {oracle_vs_cheap:+.2f}%"
    )
    print(
        f"  oracle savings vs best fixed EXPENSIVE whole-file: {oracle_vs_expensive:+.2f}%"
    )

    # Verdict
    print("\n=== verdict ===")
    if oracle_vs_cheap < 3.0 and oracle_vs_expensive < 0:
        print(
            f"CLEAN-NEGATIVE (decisive). Oracle saves only {oracle_vs_cheap:.2f}% over"
        )
        print(
            f"  the best cheap fixed codec ({best_cheap_name}, under the 3% stop"
        )
        print(
            f"  rule), AND loses by {-oracle_vs_expensive:.1f}% to the best expensive"
        )
        print(
            f"  fixed codec ({best_expensive_name}). Any real storage system that"
        )
        print(
            "  can afford to run expensive fixed compression on a whole file"
        )
        print(
            "  beats a perfect per-block selector at cheap settings. Per-block"
        )
        print(
            "  codec selection has no practical headroom — the entire learned-"
        )
        print(
            "  selector research direction is foreclosed on this corpus mix."
        )
    elif oracle_vs_cheap < 3.0:
        print(
            f"CLEAN-NEGATIVE — oracle saves only {oracle_vs_cheap:.2f}% vs best cheap"
        )
        print(
            "  fixed whole-file codec, under the 3% stop-rule floor. With"
        )
        print(
            "  PERFECT per-block codec selection from the cheap pool, the"
        )
        print(
            "  achievable savings do not justify the complexity of a learned"
        )
        print(
            "  selector. Any real selector would do worse."
        )
    elif oracle_vs_cheap >= 3.0:
        print(
            f"NOT FALSIFIED — oracle saves {oracle_vs_cheap:.1f}% vs best cheap fixed."
        )
        print(
            "  The composition's premise has measurable headroom from the"
        )
        print(
            "  cheap codec pool. Remaining question: does a learned selector"
        )
        print(
            "  capture most of this gap from cheap block features, and does"
        )
        print(
            f"  the gap survive the expensive-fixed comparison ({oracle_vs_expensive:+.1f}%)?"
        )
    return 0


if __name__ == "__main__":
    sys.exit(main())
