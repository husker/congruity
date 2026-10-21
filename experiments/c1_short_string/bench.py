"""Run all methods on all corpora and print a comparison table.

Usage:
    uv run --with brotli --with zstandard python -m experiments.c1_short_string.bench

Accounting: trained-dict and template-table sizes are NOT charged per record
(deployed once, amortised). Per-record bytes = compressed output length.
The ratio reported is sum(encoded) / sum(raw). Average is over records, not
weighted by length, so we also report per-corpus medians for context.
"""

from __future__ import annotations

import statistics
import sys

from experiments.c1_short_string.baselines import Method, all_methods_with_hpack
from experiments.c1_short_string.c1 import C1Codec
from experiments.c1_short_string.c2 import C2Codec
from experiments.c1_short_string.c3 import C3Codec
from experiments.c1_short_string.corpora import (
    CORPORA,
    real_http_lines,
    real_http_lines_by_site_split,
    split,
)
# real_http_lines_by_site_split is used inside main(); listed explicitly so
# autofixers don't strip it before the usage site is parsed.
_KEEP = (real_http_lines_by_site_split,)


def bench_method(name: str, encode, records: list[bytes]) -> dict:
    out = [encode(r) for r in records]
    raw_total = sum(len(r) for r in records)
    enc_total = sum(len(e) for e in out)
    ratios = [len(e) / max(1, len(r)) for e, r in zip(out, records)]
    return {
        "name": name,
        "raw_total": raw_total,
        "enc_total": enc_total,
        "ratio_total": enc_total / max(1, raw_total),
        "ratio_median": statistics.median(ratios),
        "expansion_p99": sorted(ratios)[int(len(ratios) * 0.99)],
        "n": len(records),
    }


def run_split(train: list[bytes], test: list[bytes]) -> list[dict]:
    """Run all methods given a pre-computed train/test split."""
    methods = all_methods_with_hpack(train)
    c1 = C1Codec.train(train, max_templates=1024, w=4)
    methods.append(Method(name="C1-proto", encode=c1.encode))
    c3 = C3Codec.train(train)
    methods.append(Method(name="C3-proto", encode=c3.encode))
    c2 = C2Codec.train(train)
    methods.append(Method(name="C2-proto", encode=c2.encode))
    results = []
    for m in methods:
        r = bench_method(m.name, m.encode, test)
        results.append(r)
    for r in results:
        if r["name"] == "C1-proto":
            r["hits"] = c1.hit_count
            r["fallbacks"] = c1.fallback_count
    return results


def run_corpus(records: list[bytes]) -> list[dict]:
    train, test = split(records, train_frac=0.5)
    return run_split(train, test)


def print_table(corpus_name: str, rows: list[dict]) -> None:
    print(f"\n## {corpus_name}  (n={rows[0]['n']})")
    header = f"  {'method':22s} {'ratio_tot':>10s} {'ratio_med':>10s} {'p99_exp':>10s} {'note':<22s}"
    print(header)
    print("  " + "-" * (len(header) - 2))
    # Sort by total ratio
    for r in sorted(rows, key=lambda x: x["ratio_total"]):
        note = ""
        if r["name"] == "C1-proto":
            note = f"hits {r['hits']}/fb {r['fallbacks']}"
        print(f"  {r['name']:22s} {r['ratio_total']:10.4f} {r['ratio_median']:10.4f} {r['expansion_p99']:10.4f}  {note:<22s}")


def main(argv: list[str]) -> int:
    n_per_corpus = 4000 if "--small" not in argv else 1000
    all_results: dict[str, list[dict]] = {}
    # Synthetic corpora first, then real HTTP headers from the scrape
    for name, fn in CORPORA.items():
        records = fn(n=n_per_corpus)
        print(f"\nRunning {name} (n={len(records)})...", flush=True)
        all_results[name] = run_corpus(records)
        print_table(name, all_results[name])
    # real_http: line-level shuffled split (leakage potential — see real_http_site for fair split)
    real_records = real_http_lines(n=n_per_corpus)
    print(f"\nRunning real_http_shuffled (n={len(real_records)})...", flush=True)
    all_results["real_http_shuf"] = run_corpus(real_records)
    print_table("real_http_shuf", all_results["real_http_shuf"])

    # real_http_site: train and test from DISJOINT sets of sites — leakage-free
    train_lines, test_lines = real_http_lines_by_site_split(train_sites_frac=0.5)
    print(
        f"\nRunning real_http_site (train={len(train_lines)}, test={len(test_lines)})...",
        flush=True,
    )
    all_results["real_http_site"] = run_split(train_lines, test_lines)
    print_table("real_http_site", all_results["real_http_site"])

    # Summary across corpora
    print("\n\n## SUMMARY (total compressed / total raw, lower is better)")
    methods_in_order = [r["name"] for r in all_results[next(iter(all_results))]]
    print(f"  {'method':22s} " + " ".join(f"{c:>12s}" for c in all_results.keys()))
    for m in methods_in_order:
        row = []
        for corpus in all_results:
            hit = next((r for r in all_results[corpus] if r["name"] == m), None)
            row.append(f"{hit['ratio_total']:12.4f}" if hit else f"{'-':>12}")
        print(f"  {m:22s} " + " ".join(row))
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
