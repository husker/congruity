"""Embedding quantization benchmark.

For each method, encode the database, reconstruct, then for each query find
the top-k by cosine similarity using reconstructed vectors. Compare to
exact (float32) top-k. Report recall@k vs bits-per-vector.
"""

from __future__ import annotations

import sys
import time

import numpy as np

from experiments.embed_quant.baselines import make_baselines
from experiments.embed_quant.c3_pca_alloc import PCABandedPQ
from experiments.embed_quant.corpus import build_or_load, split


def topk_cosine(queries: np.ndarray, db: np.ndarray, k: int = 10) -> np.ndarray:
    """Return [n_queries, k] indices of top-k most similar db items per query.

    Assumes db and queries are L2-normalized so cosine ~ dot product.

    Note: np.argpartition(a, kth)[:kth+1] returns the k+1 smallest items.
    For top-k by descending sim, kth must be k-1.
    """
    # Normalize defensively
    q = queries / (np.linalg.norm(queries, axis=1, keepdims=True) + 1e-9)
    d = db / (np.linalg.norm(db, axis=1, keepdims=True) + 1e-9)
    sims = q @ d.T  # [nq, nd]
    return np.argpartition(-sims, k - 1, axis=1)[:, :k]


def recall_at_k(exact_topk: np.ndarray, approx_topk: np.ndarray) -> float:
    """Mean fraction of exact-top-k items that appear in approx-top-k."""
    overlaps = [len(set(e).intersection(set(a))) for e, a in zip(exact_topk, approx_topk)]
    k = exact_topk.shape[1]
    return float(np.mean(overlaps) / k)


def main(argv: list[str]) -> int:
    K = 10
    print("loading embeddings...", flush=True)
    emb = build_or_load(n_docs=5000)
    db, queries = split(emb, n_query=500)
    print(f"db: {db.shape}  queries: {queries.shape}  dim: {db.shape[1]}", flush=True)

    calib_q = db[-200:]  # held-out chunk treated as calibration queries
    main_db = db[:-200]  # train PQ codebooks here; targets for retrieval

    baselines = make_baselines(dim=main_db.shape[1])
    # Add C3 variants matched to PQ bit-budgets
    c3_methods = [
        PCABandedPQ(n_bands=4, total_m=4, allocation_mode="variance", name="C3-var-32b"),
        PCABandedPQ(n_bands=4, total_m=4, allocation_mode="utility", name="C3-util-32b"),
        PCABandedPQ(n_bands=4, total_m=8, allocation_mode="variance", name="C3-var-64b"),
        PCABandedPQ(n_bands=4, total_m=8, allocation_mode="utility", name="C3-util-64b"),
        PCABandedPQ(n_bands=4, total_m=16, allocation_mode="variance", name="C3-var-128b"),
        PCABandedPQ(n_bands=4, total_m=16, allocation_mode="utility", name="C3-util-128b"),
        PCABandedPQ(n_bands=4, total_m=32, allocation_mode="variance", name="C3-var-256b"),
        PCABandedPQ(n_bands=4, total_m=32, allocation_mode="utility", name="C3-util-256b"),
    ]
    # Compute exact top-k once — main_db and queries don't change inside the
    # method loop, so re-running this per method was wasted work.
    print("computing exact top-k over main_db...", flush=True)
    exact_main = topk_cosine(queries, main_db, k=K)

    print(f"\n{'method':22s} {'bits':>5s} {'recall@10':>10s} {'fit_s':>7s} {'enc_s':>7s}")
    print("-" * 60)
    for b in baselines:
        t0 = time.time()
        b.fit(main_db)
        fit_t = time.time() - t0
        t1 = time.time()
        db_codes = b.encode(main_db)
        db_recon = b.reconstruct(db_codes)
        enc_t = time.time() - t1
        approx = topk_cosine(queries, db_recon, k=K)
        r = recall_at_k(exact_main, approx)
        print(f"{b.name:22s} {b.bits_per_vector:>5d} {r:>10.4f} {fit_t:>7.2f} {enc_t:>7.2f}")
    print()
    for c3 in c3_methods:
        t0 = time.time()
        if c3.allocation_mode == "utility":
            c3.fit(main_db, calib_queries=calib_q)
        else:
            c3.fit(main_db)
        fit_t = time.time() - t0
        # Relabel with the actual bit count after divisor rounding inside
        # _weights_to_ms; the constructor's name encoded the nominal budget
        # (e.g. "C3-var-32b") but the realized budget can be smaller.
        actual_bits = c3.bits_per_vector
        nominal_name = c3.name
        prefix, _, _suffix = nominal_name.rpartition("-")
        c3.name = f"{prefix}-{actual_bits}b" if prefix else nominal_name
        t1 = time.time()
        db_codes = c3.encode(main_db)
        db_recon = c3.reconstruct(db_codes)
        enc_t = time.time() - t1
        approx = topk_cosine(queries, db_recon, k=K)
        r = recall_at_k(exact_main, approx)
        ms_str = "/".join(str(m) for m in c3.ms)
        print(f"{c3.name:22s} {actual_bits:>5d} {r:>10.4f} {fit_t:>7.2f} {enc_t:>7.2f}  ms={ms_str}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
