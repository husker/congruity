"""Top-k correctness test for the embed_quant bench.

np.argpartition(a, kth) returns an array such that a[kth] is in its sorted
position and elements 0..kth-1 are <= a[kth]. So the top-k smallest are
a[:kth+1]. For top-k largest (most-similar), we negate and ask for the
(k-1)-th partition; the result's first k elements are the top-k.

Catches the off-by-one (kth=k vs kth=k-1) flagged in the deep-analysis.
"""

from __future__ import annotations

import numpy as np

from experiments.embed_quant.bench import topk_cosine


def _topk_argsort(sims: np.ndarray, k: int) -> np.ndarray:
    """Brute-force top-k by full sort. Slow but unambiguously correct."""
    return np.argsort(-sims, axis=1)[:, :k]


def test_argpartition_topk_matches_argsort_topk_as_set():
    rng = np.random.default_rng(0)
    sims = rng.standard_normal((10, 100))
    k = 5
    part = np.argpartition(-sims, k - 1, axis=1)[:, :k]
    full = _topk_argsort(sims, k)
    for row_part, row_full in zip(part, full):
        assert set(row_part.tolist()) == set(row_full.tolist())


def test_topk_cosine_matches_brute_force():
    rng = np.random.default_rng(0)
    n_queries, n_db, dim = 5, 50, 16
    queries = rng.standard_normal((n_queries, dim)).astype(np.float32)
    db = rng.standard_normal((n_db, dim)).astype(np.float32)

    k = 10
    fast = topk_cosine(queries, db, k=k)

    # Brute-force: compute all cosines, sort, take top-k
    q_norm = queries / np.linalg.norm(queries, axis=1, keepdims=True)
    d_norm = db / np.linalg.norm(db, axis=1, keepdims=True)
    sims = q_norm @ d_norm.T
    brute = _topk_argsort(sims, k)

    for row_fast, row_brute in zip(fast, brute):
        assert set(row_fast.tolist()) == set(row_brute.tolist())


def test_topk_cosine_k_equals_one():
    rng = np.random.default_rng(1)
    queries = rng.standard_normal((3, 8)).astype(np.float32)
    db = rng.standard_normal((20, 8)).astype(np.float32)
    fast = topk_cosine(queries, db, k=1)
    q_norm = queries / np.linalg.norm(queries, axis=1, keepdims=True)
    d_norm = db / np.linalg.norm(db, axis=1, keepdims=True)
    sims = q_norm @ d_norm.T
    brute = _topk_argsort(sims, k=1)
    np.testing.assert_array_equal(fast, brute)
