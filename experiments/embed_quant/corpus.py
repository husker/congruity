"""Embedding corpus for vector quantization benchmark.

We embed a small public text corpus with sentence-transformers MiniLM-L6-v2
(384-dimensional, fast, ~80MB model). Embeddings cached to disk so we
don't re-embed every run.

The corpus is 20 Newsgroups (~5k random documents) — public, diverse,
reasonable proxy for general-purpose sentence embeddings used in retrieval.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

CACHE_PATH = Path("data/embed_quant/minilm_embeds.npy")


def build_or_load(n_docs: int = 5000, seed: int = 0) -> np.ndarray:
    """Load cached embeddings or build them. Returns float32 [n_docs, 384]."""
    if CACHE_PATH.exists():
        arr = np.load(CACHE_PATH)
        if arr.shape[0] >= n_docs:
            return arr[:n_docs]
    CACHE_PATH.parent.mkdir(parents=True, exist_ok=True)

    from sentence_transformers import SentenceTransformer
    from sklearn.datasets import fetch_20newsgroups

    rng = np.random.default_rng(seed)
    print("fetching 20newsgroups...", flush=True)
    ng = fetch_20newsgroups(
        subset="all",
        remove=("headers", "footers", "quotes"),
        random_state=seed,
    )
    docs = [d for d in ng.data if 50 <= len(d) <= 2000]
    rng.shuffle(docs)
    docs = docs[:n_docs]
    print(f"docs selected: {len(docs)}; embedding...", flush=True)

    model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
    embeds = model.encode(
        docs, batch_size=64, show_progress_bar=True, convert_to_numpy=True, normalize_embeddings=True
    ).astype(np.float32)
    np.save(CACHE_PATH, embeds)
    print(f"saved {embeds.shape} -> {CACHE_PATH}", flush=True)
    return embeds


def split(embeds: np.ndarray, n_query: int = 500, seed: int = 1) -> tuple[np.ndarray, np.ndarray]:
    """Split into database + queries; queries are held out from db."""
    n = embeds.shape[0]
    rng = np.random.default_rng(seed)
    idx = rng.permutation(n)
    q_idx = idx[:n_query]
    db_idx = idx[n_query:]
    return embeds[db_idx], embeds[q_idx]


if __name__ == "__main__":
    emb = build_or_load(n_docs=5000)
    print(f"shape: {emb.shape}  dtype: {emb.dtype}")
    print(f"norm range: [{np.linalg.norm(emb, axis=1).min():.3f}, "
          f"{np.linalg.norm(emb, axis=1).max():.3f}]")
    db, q = split(emb, n_query=500)
    print(f"db: {db.shape}  queries: {q.shape}")
