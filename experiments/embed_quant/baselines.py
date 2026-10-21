"""Embedding quantization baselines.

All baselines expose:
    .fit(db_vectors)             # train codebooks / params
    .encode(vectors) -> bytes    # quantize
    .reconstruct(codes) -> vectors  # decode back to float
    .bits_per_vector             # int

Recall is measured by reconstructing quantized vectors and doing exact
brute-force top-k search on the reconstructions. This matches how PQ-based
ANN libraries (faiss) work under the hood (modulo asymmetric distance
computation which we skip for simplicity).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class Float32:
    """No-op baseline: store original float32 (1536 bits for 384d)."""

    name: str = "float32"

    def fit(self, db: np.ndarray) -> None:
        self.dim = db.shape[1]

    def encode(self, x: np.ndarray) -> np.ndarray:
        return x.astype(np.float32)

    def reconstruct(self, codes: np.ndarray) -> np.ndarray:
        return codes

    @property
    def bits_per_vector(self) -> int:
        return self.dim * 32


class ScalarInt8:
    """Scalar quantization: each dim independently mapped to [-128, 127]."""

    name = "scalar-int8"

    def fit(self, db: np.ndarray) -> None:
        self.dim = db.shape[1]
        # Per-dimension min/max from training data
        self.lo = db.min(axis=0)
        self.hi = db.max(axis=0)
        self.scale = (self.hi - self.lo) / 255.0
        self.scale[self.scale == 0] = 1.0

    def encode(self, x: np.ndarray) -> np.ndarray:
        u = (x - self.lo) / self.scale  # 0..255
        return np.clip(np.round(u), 0, 255).astype(np.uint8)

    def reconstruct(self, codes: np.ndarray) -> np.ndarray:
        return (codes.astype(np.float32) * self.scale + self.lo).astype(np.float32)

    @property
    def bits_per_vector(self) -> int:
        return self.dim * 8


class ScalarInt4:
    """Scalar quantization to 4 bits per dim."""

    name = "scalar-int4"

    def fit(self, db: np.ndarray) -> None:
        self.dim = db.shape[1]
        self.lo = db.min(axis=0)
        self.hi = db.max(axis=0)
        self.scale = (self.hi - self.lo) / 15.0
        self.scale[self.scale == 0] = 1.0

    def encode(self, x: np.ndarray) -> np.ndarray:
        u = (x - self.lo) / self.scale
        return np.clip(np.round(u), 0, 15).astype(np.uint8)

    def reconstruct(self, codes: np.ndarray) -> np.ndarray:
        return (codes.astype(np.float32) * self.scale + self.lo).astype(np.float32)

    @property
    def bits_per_vector(self) -> int:
        return self.dim * 4


class ProductQuantizer:
    """Product Quantization (Jégou et al.).

    Vector split into `m` subvectors. Each subspace gets a `k`-entry codebook
    via k-means. Each vector encoded as m indices, each log2(k) bits.
    """

    def __init__(self, m: int = 8, k: int = 256, name: str | None = None):
        self.m = m
        self.k = k
        self.name = name or f"PQ(m={m},k={k})"
        self.codebooks: list[np.ndarray] = []
        self.dim_per_sub: int | None = None

    def fit(self, db: np.ndarray) -> None:
        from sklearn.cluster import MiniBatchKMeans

        n, d = db.shape
        assert d % self.m == 0, f"dim {d} not divisible by m={self.m}"
        self.dim_per_sub = d // self.m
        self.codebooks = []
        for j in range(self.m):
            sub = db[:, j * self.dim_per_sub:(j + 1) * self.dim_per_sub]
            km = MiniBatchKMeans(
                n_clusters=self.k, batch_size=256, n_init=3, random_state=42 + j
            )
            km.fit(sub)
            self.codebooks.append(km.cluster_centers_.astype(np.float32))

    def encode(self, x: np.ndarray) -> np.ndarray:
        n = x.shape[0]
        codes = np.empty((n, self.m), dtype=np.uint16 if self.k > 256 else np.uint8)
        for j in range(self.m):
            sub = x[:, j * self.dim_per_sub:(j + 1) * self.dim_per_sub]
            cb = self.codebooks[j]
            # Find nearest centroid for each sub-vector
            d2 = ((sub[:, None, :] - cb[None, :, :]) ** 2).sum(axis=2)
            codes[:, j] = d2.argmin(axis=1)
        return codes

    def reconstruct(self, codes: np.ndarray) -> np.ndarray:
        n = codes.shape[0]
        d = self.m * self.dim_per_sub
        out = np.empty((n, d), dtype=np.float32)
        for j in range(self.m):
            cb = self.codebooks[j]
            out[:, j * self.dim_per_sub:(j + 1) * self.dim_per_sub] = cb[codes[:, j]]
        return out

    @property
    def bits_per_vector(self) -> int:
        return int(self.m * np.ceil(np.log2(self.k)))


class BinaryHash:
    """Trivial 1-bit-per-dim binary hash: sign(x - mean)."""

    name = "binary-1bit"

    def fit(self, db: np.ndarray) -> None:
        self.dim = db.shape[1]
        self.mean = db.mean(axis=0)

    def encode(self, x: np.ndarray) -> np.ndarray:
        return (x > self.mean).astype(np.uint8)

    def reconstruct(self, codes: np.ndarray) -> np.ndarray:
        # Sign of (codes - 0.5) → ±1, scaled to match data std loosely
        std = 1.0
        return ((codes.astype(np.float32) - 0.5) * 2 * std).astype(np.float32)

    @property
    def bits_per_vector(self) -> int:
        return self.dim


def make_baselines(dim: int = 384) -> list:
    return [
        Float32(),
        ScalarInt8(),
        ScalarInt4(),
        BinaryHash(),
        ProductQuantizer(m=4, k=256, name="PQ-32b"),    # 32 bits/vec
        ProductQuantizer(m=8, k=256, name="PQ-64b"),    # 64 bits/vec
        ProductQuantizer(m=16, k=256, name="PQ-128b"),  # 128 bits/vec
        ProductQuantizer(m=32, k=256, name="PQ-256b"),  # 256 bits/vec
    ]
