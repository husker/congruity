"""C3: PCA-banded PQ with non-uniform bit allocation.

Two variants:
    Variance:  bits per band ∝ log(eigenvalue) — JPEG-table style folklore
    Utility:   bits per band ∝ per-band recall@10 contribution from a
               held-out calibration query set — the C3-specific claim

The L3 reviewer predicts:
    - Variance allocation ≈ folklore, will probably ≈ uniform PQ
    - Utility allocation needs to beat Variance to show C3 has signal
"""

from __future__ import annotations

import numpy as np
from sklearn.cluster import MiniBatchKMeans
from sklearn.decomposition import PCA


def _train_band_pq(
    band_data: np.ndarray, m: int, k: int = 256, seed: int = 0
) -> tuple[list[np.ndarray], int]:
    """Train PQ on one band. Returns (codebooks, dim_per_sub). m must divide band dim."""
    d = band_data.shape[1]
    if m == 0:
        return [], d
    if d % m != 0:
        # Adjust m to nearest divisor
        while m > 1 and d % m != 0:
            m -= 1
    dim_per_sub = d // m
    codebooks = []
    for j in range(m):
        sub = band_data[:, j * dim_per_sub:(j + 1) * dim_per_sub]
        km = MiniBatchKMeans(n_clusters=k, batch_size=256, n_init=3, random_state=seed + j)
        km.fit(sub)
        codebooks.append(km.cluster_centers_.astype(np.float32))
    return codebooks, dim_per_sub


def _encode_band(band: np.ndarray, codebooks, dim_per_sub: int) -> np.ndarray:
    """Encode a band's vectors to codebook indices."""
    if not codebooks:
        return np.zeros((band.shape[0], 0), dtype=np.uint16)
    m = len(codebooks)
    n = band.shape[0]
    codes = np.empty((n, m), dtype=np.uint16)
    for j in range(m):
        sub = band[:, j * dim_per_sub:(j + 1) * dim_per_sub]
        cb = codebooks[j]
        d2 = ((sub[:, None, :] - cb[None, :, :]) ** 2).sum(axis=2)
        codes[:, j] = d2.argmin(axis=1)
    return codes


def _reconstruct_band(codes: np.ndarray, codebooks, dim_per_sub: int) -> np.ndarray:
    """Decode codes back to float."""
    if not codebooks:
        n = codes.shape[0]
        return np.zeros((n, 0), dtype=np.float32)
    m = len(codebooks)
    n = codes.shape[0]
    d = m * dim_per_sub
    out = np.empty((n, d), dtype=np.float32)
    for j in range(m):
        cb = codebooks[j]
        out[:, j * dim_per_sub:(j + 1) * dim_per_sub] = cb[codes[:, j]]
    return out


class PCABandedPQ:
    """PCA → equal-sized bands → PQ per band with non-uniform m allocation.

    `allocation_mode`:
        "variance": bits proportional to log(eigenvalue)
        "utility":  bits proportional to held-out per-band recall@10 contribution
        "equal":    bits equal across bands (sanity baseline)
    """

    def __init__(
        self,
        n_bands: int = 4,
        total_m: int = 8,
        k: int = 256,
        allocation_mode: str = "variance",
        name: str | None = None,
    ):
        self.n_bands = n_bands
        self.total_m = total_m
        self.k = k
        self.allocation_mode = allocation_mode
        self.name = name or f"PCA-PQ-{allocation_mode}(b={n_bands},m={total_m})"

    def fit(self, db: np.ndarray, calib_queries: np.ndarray | None = None) -> None:
        """Train PCA, allocate bits per band, train per-band PQ codebooks.

        For 'utility' mode, callers must pass calib_queries (a held-out
        slice of the db) so we can measure per-band recall@10 contribution.
        """
        d = db.shape[1]
        assert d % self.n_bands == 0, f"dim {d} not divisible by n_bands {self.n_bands}"
        self.dim_per_band = d // self.n_bands

        # PCA on db, project, split into bands
        self.pca = PCA(n_components=d, random_state=0).fit(db)
        proj = self.pca.transform(db).astype(np.float32)
        self.bands = [
            proj[:, b * self.dim_per_band:(b + 1) * self.dim_per_band]
            for b in range(self.n_bands)
        ]
        eigvals = self.pca.explained_variance_

        # Allocate m per band
        if self.allocation_mode == "equal":
            ms = [max(1, self.total_m // self.n_bands)] * self.n_bands
        elif self.allocation_mode == "variance":
            band_var = np.array([
                eigvals[b * self.dim_per_band:(b + 1) * self.dim_per_band].sum()
                for b in range(self.n_bands)
            ])
            log_var = np.log(band_var + 1e-9)
            log_var -= log_var.min()  # shift so smallest is 0
            weights = log_var + 0.5  # floor so no band gets 0
            ms = self._weights_to_ms(weights)
        elif self.allocation_mode == "utility":
            if calib_queries is None:
                raise ValueError("utility mode requires calib_queries")
            utilities = self._measure_band_utility(db, calib_queries)
            weights = np.maximum(utilities - utilities.min() + 0.5, 0.1)
            ms = self._weights_to_ms(weights)
        else:
            raise ValueError(f"unknown allocation_mode {self.allocation_mode}")
        self.ms = ms

        # Train PQ per band with the allocated m
        self.codebooks_per_band: list[tuple[list, int]] = []
        for b in range(self.n_bands):
            m = ms[b]
            if m == 0:
                self.codebooks_per_band.append(([], self.dim_per_band))
                continue
            cb, dps = _train_band_pq(self.bands[b], m=m, k=self.k, seed=42 + b)
            self.codebooks_per_band.append((cb, dps))

    def _weights_to_ms(self, weights: np.ndarray) -> list[int]:
        """Distribute `total_m` among bands by `weights`, ensuring each m divides dim_per_band."""
        n = self.n_bands
        # Allocate floor(total_m * w / sum), then top-up by remainder
        share = self.total_m * weights / max(weights.sum(), 1e-9)
        ms = np.floor(share).astype(int)
        remainder = self.total_m - ms.sum()
        # Distribute remainder to bands with highest fractional part
        frac = share - ms
        order = np.argsort(-frac)
        for i in range(remainder):
            ms[order[i % n]] += 1
        # Each m must divide dim_per_band; if not, round down to nearest divisor
        for b in range(n):
            m = ms[b]
            if m > self.dim_per_band:
                m = self.dim_per_band
            while m > 1 and self.dim_per_band % m != 0:
                m -= 1
            ms[b] = m
        return ms.tolist()

    def _measure_band_utility(self, db: np.ndarray, calib_queries: np.ndarray) -> np.ndarray:
        """Per-band recall@10 contribution when only that band's PCA dims are kept.

        Compute exact top-10 over full PCA-projected db. Then for each band b,
        compute top-10 using only band b's coords (other bands zeroed). The
        intersection-over-10 is band b's standalone retrieval utility.
        """
        proj_db = self.pca.transform(db).astype(np.float32)
        proj_q = self.pca.transform(calib_queries).astype(np.float32)
        # Normalize for cosine
        proj_db_n = proj_db / (np.linalg.norm(proj_db, axis=1, keepdims=True) + 1e-9)
        proj_q_n = proj_q / (np.linalg.norm(proj_q, axis=1, keepdims=True) + 1e-9)
        sims = proj_q_n @ proj_db_n.T
        exact_top = np.argpartition(-sims, 10, axis=1)[:, :10]

        utilities = []
        for b in range(self.n_bands):
            mask = np.zeros(proj_db.shape[1], dtype=np.float32)
            mask[b * self.dim_per_band:(b + 1) * self.dim_per_band] = 1.0
            d_band = proj_db * mask
            q_band = proj_q * mask
            d_band_n = d_band / (np.linalg.norm(d_band, axis=1, keepdims=True) + 1e-9)
            q_band_n = q_band / (np.linalg.norm(q_band, axis=1, keepdims=True) + 1e-9)
            sims_b = q_band_n @ d_band_n.T
            approx_top = np.argpartition(-sims_b, 10, axis=1)[:, :10]
            overlap = np.mean([
                len(set(e).intersection(set(a))) / 10
                for e, a in zip(exact_top, approx_top)
            ])
            utilities.append(float(overlap))
        return np.array(utilities)

    def encode(self, x: np.ndarray) -> np.ndarray:
        """Encode each band separately. Returns concatenated codes (max-padded)."""
        proj = self.pca.transform(x).astype(np.float32)
        all_codes = []
        for b in range(self.n_bands):
            band = proj[:, b * self.dim_per_band:(b + 1) * self.dim_per_band]
            cb, dps = self.codebooks_per_band[b]
            codes = _encode_band(band, cb, dps)
            all_codes.append(codes)
        return np.concatenate(all_codes, axis=1)

    def reconstruct(self, codes: np.ndarray) -> np.ndarray:
        """Decode codes into reconstructed PCA-space vector, then un-rotate."""
        n = codes.shape[0]
        out = np.empty((n, self.n_bands * self.dim_per_band), dtype=np.float32)
        offset = 0
        for b in range(self.n_bands):
            cb, dps = self.codebooks_per_band[b]
            m = self.ms[b]
            band_codes = codes[:, offset:offset + m]
            offset += m
            band_recon = _reconstruct_band(band_codes, cb, dps)
            # Pad with zeros if m < dim_per_band / dps
            if band_recon.shape[1] < self.dim_per_band:
                pad = np.zeros(
                    (n, self.dim_per_band - band_recon.shape[1]), dtype=np.float32
                )
                band_recon = np.concatenate([band_recon, pad], axis=1)
            out[:, b * self.dim_per_band:(b + 1) * self.dim_per_band] = band_recon
        return self.pca.inverse_transform(out).astype(np.float32)

    @property
    def bits_per_vector(self) -> int:
        return int(sum(self.ms) * np.ceil(np.log2(self.k)))
