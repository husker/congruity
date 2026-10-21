"""C1: smooth-min + repulsion + boundary penalty, gradient flow.

The L3 reviewer's diagnosis: standard SA gets stuck because the min-triangle
objective is non-smooth — only the *worst* triangle's gradient matters.

C1's fix:
    loss = -softmin(τ, areas) + α * Riesz_repulsion + β * boundary_penalty
    softmin_τ(a) = -(1/τ) * log(sum(exp(-τ * a)))    → min(a) as τ → ∞

Anneal τ from low (smooth, all triangles contribute) to high (sharp min).
Anneal α from medium (push points apart early) to zero (let real objective dominate late).
β stays high throughout (keep points inside [0,1]²).
"""

from __future__ import annotations

import itertools
import time

import numpy as np


def _triplet_indices(n: int) -> np.ndarray:
    return np.array(list(itertools.combinations(range(n), 3)), dtype=np.int32)


def true_min_area(flat_pts: np.ndarray, n: int) -> float:
    triplets = _triplet_indices(n)
    pts = flat_pts.reshape(n, 2)
    a = pts[triplets[:, 0]]
    b = pts[triplets[:, 1]]
    c = pts[triplets[:, 2]]
    cross = (b[:, 0] - a[:, 0]) * (c[:, 1] - a[:, 1]) - (b[:, 1] - a[:, 1]) * (c[:, 0] - a[:, 0])
    return float(0.5 * np.min(np.abs(cross)))


def _make_value_and_grad(n: int):
    import jax
    import jax.numpy as jnp
    triplets = jnp.asarray(_triplet_indices(n))
    pair_indices = jnp.asarray(np.array(list(itertools.combinations(range(n), 2)), dtype=np.int32))
    eps = 1e-6

    def loss(flat_pts, tau, alpha, beta):
        pts = flat_pts.reshape(n, 2)
        a = pts[triplets[:, 0]]
        b = pts[triplets[:, 1]]
        c = pts[triplets[:, 2]]
        cross = (b[:, 0] - a[:, 0]) * (c[:, 1] - a[:, 1]) - (b[:, 1] - a[:, 1]) * (c[:, 0] - a[:, 0])
        areas = 0.5 * jnp.abs(cross) + 1e-10
        # -softmin_τ(a) = (1/τ) * logsumexp(-τ * areas), this is what we MINIMIZE
        # so the gradient pushes the smallest areas UP.
        x = -tau * areas
        m = jnp.max(x)
        neg_softmin = (m + jnp.log(jnp.sum(jnp.exp(x - m)))) / tau
        # Riesz s=1 (Coulomb) repulsion, log-clipped to avoid blowups when points coincide
        dp = pts[pair_indices[:, 0]] - pts[pair_indices[:, 1]]
        d2 = jnp.sum(dp * dp, axis=1) + eps
        repulsion = jnp.sum(1.0 / jnp.sqrt(d2))
        # Boundary penalty
        b_pen = jnp.sum(jnp.maximum(0.0, -pts) ** 2) + jnp.sum(jnp.maximum(0.0, pts - 1.0) ** 2)
        return neg_softmin + alpha * repulsion + beta * b_pen

    return jax.jit(jax.value_and_grad(loss, argnums=0))


def c1_run(
    n_points: int,
    n_steps: int = 5_000,
    n_restarts: int = 8,
    lr: float = 0.005,
    tau_start: float = 10.0,
    tau_end: float = 5_000.0,
    alpha_start: float = 1e-4,
    alpha_end: float = 1e-8,
    beta: float = 50.0,
    seed: int = 0,
    verbose: bool = False,
) -> tuple[np.ndarray, float]:
    import jax.numpy as jnp
    value_and_grad = _make_value_and_grad(n_points)
    rng = np.random.default_rng(seed)
    best_pts = None
    best_area = 0.0
    for r in range(n_restarts):
        pts = rng.random(n_points * 2).astype(np.float32)
        for step in range(n_steps):
            frac = step / max(1, n_steps - 1)
            tau = tau_start * (tau_end / tau_start) ** frac
            alpha_cur = alpha_start * (1 - frac) + alpha_end * frac
            v, g = value_and_grad(jnp.asarray(pts), float(tau), float(alpha_cur), float(beta))
            g = np.asarray(g)
            if not np.isfinite(g).all():
                # numerical blowup — reset
                pts = rng.random(n_points * 2).astype(np.float32)
                continue
            pts = pts - lr * g
            # Soft clip toward [0,1] every step
            pts = np.clip(pts, 0.0, 1.0)
        area = true_min_area(pts, n_points)
        if verbose:
            print(f"   restart {r}: area={area:.5f}")
        if area > best_area:
            best_area = area
            best_pts = pts.reshape(n_points, 2).copy()
    return best_pts, best_area


if __name__ == "__main__":
    for n in (8, 10, 12, 15):
        t0 = time.time()
        _, area = c1_run(n, n_steps=2000, n_restarts=4, seed=42, verbose=True)
        print(f"n={n:3d}  C1 best area={area:.5f}  ({time.time() - t0:.1f}s)")
