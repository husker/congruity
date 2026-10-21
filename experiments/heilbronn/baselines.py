"""Heilbronn baseline optimizers.

We implement two reference methods:
    sa_uniform     — simulated annealing from random uniform init
    sa_lds         — SA from low-discrepancy (Halton/Sobol-like) init

Each runs `n_starts` independent SA passes and reports the best result.
Per-pass budget is `n_steps`. Use these as the floor to beat.

Each method exposes:
    .run(n_points, n_steps, n_starts, seed) -> (points, area)
"""

from __future__ import annotations

import itertools
import math
import time

import numpy as np

from experiments.heilbronn.evaluator import min_triangle_area


def _halton_2d(n: int, base1: int = 2, base2: int = 3) -> np.ndarray:
    """First-n Halton low-discrepancy sequence in [0,1]²."""
    def vdc(i: int, b: int) -> float:
        q = 0.0
        bk = 1.0 / b
        while i > 0:
            q += (i % b) * bk
            i //= b
            bk /= b
        return q

    return np.array([[vdc(i + 1, base1), vdc(i + 1, base2)] for i in range(n)])


def sa_run(
    n_points: int,
    n_steps: int = 50_000,
    seed: int = 0,
    init: str = "uniform",
    T_start: float = 0.05,
    T_end: float = 1e-5,
) -> tuple[np.ndarray, float]:
    """Single SA pass: perturb one point, accept by Metropolis on min-area objective.

    init: 'uniform' (random) or 'halton' (low-discrepancy).
    """
    rng = np.random.default_rng(seed)
    if init == "halton":
        pts = _halton_2d(n_points)
    else:
        pts = rng.random((n_points, 2))
    best_pts = pts.copy()
    best_area = min_triangle_area(pts)
    cur_area = best_area
    # Geometric cooling
    log_ratio = math.log(T_end / T_start)
    for step in range(n_steps):
        T = T_start * math.exp(log_ratio * step / max(1, n_steps - 1))
        # Perturb one point
        i = rng.integers(0, n_points)
        old = pts[i].copy()
        sigma = max(0.005, 0.15 * T / T_start)
        pts[i] = np.clip(pts[i] + rng.normal(0, sigma, 2), 0.0, 1.0)
        new_area = min_triangle_area(pts)
        d = new_area - cur_area
        if d > 0 or rng.random() < math.exp(d / T):
            cur_area = new_area
            if new_area > best_area:
                best_area = new_area
                best_pts = pts.copy()
        else:
            pts[i] = old
    return best_pts, best_area


def sa_uniform(
    n_points: int,
    n_steps: int = 50_000,
    n_starts: int = 8,
    seed: int = 0,
) -> tuple[np.ndarray, float]:
    """Multi-restart SA from random uniform initializations."""
    best_pts: np.ndarray | None = None
    best_area = 0.0
    for s in range(n_starts):
        pts, area = sa_run(n_points, n_steps=n_steps, seed=seed + s, init="uniform")
        if area > best_area:
            best_area = area
            best_pts = pts
    return best_pts, best_area  # type: ignore


def sa_halton(
    n_points: int,
    n_steps: int = 50_000,
    n_starts: int = 8,
    seed: int = 0,
) -> tuple[np.ndarray, float]:
    """Multi-restart SA from low-discrepancy Halton initializations.

    The Halton init only matters for the first run; subsequent restarts
    add small jitter to break ties.
    """
    best_pts: np.ndarray | None = None
    best_area = 0.0
    for s in range(n_starts):
        if s == 0:
            pts, area = sa_run(n_points, n_steps=n_steps, seed=seed + s, init="halton")
        else:
            pts, area = sa_run(n_points, n_steps=n_steps, seed=seed + s, init="uniform")
        if area > best_area:
            best_area = area
            best_pts = pts
    return best_pts, best_area  # type: ignore


def scipy_basinhopping(
    n_points: int,
    n_steps: int = 50_000,
    n_starts: int = 8,
    seed: int = 0,
) -> tuple[np.ndarray, float]:
    """SciPy basin-hopping with L-BFGS-B local minimizer.

    Stronger baseline than naive SA: combines random hops with proper local
    gradient descent on a smoothed objective. We use logsumexp-softmin so
    L-BFGS gets a useful gradient. `n_steps * n_starts / 1000` outer iters,
    each with default ~100-step L-BFGS inner.
    """
    from scipy.optimize import basinhopping

    rng = np.random.default_rng(seed)
    triplets = np.array(list(itertools.combinations(range(n_points), 3)), dtype=np.int32)

    def neg_softmin(flat_pts, tau=200.0):
        pts = flat_pts.reshape(n_points, 2)
        # Penalize outside [0,1]
        outside = np.clip(-pts, 0, None).sum() + np.clip(pts - 1, 0, None).sum()
        if outside > 0:
            return 1e3 * outside
        a = pts[triplets[:, 0]]
        b = pts[triplets[:, 1]]
        c = pts[triplets[:, 2]]
        cross = (b[:, 0] - a[:, 0]) * (c[:, 1] - a[:, 1]) - (b[:, 1] - a[:, 1]) * (
            c[:, 0] - a[:, 0]
        )
        areas = 0.5 * np.abs(cross) + 1e-10
        x = -tau * areas
        m = x.max()
        return (m + np.log(np.exp(x - m).sum())) / tau

    # Outer-loop budget: roughly map (n_starts × n_steps) to basin-hopping outer steps.
    outer_steps = max(1, n_starts * n_steps // 500)
    best_pts = None
    best_area = 0.0
    for s in range(min(8, n_starts)):
        x0 = rng.random(n_points * 2)
        try:
            result = basinhopping(
                neg_softmin,
                x0,
                minimizer_kwargs={"method": "L-BFGS-B"},
                niter=outer_steps // max(1, min(8, n_starts)),
                seed=seed + s,
                disp=False,
            )
            pts = np.clip(result.x, 0, 1)
            area = min_triangle_area(pts.reshape(n_points, 2))
            if area > best_area:
                best_area = area
                best_pts = pts.reshape(n_points, 2)
        except Exception:
            continue
    return best_pts, best_area  # type: ignore


if __name__ == "__main__":
    for n in (6, 8, 10, 12, 15):
        t0 = time.time()
        _, area = sa_uniform(n, n_steps=10_000, n_starts=4, seed=42)
        print(f"n={n:3d}  area={area:.5f}  ({time.time() - t0:.1f}s)")
