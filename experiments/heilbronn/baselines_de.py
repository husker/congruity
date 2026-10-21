"""Differential-evolution baseline for the Heilbronn small-points problem.

Uses scipy.optimize.differential_evolution over the flattened point
coordinates with unit-square bounds. DE is a standard derivative-free
optimizer that should beat naive multi-restart SA on smooth landscapes
and serves as a second "standard tough" baseline alongside CMA-ES.
"""

from __future__ import annotations

import numpy as np
from scipy.optimize import differential_evolution

from experiments.heilbronn.evaluator import min_triangle_area


def _objective(flat: np.ndarray, n_points: int) -> float:
    pts = np.clip(flat.reshape(n_points, 2), 0.0, 1.0)
    return -float(min_triangle_area(pts))


def de_run(
    n_points: int,
    n_steps: int = 50_000,
    n_starts: int = 8,
    seed: int = 0,
) -> tuple[np.ndarray, float]:
    """Multi-restart differential-evolution.

    Each restart gets `n_steps // n_starts` function-evaluation budget.
    `n_starts > 1` is useful because DE is sometimes seed-sensitive on
    this objective.

    Returns (best_pts, best_area).
    """
    budget_per_start = max(100, n_steps // n_starts)
    bounds = [(0.0, 1.0)] * (2 * n_points)
    best_pts: np.ndarray | None = None
    best_area = 0.0
    for s in range(n_starts):
        result = differential_evolution(
            lambda v: _objective(v, n_points),
            bounds=bounds,
            seed=int(seed + s),
            maxiter=max(1, budget_per_start // (15 * 2 * n_points)),
            popsize=15,
            tol=1e-12,
            polish=True,
            updating="deferred",
        )
        pts = np.clip(result.x.reshape(n_points, 2), 0.0, 1.0)
        area = min_triangle_area(pts)
        if area > best_area:
            best_area = area
            best_pts = pts
    if best_pts is None:
        rng = np.random.default_rng(seed)
        best_pts = rng.random((n_points, 2))
    return best_pts, best_area
