"""CMA-ES baseline for the Heilbronn small-points problem.

We flatten the [n, 2] point coordinates into a length-2n vector in
[0, 1]^(2n) and let CMA-ES maximize min_triangle_area by minimizing
its negative. Each candidate vector is clipped to the unit square
before evaluation; the area we report is the true (non-smoothed)
minimum over all triplets.

CMA-ES is a strong derivative-free global optimizer that typically
beats hand-tuned SA on smooth, low-dimensional landscapes. It serves
here as a "tough but standard" baseline that the C1 gradient-flow
method must clear to claim a real algorithmic win.
"""

from __future__ import annotations

import cma
import numpy as np

from experiments.heilbronn.evaluator import min_triangle_area


def _objective(flat: np.ndarray, n_points: int) -> float:
    pts = np.clip(flat.reshape(n_points, 2), 0.0, 1.0)
    return -float(min_triangle_area(pts))


def cmaes_run(
    n_points: int,
    n_steps: int = 50_000,
    n_starts: int = 8,
    seed: int = 0,
    sigma0: float = 0.15,
) -> tuple[np.ndarray, float]:
    """Multi-restart CMA-ES on the Heilbronn objective.

    n_steps is interpreted as the total function-evaluation budget across
    all restarts. Each restart gets n_steps // n_starts evaluations.

    Returns (best_pts, best_area). Matches the signature used by the
    other Heilbronn baselines so the bench can swap them freely.
    """
    budget_per_start = max(100, n_steps // n_starts)
    best_pts: np.ndarray | None = None
    best_area = 0.0
    rng = np.random.default_rng(seed)
    for s in range(n_starts):
        x0 = rng.random(2 * n_points)
        opts = {
            "seed": int(seed + s) + 1,  # cma rejects seed=0
            "maxfevals": budget_per_start,
            "bounds": [0.0, 1.0],
            "verbose": -9,
            "verb_disp": 0,
            "verb_log": 0,
            "verb_filenameprefix": "/tmp/cma_",
        }
        es = cma.CMAEvolutionStrategy(x0, sigma0, opts)
        es.optimize(lambda v: _objective(v, n_points))
        flat = np.asarray(es.result.xbest if es.result.xbest is not None else x0)
        pts = np.clip(flat.reshape(n_points, 2), 0.0, 1.0)
        area = min_triangle_area(pts)
        if area > best_area:
            best_area = area
            best_pts = pts
    if best_pts is None:
        best_pts = rng.random((n_points, 2))
    return best_pts, best_area
