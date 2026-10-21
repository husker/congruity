"""Heilbronn objective: min triangle area over all C(n,3) triples.

Vectorized via numpy. For n=30 we have C(30,3)=4060 triangles; for n=50 we have 19600.
All small enough to evaluate in microseconds.
"""

from __future__ import annotations

import itertools

import numpy as np


def min_triangle_area(points: np.ndarray) -> float:
    """Return the min triangle area over all triples in `points`.

    points: shape [n, 2], real-valued. Points outside [0,1]² are allowed
    (callers must clip if they need the standard Heilbronn objective).
    """
    n = points.shape[0]
    if n < 3:
        raise ValueError(f"need >=3 points, got {n}")
    # Build index combinations
    idx = np.array(list(itertools.combinations(range(n), 3)), dtype=np.int32)
    a = points[idx[:, 0]]
    b = points[idx[:, 1]]
    c = points[idx[:, 2]]
    # Signed area = 0.5 * | (b-a) × (c-a) |
    cross = (b[:, 0] - a[:, 0]) * (c[:, 1] - a[:, 1]) - (b[:, 1] - a[:, 1]) * (c[:, 0] - a[:, 0])
    return float(0.5 * np.min(np.abs(cross)))


def all_triangle_areas(points: np.ndarray) -> np.ndarray:
    """All C(n,3) absolute triangle areas (useful for soft-min objectives)."""
    n = points.shape[0]
    idx = np.array(list(itertools.combinations(range(n), 3)), dtype=np.int32)
    a = points[idx[:, 0]]
    b = points[idx[:, 1]]
    c = points[idx[:, 2]]
    cross = (b[:, 0] - a[:, 0]) * (c[:, 1] - a[:, 1]) - (b[:, 1] - a[:, 1]) * (c[:, 0] - a[:, 0])
    return 0.5 * np.abs(cross)


def is_in_unit_square(points: np.ndarray, tol: float = 1e-9) -> bool:
    return bool(((points >= -tol) & (points <= 1.0 + tol)).all())


def clip_to_unit_square(points: np.ndarray) -> np.ndarray:
    return np.clip(points, 0.0, 1.0)


if __name__ == "__main__":
    # Sanity: 4 corners of unit square → min triangle area = 0.5 (half the square)
    corners = np.array([[0, 0], [1, 0], [0, 1], [1, 1]], dtype=float)
    print(f"4 corners: min area = {min_triangle_area(corners):.6f} (expected 0.5)")

    # 6 known optimal: 1/8 = 0.125 (Yang et al.)
    rng = np.random.default_rng(0)
    random6 = rng.random((6, 2))
    print(f"6 random:  min area = {min_triangle_area(random6):.6f}")
