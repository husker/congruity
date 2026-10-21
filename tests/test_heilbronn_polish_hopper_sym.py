"""Tests for the symmetry-seeded hopper (experiments/heilbronn/polish_hopper_sym.py)."""

from __future__ import annotations

import numpy as np

from experiments.heilbronn.evaluator import is_in_unit_square
from experiments.heilbronn.polish_hopper_sym import (
    _mirror_start,
    _rect_start,
    _rot180_start,
    _symmetrize_mirror,
    _symmetrize_rot180,
    solve,
)


def test_symmetric_generators_have_their_symmetry():
    rng = np.random.default_rng(5)
    for n in (14, 16, 13):
        m = _mirror_start(rng, n)
        assert m.shape == (n, 2) and is_in_unit_square(m)
        mirrored = np.column_stack([1.0 - m[:, 0], m[:, 1]])
        assert _set_equal(m, mirrored)

        r = _rot180_start(rng, n)
        assert r.shape == (n, 2) and is_in_unit_square(r)
        assert _set_equal(r, 1.0 - r)

        q = _rect_start(rng, n)
        assert q.shape == (n, 2) and is_in_unit_square(q)
        assert _set_equal(q, np.column_stack([1.0 - q[:, 0], q[:, 1]]))
        assert _set_equal(q, np.column_stack([q[:, 0], 1.0 - q[:, 1]]))


def _set_equal(a: np.ndarray, b: np.ndarray, tol: float = 1e-9) -> bool:
    """Point sets equal as multisets (nearest-neighbor match)."""
    used: set[int] = set()
    for p in a:
        d = np.linalg.norm(b - p, axis=1)
        if used:
            d[list(used)] = np.inf
        j = int(np.argmin(d))
        if d[j] > tol:
            return False
        used.add(j)
    return True


def test_symmetrize_kicks_produce_symmetric_configs():
    rng = np.random.default_rng(9)
    pts = rng.random((14, 2))
    m = _symmetrize_mirror(pts)
    assert _set_equal(m, np.column_stack([1.0 - m[:, 0], m[:, 1]]))
    r = _symmetrize_rot180(pts)
    assert _set_equal(r, 1.0 - r)


def test_solve_deterministic_and_valid():
    a = solve(10, 3, 8000)
    b = solve(10, 3, 8000)
    assert np.array_equal(np.array(a), np.array(b))
    arr = np.array(a)
    assert arr.shape == (10, 2)
    assert is_in_unit_square(arr)
