"""Tests for the epigraph max-min polish (experiments/heilbronn/exact_polish.py)."""

from __future__ import annotations

import numpy as np
import pytest

from experiments.heilbronn.evaluator import is_in_unit_square, min_triangle_area
from experiments.heilbronn.exact_polish import (
    exact_polish,
    load_karpov,
    signed_area_jacobian,
    signed_areas,
    triangle_index,
)


def test_signed_area_jacobian_matches_finite_differences():
    rng = np.random.default_rng(7)
    pts = rng.random((6, 2))
    idx = triangle_index(6)
    jac = signed_area_jacobian(pts, idx)
    eps = 1e-7
    flat = pts.ravel().copy()
    for col in range(flat.size):
        hi, lo = flat.copy(), flat.copy()
        hi[col] += eps
        lo[col] -= eps
        num = (
            signed_areas(hi.reshape(-1, 2), idx) - signed_areas(lo.reshape(-1, 2), idx)
        ) / (2 * eps)
        np.testing.assert_allclose(jac[:, col], num, atol=1e-8)


def test_perturbed_corners_recover_half():
    # n=4 optimum is the 4 corners (min triangle area 1/2). From a
    # perturbed square the polish must climb back to 0.5 exactly.
    rng = np.random.default_rng(3)
    corners = np.array([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0], [1.0, 1.0]])
    start = np.clip(corners + rng.uniform(-0.06, 0.06, size=(4, 2)), 0.0, 1.0)
    res = exact_polish(start)
    assert res.area >= 0.5 - 1e-9
    assert is_in_unit_square(res.points)


@pytest.mark.parametrize("seed", [0, 1, 2])
def test_polish_never_degrades_and_stays_in_box(seed):
    rng = np.random.default_rng(seed)
    start = rng.random((7, 2))
    start_area = min_triangle_area(start)
    res = exact_polish(start)
    assert res.area >= start_area - 1e-15
    assert res.start_area == pytest.approx(start_area)
    assert is_in_unit_square(res.points)
    # polished area is recomputed from coordinates, not solver-reported
    assert res.area == pytest.approx(min_triangle_area(res.points))


def test_polish_improves_a_random_start_substantially():
    # Random 7-point configs are far from stationary; one polish should
    # add a large margin (typical random min area ~1e-3, stationary ~1e-2+).
    rng = np.random.default_rng(11)
    start = rng.random((7, 2))
    res = exact_polish(start)
    assert res.gain > 1e-3


def test_karpov_s13_holds_under_polish():
    pts, measured = load_karpov(13)
    res = exact_polish(pts, max_rounds=4)
    # A record-grade config must never come out worse; tiny numerical
    # refinement is allowed (same basin), a big jump would mean a bug.
    assert res.area >= measured - 1e-12
