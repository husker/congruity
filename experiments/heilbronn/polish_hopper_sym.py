"""Symmetry-seeded variant of the polish_hopper candidate (v2 lane).

Motivation (campaign log, stints 1-5): the v1 hopper matched the n=10/
13/15 records from cold but sits far below the SYMMETRIC records —
n=14 (Beyleveld 2006, "symmetries of a rectangle") and n=16 (Beyleveld
2006, 180-degree rotational). Random starts almost never land in
high-symmetry basins, so v2 adds:

- symmetric start families: mirror about x=1/2 (pairs + on-axis
  points), 180-degree rotation about the center (pairs + optional
  center point), and the full rectangle group (quadruples + axis
  points + center);
- two symmetrizing kicks (mirror-half, rotate-half) that project the
  incumbent onto a nearby symmetric configuration.

Polishing stays UNCONSTRAINED in the full 2n-dimensional space: a
symmetric start is a doorway, not a cage — the NLP is free to break
symmetry if that is where the basin's optimum lies.

Same record_run interface and determinism contract as v1; run it as a
separate lane with its own --state file so the v1 sample stays clean.
"""

from __future__ import annotations

import numpy as np

from experiments.heilbronn import polish_hopper as v1
from experiments.heilbronn.evaluator import clip_to_unit_square, min_triangle_area
from experiments.heilbronn.exact_polish import exact_polish, load_karpov, triangle_index

_STAGNATION_RESTART = 12


def _mirror_start(rng: np.random.Generator, n: int) -> np.ndarray:
    """Mirror-symmetric about x=1/2: k pairs + (n-2k) on-axis points."""
    max_pairs = n // 2
    k = int(rng.integers(max(1, max_pairs - 1), max_pairs + 1))
    pts = []
    for _ in range(k):
        x, y = rng.random() * 0.5, rng.random()
        pts.append([x, y])
        pts.append([1.0 - x, y])
    for _ in range(n - 2 * k):
        pts.append([0.5, rng.random()])
    return np.array(pts[:n])


def _rot180_start(rng: np.random.Generator, n: int) -> np.ndarray:
    """180-degree rotationally symmetric about (1/2, 1/2)."""
    pts = []
    if n % 2 == 1:
        pts.append([0.5, 0.5])
    while len(pts) < n:
        p = rng.random(2)
        pts.append(list(p))
        pts.append([1.0 - p[0], 1.0 - p[1]])
    return np.array(pts[:n])


def _rect_start(rng: np.random.Generator, n: int) -> np.ndarray:
    """Full rectangle group: quadruples, axis pairs, optional center."""
    pts: list[list[float]] = []
    remaining = n
    if remaining % 2 == 1:
        pts.append([0.5, 0.5])
        remaining -= 1
    while remaining >= 4 and rng.random() < 0.75:
        x, y = rng.random() * 0.5, rng.random() * 0.5
        pts += [[x, y], [1 - x, y], [x, 1 - y], [1 - x, 1 - y]]
        remaining -= 4
    while remaining >= 2:
        c = rng.random() * 0.5
        if rng.random() < 0.5:
            pts += [[c, 0.5], [1 - c, 0.5]]  # pair on the horizontal axis
        else:
            pts += [[0.5, c], [0.5, 1 - c]]  # pair on the vertical axis
        remaining -= 2
    return np.array(pts[:n])


def _symmetrize_mirror(pts: np.ndarray) -> np.ndarray:
    """Project onto a mirror-symmetric config: left half mirrors right."""
    out = pts.copy()
    n = len(out)
    order = np.argsort(out[:, 0])
    half = n // 2
    left = order[:half]
    right = order[n - half :]
    out[right[::-1]] = np.column_stack([1.0 - out[left, 0], out[left, 1]])
    for i in order[half : n - half]:  # odd n: middle point onto the axis
        out[i, 0] = 0.5
    return clip_to_unit_square(out)


def _symmetrize_rot180(pts: np.ndarray) -> np.ndarray:
    """Project onto a 180-degree-symmetric config via center distance."""
    out = pts.copy()
    n = len(out)
    d = np.linalg.norm(out - 0.5, axis=1)
    order = np.argsort(-d)  # outermost points keep their positions
    half = n // 2
    keep = order[:half]
    replace = order[n - half :]
    out[replace[::-1]] = 1.0 - out[keep]
    for i in order[half : n - half]:
        out[i] = [0.5, 0.5]
    return clip_to_unit_square(out)


def _make_start(rng: np.random.Generator, n: int) -> np.ndarray:
    roll = rng.random()
    if roll < 0.60:
        return v1._make_start(rng, n)
    if roll < 0.75:
        return _mirror_start(rng, n)
    if roll < 0.90:
        return _rot180_start(rng, n)
    return _rect_start(rng, n)


def _kick(pts: np.ndarray, idx: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    roll = rng.random()
    if roll < 0.10:
        return _symmetrize_mirror(pts)
    if roll < 0.20:
        return _symmetrize_rot180(pts)
    return v1._kick(pts, idx, rng)


def solve(n: int, seed: int, budget: int):
    """Deterministic symmetric-start multistart + basin-hop (v2 lane)."""
    rng = np.random.default_rng([seed, n, 2])
    polish_calls = max(16, int(budget) // 400)
    idx = triangle_index(n)

    best_pts: np.ndarray | None = None
    best_area = -1.0
    spent = 0

    def polish_into_best(start: np.ndarray) -> tuple[np.ndarray, float]:
        nonlocal best_pts, best_area, spent
        res = exact_polish(start)
        spent += 1
        if res.area > best_area:
            best_pts, best_area = res.points, res.area
        return res.points, res.area

    for _ in range(max(4, polish_calls // 2)):
        if spent >= polish_calls:
            break
        polish_into_best(_make_start(rng, n))

    cur_pts, cur_area = best_pts.copy(), best_area
    stall = 0
    while spent < polish_calls:
        if stall >= _STAGNATION_RESTART:
            stall = 0
            if n in (13, 15) and rng.random() < 0.25:
                cur_pts, _ = load_karpov(n)
                cur_pts = cur_pts.copy()
            else:
                cur_pts = _make_start(rng, n)
            cur_area = min_triangle_area(cur_pts)
        pts, area = polish_into_best(_kick(cur_pts, idx, rng))
        if area > cur_area + 1e-12:
            cur_pts, cur_area = pts, area
            stall = 0
        else:
            stall += 1

    return best_pts.tolist()
