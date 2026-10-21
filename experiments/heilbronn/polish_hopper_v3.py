"""Continuation-seeded hopper (v3 lane): record configs at m = n+-1 as starts.

The v1/v2 lanes sit exactly AT the n=13/15 record bars (campaign log,
stints 1-8) — going above needs basins that random and symmetric starts
do not reach. v3 adds CONTINUATION starts: take a known record-grade
configuration at a neighboring m (Karpov's n=13/15, the reconstructed
Beyleveld n=14 — all in record_sources/), then

- drop the least-constrained point (m = n+1), or
- insert a point at the largest empty spot (m = n-1),

jitter slightly, and let the exact polish equalize. Record-grade
structure at m is the strongest prior available for good basins at n.
Start menu: ~30% continuation, ~30% v2 symmetric families, ~40% v1;
kicks and hop loop are v2's. Same record_run contract, own state file.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from experiments.heilbronn import polish_hopper as v1
from experiments.heilbronn import polish_hopper_sym as v2
from experiments.heilbronn.evaluator import clip_to_unit_square, min_triangle_area
from experiments.heilbronn.exact_polish import (
    RECORD_SOURCES,
    exact_polish,
    load_karpov,
    triangle_index,
)

_STAGNATION_RESTART = 12


def _load_sources() -> dict[int, np.ndarray]:
    """All record-grade configs in record_sources/, keyed by their n."""
    out: dict[int, np.ndarray] = {}
    for path in sorted(Path(RECORD_SOURCES).glob("heilbronn_s*_*.json")):
        data = json.loads(path.read_text())
        out[int(data["n"])] = np.array(data["points"], dtype=float)
    return out


_SOURCES = _load_sources()


def _drop_point(pts: np.ndarray) -> np.ndarray:
    """Remove the point whose removal hurts the min area least."""
    best_pts, best_area = None, -1.0
    for i in range(len(pts)):
        cand = np.delete(pts, i, axis=0)
        area = min_triangle_area(cand)
        if area > best_area:
            best_area, best_pts = area, cand
    assert best_pts is not None
    return best_pts


def _insert_point(pts: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Add a point at the candidate spot farthest from existing points."""
    cands = rng.random((48, 2))
    d = np.min(
        np.linalg.norm(cands[:, None, :] - pts[None, :, :], axis=2), axis=1
    )
    return np.vstack([pts, cands[int(np.argmax(d))]])


def continuation_start(rng: np.random.Generator, n: int) -> np.ndarray | None:
    """A start derived from a record-grade config at a neighboring m."""
    options = [m for m in _SOURCES if m != n and abs(m - n) <= 2]
    if not options:
        return None
    m = options[int(rng.integers(len(options)))]
    pts = _SOURCES[m].copy()
    while len(pts) > n:
        pts = _drop_point(pts)
    while len(pts) < n:
        pts = _insert_point(pts, rng)
    pts += rng.normal(0.0, 0.02, size=pts.shape)
    return clip_to_unit_square(pts)


def _make_start(rng: np.random.Generator, n: int) -> np.ndarray:
    roll = rng.random()
    if roll < 0.30:
        start = continuation_start(rng, n)
        if start is not None:
            return start
    if roll < 0.60:
        return v2._make_start(rng, n)
    return v1._make_start(rng, n)


def solve(n: int, seed: int, budget: int):
    """Deterministic continuation + symmetric + free multistart/hop (v3)."""
    rng = np.random.default_rng([seed, n, 3])
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
            if n in (13, 15) and rng.random() < 0.2:
                cur_pts, _ = load_karpov(n)
                cur_pts = cur_pts.copy()
            else:
                cur_pts = _make_start(rng, n)
            cur_area = min_triangle_area(cur_pts)
        pts, area = polish_into_best(v2._kick(cur_pts, idx, rng))
        if area > cur_area + 1e-12:
            cur_pts, cur_area = pts, area
            stall = 0
        else:
            stall += 1

    return best_pts.tolist()
