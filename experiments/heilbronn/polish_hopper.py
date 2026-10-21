"""Multistart + basin-hopping Heilbronn candidate over the exact polish.

The M1 calibration (findings/heilbronn-record-campaign-20260706.md)
showed the epigraph polish recovers proven optima exactly whenever a
start lands in the right basin, and that the 2026-06 campaign bests are
already stationary (basin-limited, not optimizer-limited). This
candidate therefore spends its entire budget on BASIN DISCOVERY, using
``exact_polish`` as the local step:

- Phase 1 multistart: uniform / boundary-biased starts (about 45% of the
  points in Karpov's record configs sit on the square's boundary).
- Phase 2 basin-hopping: structured kicks from the incumbent — teleport
  the most-constrained vertex, global jitter, edge snaps, corner pulls —
  polish after each kick, accept strict improvements, restart on
  stagnation. For n in {13, 15} some hop chains start from Karpov's
  published record configuration: its neighborhood is the
  highest-prior region for a strictly better basin.

record_run.py interface: ``solve(n, seed, budget)`` — deterministic in
(n, seed, budget); ``budget`` is the pack's eval-ish unit, mapped here
to ``max(16, budget // 400)`` exact-polish calls (budget 80000 -> 200
polishes, roughly 10-30 s at n=13).
"""

from __future__ import annotations

import numpy as np

from experiments.heilbronn.evaluator import (
    all_triangle_areas,
    clip_to_unit_square,
    min_triangle_area,
)
from experiments.heilbronn.exact_polish import (
    exact_polish,
    load_karpov,
    triangle_index,
)

_JITTER_SIGMAS = (0.02, 0.05, 0.12)
_STAGNATION_RESTART = 12


def _make_start(rng: np.random.Generator, n: int) -> np.ndarray:
    """Random start: uniform, or with a boundary fraction of 0.35/0.6."""
    pts = rng.random((n, 2))
    mode = int(rng.integers(3))
    frac = (0.0, 0.35, 0.6)[mode]
    if frac > 0.0:
        for i in range(n):
            if rng.random() < frac:
                edge = int(rng.integers(4))
                if edge == 0:
                    pts[i, 0] = 0.0
                elif edge == 1:
                    pts[i, 0] = 1.0
                elif edge == 2:
                    pts[i, 1] = 0.0
                else:
                    pts[i, 1] = 1.0
    return pts


def _worst_vertex(pts: np.ndarray, idx: np.ndarray) -> int:
    """Vertex participating in the most near-minimal triangles."""
    areas = all_triangle_areas(pts)
    near = areas <= areas.min() * 1.5 + 1e-15
    counts = np.bincount(idx[near].ravel(), minlength=pts.shape[0])
    return int(np.argmax(counts))


def _kick(pts: np.ndarray, idx: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    out = pts.copy()
    n = pts.shape[0]
    move = int(rng.integers(4))
    if move == 0:  # teleport the most-constrained vertex
        v = _worst_vertex(out, idx)
        if rng.random() < 0.5:  # onto a random edge
            edge = int(rng.integers(4))
            coord = rng.random()
            out[v] = [
                (0.0, coord),
                (1.0, coord),
                (coord, 0.0),
                (coord, 1.0),
            ][edge]
        else:
            out[v] = rng.random(2)
    elif move == 1:  # global jitter
        sigma = _JITTER_SIGMAS[int(rng.integers(len(_JITTER_SIGMAS)))]
        out += rng.normal(0.0, sigma, size=out.shape)
    elif move == 2:  # snap a random interior point to a random edge
        interior = [i for i in range(n) if 0.0 < out[i].min() and out[i].max() < 1.0]
        if interior:
            v = interior[int(rng.integers(len(interior)))]
            edge = int(rng.integers(4))
            if edge == 0:
                out[v, 0] = 0.0
            elif edge == 1:
                out[v, 0] = 1.0
            elif edge == 2:
                out[v, 1] = 0.0
            else:
                out[v, 1] = 1.0
        else:
            out += rng.normal(0.0, 0.05, size=out.shape)
    else:  # pull a random point toward its nearest corner
        v = int(rng.integers(n))
        corner = np.round(out[v])
        out[v] = out[v] + (corner - out[v]) * rng.random()
    return clip_to_unit_square(out)


def solve(n: int, seed: int, budget: int):
    """Deterministic multistart + basin-hop; returns an [n,2] point list."""
    rng = np.random.default_rng([seed, n])
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

    # Phase 1: multistart (half the budget).
    for _ in range(max(4, polish_calls // 2)):
        if spent >= polish_calls:
            break
        polish_into_best(_make_start(rng, n))

    # Phase 2: basin-hop chains with stagnation restarts.
    cur_pts, cur_area = best_pts.copy(), best_area
    stall = 0
    while spent < polish_calls:
        if stall >= _STAGNATION_RESTART:
            stall = 0
            if n in (13, 15) and rng.random() < 0.25:
                cur_pts, _ = load_karpov(n)  # explore the record neighborhood
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
