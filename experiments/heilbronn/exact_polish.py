"""Exact max-min polish for Heilbronn configurations (epigraph NLP).

Maximize t subject to  s_ijk * signed_area_ijk(x) >= t  for ALL C(n,3)
triples and x in [0,1]^2, where s_ijk fixes each triangle's orientation
at the starting configuration. Solved with SLSQP and analytic
jacobians; an outer loop re-fixes orientations and re-solves until the
validated min area stops improving.

Why: the record attempts (findings/heilbronn-record-attempt-20260618.md)
concluded "beating the record needs a stronger candidate, not more
seeds", and the g2_e1 candidate family polishes only the currently
smallest triangles. Max-min optima are heavily degenerate — Karpov's
n=13 record has 16 triangles tight within 1e-6 relative (18 for n=15;
see computable_packs/small_points_optimization/record_sources/) — so a
local step that ignores the near-active set chatters instead of
equalizing it. The epigraph formulation carries every triple with exact
gradients, so SLSQP converges to the basin's true local optimum.

The reported area is always recomputed from the (clipped) coordinates
via evaluator.min_triangle_area — never trusted from the solver.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from dataclasses import dataclass
from itertools import combinations
from pathlib import Path

# Single-threaded BLAS: matrices here are tiny and callers parallelize at
# the process level (same rationale + gotcha as record_run.py). Only
# effective when this module is imported before numpy — harmless otherwise.
for _v in (
    "OMP_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "MKL_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS",
    "NUMEXPR_NUM_THREADS",
):
    os.environ.setdefault(_v, "1")

import numpy as np  # noqa: E402
from scipy.optimize import minimize  # noqa: E402

from experiments.heilbronn.evaluator import (  # noqa: E402
    clip_to_unit_square,
    min_triangle_area,
)

RECORD_SOURCES = (
    Path(__file__).resolve().parents[2]
    / "computable_packs"
    / "small_points_optimization"
    / "record_sources"
)


def triangle_index(n: int) -> np.ndarray:
    """All C(n,3) triples as an [m,3] int array."""
    return np.array(list(combinations(range(n), 3)), dtype=np.int32)


def signed_areas(points: np.ndarray, idx: np.ndarray) -> np.ndarray:
    """Signed triangle areas (0.5 * cross) for every triple in idx."""
    a = points[idx[:, 0]]
    b = points[idx[:, 1]]
    c = points[idx[:, 2]]
    cross = (b[:, 0] - a[:, 0]) * (c[:, 1] - a[:, 1]) - (b[:, 1] - a[:, 1]) * (
        c[:, 0] - a[:, 0]
    )
    return 0.5 * cross


def signed_area_jacobian(points: np.ndarray, idx: np.ndarray) -> np.ndarray:
    """Dense jacobian of signed areas wrt flattened coords: [m, 2n]."""
    m = idx.shape[0]
    n = points.shape[0]
    a = points[idx[:, 0]]
    b = points[idx[:, 1]]
    c = points[idx[:, 2]]
    jac = np.zeros((m, 2 * n))
    rows = np.arange(m)
    jac[rows, 2 * idx[:, 0]] = 0.5 * (b[:, 1] - c[:, 1])
    jac[rows, 2 * idx[:, 0] + 1] = 0.5 * (c[:, 0] - b[:, 0])
    jac[rows, 2 * idx[:, 1]] = 0.5 * (c[:, 1] - a[:, 1])
    jac[rows, 2 * idx[:, 1] + 1] = 0.5 * (a[:, 0] - c[:, 0])
    jac[rows, 2 * idx[:, 2]] = 0.5 * (a[:, 1] - b[:, 1])
    jac[rows, 2 * idx[:, 2] + 1] = 0.5 * (b[:, 0] - a[:, 0])
    return jac


@dataclass
class PolishResult:
    points: np.ndarray
    area: float
    start_area: float
    rounds: int
    converged: bool

    @property
    def gain(self) -> float:
        return self.area - self.start_area


def _solve_round(
    points: np.ndarray,
    idx: np.ndarray,
    signs: np.ndarray,
    maxiter: int,
    ftol: float,
) -> np.ndarray | None:
    """One SLSQP solve with fixed orientations. Returns new points or None."""
    n = points.shape[0]
    dim = 2 * n + 1  # coords + epigraph t

    def cons_fun(z: np.ndarray) -> np.ndarray:
        pts = z[:-1].reshape(n, 2)
        return signs * signed_areas(pts, idx) - z[-1]

    def cons_jac(z: np.ndarray) -> np.ndarray:
        pts = z[:-1].reshape(n, 2)
        full = np.empty((idx.shape[0], dim))
        full[:, :-1] = signs[:, None] * signed_area_jacobian(pts, idx)
        full[:, -1] = -1.0
        return full

    t0 = float(np.min(signs * signed_areas(points, idx)))
    z0 = np.concatenate([points.ravel(), [t0]])
    grad = np.zeros(dim)
    grad[-1] = -1.0
    res = minimize(
        lambda z: -z[-1],
        z0,
        jac=lambda z: grad,
        method="SLSQP",
        bounds=[(0.0, 1.0)] * (2 * n) + [(0.0, 0.5)],
        constraints=[{"type": "ineq", "fun": cons_fun, "jac": cons_jac}],
        options={"maxiter": maxiter, "ftol": ftol},
    )
    if not np.all(np.isfinite(res.x)):
        return None
    return clip_to_unit_square(res.x[:-1].reshape(n, 2))


def exact_polish(
    points: np.ndarray,
    max_rounds: int = 12,
    slsqp_maxiter: int = 400,
    ftol: float = 1e-14,
    tol: float = 1e-12,
) -> PolishResult:
    """Polish a configuration to its basin's max-min stationary point.

    Each round fixes triangle orientations at the current iterate,
    solves the epigraph NLP, and accepts the result iff the recomputed
    min area improves. Stops when a round yields no improvement (or
    ``max_rounds``). Never returns anything worse than the input.
    """
    pts = clip_to_unit_square(np.asarray(points, dtype=float).copy())
    n = pts.shape[0]
    idx = triangle_index(n)
    start_area = min_triangle_area(pts)
    best_pts, best_area = pts, start_area
    rounds = 0
    converged = False
    for rounds in range(1, max_rounds + 1):
        areas = signed_areas(best_pts, idx)
        signs = np.where(areas >= 0.0, 1.0, -1.0)
        cand = _solve_round(best_pts, idx, signs, slsqp_maxiter, ftol)
        if cand is None:
            break
        cand_area = min_triangle_area(cand)
        if cand_area > best_area + tol:
            best_pts, best_area = cand, cand_area
            continue
        converged = True  # a full solve produced no improvement: stationary
        break
    return PolishResult(
        points=best_pts,
        area=best_area,
        start_area=start_area,
        rounds=rounds,
        converged=converged,
    )


def load_karpov(n: int) -> tuple[np.ndarray, float]:
    """Karpov's published record configuration for n in {13, 15}."""
    path = RECORD_SOURCES / f"heilbronn_s{n}_karpov2011.json"
    data = json.loads(path.read_text())
    return np.array(data["points"], dtype=float), float(data["measured_min_area"])


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="Exact max-min polish (epigraph SLSQP).")
    src = p.add_mutually_exclusive_group(required=True)
    src.add_argument("--json", help="path to a JSON file with a 'points' array")
    src.add_argument("--karpov", type=int, choices=(13, 15), help="polish Karpov's record config")
    p.add_argument("--max-rounds", type=int, default=12)
    args = p.parse_args(argv if argv is not None else sys.argv[1:])

    if args.karpov:
        pts, ref = load_karpov(args.karpov)
        print(f"input: Karpov S{args.karpov}, measured {ref:.15f}")
    else:
        data = json.loads(Path(args.json).read_text())
        pts = np.array(data["points"] if isinstance(data, dict) else data, dtype=float)
        print(f"input: {args.json}, start area {min_triangle_area(pts):.15f}")

    res = exact_polish(pts, max_rounds=args.max_rounds)
    print(
        f"polished: area={res.area:.15f} gain={res.gain:+.3e} "
        f"rounds={res.rounds} converged={res.converged}"
    )
    print(json.dumps({"area": res.area, "points": res.points.tolist()}))
    return 0


if __name__ == "__main__":
    sys.exit(main())
