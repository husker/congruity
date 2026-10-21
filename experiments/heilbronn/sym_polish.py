"""Symmetry-constrained exact max-min polish (fundamental-domain NLP).

Points are generated from free parameters theta through an AFFINE
symmetry map pts = (A @ theta + b).reshape(n, 2), so the epigraph NLP
(maximize t s.t. oriented triangle areas >= t) runs in the fundamental
domain: n=14 under the rectangle group drops from 28 coordinates to ~8,
which makes symmetric stationary points cheap to enumerate by
multistart.

Purpose (campaign log stints 6-8): the n=14/16 records (Beyleveld 2006)
are SYMMETRIC configurations that free search keeps missing — n=14
free search is stuck on a magnetic 0.02341751 shelf, 3.6% below the
record. Reconstructing the record configurations from their published
symmetry + value gives (a) coordinates for the record_sources corpus,
(b) continuation seeds for the neighboring cells, (c) a check that our
machinery can represent those optima at all.

CLI sweep: for each symmetry structure compatible with n, run K
multistarts of the reduced NLP, then re-polish winners UNCONSTRAINED in
the full space (a symmetric stationary point need not be a free local
optimum). Reports best per structure vs the pack record value.
"""

from __future__ import annotations

import argparse
import sys
import zlib
from dataclasses import dataclass

import numpy as np
from scipy.optimize import minimize

from experiments.heilbronn.calibrate_exact_polish import load_records
from experiments.heilbronn.evaluator import clip_to_unit_square, min_triangle_area
from experiments.heilbronn.exact_polish import (
    exact_polish,
    signed_area_jacobian,
    signed_areas,
    triangle_index,
)


@dataclass
class SymMap:
    """pts.ravel() = A @ theta + b, theta in [0,1]^dim."""

    name: str
    n: int
    A: np.ndarray  # [2n, dim]
    b: np.ndarray  # [2n]

    @property
    def dim(self) -> int:
        return self.A.shape[1]

    def points(self, theta: np.ndarray) -> np.ndarray:
        return (self.A @ theta + self.b).reshape(self.n, 2)


def _add_free_xy(rows_a: list, rows_b: list, dim: int, slot: int) -> None:
    """One generated coordinate = theta[slot] (helper for map building)."""
    row = [0.0] * dim
    row[slot] = 1.0
    rows_a.append(row)
    rows_b.append(0.0)


def _add_mirror_x(rows_a: list, rows_b: list, dim: int, slot: int) -> None:
    """One generated coordinate = 1 - theta[slot]."""
    row = [0.0] * dim
    row[slot] = -1.0
    rows_a.append(row)
    rows_b.append(1.0)


def _add_const(rows_a: list, rows_b: list, dim: int, value: float) -> None:
    rows_a.append([0.0] * dim)
    rows_b.append(value)


def mirror_map(n: int, pairs: int) -> SymMap:
    """Mirror about x=1/2: `pairs` free pairs + (n-2*pairs) on-axis points."""
    axis = n - 2 * pairs
    assert axis >= 0
    dim = 2 * pairs + axis  # (x,y) per pair + y per axis point
    ra: list[list[float]] = []
    rb: list[float] = []
    for i in range(pairs):
        xs, ys = 2 * i, 2 * i + 1
        _add_free_xy(ra, rb, dim, xs)  # (x, y)
        _add_free_xy(ra, rb, dim, ys)
        _add_mirror_x(ra, rb, dim, xs)  # (1-x, y)
        _add_free_xy(ra, rb, dim, ys)
    for j in range(axis):
        _add_const(ra, rb, dim, 0.5)  # x = 1/2
        _add_free_xy(ra, rb, dim, 2 * pairs + j)
    return SymMap(f"mirror(p{pairs},a{axis})", n, np.array(ra), np.array(rb))


def rot180_map(n: int) -> SymMap:
    """180-degree rotation about the center; odd n pins one point there."""
    pairs = n // 2
    center = n - 2 * pairs
    dim = 2 * pairs
    ra: list[list[float]] = []
    rb: list[float] = []
    for i in range(pairs):
        xs, ys = 2 * i, 2 * i + 1
        _add_free_xy(ra, rb, dim, xs)
        _add_free_xy(ra, rb, dim, ys)
        _add_mirror_x(ra, rb, dim, xs)  # (1-x, 1-y)
        _add_mirror_x(ra, rb, dim, ys)
    for _ in range(center):
        _add_const(ra, rb, dim, 0.5)
        _add_const(ra, rb, dim, 0.5)
    return SymMap(f"rot180(p{pairs},c{center})", n, np.array(ra), np.array(rb))


def rect_map(n: int, quads: int, xpairs: int, ypairs: int) -> SymMap:
    """Full rectangle group: quads + pairs on each axis (+ center if odd)."""
    center = n - 4 * quads - 2 * xpairs - 2 * ypairs
    assert center in (0, 1)
    dim = 2 * quads + xpairs + ypairs
    ra: list[list[float]] = []
    rb: list[float] = []
    for i in range(quads):
        xs, ys = 2 * i, 2 * i + 1
        for mx in (False, True):
            for my in (False, True):
                (_add_mirror_x if mx else _add_free_xy)(ra, rb, dim, xs)
                (_add_mirror_x if my else _add_free_xy)(ra, rb, dim, ys)
    for j in range(xpairs):  # pair on the horizontal axis y=1/2
        slot = 2 * quads + j
        _add_free_xy(ra, rb, dim, slot)
        _add_const(ra, rb, dim, 0.5)
        _add_mirror_x(ra, rb, dim, slot)
        _add_const(ra, rb, dim, 0.5)
    for j in range(ypairs):  # pair on the vertical axis x=1/2
        slot = 2 * quads + xpairs + j
        _add_const(ra, rb, dim, 0.5)
        _add_free_xy(ra, rb, dim, slot)
        _add_const(ra, rb, dim, 0.5)
        _add_mirror_x(ra, rb, dim, slot)
    for _ in range(center):
        _add_const(ra, rb, dim, 0.5)
        _add_const(ra, rb, dim, 0.5)
    return SymMap(f"rect(q{quads},x{xpairs},y{ypairs},c{center})", n, np.array(ra), np.array(rb))


def rot180_pinned_map(n: int, left_pairs: int, bottom_pairs: int) -> SymMap:
    """rot180 with some antipodal pairs pinned to boundary edges.

    A left-pinned pair is (0, t) / (1, 1-t); a bottom-pinned pair is
    (t, 0) / (1-t, 1) — each costs 1 dof instead of 2. Every known
    record configuration puts 40-57% of its points on the boundary, so
    pinning is the structural cut that shrinks rot180's search space
    where the prior is strongest. Even n only (odd needs the center,
    which is collinear with every antipodal pair).
    """
    assert n % 2 == 0
    free = n // 2 - left_pairs - bottom_pairs
    assert free >= 0
    dim = left_pairs + bottom_pairs + 2 * free
    ra: list[list[float]] = []
    rb: list[float] = []
    for j in range(left_pairs):
        _add_const(ra, rb, dim, 0.0)  # (0, t)
        _add_free_xy(ra, rb, dim, j)
        _add_const(ra, rb, dim, 1.0)  # (1, 1-t)
        _add_mirror_x(ra, rb, dim, j)
    for j in range(bottom_pairs):
        slot = left_pairs + j
        _add_free_xy(ra, rb, dim, slot)  # (t, 0)
        _add_const(ra, rb, dim, 0.0)
        _add_mirror_x(ra, rb, dim, slot)  # (1-t, 1)
        _add_const(ra, rb, dim, 1.0)
    for i in range(free):
        xs = left_pairs + bottom_pairs + 2 * i
        ys = xs + 1
        _add_free_xy(ra, rb, dim, xs)
        _add_free_xy(ra, rb, dim, ys)
        _add_mirror_x(ra, rb, dim, xs)
        _add_mirror_x(ra, rb, dim, ys)
    return SymMap(
        f"rot180pin(L{left_pairs},B{bottom_pairs},f{free})", n, np.array(ra), np.array(rb)
    )


def diag_map(n: int, pairs: int) -> SymMap:
    """Mirror about the main diagonal y=x: pairs (x,y)/(y,x) + on-diagonal (t,t)."""
    diag = n - 2 * pairs
    assert diag >= 0
    dim = 2 * pairs + diag
    ra: list[list[float]] = []
    rb: list[float] = []
    for i in range(pairs):
        xs, ys = 2 * i, 2 * i + 1
        _add_free_xy(ra, rb, dim, xs)  # (x, y)
        _add_free_xy(ra, rb, dim, ys)
        _add_free_xy(ra, rb, dim, ys)  # (y, x)
        _add_free_xy(ra, rb, dim, xs)
    for j in range(diag):
        slot = 2 * pairs + j
        _add_free_xy(ra, rb, dim, slot)  # (t, t)
        _add_free_xy(ra, rb, dim, slot)
    return SymMap(f"diag(p{pairs},d{diag})", n, np.array(ra), np.array(rb))


def antidiag_map(n: int, pairs: int) -> SymMap:
    """Mirror about y=1-x: pairs (x,y)/(1-y,1-x) + fixed points (t,1-t)."""
    fixed = n - 2 * pairs
    assert fixed >= 0
    dim = 2 * pairs + fixed
    ra: list[list[float]] = []
    rb: list[float] = []
    for i in range(pairs):
        xs, ys = 2 * i, 2 * i + 1
        _add_free_xy(ra, rb, dim, xs)  # (x, y)
        _add_free_xy(ra, rb, dim, ys)
        _add_mirror_x(ra, rb, dim, ys)  # (1-y, 1-x)
        _add_mirror_x(ra, rb, dim, xs)
    for j in range(fixed):
        slot = 2 * pairs + j
        _add_free_xy(ra, rb, dim, slot)  # (t, 1-t)
        _add_mirror_x(ra, rb, dim, slot)
    return SymMap(f"antidiag(p{pairs},d{fixed})", n, np.array(ra), np.array(rb))


def structures_for(n: int) -> list[SymMap]:
    """A reasonable menu of symmetry structures for one n."""
    out: list[SymMap] = []
    for pairs in range(max(1, n // 2 - 2), n // 2 + 1):
        out.append(mirror_map(n, pairs))
    out.append(rot180_map(n))
    for pairs in range(max(1, n // 2 - 1), n // 2 + 1):
        if n - 2 * pairs <= 2:  # 3+ points on one line are collinear-dead
            out.append(diag_map(n, pairs))
            out.append(antidiag_map(n, pairs))
    if n % 2 == 0:
        for b in range(2, min(5, n // 2) + 1):  # 4-10 boundary points
            for left in range(b + 1):
                # 3+ pinned points per edge line are collinear-dead
                if left <= 2 and (b - left) <= 2:
                    out.append(rot180_pinned_map(n, left, b - left))
    for quads in range(n // 4, -1, -1):
        rest = n - 4 * quads
        for xpairs in range(rest // 2 + 1):
            rest2 = rest - 2 * xpairs
            ypairs = rest2 // 2
            if 4 * quads + 2 * xpairs + 2 * ypairs in (n, n - 1):
                try:
                    out.append(rect_map(n, quads, xpairs, ypairs))
                except AssertionError:
                    continue
    # dedupe by name
    seen: set[str] = set()
    uniq = []
    for s in out:
        if s.name not in seen:
            seen.add(s.name)
            uniq.append(s)
    return uniq


def sym_polish(sym: SymMap, theta0: np.ndarray, maxiter: int = 300) -> tuple[np.ndarray, float]:
    """Epigraph SLSQP in the fundamental domain; returns (points, area)."""
    n = sym.n
    idx = triangle_index(n)
    pts0 = clip_to_unit_square(sym.points(theta0))
    signs = np.where(signed_areas(pts0, idx) >= 0.0, 1.0, -1.0)
    dim = sym.dim + 1

    def cons_fun(z: np.ndarray) -> np.ndarray:
        pts = sym.points(z[:-1])
        return signs * signed_areas(pts, idx) - z[-1]

    def cons_jac(z: np.ndarray) -> np.ndarray:
        pts = sym.points(z[:-1])
        full = np.empty((idx.shape[0], dim))
        # chain rule through the affine map: d(area)/d(theta) = J_pts @ A
        full[:, :-1] = (signs[:, None] * signed_area_jacobian(pts, idx)) @ sym.A
        full[:, -1] = -1.0
        return full

    t0 = float(np.min(signs * signed_areas(pts0, idx)))
    z0 = np.concatenate([theta0, [max(t0, 0.0)]])
    grad = np.zeros(dim)
    grad[-1] = -1.0
    res = minimize(
        lambda z: -z[-1],
        z0,
        jac=lambda z: grad,
        method="SLSQP",
        bounds=[(0.0, 1.0)] * sym.dim + [(0.0, 0.5)],
        constraints=[{"type": "ineq", "fun": cons_fun, "jac": cons_jac}],
        options={"maxiter": maxiter, "ftol": 1e-14},
    )
    pts = clip_to_unit_square(sym.points(np.clip(res.x[:-1], 0.0, 1.0)))
    return pts, min_triangle_area(pts)


def reduced_hop(
    sym: SymMap,
    rng: np.random.Generator,
    polish_calls: int,
    stagnation: int = 15,
) -> tuple[np.ndarray, float]:
    """Basin-hop entirely inside a symmetry's fundamental domain.

    Unlike ``sweep`` (pure multistart) and polish_hopper_sym (symmetric
    START, full-space polish that immediately breaks symmetry), this
    perturbs the free parameters theta and re-polishes IN THE REDUCED
    SPACE, so every iterate is exactly symmetric. This is the right
    search when the target itself is symmetric: the n=12 rot180 record
    is reachable by free hopping but pure rot180 multistart undersamples
    it (91.5% at 400 starts) — a reduced-space hopper closes that gap.

    Returns (full_space_points, area) — the incumbent gets a final
    unconstrained polish, since a symmetric stationary point need not be
    a free local optimum.
    """
    best_theta = rng.random(sym.dim)
    _, best_area = sym_polish(sym, best_theta)
    cur_theta, cur_area = best_theta.copy(), best_area
    stall = 0
    spent = 0
    while spent < polish_calls:
        theta = cur_theta.copy()
        move = rng.random()
        if move < 0.5:  # nudge a few coordinates
            m = 1 + int(rng.integers(max(1, sym.dim // 3)))
            js = rng.choice(sym.dim, size=min(m, sym.dim), replace=False)
            theta[js] = np.clip(
                theta[js] + rng.normal(0.0, 0.08, size=len(js)), 0.0, 1.0
            )
        elif move < 0.8:  # teleport one coordinate
            theta[int(rng.integers(sym.dim))] = rng.random()
        else:  # global jitter
            theta = np.clip(theta + rng.normal(0.0, 0.03, sym.dim), 0.0, 1.0)
        pts, area = sym_polish(sym, theta)
        spent += 1
        # recover theta of the polished config is nontrivial; keep theta
        # as the perturbed seed but track area from the polish.
        if area > cur_area + 1e-12:
            cur_theta, cur_area = theta, area
            stall = 0
            if area > best_area:
                best_area, best_theta = area, theta.copy()
        else:
            stall += 1
            if stall >= stagnation:
                cur_theta = rng.random(sym.dim)
                _, cur_area = sym_polish(sym, cur_theta)
                spent += 1
                stall = 0
    pts, _ = sym_polish(sym, best_theta)
    full = exact_polish(pts)
    return full.points, full.area


def hop(
    n: int, structure: str, budget: int, seed0: int, restarts: int
) -> None:
    """Reduced-space basin-hop over the matching structures for one n."""
    records = load_records()
    target = float(records[n]["value"]) if n in records else None
    menu = [s for s in structures_for(n) if structure in s.name]
    if not menu:
        print(f"no structures match {structure!r} for n={n}")
        return
    calls = max(20, budget // 400)
    overall, overall_pts, overall_src = -1.0, None, ""
    for sym in menu:
        best = -1.0
        for r in range(restarts):
            rng = np.random.default_rng([seed0, n, zlib.crc32(sym.name.encode()), r])
            pts, area = reduced_hop(sym, rng, calls)
            if area > best:
                best = area
            if area > overall:
                overall, overall_pts, overall_src = area, pts, sym.name
        pct = f"{100 * best / target:.3f}%" if target else "?"
        print(f"  {sym.name:24s} dim={sym.dim:2d} hop-best={best:.10f} ({pct})", flush=True)
    flag = ""
    if target and overall >= target * (1 - 1e-9):
        flag = "  <<< AT/ABOVE RECORD — hand-check protocol"
    print(f"n={n} HOP OVERALL best={overall:.12f} via {overall_src}{flag}")
    if overall_pts is not None and target and overall >= target * (1 - 1e-6):
        print(np.array2string(overall_pts, precision=15))


def sweep(
    n: int, starts: int, seed0: int, name_filter: list[str] | None = None
) -> None:
    records = load_records()
    target = float(records[n]["value"]) if n in records else None
    print(f"n={n} target={target}")
    overall_best, overall_pts, overall_src = -1.0, None, ""
    menu = structures_for(n)
    if name_filter:
        menu = [s for s in menu if any(f in s.name for f in name_filter)]
    for sym in menu:
        best, best_theta = -1.0, None
        for k in range(starts):
            rng = np.random.default_rng([seed0, n, zlib.crc32(sym.name.encode()), k])
            theta0 = rng.random(sym.dim)
            _, area = sym_polish(sym, theta0)
            if area > best:
                best = area
                best_theta = theta0
        # winners get an UNCONSTRAINED full-space polish on top
        pts, _ = sym_polish(sym, best_theta)
        full = exact_polish(pts)
        pct = f"{100 * full.area / target:.3f}%" if target else "?"
        print(
            f"  {sym.name:24s} dim={sym.dim:2d} sym-best={best:.10f} "
            f"full-polished={full.area:.10f} ({pct})",
            flush=True,
        )
        if full.area > overall_best:
            overall_best, overall_pts, overall_src = full.area, full.points, sym.name
    flag = ""
    if target and overall_best >= target * (1 - 1e-9):
        flag = "  <<< AT/ABOVE RECORD — hand-check protocol applies"
    print(f"n={n} OVERALL best={overall_best:.12f} via {overall_src}{flag}")
    if overall_pts is not None and target and overall_best >= target * (1 - 1e-6):
        print(np.array2string(overall_pts, precision=15))


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="Symmetry-constrained polish sweeps.")
    p.add_argument("--ns", default="14,16")
    p.add_argument("--starts", type=int, default=120)
    p.add_argument("--seed0", type=int, default=1)
    p.add_argument(
        "--structures",
        default=None,
        help="comma-separated substrings; only matching structures run",
    )
    p.add_argument(
        "--hop",
        default=None,
        help="reduced-space basin-hop mode: structure substring to hop (e.g. rot180)",
    )
    p.add_argument("--budget", type=int, default=80000, help="hop budget (calls=budget//400)")
    p.add_argument("--restarts", type=int, default=8, help="hop restarts per structure")
    args = p.parse_args(argv if argv is not None else sys.argv[1:])
    ns = [int(x) for x in args.ns.split(",") if x.strip()]
    if args.hop:
        for n in ns:
            hop(n, args.hop, args.budget, args.seed0, args.restarts)
        return 0
    filt = [f for f in (args.structures or "").split(",") if f.strip()] or None
    for n in ns:
        sweep(n, args.starts, args.seed0, filt)
    return 0


if __name__ == "__main__":
    sys.exit(main())
