"""CP-SAT lattice search for Heilbronn records with rational values.

Some Heilbronn best-knowns are exact rationals (n=6: 1/8, n=11: 1/27,
n=16: 7/341) — evidence of a lattice/algebraic construction rather than
the cubic-irrational optima at n=7/8/10/12/14. For a value p/q, if the
points lie on a grid x=i/GX, y=j/GY then every triangle area is
|D|/(2*GX*GY) with D the integer determinant, so the record corresponds
to a specific minimum integer determinant. This tool maximizes that
minimum (optionally under 180-degree rotational symmetry) with CP-SAT:
a bound below the record's target refutes the grid; reaching it yields
exact coordinates.

Example (n=16 record 7/341 = 14/682, grid 11x31 needs min|D| >= 14):
    python -m experiments.heilbronn.lattice_search --n 16 --gx 11 --gy 31 \
        --target 14 --mode sym --seconds 600
"""

from __future__ import annotations

import argparse
import sys
from itertools import combinations

from ortools.sat.python import cp_model


def build_and_solve(
    n: int, gx: int, gy: int, mode: str, seconds: float, workers: int
) -> None:
    m = cp_model.CpModel()
    X = [m.NewIntVar(0, gx, f"x{k}") for k in range(n)]
    Y = [m.NewIntVar(0, gy, f"y{k}") for k in range(n)]

    if mode == "sym":
        if n % 2:
            raise SystemExit("sym mode requires even n (rot180 pairs)")
        for p in range(n // 2):
            m.Add(X[2 * p + 1] == gx - X[2 * p])
            m.Add(Y[2 * p + 1] == gy - Y[2 * p])

    code = []
    for k in range(n):
        c = m.NewIntVar(0, (gx + 1) * (gy + 1), f"c{k}")
        m.Add(c == X[k] * (gy + 1) + Y[k])
        code.append(c)
    m.AddAllDifferent(code)
    if mode == "sym":
        for p in range(n // 2 - 1):
            m.Add(code[2 * p] < code[2 * (p + 1)])  # order representatives

    prod: dict[tuple[int, int], cp_model.IntVar] = {}
    for a in range(n):
        for b in range(n):
            if a != b:
                pab = m.NewIntVar(-gx * gy, gx * gy, f"P{a}_{b}")
                m.AddMultiplicationEquality(pab, X[a], Y[b])
                prod[(a, b)] = pab

    L = m.NewIntVar(0, gx * gy, "L")
    for a, b, c in combinations(range(n), 3):
        det = (
            prod[(a, b)] - prod[(a, c)] + prod[(b, c)]
            - prod[(b, a)] + prod[(c, a)] - prod[(c, b)]
        )
        z = m.NewBoolVar(f"z{a}_{b}_{c}")
        m.Add(det >= L).OnlyEnforceIf(z)
        m.Add(det <= -L).OnlyEnforceIf(z.Not())
        m.Add(det >= 1).OnlyEnforceIf(z)  # forbid degeneracy in either orientation
        m.Add(det <= -1).OnlyEnforceIf(z.Not())
    m.Maximize(L)

    class CB(cp_model.CpSolverSolutionCallback):
        def on_solution_callback(self) -> None:
            v = self.Value(L)
            print(f"  incumbent min|D| = {v} (area {v}/{2 * gx * gy})", flush=True)

    solver = cp_model.CpSolver()
    solver.parameters.max_time_in_seconds = seconds
    solver.parameters.num_search_workers = workers
    print(f"maximize min|D|  n={n} mode={mode} grid={gx}x{gy} (<= {seconds}s)", flush=True)
    status = solver.Solve(m, CB())
    print(f"STATUS: {solver.StatusName(status)}")
    feasible = status in (cp_model.OPTIMAL, cp_model.FEASIBLE)
    print(f"best min|D| (lower bound) = {solver.Value(L) if feasible else 'none'}")
    print(f"proven upper bound        = {solver.best_objective_bound}")
    if feasible:
        pts = [(solver.Value(X[k]), solver.Value(Y[k])) for k in range(n)]
        print("grid (i,j):", pts)
        print("continuous:", [(x / gx, y / gy) for x, y in pts])


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="CP-SAT lattice search for Heilbronn.")
    p.add_argument("--n", type=int, default=16)
    p.add_argument("--gx", type=int, default=11)
    p.add_argument("--gy", type=int, default=31)
    p.add_argument("--mode", choices=("sym", "free"), default="sym")
    p.add_argument("--target", type=int, default=None, help="record min|D| (for reference)")
    p.add_argument("--seconds", type=float, default=300.0)
    p.add_argument("--workers", type=int, default=6)
    args = p.parse_args(argv if argv is not None else sys.argv[1:])
    if args.target:
        print(f"record target min|D| >= {args.target}")
    build_and_solve(args.n, args.gx, args.gy, args.mode, args.seconds, args.workers)
    return 0


if __name__ == "__main__":
    sys.exit(main())
