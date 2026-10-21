"""Stage-0 restricted master LP over the four-subcube fragment pool.

Column generation (Dantzig-Wolfe) for the layered snake, LP-first: no C
changes, columns are the CELL FRAGMENTS of assemblies we already have
on disk (router --out paths, joint-construction certificates). Q9 =
four Q7 cells indexed by (bit8, bit7). Constraints:

  - convexity: one fragment per cell (relaxed to sum = 1 over x >= 0)
  - seam label-disjointness: for each Q7 label and each of the four
    seams (bit7 within each Q8, bit8 within each column), the two
    adjacent cells may jointly use the label at most once — EXCEPT via
    a junction slack y[seam,label] (a crossing uses the same label on
    both sides), budgeted at one junction per seam and worth +1 edge
    in the objective. Every true 9:1,8:1 assembly is feasible, so the
    LP value is a valid upper bound for the pool x architecture.

Objective: maximize fragment edges + junction edges. Output: LP value,
and the label duals — shadow prices for bottom generation: a new
bottom's reduced cost is len(bottom) - sum(price over its labels).
"""

from __future__ import annotations

import glob
import json
from collections import defaultdict

import numpy as np
from scipy.optimize import linprog

CELLS = ((0, 0), (0, 1), (1, 0), (1, 1))    # (bit8, bit7)
SEAMS = (((0, 0), (0, 1)), ((1, 0), (1, 1)),   # bit7 seams
         ((0, 0), (1, 0)), ((0, 1), (1, 1)))   # bit8 seams


def split_cells(path: list[int]) -> dict[tuple[int, int], list[int]]:
    """Contiguous runs of the assembly per cell, as Q7 label lists."""
    out: dict[tuple[int, int], list[int]] = defaultdict(list)
    for v in path:
        out[((v >> 8) & 1, (v >> 7) & 1)].append(v & 127)
    return out


def load_pool(patterns: list[str]) -> list[dict]:
    cols, seen = [], set()
    for pat in patterns:
        for fp in sorted(glob.glob(pat)):
            d = json.load(open(fp))
            path = d.get("path") or d.get("seq")
            if not path:
                continue
            for cell, labels in split_cells(path).items():
                key = (cell, tuple(sorted(set(labels))))
                if key in seen or len(labels) < 2:
                    continue
                seen.add(key)
                cols.append({"cell": cell, "labels": set(labels),
                             "edges": len(labels) - 1, "src": fp})
    return cols


def solve(cols: list[dict]):
    n = len(cols)
    # variables: x (fragments) then y (junction slacks, one per
    # (seam,label) that appears on both sides is enough, but per
    # (seam,label) generally is fine — budgeted per seam)
    seam_labels = []
    for ca, cb in SEAMS:
        labels = set()
        for c in cols:
            if c["cell"] in (ca, cb):
                labels |= c["labels"]
        seam_labels.append(sorted(labels))
    y_index = [(si, lab) for si, labs in enumerate(seam_labels)
               for lab in labs]
    ny = len(y_index)
    a_eq, b_eq = [], []
    for cell in CELLS:
        a_eq.append([1.0 if c["cell"] == cell else 0.0 for c in cols]
                    + [0.0] * ny)
        b_eq.append(1.0)
    a_ub, b_ub, row_key = [], [], []
    for yi, (si, lab) in enumerate(y_index):
        ca, cb = SEAMS[si]
        row = [1.0 if (c["cell"] in (ca, cb) and lab in c["labels"])
               else 0.0 for c in cols] + [0.0] * ny
        row[n + yi] = -1.0                      # usage - y <= 1
        a_ub.append(row)
        b_ub.append(1.0)
        row_key.append((si, lab))
    for si in range(len(SEAMS)):                # junction budget per seam
        row = [0.0] * n + [1.0 if s == si else 0.0 for s, _ in y_index]
        a_ub.append(row)
        b_ub.append(1.0)
        row_key.append(("budget", si))
    res = linprog(c=[-c["edges"] for c in cols] + [-1.0] * ny,
                  A_ub=np.array(a_ub), b_ub=b_ub,
                  A_eq=np.array(a_eq), b_eq=b_eq,
                  bounds=[(0, 1)] * n + [(0, 1)] * ny, method="highs")
    assert res.status == 0, res.message
    duals = {}
    if res.ineqlin is not None:
        for key, m in zip(row_key, res.ineqlin.marginals):
            if abs(m) > 1e-9 and key[0] != "budget":
                duals[key] = -float(m)      # price >= 0
    return -res.fun, res.x[:n], duals


def main() -> int:
    cols = load_pool(["data/snake_in_box/hrouter_*.json",
                      "data/snake_in_box/certificates/q9_len1*_joint.json",
                      "data/snake_in_box/certificates/q9_len190_wynn.json"])
    from collections import Counter
    print(f"pool: {len(cols)} distinct cell-fragments "
          f"({Counter(c['cell'] for c in cols)})")
    val, x, duals = solve(cols)
    print(f"LP value (pool x architecture upper bound, incl. junction "
          f"budget): {val:.1f}")
    picked = [(c, xi) for c, xi in zip(cols, x) if xi > 1e-6]
    for c, xi in sorted(picked, key=lambda t: -t[1]):
        print(f"  x={xi:.3f} cell{c['cell']} {c['edges']}e  {c['src']}")
    hot = sorted(duals.items(), key=lambda kv: -kv[1])[:20]
    print("hottest label prices (seam, label) -> price:")
    for (si, lab), p in hot:
        print(f"  seam{si} label {lab:3d}: {p:.3f}")
    # reduced-cost view of each source assembly's bottom
    print("\nbottom scores under duals (edges - shadow price paid):")
    for fp in sorted(set(c["src"] for c in cols)):
        d = json.load(open(fp))
        path = d.get("path") or d.get("seq")
        cells = split_cells(path)
        for cell in ((0, 0), (0, 1)):
            labs = set(cells.get(cell, []))
            seams = {i for i, (ca, cb) in enumerate(SEAMS)
                     if cell in (ca, cb)}
            paid = sum(p for (si, lab), p in duals.items()
                       if si in seams and lab in labs)
            if labs:
                print(f"  {fp.split('/')[-1]:34s} cell{cell} "
                      f"{len(labs)-1:3d}e  pays {paid:6.2f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
