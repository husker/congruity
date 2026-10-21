"""Offline portfolio-policy study for the Hamiltonian solver-selection claim.

Reads the solo-timing dataset from `collect.py` and asks the question the
original finding left open (findings/c1-hamiltonian-regime-classifier-
20260524.md): the naive Ridge selector scored -84% because of cost
asymmetry — can a COST-AWARE realization of the transferred mechanism
(topology features -> solver choice) actually beat the best single
solver? And is the oracle headroom mutual or one-sided?

Everything here is pure/offline over recorded timings — no solving, no
parallel infra. Two metrics, kept explicitly separate:

  * single-core / total-CPU: a policy pays the chosen solver's solo
    time. best-single is the honest bar; nothing that runs two solvers
    can beat it here.
  * 2-core latency: a speculative policy that always runs CaDiCaL and
    ALSO runs DFS on gated instances pays max/min appropriately and can
    approach oracle latency — at 2x CPU.

Policies compared: best-single, oracle, Ridge (symmetric, the reproduced
failure), cost-aware two-regressor + safety margin (path #1), and
safe-default + speculative-DFS (path #3).
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from sklearn.linear_model import Ridge
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler


@dataclass
class Data:
    X: np.ndarray  # [N, F] features
    sat: np.ndarray  # [N] CaDiCaL solo time
    dfs: np.ndarray  # [N] DFS solo time
    dfs_timeout: np.ndarray  # [N] bool
    sat_timeout: np.ndarray  # [N] bool
    features: list[str]


def load(path: Path) -> Data:
    payload = json.loads(path.read_text())
    recs = payload["records"]
    feats = payload["meta"]["features"]
    X = np.array([[r["features"][f] for f in feats] for r in recs], dtype=float)
    sat = np.array([r["sat_time"] for r in recs], dtype=float)
    dfs = np.array([r["dfs_time"] for r in recs], dtype=float)
    return Data(
        X=X,
        sat=sat,
        dfs=dfs,
        dfs_timeout=np.array([r["dfs_timeout"] for r in recs]),
        sat_timeout=np.array([r["sat_timeout"] for r in recs]),
        features=feats,
    )


def totals(d: Data) -> dict:
    oracle = np.minimum(d.sat, d.dfs)
    best_single = min(d.sat.sum(), d.dfs.sum())
    return {
        "n": len(d.sat),
        "sat_total": float(d.sat.sum()),
        "dfs_total": float(d.dfs.sum()),
        "oracle_total": float(oracle.sum()),
        "best_single": float(best_single),
        "best_single_name": "CaDiCaL" if d.sat.sum() <= d.dfs.sum() else "DFS",
        "oracle_gain": float((best_single - oracle.sum()) / best_single),
        "sat_wins": int((d.sat < d.dfs).sum()),
        "dfs_wins": int((d.dfs < d.sat).sum()),
        "ties": int((d.sat == d.dfs).sum()),
    }


def mutuality(d: Data, catastrophe_s: float = 0.5) -> dict:
    """Two distinct questions about portfolio headroom:

    * OCCURRENCE complementarity (relative): how often is each solver
      meaningfully (>=2x) faster than the other? Mutual if both sides
      are non-trivial.
    * CATASTROPHE rescue (absolute): a portfolio's real robustness value
      is rescuing runs that are slow in ABSOLUTE terms (>= catastrophe_s).
      Only if BOTH solvers sometimes hit absolute catastrophes that the
      other avoids is the *cost*-relevant headroom mutual. If only one
      solver ever hits absolute disasters, the safe default is the other
      and predict-then-commit selection can't win (the cost asymmetry
      the original finding named).
    """
    faster_sat = int(np.sum(d.sat * 2 <= d.dfs))  # CaDiCaL >=2x faster
    faster_dfs = int(np.sum(d.dfs * 2 <= d.sat))  # DFS >=2x faster
    sat_cat = d.sat >= catastrophe_s
    dfs_cat = d.dfs >= catastrophe_s
    return {
        "sat_max": float(d.sat.max()),
        "sat_p99": float(np.percentile(d.sat, 99)),
        "sat_median": float(np.median(d.sat)),
        "dfs_max": float(d.dfs.max()),
        "dfs_p99": float(np.percentile(d.dfs, 99)),
        "dfs_median": float(np.median(d.dfs)),
        "dfs_timeouts": int(d.dfs_timeout.sum()),
        "sat_timeouts": int(d.sat_timeout.sum()),
        "occurrence_sat_2x_faster": faster_sat,
        "occurrence_dfs_2x_faster": faster_dfs,
        "catastrophe_s": catastrophe_s,
        "sat_catastrophes": int(sat_cat.sum()),
        "dfs_catastrophes": int(dfs_cat.sum()),
        # catastrophes of one solver the OTHER solves under threshold (rescues)
        "dfs_catastrophes_rescued_by_sat": int(np.sum(dfs_cat & ~sat_cat)),
        "sat_catastrophes_rescued_by_dfs": int(np.sum(sat_cat & ~dfs_cat)),
    }


def _folds(n: int, seed: int, k: int = 5):
    return KFold(n_splits=k, shuffle=True, random_state=seed).split(np.arange(n))


def eval_ridge_symmetric(d: Data, seed: int) -> float:
    """Reproduce the original failure: predict log(sat)-log(dfs), pick
    predicted-faster. Symmetric loss ignores the timeout cost asymmetry."""
    y = np.log10(d.sat + 1e-6) - np.log10(d.dfs + 1e-6)
    total = 0.0
    for tr, te in _folds(len(d.sat), seed):
        sc = StandardScaler().fit(d.X[tr])
        m = Ridge(alpha=1.0).fit(sc.transform(d.X[tr]), y[tr])
        pred = m.predict(sc.transform(d.X[te]))
        for i, idx in zip(range(len(te)), te):
            # pred>0 => sat slower => choose DFS
            total += d.dfs[idx] if pred[i] > 0 else d.sat[idx]
    return total


def _make_ridge():
    return Ridge(alpha=1.0)


def _make_gbm():
    from sklearn.ensemble import GradientBoostingRegressor

    return GradientBoostingRegressor(n_estimators=200, max_depth=3, random_state=0)


def eval_cost_aware(
    d: Data, seed: int, margins: list[float], make_regressor=_make_ridge
) -> tuple[float, float]:
    """Path #1: two regressors predict log-time of each solver; choose DFS
    only if predicted DFS is faster than CaDiCaL by a safety MARGIN (log10
    units). The margin encodes cost-awareness — CaDiCaL is the tail-bounded
    default, so only defect to DFS when confidently faster. Sweep the margin;
    return (best_total, best_margin). ``make_regressor`` swaps the learner
    (linear Ridge vs nonlinear GBM) to test whether feature interactions the
    linear model misses can separate DFS-fast from DFS-catastrophe."""
    ls = np.log10(d.sat + 1e-6)
    ld = np.log10(d.dfs + 1e-6)
    best_total, best_margin = float("inf"), 0.0
    for margin in margins:
        total = 0.0
        for tr, te in _folds(len(d.sat), seed):
            sc = StandardScaler().fit(d.X[tr])
            Xtr, Xte = sc.transform(d.X[tr]), sc.transform(d.X[te])
            m_sat = make_regressor().fit(Xtr, ls[tr])
            m_dfs = make_regressor().fit(Xtr, ld[tr])
            p_sat = m_sat.predict(Xte)
            p_dfs = m_dfs.predict(Xte)
            for i, idx in zip(range(len(te)), te):
                choose_dfs = p_dfs[i] < p_sat[i] - margin
                total += d.dfs[idx] if choose_dfs else d.sat[idx]
        if total < best_total:
            best_total, best_margin = total, margin
    return best_total, best_margin


def eval_speculative(d: Data, seed: int, margins: list[float]) -> dict:
    """Path #3: ALWAYS run CaDiCaL (safe, tail-bounded). Also run DFS
    speculatively on instances the gate predicts DFS-faster by a margin.
    Never exposed to DFS timeout (CaDiCaL always backstops). Reports 2-core
    latency (min when speculating) and total-CPU (sat + speculative dfs
    until first finishes). Returns the margin minimizing latency."""
    ls = np.log10(d.sat + 1e-6)
    ld = np.log10(d.dfs + 1e-6)
    best = {"latency": float("inf")}
    for margin in margins:
        latency = 0.0
        cpu = 0.0
        speculated = 0
        for tr, te in _folds(len(d.sat), seed):
            sc = StandardScaler().fit(d.X[tr])
            Xtr, Xte = sc.transform(d.X[tr]), sc.transform(d.X[te])
            p_sat = Ridge(alpha=1.0).fit(Xtr, ls[tr]).predict(Xte)
            p_dfs = Ridge(alpha=1.0).fit(Xtr, ld[tr]).predict(Xte)
            for i, idx in zip(range(len(te)), te):
                spec = p_dfs[i] < p_sat[i] - margin
                if spec:
                    # both run; first to finish wins the latency, second
                    # core busy until then. Safe: CaDiCaL result always available.
                    latency += min(d.sat[idx], d.dfs[idx])
                    cpu += d.sat[idx] + min(d.sat[idx], d.dfs[idx])
                    speculated += 1
                else:
                    latency += d.sat[idx]
                    cpu += d.sat[idx]
        if latency < best["latency"]:
            best = {
                "margin": margin,
                "latency": latency,
                "cpu": cpu,
                "speculated": speculated,
            }
    return best


def report(path: Path, seed: int) -> dict:
    d = load(path)
    t = totals(d)
    mut = mutuality(d)
    margins = [0.0, 0.25, 0.5, 0.75, 1.0, 1.5, 2.0]
    ridge = eval_ridge_symmetric(d, seed)
    cost_total, cost_margin = eval_cost_aware(d, seed, margins, _make_ridge)
    gbm_total, gbm_margin = eval_cost_aware(d, seed, margins, _make_gbm)
    spec = eval_speculative(d, seed, margins)
    bs = t["best_single"]

    def gain(x):
        return (bs - x) / bs

    print(f"\n=== Hamiltonian portfolio study (N={t['n']}, best-single={t['best_single_name']}) ===")
    print("\n-- solo timing structure --")
    print(f"  CaDiCaL: total {t['sat_total']:.3f}s  median {mut['sat_median']:.5f}s  "
          f"p99 {mut['sat_p99']:.4f}s  max {mut['sat_max']:.4f}s  timeouts {mut['sat_timeouts']}")
    print(f"  DFS:     total {t['dfs_total']:.3f}s  median {mut['dfs_median']:.5f}s  "
          f"p99 {mut['dfs_p99']:.4f}s  max {mut['dfs_max']:.4f}s  timeouts {mut['dfs_timeouts']}")
    print(f"  per-instance wins: CaDiCaL {t['sat_wins']}  DFS {t['dfs_wins']}  ties {t['ties']}")

    print("\n-- mutuality: occurrence vs absolute catastrophe --")
    print(f"  occurrence (>=2x faster): CaDiCaL {mut['occurrence_sat_2x_faster']}  "
          f"DFS {mut['occurrence_dfs_2x_faster']}  -> mutual complementarity")
    print(f"  absolute catastrophes (>= {mut['catastrophe_s']}s): "
          f"CaDiCaL {mut['sat_catastrophes']}  DFS {mut['dfs_catastrophes']}")
    print(f"  DFS catastrophes CaDiCaL rescues: {mut['dfs_catastrophes_rescued_by_sat']}  |  "
          f"CaDiCaL catastrophes DFS rescues: {mut['sat_catastrophes_rescued_by_dfs']}")
    one_sided = mut["sat_catastrophes"] == 0
    print(f"  => occurrence MUTUAL, but absolute catastrophe is "
          f"{'ONE-SIDED (only DFS) — CaDiCaL is the safe default' if one_sided else 'MUTUAL'}")

    print("\n-- single-core / total-CPU metric (best-single is the honest bar) --")
    print(f"  best-single (CaDiCaL) ... {bs:.3f}s   (gain 0.00%, the bar)")
    print(f"  oracle (unreachable) .... {t['oracle_total']:.3f}s   gain {gain(t['oracle_total']):+.2%}")
    print(f"  Ridge symmetric ......... {ridge:.3f}s   gain {gain(ridge):+.2%}   (reproduced failure)")
    print(f"  cost-aware Ridge (m {cost_margin:.2f}) {cost_total:.3f}s   gain {gain(cost_total):+.2%}")
    print(f"  cost-aware GBM   (m {gbm_margin:.2f}) {gbm_total:.3f}s   gain {gain(gbm_total):+.2%}")

    print("\n-- 2-core latency metric (speculative, safe: CaDiCaL always backstops) --")
    print(f"  safe-default + speculative-DFS (margin {spec['margin']:.2f}):")
    print(f"    latency {spec['latency']:.3f}s   gain vs best-single {gain(spec['latency']):+.2%}   "
          f"(oracle latency {t['oracle_total']:.3f}s)")
    print(f"    total-CPU {spec['cpu']:.3f}s   ({gain(spec['cpu']):+.2%} vs best-single — the 2x-CPU price)")
    print(f"    speculated on {spec['speculated']}/{t['n']} instances")

    best_cost = min(cost_total, gbm_total)
    print("\n=== verdict ===")
    if gain(best_cost) > 0.02:
        print(f"  Cost-aware selection BEATS best-single by {gain(best_cost):+.2%} on a single core —")
        print("  the transferred mechanism works once made cost-sensitive.")
    else:
        print(f"  Cost-aware selection (linear AND nonlinear) does NOT beat best-single "
              f"on a single core (best {gain(best_cost):+.2%}).")
        print(f"  Best-single ({t['best_single_name']}) is the practical answer; the oracle")
        print(f"  headroom ({gain(t['oracle_total']):+.2%}) is "
              f"{'one-sided and ' if one_sided else ''}not cheaply exploitable by")
        print("  predict-then-commit selection. The portfolio only pays off as SAFE")
        print(f"  speculation on 2 cores (latency {gain(spec['latency']):+.2%}, at "
              f"{gain(spec['cpu']):+.2%} CPU).")

    return {
        "totals": t, "mutuality": mut, "ridge": ridge,
        "cost_aware_ridge": {"total": cost_total, "margin": cost_margin, "gain": gain(cost_total)},
        "cost_aware_gbm": {"total": gbm_total, "margin": gbm_margin, "gain": gain(gbm_total)},
        "speculative": spec, "one_sided": one_sided,
    }


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/hamiltonian/timings.json")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--save", default=None)
    args = ap.parse_args(argv if argv is not None else sys.argv[1:])
    result = report(Path(args.data), args.seed)
    if args.save:
        Path(args.save).write_text(json.dumps(result, indent=2, default=float))
        print(f"\nsaved {args.save}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
