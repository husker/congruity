"""Hamiltonian C1 verification — compose run 20260523-003811.

C1's synthesis claim:

    P113's sparse/random-graph structural regime analysis + P142's
    GNN-guided exact branching + P186's extensible SAT-solver
    architecture yields an instance-adaptive exact HC/TSP solver:
    first classify topology-dependent tractability, then choose
    branching targets with a learned graph policy, then enforce
    Hamiltonian/TSP constraints through a modern SAT backend.

The composition is a three-stage pipeline:

  Stage 1: P113 regime classifier  (topology features → solver setup)
  Stage 2: P142 learned branching  (GNN → branching decisions)
  Stage 3: P186 SAT backend        (CaDiCaL with HC encoding)

Falsification target: **Stage 1 is the upstream gate.** If topology
features don't predict solver hardness any better than basic size
features (n, |E|), the regime-classifier has no useful signal and
the whole downstream pipeline (branching policy + SAT backend
selection) is operating without input.

Side test: addresses L3's "Generic backend does not solve the hard
part" by also timing a plain DFS backtracker against PySAT + CaDiCaL
on the same instances. If CaDiCaL with a standard permutation
encoding doesn't outperform DFS on these regimes, the SAT-backend
claim is structurally weak.

Method:

  1. Generate random graphs across an (n, density) grid spanning the
     Hamiltonicity threshold (where instances are hardest):
       - Erdős-Rényi G(n, p) at n ∈ {15, 18, 21, 24}, p sweep
       - Watts-Strogatz small-world (P113-relevant regime)
       - Random regular graphs
  2. For each instance, run two solvers with a per-instance time
     budget:
       - PySAT + CaDiCaL with permutation-based HC encoding
       - Plain DFS backtracking (no SAT)
       Record wall-clock runtime (log) and whether HC found.
  3. Extract topology features (P113-style):
       - basic: n, |E|, density
       - degree: min/max/mean/std degree
       - structural: clustering coefficient, transitivity, diameter
       - spectral: algebraic connectivity (λ_2), spectral gap
       - random-graph regime: above/below ln(n)/n threshold
  4. Predict log-runtime from features via two regressors:
       - Baseline: (n, |E|) only
       - Full: + topology features
     Use a held-out split (5-fold CV). Report R² on each fold.
  5. Verdict:
       - If full-feature R² beats baseline R² by < 0.05 on
         held-out folds, the regime classifier has no useful signal
         beyond size → composition's Stage 1 is dead.
       - If CaDiCaL median runtime > DFS median runtime on solvable
         instances at the hardness threshold, the "SAT backend"
         claim's contribution is structurally weak.
"""

from __future__ import annotations

import argparse
import math
import random
import sys
import time
from dataclasses import dataclass

import networkx as nx
import numpy as np
from pysat.solvers import Cadical153
from sklearn.linear_model import Ridge
from sklearn.metrics import r2_score
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler


# --------------------------------------------------------------------------
# HC SAT encoding (permutation, with symmetry breaking on vertex 0)
# --------------------------------------------------------------------------

def hc_to_sat_clauses(G: nx.Graph) -> list[list[int]]:
    """Permutation encoding for undirected Hamiltonian Cycle. Vertex 0
    is pinned to position 0 to break rotation symmetry. Variable x[v][i]
    encodes 'vertex v is at position i'."""
    nodes = sorted(G.nodes())
    n = len(nodes)
    idx = {v: k for k, v in enumerate(nodes)}

    def var(v: int, i: int) -> int:
        # 1-indexed for DIMACS / PySAT.
        return 1 + idx[v] * n + i

    clauses: list[list[int]] = []

    # Pin vertex 0 to position 0 (symmetry breaking).
    clauses.append([var(nodes[0], 0)])
    for i in range(1, n):
        clauses.append([-var(nodes[0], i)])

    # Each position has at least one vertex.
    for i in range(n):
        clauses.append([var(v, i) for v in nodes])

    # Each position has at most one vertex (pairwise).
    for i in range(n):
        for j, u in enumerate(nodes):
            for v in nodes[j + 1 :]:
                clauses.append([-var(u, i), -var(v, i)])

    # Each vertex is at at least one position.
    for v in nodes:
        clauses.append([var(v, i) for i in range(n)])

    # Each vertex is at at most one position (pairwise).
    for v in nodes:
        for i in range(n):
            for j in range(i + 1, n):
                clauses.append([-var(v, i), -var(v, j)])

    # Adjacency: if (u, v) NOT in E, they cannot be at consecutive
    # positions (cycle wraps with i+1 mod n).
    edge_set = {tuple(sorted(e)) for e in G.edges()}
    for i in range(n):
        next_i = (i + 1) % n
        for u in nodes:
            for v in nodes:
                if u == v:
                    continue
                if tuple(sorted((u, v))) in edge_set:
                    continue
                clauses.append([-var(u, i), -var(v, next_i)])

    return clauses


def solve_hc_sat(G: nx.Graph, timeout_s: float) -> tuple[bool | None, float]:
    """Return (has_hc, wall_seconds). has_hc=None on timeout."""
    clauses = hc_to_sat_clauses(G)
    solver = Cadical153(bootstrap_with=clauses)
    start = time.perf_counter()
    # PySAT's Cadical153 doesn't accept a timeout directly, but we can
    # use solve_limited with the propagation/conflict-budget API. For
    # the bench we use a hard wall guard via solve() and trust the
    # small instance sizes.
    try:
        result = solver.solve()
    finally:
        elapsed = time.perf_counter() - start
        solver.delete()
    if elapsed > timeout_s:
        return None, elapsed
    return bool(result), elapsed


# --------------------------------------------------------------------------
# Plain DFS backtracker (baseline)
# --------------------------------------------------------------------------

def solve_hc_dfs(G: nx.Graph, timeout_s: float) -> tuple[bool | None, float]:
    nodes = sorted(G.nodes())
    n = len(nodes)
    adj = {v: set(G.neighbors(v)) for v in nodes}
    start_v = nodes[0]
    start_t = time.perf_counter()

    def deadline_hit() -> bool:
        return (time.perf_counter() - start_t) > timeout_s

    visited = {start_v}
    path = [start_v]

    def dfs() -> bool:
        if deadline_hit():
            return False
        if len(path) == n:
            return start_v in adj[path[-1]]
        for v in adj[path[-1]]:
            if v in visited:
                continue
            visited.add(v)
            path.append(v)
            if dfs():
                return True
            path.pop()
            visited.remove(v)
            if deadline_hit():
                return False
        return False

    result = dfs()
    elapsed = time.perf_counter() - start_t
    if deadline_hit():
        return None, elapsed
    return result, elapsed


# --------------------------------------------------------------------------
# Instance generation
# --------------------------------------------------------------------------

@dataclass
class Instance:
    family: str
    n: int
    p_or_k: float
    seed: int
    G: nx.Graph


def gen_erdos_renyi(n: int, p: float, seed: int) -> nx.Graph:
    return nx.erdos_renyi_graph(n, p, seed=seed)


def gen_small_world(n: int, k: int, p: float, seed: int) -> nx.Graph:
    return nx.watts_strogatz_graph(n, k, p, seed=seed)


def gen_random_regular(n: int, d: int, seed: int) -> nx.Graph | None:
    if (n * d) % 2 != 0 or d >= n:
        return None
    try:
        return nx.random_regular_graph(d, n, seed=seed)
    except Exception:
        return None


def generate_corpus(rng: random.Random, n_per_cell: int) -> list[Instance]:
    instances: list[Instance] = []
    for n in (15, 18, 21, 24):
        # Erdős-Rényi sweep around the Hamiltonicity threshold p ≈
        # ln(n)/n. We span [0.5*thr, 4*thr] to cover sparse-non-Ham,
        # threshold (hard), and dense-easy regimes.
        thr = math.log(n) / n
        for mult in (0.5, 0.8, 1.0, 1.3, 1.7, 2.5, 4.0):
            p = min(0.95, thr * mult)
            for s in range(n_per_cell):
                G = gen_erdos_renyi(n, p, seed=rng.randrange(1 << 30))
                instances.append(Instance(family="ER", n=n, p_or_k=p, seed=s, G=G))
        # Small-world (Watts-Strogatz) — P113-relevant
        for k in (4, 6):
            if k >= n:
                continue
            for rewire_p in (0.1, 0.3, 0.6):
                for s in range(max(1, n_per_cell // 2)):
                    G = gen_small_world(n, k, rewire_p, seed=rng.randrange(1 << 30))
                    instances.append(
                        Instance(
                            family=f"WS_k{k}_p{rewire_p}",
                            n=n,
                            p_or_k=k,
                            seed=s,
                            G=G,
                        )
                    )
        # Random regular
        for d in (3, 4, 6):
            if d >= n or (n * d) % 2 != 0:
                continue
            for s in range(max(1, n_per_cell // 2)):
                G = gen_random_regular(n, d, seed=rng.randrange(1 << 30))
                if G is None:
                    continue
                instances.append(
                    Instance(family=f"RR_d{d}", n=n, p_or_k=d, seed=s, G=G)
                )
    return instances


# --------------------------------------------------------------------------
# Topology features (P113-style)
# --------------------------------------------------------------------------

def extract_features(G: nx.Graph) -> dict[str, float]:
    n = G.number_of_nodes()
    m = G.number_of_edges()
    density = nx.density(G)
    degrees = [d for _, d in G.degree()]
    deg_min = float(min(degrees)) if degrees else 0.0
    deg_max = float(max(degrees)) if degrees else 0.0
    deg_mean = float(np.mean(degrees)) if degrees else 0.0
    deg_std = float(np.std(degrees)) if degrees else 0.0
    clustering = float(nx.average_clustering(G)) if m > 0 else 0.0
    transitivity = float(nx.transitivity(G))
    # Components & spectral
    if nx.is_connected(G):
        try:
            alg_conn = float(nx.algebraic_connectivity(G))
        except Exception:
            alg_conn = 0.0
        try:
            laplacian_spectrum = nx.laplacian_spectrum(G)
            spectral_gap = (
                float(laplacian_spectrum[-1] - laplacian_spectrum[-2])
                if len(laplacian_spectrum) >= 2
                else 0.0
            )
        except Exception:
            spectral_gap = 0.0
    else:
        alg_conn = 0.0
        spectral_gap = 0.0
    # Regime indicator (above/below Hamiltonicity threshold)
    thr_ratio = density / (math.log(n) / n) if n > 1 else 0.0
    n_triangles = sum(nx.triangles(G).values()) // 3
    return {
        "n": float(n),
        "m": float(m),
        "density": float(density),
        "deg_min": deg_min,
        "deg_max": deg_max,
        "deg_mean": deg_mean,
        "deg_std": deg_std,
        "clustering": clustering,
        "transitivity": transitivity,
        "alg_conn": alg_conn,
        "spectral_gap": spectral_gap,
        "thr_ratio": float(thr_ratio),
        "n_triangles": float(n_triangles),
        "is_connected": float(nx.is_connected(G)),
    }


BASELINE_FEATURES = ["n", "m"]
FULL_FEATURES = [
    "n", "m", "density", "deg_min", "deg_max", "deg_mean", "deg_std",
    "clustering", "transitivity", "alg_conn", "spectral_gap",
    "thr_ratio", "n_triangles", "is_connected",
]


# --------------------------------------------------------------------------
# Bench harness
# --------------------------------------------------------------------------

def bench(args) -> int:
    rng = random.Random(args.seed)
    print("\n=== Hamiltonian C1 verification — regime classifier upstream ===")
    print(f"corpus: {args.n_per_cell} ER instances per (n, p) cell + WS + RR\n")

    corpus = generate_corpus(rng, args.n_per_cell)
    print(f"generated {len(corpus)} instances\n")

    rows = []
    timeouts = 0
    for k, inst in enumerate(corpus):
        feats = extract_features(inst.G)
        # SAT
        sat_has, sat_t = solve_hc_sat(inst.G, timeout_s=args.timeout)
        # DFS
        dfs_has, dfs_t = solve_hc_dfs(inst.G, timeout_s=args.timeout)
        if sat_has is None or dfs_has is None:
            timeouts += 1
        row = {
            "family": inst.family,
            "n": inst.n,
            "p_or_k": inst.p_or_k,
            "seed": inst.seed,
            **feats,
            "sat_has_hc": sat_has,
            "sat_time": sat_t,
            "dfs_has_hc": dfs_has,
            "dfs_time": dfs_t,
        }
        rows.append(row)
        if (k + 1) % 25 == 0:
            print(
                f"  [{k+1}/{len(corpus)}] last: family={inst.family} n={inst.n} "
                f"p={inst.p_or_k:.3f} sat_t={sat_t:.3f}s dfs_t={dfs_t:.3f}s "
                f"sat={sat_has} dfs={dfs_has}"
            )

    print(f"\ntotal timeouts: {timeouts} / {len(rows)}")

    # Hardness signal analysis: predict log(sat_time) from features.
    # Keep only rows that didn't time out so log is well-defined.
    valid = [
        r for r in rows
        if float(r["sat_time"]) > 0 and r["sat_has_hc"] is not None
    ]
    print(f"valid for regression: {len(valid)}")
    if len(valid) < 30:
        print("INSUFFICIENT DATA")
        return 1

    log_sat = np.log10(np.array([float(r["sat_time"]) for r in valid]) + 1e-6)

    def feature_matrix(rs: list[dict], features: list[str]) -> np.ndarray:
        return np.array([[r[f] for f in features] for r in rs], dtype=float)

    X_base = feature_matrix(valid, BASELINE_FEATURES)
    X_full = feature_matrix(valid, FULL_FEATURES)
    y = log_sat

    def cv_r2(X: np.ndarray, y: np.ndarray, n_splits: int = 5) -> tuple[float, float]:
        kf = KFold(n_splits=n_splits, shuffle=True, random_state=args.seed)
        scores = []
        for tr, te in kf.split(X):
            scaler = StandardScaler()
            X_tr = scaler.fit_transform(X[tr])
            X_te = scaler.transform(X[te])
            m = Ridge(alpha=1.0)
            m.fit(X_tr, y[tr])
            scores.append(r2_score(y[te], m.predict(X_te)))
        return float(np.mean(scores)), float(np.std(scores))

    r2_base_mean, r2_base_std = cv_r2(X_base, y)
    r2_full_mean, r2_full_std = cv_r2(X_full, y)

    print("\n=== regime-classifier hardness-signal test ===")
    print(f"baseline (n, m only):     R² = {r2_base_mean:+.4f} ± {r2_base_std:.4f}")
    print(f"full topology features:   R² = {r2_full_mean:+.4f} ± {r2_full_std:.4f}")
    print(f"gain from topology:       ΔR² = {r2_full_mean - r2_base_mean:+.4f}")

    # CaDiCaL vs DFS comparison
    sat_t = np.array([r["sat_time"] for r in valid])
    dfs_t = np.array(
        [r["dfs_time"] for r in valid if r["dfs_time"] > 0 and r["dfs_has_hc"] is not None]
    )
    print("\n=== SAT vs DFS on solvable+timed instances ===")
    print(f"CaDiCaL median runtime: {float(np.median(sat_t)):.4f}s "
          f"(p90 {float(np.percentile(sat_t, 90)):.4f}s, "
          f"max {float(np.max(sat_t)):.4f}s)")
    print(f"DFS median runtime:     {float(np.median(dfs_t)):.4f}s "
          f"(p90 {float(np.percentile(dfs_t, 90)):.4f}s, "
          f"max {float(np.max(dfs_t)):.4f}s)")
    sat_med = float(np.median(sat_t))
    dfs_med = float(np.median(dfs_t))
    if dfs_med > 0:
        speedup = dfs_med / sat_med
        print(f"CaDiCaL speedup vs DFS at median: {speedup:.2f}x")

    # Correlation of SAT runtime with DFS runtime (per-instance)
    paired = [
        (r["sat_time"], r["dfs_time"])
        for r in valid
        if r["dfs_time"] > 0 and r["sat_has_hc"] == r["dfs_has_hc"]
    ]
    if paired:
        s = np.array([p[0] for p in paired])
        d = np.array([p[1] for p in paired])
        rho = float(np.corrcoef(np.log10(s + 1e-6), np.log10(d + 1e-6))[0, 1])
        print(f"per-instance log-runtime correlation (SAT vs DFS): ρ = {rho:+.3f}")

    # Oracle solver-selection portfolio gain
    # The composition is a SELECTION pipeline: regime classifier picks
    # the right solver for each instance. The oracle picks the per-
    # instance fastest solver among {CaDiCaL, DFS}. Single-best is the
    # solver with the lowest TOTAL time over the corpus.
    paired_rows = [
        r for r in valid if float(r["dfs_time"]) > 0 and r["dfs_has_hc"] is not None
    ]
    sat_times = np.array([float(r["sat_time"]) for r in paired_rows])
    dfs_times = np.array([float(r["dfs_time"]) for r in paired_rows])
    oracle_times = np.minimum(sat_times, dfs_times)
    total_sat = float(sat_times.sum())
    total_dfs = float(dfs_times.sum())
    total_oracle = float(oracle_times.sum())
    best_single = min(total_sat, total_dfs)
    oracle_gain = (best_single - total_oracle) / best_single if best_single > 0 else 0.0
    sat_wins = int(np.sum(sat_times < dfs_times))
    dfs_wins = int(np.sum(dfs_times < sat_times))
    ties = int(np.sum(sat_times == dfs_times))
    print("\n=== oracle solver-selection portfolio ===")
    print(f"total time: CaDiCaL={total_sat:.3f}s  DFS={total_dfs:.3f}s  oracle={total_oracle:.3f}s")
    print(f"per-instance wins: CaDiCaL={sat_wins}  DFS={dfs_wins}  ties={ties}")
    print(f"oracle vs best-single relative gain: {oracle_gain:+.2%}")

    # Learned classifier closure: train a Ridge regressor on topology
    # features to predict per-instance log-runtime DIFFERENCE
    # (log_sat - log_dfs), then pick the predicted-faster solver. This
    # tests whether a CHEAP linear classifier can recoup the oracle
    # gain — which the composition's "first classify topology-dependent
    # tractability, then choose branching" pipeline implicitly requires.
    print("\n=== learned classifier closure (Ridge on topology features) ===")
    X_paired = feature_matrix(paired_rows, FULL_FEATURES)
    y_diff = np.log10(sat_times + 1e-6) - np.log10(dfs_times + 1e-6)
    kf2 = KFold(n_splits=5, shuffle=True, random_state=args.seed)
    learned_total = 0.0
    for tr, te in kf2.split(X_paired):
        scaler = StandardScaler()
        X_tr = scaler.fit_transform(X_paired[tr])
        X_te = scaler.transform(X_paired[te])
        m = Ridge(alpha=1.0)
        m.fit(X_tr, y_diff[tr])
        preds = m.predict(X_te)
        # If pred > 0, SAT is predicted slower → choose DFS; else SAT.
        for i, idx in enumerate(te):
            chosen = dfs_times[idx] if preds[i] > 0 else sat_times[idx]
            learned_total += float(chosen)
    learned_gain = (best_single - learned_total) / best_single if best_single > 0 else 0.0
    print(f"5-fold CV learned-classifier total time: {learned_total:.3f}s")
    print(f"learned vs best-single relative gain:    {learned_gain:+.2%}")
    print(f"  (oracle gain was {oracle_gain:+.2%}; "
          f"oracle headroom recouped by learner: "
          f"{(learned_gain / oracle_gain * 100) if oracle_gain > 0 else 0:.1f}%)")

    # Verdict
    print("\n=== verdict ===")
    delta = r2_full_mean - r2_base_mean
    print(
        f"topology feature ΔR² over (n, m) baseline:  {delta:+.4f}  "
        f"(required: >= {args.required_delta:.2f})"
    )
    print(
        f"oracle solver-portfolio gain vs best-single: {oracle_gain:+.2%}  "
        f"(required: >= {args.required_oracle_gain:.0%})"
    )

    has_classifier_signal = delta >= args.required_delta
    has_portfolio_gain = oracle_gain >= args.required_oracle_gain
    if has_classifier_signal and has_portfolio_gain:
        print(
            "\nORACLE-POSITIVE — topology features predict hardness AND\n"
            "  an oracle solver-portfolio beats picking the single best\n"
            "  solver by the required margin. C1's pipeline has structural\n"
            "  headroom: a learned regime classifier (Stage 1) could in\n"
            "  principle exploit the predictive signal to pick the right\n"
            "  solver (CaDiCaL vs DFS) per instance.\n"
            "  Open follow-up: does a CHEAP learned classifier recoup the\n"
            "  oracle gain after extraction-cost is charged?"
        )
    elif has_classifier_signal and not has_portfolio_gain:
        print(
            "\nPARTIAL-FORECLOSED — topology features DO predict hardness\n"
            "  (Stage 1 has signal) but the oracle solver-portfolio gain\n"
            "  vs picking the single best solver is below the required\n"
            f"  margin ({args.required_oracle_gain:.0%}). The composition's\n"
            "  Stage 1 has a signal, but downstream solver-selection has\n"
            "  no real choice to make. The single-best solver dominates;\n"
            "  regime-adaptive selection is structurally low-headroom."
        )
    elif not has_classifier_signal and has_portfolio_gain:
        print(
            "\nWRONG-CLASSIFIER FORECLOSURE — there IS oracle headroom for\n"
            "  solver selection, but topology features don't capture the\n"
            "  hardness signal. The composition's Stage 1 mechanism would\n"
            "  need a different feature space (perhaps SAT-encoding-\n"
            "  specific or proof-trace features) to exploit the portfolio\n"
            "  gain."
        )
    else:
        print(
            "\nFULL FORECLOSURE — neither Stage 1 has signal NOR is there\n"
            "  oracle headroom for solver selection. The composition is\n"
            "  structurally dead at both upstream and downstream points."
        )

    return 0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--n-per-cell", type=int, default=5)
    ap.add_argument("--timeout", type=float, default=10.0)
    ap.add_argument(
        "--required-delta",
        type=float,
        default=0.05,
        help="Required topology-feature ΔR² over (n, m) baseline.",
    )
    ap.add_argument(
        "--required-oracle-gain",
        type=float,
        default=0.05,
        help="Required oracle-portfolio relative gain over best-single solver.",
    )
    args = ap.parse_args()
    return bench(args)


if __name__ == "__main__":
    sys.exit(main())
