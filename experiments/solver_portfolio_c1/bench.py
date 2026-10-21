"""Solver-portfolio C1 verification — compose run 20260525-011650.

C1's plain-language hypothesis:

    If route instances have measurable structural fingerprints, those
    fingerprints should help choose which cheap solver strategy to run.

This harness tests the cheapest CPU-only version of that claim on generated
metric TSP instances. It does not train GNNs or call external solvers. Instead,
it asks whether classical geometry/graph features plus unsupervised
density-peak regimes can recover enough solver-behavior signal to beat simple
portfolio baselines.

The benchmark:

  1. Generate Euclidean TSP-like instances across several local families.
  2. Run a small portfolio of deterministic construction/improvement
     heuristics on every instance.
  3. Convert length/effort tradeoffs into a per-instance solver loss matrix.
  4. Compare selectors:
       - single-best solver from the training fold
       - generator-family selector
       - direct supervised selector over features
       - density-peak cluster-first selector over features
  5. Report selection regret against the per-instance oracle.

Stop rule:

  - clean-positive if density-peak cluster selection beats single-best by the
    requested margin and is competitive with direct supervised selection.
  - partial-positive if the direct selector works but the cluster-first
    mechanism does not.
  - clean-negative if neither representation nor cluster regimes beat
    single-best.

Run:
    uv run python -m experiments.solver_portfolio_c1.bench --smoke
    uv run python -m experiments.solver_portfolio_c1.bench --write-findings
"""

from __future__ import annotations

import argparse
import math
import sys
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler

RUN_ID = "20260525-011650"
COMPOSITION_ID = "C1"
SOLVER_NAMES = [
    "nearest_neighbor",
    "angle_sweep",
    "axis_sweep",
    "farthest_insertion",
    "cheapest_insertion",
    "two_opt_light",
    "two_opt_full",
]
FEATURE_NAMES = [
    "n",
    "bbox_aspect",
    "bbox_area",
    "centroid_radius_mean",
    "centroid_radius_std",
    "cov_anisotropy",
    "pair_dist_mean",
    "pair_dist_std",
    "pair_dist_cv",
    "nn_mean",
    "nn_std",
    "nn_cv",
    "nn_p90",
    "mst_length",
    "mst_edge_mean",
    "mst_edge_std",
    "mst_edge_max",
]


@dataclass(frozen=True)
class RouteInstance:
    family: str
    n: int
    seed: int
    coords: np.ndarray


@dataclass(frozen=True)
class SolverRun:
    name: str
    length: float
    effort_units: float
    elapsed_s: float


@dataclass(frozen=True)
class DensityPeakModel:
    centers: np.ndarray
    center_features: np.ndarray
    scaler_mean: np.ndarray
    scaler_scale: np.ndarray
    labels: np.ndarray


@dataclass(frozen=True)
class SelectorResult:
    name: str
    mean_loss: float
    mean_oracle_loss: float
    mean_regret: float
    relative_regret_reduction: float
    chosen_solver_names: list[str]


@dataclass(frozen=True)
class FoldResult:
    fold: int
    single_best: SelectorResult
    family: SelectorResult
    direct_supervised: SelectorResult
    density_cluster: SelectorResult
    cluster_purity: float
    random_cluster_purity: float
    cluster_loss_variance: float
    random_loss_variance: float


@dataclass(frozen=True)
class BenchReport:
    n_instances: int
    n_folds: int
    effort_penalty: float
    improvement_margin: float
    supervised_tolerance: float
    fold_results: list[FoldResult]
    summary: dict[str, float | str]
    verdict: str


# ---------------------------------------------------------------------------
# TSP heuristics
# ---------------------------------------------------------------------------


def distance_matrix(coords: np.ndarray) -> np.ndarray:
    diff = coords[:, None, :] - coords[None, :, :]
    return np.sqrt(np.sum(diff * diff, axis=2))


def tour_length(tour: list[int], dist: np.ndarray) -> float:
    return float(sum(dist[tour[i], tour[(i + 1) % len(tour)]] for i in range(len(tour))))


def nearest_neighbor_tour(dist: np.ndarray, start: int = 0) -> list[int]:
    n = dist.shape[0]
    unvisited = set(range(n))
    tour = [start]
    unvisited.remove(start)
    while unvisited:
        last = tour[-1]
        nxt = min(unvisited, key=lambda j: (dist[last, j], j))
        tour.append(int(nxt))
        unvisited.remove(nxt)
    return tour


def farthest_insertion_tour(dist: np.ndarray) -> list[int]:
    n = dist.shape[0]
    a, b = np.unravel_index(np.argmax(dist), dist.shape)
    tour = [int(a), int(b)]
    remaining = set(range(n)) - set(tour)

    while remaining:
        nxt = max(remaining, key=lambda j: min(dist[j, i] for i in tour))
        best_pos = 0
        best_delta = math.inf
        for pos in range(len(tour)):
            left = tour[pos]
            right = tour[(pos + 1) % len(tour)]
            delta = dist[left, nxt] + dist[nxt, right] - dist[left, right]
            if delta < best_delta:
                best_delta = float(delta)
                best_pos = pos + 1
        tour.insert(best_pos, int(nxt))
        remaining.remove(nxt)
    return tour


def cheapest_insertion_tour(dist: np.ndarray) -> list[int]:
    n = dist.shape[0]
    a, b = np.unravel_index(np.argmax(dist), dist.shape)
    tour = [int(a), int(b)]
    remaining = set(range(n)) - set(tour)

    while remaining:
        best_node = -1
        best_pos = 0
        best_delta = math.inf
        for node in remaining:
            for pos in range(len(tour)):
                left = tour[pos]
                right = tour[(pos + 1) % len(tour)]
                delta = dist[left, node] + dist[node, right] - dist[left, right]
                if delta < best_delta:
                    best_delta = float(delta)
                    best_node = int(node)
                    best_pos = pos + 1
        tour.insert(best_pos, best_node)
        remaining.remove(best_node)
    return tour


def angle_sweep_tour(coords: np.ndarray) -> list[int]:
    center = coords.mean(axis=0)
    angles = np.arctan2(coords[:, 1] - center[1], coords[:, 0] - center[0])
    return [int(i) for i in np.argsort(angles)]


def axis_sweep_tour(coords: np.ndarray) -> list[int]:
    centered = coords - coords.mean(axis=0)
    cov = np.cov(centered.T) if len(coords) > 2 else np.eye(2)
    eigvals, eigvecs = np.linalg.eigh(cov)
    axis = eigvecs[:, int(np.argmax(eigvals))]
    projection = centered @ axis
    return [int(i) for i in np.argsort(projection)]


def two_opt(tour: list[int], dist: np.ndarray, max_passes: int = 5) -> list[int]:
    out = list(tour)
    n = len(out)
    if n < 4:
        return out

    for _ in range(max_passes):
        improved = False
        for i in range(n - 1):
            a = out[i]
            b = out[(i + 1) % n]
            for k in range(i + 2, n):
                if i == 0 and k == n - 1:
                    continue
                c = out[k]
                d = out[(k + 1) % n]
                old = dist[a, b] + dist[c, d]
                new = dist[a, c] + dist[b, d]
                if new + 1e-12 < old:
                    out[i + 1 : k + 1] = reversed(out[i + 1 : k + 1])
                    improved = True
        if not improved:
            break
    return out


def run_solvers(coords: np.ndarray) -> list[SolverRun]:
    dist = distance_matrix(coords)
    n = len(coords)
    solver_specs = [
        ("nearest_neighbor", lambda: nearest_neighbor_tour(dist), 1.0),
        ("angle_sweep", lambda: angle_sweep_tour(coords), 0.8),
        ("axis_sweep", lambda: axis_sweep_tour(coords), 0.8),
        ("farthest_insertion", lambda: farthest_insertion_tour(dist), 2.5),
        ("cheapest_insertion", lambda: cheapest_insertion_tour(dist), 4.0),
        ("two_opt_light", lambda: two_opt(nearest_neighbor_tour(dist), dist, max_passes=2), 4.5),
        ("two_opt_full", lambda: two_opt(nearest_neighbor_tour(dist), dist, max_passes=10), 9.0),
    ]
    runs: list[SolverRun] = []
    for name, fn, effort_multiplier in solver_specs:
        start = time.perf_counter()
        tour = fn()
        elapsed = time.perf_counter() - start
        runs.append(
            SolverRun(
                name=name,
                length=tour_length(tour, dist),
                effort_units=effort_multiplier * n * n,
                elapsed_s=elapsed,
            )
        )
    return runs


def solver_losses(runs: list[SolverRun], n: int, effort_penalty: float) -> np.ndarray:
    best_length = min(r.length for r in runs)
    out = []
    for run in runs:
        length_gap = run.length / best_length - 1.0
        effort_charge = effort_penalty * (run.effort_units / (n * n))
        out.append(length_gap + effort_charge)
    return np.array(out, dtype=float)


# ---------------------------------------------------------------------------
# Instance generation and features
# ---------------------------------------------------------------------------


def _normalize_coords(coords: np.ndarray) -> np.ndarray:
    lo = coords.min(axis=0)
    hi = coords.max(axis=0)
    span = np.maximum(hi - lo, 1e-9)
    return (coords - lo) / span


def _instance_coords(family: str, n: int, rng: np.random.Generator) -> np.ndarray:
    if family == "uniform":
        return rng.random((n, 2))

    if family == "clustered":
        k = int(rng.integers(2, 5))
        centers = rng.random((k, 2))
        labels = rng.integers(0, k, size=n)
        coords = centers[labels] + rng.normal(0.0, 0.055, size=(n, 2))
        return np.clip(coords, 0.0, 1.0)

    if family == "line":
        x = np.sort(rng.random(n))
        slope = rng.uniform(-0.5, 0.5)
        y = 0.5 + slope * (x - 0.5) + rng.normal(0.0, 0.035, size=n)
        return _normalize_coords(np.column_stack([x, y]))

    if family == "grid":
        side = int(math.ceil(math.sqrt(n)))
        grid = np.array([(i, j) for i in range(side) for j in range(side)], dtype=float)
        picks = rng.choice(len(grid), size=n, replace=False)
        coords = grid[picks] / max(1, side - 1)
        coords += rng.normal(0.0, 0.025, size=(n, 2))
        return np.clip(coords, 0.0, 1.0)

    if family == "ring":
        theta = np.sort(rng.random(n) * 2 * math.pi)
        radius = 0.38 + rng.normal(0.0, 0.035, size=n)
        coords = np.column_stack([0.5 + radius * np.cos(theta), 0.5 + radius * np.sin(theta)])
        return np.clip(coords, 0.0, 1.0)

    raise ValueError(f"unknown family {family!r}")


def generate_instances(
    sizes: list[int],
    seeds_per_family: int,
    base_seed: int,
    families: list[str] | None = None,
) -> list[RouteInstance]:
    families = families or ["uniform", "clustered", "line", "grid", "ring"]
    family_offsets = {family: i * 1_000_003 for i, family in enumerate(families)}
    instances: list[RouteInstance] = []
    for family in families:
        for n in sizes:
            for seed in range(seeds_per_family):
                mixed_seed = base_seed + 10_003 * seed + 101 * n + family_offsets[family]
                rng = np.random.default_rng(mixed_seed)
                instances.append(
                    RouteInstance(
                        family=family,
                        n=n,
                        seed=seed,
                        coords=_instance_coords(family, n, rng),
                    )
                )
    return instances


def mst_edge_lengths(dist: np.ndarray) -> np.ndarray:
    n = dist.shape[0]
    selected = np.zeros(n, dtype=bool)
    selected[0] = True
    best = dist[0].copy()
    best[0] = math.inf
    edges: list[float] = []
    for _ in range(n - 1):
        j = int(np.argmin(np.where(selected, math.inf, best)))
        edges.append(float(best[j]))
        selected[j] = True
        best = np.minimum(best, dist[j])
    return np.array(edges, dtype=float)


def extract_features(coords: np.ndarray) -> np.ndarray:
    n = len(coords)
    dist = distance_matrix(coords)
    nonzero = dist[np.triu_indices(n, k=1)]
    nn = np.partition(np.where(dist == 0.0, math.inf, dist), kth=1, axis=1)[:, 1]
    edges = mst_edge_lengths(dist)
    centered = coords - coords.mean(axis=0)
    cov = np.cov(centered.T) if n > 2 else np.eye(2)
    eig = np.linalg.eigvalsh(cov)
    cov_anisotropy = float((eig[-1] + 1e-9) / (eig[0] + 1e-9))
    bbox = coords.max(axis=0) - coords.min(axis=0)
    bbox_aspect = float(max(bbox) / max(min(bbox), 1e-9))
    bbox_area = float(max(bbox[0] * bbox[1], 1e-9))
    radii = np.sqrt(np.sum(centered * centered, axis=1))

    pair_mean = float(np.mean(nonzero))
    pair_std = float(np.std(nonzero))
    nn_mean = float(np.mean(nn))
    nn_std = float(np.std(nn))
    edge_mean = float(np.mean(edges))
    edge_std = float(np.std(edges))

    return np.array(
        [
            float(n),
            bbox_aspect,
            bbox_area,
            float(np.mean(radii)),
            float(np.std(radii)),
            cov_anisotropy,
            pair_mean,
            pair_std,
            pair_std / max(pair_mean, 1e-9),
            nn_mean,
            nn_std,
            nn_std / max(nn_mean, 1e-9),
            float(np.percentile(nn, 90)),
            float(np.sum(edges)),
            edge_mean,
            edge_std,
            float(np.max(edges)),
        ],
        dtype=float,
    )


# ---------------------------------------------------------------------------
# Density-peak clustering
# ---------------------------------------------------------------------------


def _standardize(features: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    mean = features.mean(axis=0)
    scale = features.std(axis=0)
    scale = np.where(scale < 1e-12, 1.0, scale)
    return (features - mean) / scale, mean, scale


def fit_density_peak(
    features: np.ndarray,
    n_clusters: int,
    dc_percentile: float = 20.0,
) -> DensityPeakModel:
    if len(features) < n_clusters:
        raise ValueError("n_clusters cannot exceed number of feature rows")

    x, mean, scale = _standardize(features)
    diff = x[:, None, :] - x[None, :, :]
    dist = np.sqrt(np.sum(diff * diff, axis=2))
    upper = dist[np.triu_indices(len(x), k=1)]
    dc = float(np.percentile(upper, dc_percentile)) if len(upper) else 1.0
    dc = max(dc, 1e-9)
    rho = np.exp(-((dist / dc) ** 2)).sum(axis=1) - 1.0

    order = np.argsort(-rho)
    delta = np.zeros(len(x), dtype=float)
    nearest_higher = np.full(len(x), -1, dtype=int)
    max_dist = float(np.max(dist)) if len(x) > 1 else 0.0
    delta[order[0]] = max_dist
    for rank in range(1, len(order)):
        idx = order[rank]
        higher = order[:rank]
        nearest_pos = int(np.argmin(dist[idx, higher]))
        nearest_higher[idx] = int(higher[nearest_pos])
        delta[idx] = float(dist[idx, nearest_higher[idx]])

    gamma = rho * delta
    centers = np.array(sorted(np.argsort(-gamma)[:n_clusters]), dtype=int)
    labels = np.full(len(x), -1, dtype=int)
    center_to_label = {int(c): label for label, c in enumerate(centers)}
    for c, label in center_to_label.items():
        labels[c] = label
    for idx in order:
        if labels[idx] >= 0:
            continue
        parent = nearest_higher[idx]
        labels[idx] = labels[parent] if parent >= 0 else int(np.argmin(dist[idx, centers]))

    return DensityPeakModel(
        centers=centers,
        center_features=x[centers],
        scaler_mean=mean,
        scaler_scale=scale,
        labels=labels,
    )


def predict_density_peak(model: DensityPeakModel, features: np.ndarray) -> np.ndarray:
    x = (features - model.scaler_mean) / model.scaler_scale
    diff = x[:, None, :] - model.center_features[None, :, :]
    dist = np.sqrt(np.sum(diff * diff, axis=2))
    return np.argmin(dist, axis=1).astype(int)


# ---------------------------------------------------------------------------
# Selector evaluation
# ---------------------------------------------------------------------------


def _result_from_choices(
    name: str,
    losses: np.ndarray,
    test_idx: np.ndarray,
    chosen_solver_idx: np.ndarray,
    baseline_regret: float | None = None,
    solver_names: list[str] | None = None,
) -> SelectorResult:
    solver_names = solver_names or [f"solver_{i}" for i in range(losses.shape[1])]
    test_losses = losses[test_idx]
    chosen_loss = test_losses[np.arange(len(test_idx)), chosen_solver_idx]
    oracle_loss = np.min(test_losses, axis=1)
    regret = float(np.mean(chosen_loss - oracle_loss))
    if baseline_regret is None or baseline_regret <= 1e-12:
        reduction = 0.0
    else:
        reduction = (baseline_regret - regret) / baseline_regret
    return SelectorResult(
        name=name,
        mean_loss=float(np.mean(chosen_loss)),
        mean_oracle_loss=float(np.mean(oracle_loss)),
        mean_regret=regret,
        relative_regret_reduction=float(reduction),
        chosen_solver_names=[solver_names[int(i)] for i in chosen_solver_idx],
    )


def evaluate_single_best(
    losses: np.ndarray,
    train_idx: np.ndarray,
    test_idx: np.ndarray,
    solver_names: list[str] | None = None,
) -> SelectorResult:
    best = int(np.argmin(np.mean(losses[train_idx], axis=0)))
    choices = np.full(len(test_idx), best, dtype=int)
    return _result_from_choices("single_best", losses, test_idx, choices, solver_names=solver_names)


def evaluate_family_selector(
    losses: np.ndarray,
    train_idx: np.ndarray,
    test_idx: np.ndarray,
    families: np.ndarray,
    default_solver: int,
    baseline_regret: float,
    solver_names: list[str],
) -> SelectorResult:
    choices = []
    for family in families[test_idx]:
        matching = train_idx[families[train_idx] == family]
        if len(matching):
            choices.append(int(np.argmin(np.mean(losses[matching], axis=0))))
        else:
            choices.append(default_solver)
    return _result_from_choices(
        "family_selector",
        losses,
        test_idx,
        np.array(choices, dtype=int),
        baseline_regret=baseline_regret,
        solver_names=solver_names,
    )


def evaluate_cluster_selector(
    losses: np.ndarray,
    train_idx: np.ndarray,
    test_idx: np.ndarray,
    train_clusters: np.ndarray,
    test_clusters: np.ndarray,
    baseline_regret: float | None = None,
    solver_names: list[str] | None = None,
) -> SelectorResult:
    default_solver = int(np.argmin(np.mean(losses[train_idx], axis=0)))
    cluster_solver: dict[int, int] = {}
    for cluster in sorted(set(int(c) for c in train_clusters)):
        matching = train_idx[train_clusters == cluster]
        cluster_solver[cluster] = int(np.argmin(np.mean(losses[matching], axis=0)))
    choices = np.array(
        [cluster_solver.get(int(cluster), default_solver) for cluster in test_clusters],
        dtype=int,
    )
    return _result_from_choices(
        "density_cluster",
        losses,
        test_idx,
        choices,
        baseline_regret=baseline_regret,
        solver_names=solver_names,
    )


def evaluate_direct_supervised(
    features: np.ndarray,
    losses: np.ndarray,
    train_idx: np.ndarray,
    test_idx: np.ndarray,
    baseline_regret: float,
    seed: int,
    solver_names: list[str],
) -> SelectorResult:
    y_train = np.argmin(losses[train_idx], axis=1)
    default_solver = int(np.argmin(np.mean(losses[train_idx], axis=0)))
    if len(set(int(x) for x in y_train)) < 2:
        choices = np.full(len(test_idx), default_solver, dtype=int)
    else:
        scaler = StandardScaler()
        x_train = scaler.fit_transform(features[train_idx])
        x_test = scaler.transform(features[test_idx])
        clf = RandomForestClassifier(
            n_estimators=120,
            max_depth=5,
            min_samples_leaf=3,
            random_state=seed,
            class_weight="balanced",
        )
        clf.fit(x_train, y_train)
        choices = clf.predict(x_test).astype(int)
    return _result_from_choices(
        "direct_supervised",
        losses,
        test_idx,
        choices,
        baseline_regret=baseline_regret,
        solver_names=solver_names,
    )


def cluster_purity(labels: np.ndarray, best_solver: np.ndarray) -> float:
    total = 0
    majority = 0
    for label in sorted(set(int(x) for x in labels)):
        sub = best_solver[labels == label]
        if len(sub) == 0:
            continue
        counts = np.bincount(sub, minlength=len(SOLVER_NAMES))
        majority += int(np.max(counts))
        total += len(sub)
    return float(majority / total) if total else 0.0


def cluster_loss_variance(labels: np.ndarray, losses: np.ndarray) -> float:
    variances = []
    weights = []
    for label in sorted(set(int(x) for x in labels)):
        sub = losses[labels == label]
        if len(sub) <= 1:
            continue
        variances.append(float(np.mean(np.var(sub, axis=0))))
        weights.append(len(sub))
    if not variances:
        return 0.0
    return float(np.average(variances, weights=weights))


def _random_cluster_labels(n: int, n_clusters: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    labels = np.arange(n) % n_clusters
    rng.shuffle(labels)
    return labels.astype(int)


def evaluate_portfolios(
    features: np.ndarray,
    losses: np.ndarray,
    families: np.ndarray,
    n_folds: int,
    n_clusters: int,
    seed: int,
) -> list[FoldResult]:
    splitter = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=seed)
    fold_results: list[FoldResult] = []
    best_solver = np.argmin(losses, axis=1)

    for fold, (train_idx, test_idx) in enumerate(splitter.split(features, families), start=1):
        single = evaluate_single_best(losses, train_idx, test_idx, solver_names=SOLVER_NAMES)
        default_solver = int(np.argmin(np.mean(losses[train_idx], axis=0)))
        family = evaluate_family_selector(
            losses,
            train_idx,
            test_idx,
            families,
            default_solver,
            single.mean_regret,
            SOLVER_NAMES,
        )
        direct = evaluate_direct_supervised(
            features,
            losses,
            train_idx,
            test_idx,
            single.mean_regret,
            seed + fold,
            SOLVER_NAMES,
        )

        model = fit_density_peak(features[train_idx], n_clusters=n_clusters)
        train_clusters = model.labels
        test_clusters = predict_density_peak(model, features[test_idx])
        density = evaluate_cluster_selector(
            losses,
            train_idx,
            test_idx,
            train_clusters,
            test_clusters,
            baseline_regret=single.mean_regret,
            solver_names=SOLVER_NAMES,
        )

        random_train_clusters = _random_cluster_labels(len(train_idx), n_clusters, seed + fold)
        fold_results.append(
            FoldResult(
                fold=fold,
                single_best=single,
                family=family,
                direct_supervised=direct,
                density_cluster=density,
                cluster_purity=cluster_purity(train_clusters, best_solver[train_idx]),
                random_cluster_purity=cluster_purity(
                    random_train_clusters, best_solver[train_idx]
                ),
                cluster_loss_variance=cluster_loss_variance(
                    train_clusters, losses[train_idx]
                ),
                random_loss_variance=cluster_loss_variance(
                    random_train_clusters, losses[train_idx]
                ),
            )
        )
    return fold_results


def summarize(
    fold_results: list[FoldResult],
    improvement_margin: float,
    supervised_tolerance: float,
) -> tuple[dict[str, float | str], str]:
    def mean(selector: str, metric: str) -> float:
        vals = [getattr(getattr(f, selector), metric) for f in fold_results]
        return float(np.mean(vals))

    summary: dict[str, float | str] = {
        "single_best_regret": mean("single_best", "mean_regret"),
        "family_regret": mean("family", "mean_regret"),
        "direct_supervised_regret": mean("direct_supervised", "mean_regret"),
        "density_cluster_regret": mean("density_cluster", "mean_regret"),
        "density_vs_single_regret_reduction": mean(
            "density_cluster", "relative_regret_reduction"
        ),
        "direct_vs_single_regret_reduction": mean(
            "direct_supervised", "relative_regret_reduction"
        ),
        "cluster_purity": float(np.mean([f.cluster_purity for f in fold_results])),
        "random_cluster_purity": float(
            np.mean([f.random_cluster_purity for f in fold_results])
        ),
        "cluster_loss_variance": float(
            np.mean([f.cluster_loss_variance for f in fold_results])
        ),
        "random_loss_variance": float(
            np.mean([f.random_loss_variance for f in fold_results])
        ),
    }
    density_gain = float(summary["density_vs_single_regret_reduction"])
    direct_gain = float(summary["direct_vs_single_regret_reduction"])
    density_regret = float(summary["density_cluster_regret"])
    direct_regret = float(summary["direct_supervised_regret"])
    purity_gain = float(summary["cluster_purity"]) - float(summary["random_cluster_purity"])
    variance_gain = float(summary["random_loss_variance"]) - float(summary["cluster_loss_variance"])

    density_beats_single = density_gain >= improvement_margin
    density_competitive = density_regret <= direct_regret * (1.0 + supervised_tolerance)
    regimes_have_signal = purity_gain > 0.02 or variance_gain > 0.0
    direct_beats_single = direct_gain >= improvement_margin

    if density_beats_single and density_competitive and regimes_have_signal:
        verdict = "clean-positive"
    elif direct_beats_single and not density_beats_single:
        verdict = "partial-positive"
    elif density_beats_single:
        verdict = "partial-positive"
    else:
        verdict = "clean-negative"
    summary["verdict"] = verdict
    return summary, verdict


def run_benchmark(args: argparse.Namespace) -> BenchReport:
    sizes = args.sizes
    if args.smoke:
        sizes = [18, 24]
        args.seeds_per_family = min(args.seeds_per_family, 3)
        args.folds = min(args.folds, 3)

    instances = generate_instances(sizes, args.seeds_per_family, args.seed)
    features = np.vstack([extract_features(inst.coords) for inst in instances])
    families = np.array([inst.family for inst in instances])
    losses = []
    run_rows: list[list[SolverRun]] = []
    for idx, inst in enumerate(instances, start=1):
        runs = run_solvers(inst.coords)
        run_rows.append(runs)
        losses.append(solver_losses(runs, inst.n, args.effort_penalty))
        if args.verbose and (idx == 1 or idx % 25 == 0 or idx == len(instances)):
            best = min(runs, key=lambda r: r.length)
            print(
                f"  [{idx:3d}/{len(instances)}] {inst.family:9s} n={inst.n:2d} "
                f"best_length={best.name:18s} len={best.length:.3f}"
            )
    loss_matrix = np.vstack(losses)
    fold_results = evaluate_portfolios(
        features,
        loss_matrix,
        families,
        n_folds=args.folds,
        n_clusters=args.clusters,
        seed=args.seed,
    )
    summary, verdict = summarize(
        fold_results,
        improvement_margin=args.improvement_margin,
        supervised_tolerance=args.supervised_tolerance,
    )
    return BenchReport(
        n_instances=len(instances),
        n_folds=args.folds,
        effort_penalty=args.effort_penalty,
        improvement_margin=args.improvement_margin,
        supervised_tolerance=args.supervised_tolerance,
        fold_results=fold_results,
        summary=summary,
        verdict=verdict,
    )


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------


def _fmt_pct(x: float) -> str:
    return f"{x * 100:+.1f}%"


def print_report(report: BenchReport) -> None:
    print("\n=== Solver-portfolio C1 verification ===")
    print(f"compose run: {RUN_ID}, composition: {COMPOSITION_ID}")
    print(f"instances: {report.n_instances}, folds: {report.n_folds}")
    print(
        "loss = tour-length gap vs per-instance best + "
        f"{report.effort_penalty:g} * normalized effort"
    )
    print("\n=== fold regrets vs per-instance oracle (lower is better) ===")
    print(
        "fold | single-best | family     | direct ML  | density cluster | "
        "cluster purity"
    )
    print("-" * 88)
    for fold in report.fold_results:
        print(
            f"{fold.fold:>4d} | "
            f"{fold.single_best.mean_regret:11.4f} | "
            f"{fold.family.mean_regret:10.4f} | "
            f"{fold.direct_supervised.mean_regret:10.4f} | "
            f"{fold.density_cluster.mean_regret:15.4f} | "
            f"{fold.cluster_purity:6.1%}"
        )

    s = report.summary
    print("\n=== aggregate ===")
    print(f"single-best regret:          {float(s['single_best_regret']):.4f}")
    print(f"family selector regret:      {float(s['family_regret']):.4f}")
    print(f"direct supervised regret:    {float(s['direct_supervised_regret']):.4f}")
    print(f"density-cluster regret:      {float(s['density_cluster_regret']):.4f}")
    print(
        "density regret reduction vs single-best: "
        f"{_fmt_pct(float(s['density_vs_single_regret_reduction']))}"
    )
    print(
        "direct-ML regret reduction vs single-best: "
        f"{_fmt_pct(float(s['direct_vs_single_regret_reduction']))}"
    )
    print(
        f"cluster best-solver purity:  {float(s['cluster_purity']):.1%} "
        f"(random {float(s['random_cluster_purity']):.1%})"
    )
    print(
        f"within-cluster loss variance: {float(s['cluster_loss_variance']):.5f} "
        f"(random {float(s['random_loss_variance']):.5f})"
    )
    print("\n=== verdict ===")
    print(report.verdict)


def render_markdown(report: BenchReport) -> str:
    s = report.summary
    lines = [
        "# Solver-Portfolio C1 Verification",
        "",
        f"- Originating run: `{RUN_ID}`",
        f"- Composition: `{COMPOSITION_ID}`",
        f"- Outcome: **{report.verdict}**",
        f"- Instances: {report.n_instances}",
        f"- Cross-validation folds: {report.n_folds}",
        "",
        "## Question",
        "",
        "Can cheap structural fingerprints of routing instances help choose a solver "
        "strategy better than simple baselines?",
        "",
        "## Method",
        "",
        "The harness generated local Euclidean TSP-style instances across uniform, "
        "clustered, line-like, grid-like, and ring-like families. Each instance was "
        "run through seven deterministic heuristics: nearest neighbor, angle sweep, "
        "axis sweep, farthest insertion, cheapest insertion, light 2-opt, and fuller "
        "2-opt. The selector loss charged both tour-length gap and a normalized "
        f"effort penalty of {report.effort_penalty:g}.",
        "",
        "Selectors compared:",
        "",
        "- single best solver from the training fold",
        "- generator-family selector",
        "- direct supervised random-forest selector",
        "- density-peak cluster-first selector",
        "",
        "## Results",
        "",
        "| selector | mean regret vs oracle | regret reduction vs single-best |",
        "|---|---:|---:|",
        f"| single best | {float(s['single_best_regret']):.4f} | 0.0% |",
        f"| family | {float(s['family_regret']):.4f} | "
        f"{_fmt_pct((float(s['single_best_regret']) - float(s['family_regret'])) / max(float(s['single_best_regret']), 1e-12))} |",
        f"| direct supervised | {float(s['direct_supervised_regret']):.4f} | "
        f"{_fmt_pct(float(s['direct_vs_single_regret_reduction']))} |",
        f"| density cluster | {float(s['density_cluster_regret']):.4f} | "
        f"{_fmt_pct(float(s['density_vs_single_regret_reduction']))} |",
        "",
        f"- Density-cluster best-solver purity: {float(s['cluster_purity']):.1%}",
        f"- Random-cluster best-solver purity: {float(s['random_cluster_purity']):.1%}",
        f"- Density-cluster loss variance: {float(s['cluster_loss_variance']):.5f}",
        f"- Random-cluster loss variance: {float(s['random_loss_variance']):.5f}",
        "",
        "## Interpretation",
        "",
    ]
    if report.verdict == "clean-positive":
        lines.extend(
            [
                "This is a positive local result for C1. The unsupervised density-regime "
                "mechanism beat the single-best solver baseline and stayed competitive "
                "with the direct supervised selector. That means the C1 idea has "
                "cheap, testable signal before any GNN or external solver work.",
            ]
        )
    elif report.verdict == "partial-positive":
        lines.extend(
            [
                "This is a partial result. The generated feature space contains solver-"
                "selection signal, but the density-cluster mechanism is not yet the "
                "strongest way to use it. C1 remains interesting, but the next version "
                "should treat clustering as an explanation layer or feature rather than "
                "the primary selector.",
            ]
        )
    else:
        lines.extend(
            [
                "This falsifies the cheap version of C1 on this local benchmark. The "
                "tested fingerprints and density-regime selector did not beat the "
                "single-best baseline by the required margin. A stronger follow-up "
                "would need richer features, harder solver choices, or a different "
                "instance distribution.",
            ]
        )
    lines.extend(
        [
            "",
            "## Reproduction",
            "",
            "```bash",
            "uv run python -m experiments.solver_portfolio_c1.bench "
            f"--effort-penalty {report.effort_penalty:g} --write-findings",
            "```",
            "",
        ]
    )
    return "\n".join(lines)


def add_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--sizes", type=int, nargs="+", default=[24, 36, 48, 64])
    parser.add_argument("--seeds-per-family", type=int, default=8)
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--clusters", type=int, default=5)
    parser.add_argument(
        "--effort-penalty",
        type=float,
        default=0.04,
        help="Penalty charged per normalized n^2 effort unit.",
    )
    parser.add_argument(
        "--improvement-margin",
        type=float,
        default=0.05,
        help="Required regret reduction vs single-best for support.",
    )
    parser.add_argument(
        "--supervised-tolerance",
        type=float,
        default=0.20,
        help="How much worse density clusters may be than direct ML and still count.",
    )
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--verbose", action="store_true")
    parser.add_argument(
        "--write-findings",
        action="store_true",
        help="Write findings/solver-portfolio-c1-verification-20260525.md.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("findings/solver-portfolio-c1-verification-20260525.md"),
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    add_args(parser)
    args = parser.parse_args(argv)
    report = run_benchmark(args)
    print_report(report)
    if args.write_findings:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(render_markdown(report), encoding="utf-8")
        print(f"\nwrote {args.output}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
