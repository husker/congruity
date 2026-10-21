"""Solver-portfolio C1b robustness verification.

This is the follow-up to ``experiments.solver_portfolio_c1``. C1 found a
partial-positive result: route-instance fingerprints contain solver-selection
signal, but the density-cluster-first mechanism was weak and a generator-family
selector was suspiciously strong.

C1b asks the harder question:

    Is the signal structural, or did the first benchmark mostly learn obvious
    labels such as "grid", "ring", and "clustered"?

The test intentionally withholds families:

  - leave-one-family-out on the original base families
  - train on all base families, then test on hybrid unseen families

The main selector is cost-sensitive: it predicts each solver's expected loss
and chooses the lowest predicted-loss solver. This matches the decision we
actually care about better than "classify the winning solver".

Run:
    uv run python -m experiments.solver_portfolio_c1b.bench --smoke
    uv run python -m experiments.solver_portfolio_c1b.bench --write-findings
"""

from __future__ import annotations

import argparse
import math
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler

from experiments.solver_portfolio_c1 import bench as c1

RUN_ID = "20260525-011650"
COMPOSITION_ID = "C1b"
BASE_FAMILIES = ["uniform", "clustered", "line", "grid", "ring"]
HYBRID_FAMILIES = [
    "clustered_ring",
    "jittered_grid",
    "line_cluster",
    "anisotropic_blobs",
    "outlier_mix",
]


@dataclass(frozen=True)
class RobustSplitResult:
    name: str
    train_families: tuple[str, ...]
    test_families: tuple[str, ...]
    single_best: c1.SelectorResult
    direct_supervised: c1.SelectorResult
    density_cluster: c1.SelectorResult
    cost_sensitive: c1.SelectorResult
    feature_importances: list[tuple[str, float]]


@dataclass(frozen=True)
class RobustReport:
    base_instances: int
    hybrid_instances: int
    effort_penalty: float
    improvement_margin: float
    splits: list[RobustSplitResult]
    summary: dict[str, float | str]
    top_features: list[tuple[str, float]]
    verdict: str


# ---------------------------------------------------------------------------
# Hybrid instance generation
# ---------------------------------------------------------------------------


def _normalize(coords: np.ndarray) -> np.ndarray:
    lo = coords.min(axis=0)
    hi = coords.max(axis=0)
    span = np.maximum(hi - lo, 1e-9)
    return np.clip((coords - lo) / span, 0.0, 1.0)


def _rotation(theta: float) -> np.ndarray:
    c = math.cos(theta)
    s = math.sin(theta)
    return np.array([[c, -s], [s, c]], dtype=float)


def _hybrid_coords(family: str, n: int, rng: np.random.Generator) -> np.ndarray:
    if family == "clustered_ring":
        k = int(rng.integers(4, 8))
        theta = np.linspace(0.0, 2.0 * math.pi, k, endpoint=False)
        theta += rng.normal(0.0, 0.10, size=k)
        centers = np.column_stack([0.5 + 0.34 * np.cos(theta), 0.5 + 0.34 * np.sin(theta)])
        labels = rng.integers(0, k, size=n)
        coords = centers[labels] + rng.normal(0.0, 0.035, size=(n, 2))
        return np.clip(coords, 0.0, 1.0)

    if family == "jittered_grid":
        side = int(math.ceil(math.sqrt(n * 1.4)))
        grid = np.array([(i, j) for i in range(side) for j in range(side)], dtype=float)
        picks = rng.choice(len(grid), size=n, replace=False)
        coords = grid[picks] / max(1, side - 1)
        coords += rng.normal(0.0, 0.055, size=(n, 2))
        shear = np.array([[1.0, rng.uniform(-0.25, 0.25)], [0.0, 1.0]], dtype=float)
        return _normalize(coords @ shear.T)

    if family == "line_cluster":
        n_line = int(round(n * 0.68))
        x = np.sort(rng.random(n_line))
        slope = rng.uniform(-0.75, 0.75)
        line = np.column_stack([x, 0.48 + slope * (x - 0.5)])
        line += rng.normal(0.0, 0.025, size=line.shape)
        n_cluster = n - n_line
        center = rng.uniform(0.20, 0.80, size=2)
        cluster = center + rng.normal(0.0, 0.065, size=(n_cluster, 2))
        return _normalize(np.vstack([line, cluster]))

    if family == "anisotropic_blobs":
        k = int(rng.integers(2, 5))
        centers = rng.random((k, 2))
        labels = rng.integers(0, k, size=n)
        coords = np.empty((n, 2), dtype=float)
        for i, label in enumerate(labels):
            theta = rng.uniform(0.0, math.pi)
            rot = _rotation(theta)
            scale = np.diag([rng.uniform(0.025, 0.055), rng.uniform(0.12, 0.22)])
            coords[i] = centers[label] + rng.normal(size=2) @ scale @ rot.T
        return _normalize(coords)

    if family == "outlier_mix":
        n_core = int(round(n * 0.82))
        center = rng.uniform(0.25, 0.75, size=2)
        core = center + rng.normal(0.0, 0.095, size=(n_core, 2))
        outliers = rng.random((n - n_core, 2))
        return np.clip(np.vstack([core, outliers]), 0.0, 1.0)

    raise ValueError(f"unknown hybrid family {family!r}")


def generate_hybrid_instances(
    sizes: list[int],
    seeds_per_family: int,
    base_seed: int,
    families: list[str] | None = None,
) -> list[c1.RouteInstance]:
    families = families or HYBRID_FAMILIES
    family_offsets = {family: i * 1_000_003 for i, family in enumerate(families)}
    instances: list[c1.RouteInstance] = []
    for family in families:
        for n in sizes:
            for seed in range(seeds_per_family):
                mixed_seed = base_seed + 20_003 * seed + 199 * n + family_offsets[family]
                rng = np.random.default_rng(mixed_seed)
                instances.append(
                    c1.RouteInstance(
                        family=family,
                        n=n,
                        seed=seed,
                        coords=_hybrid_coords(family, n, rng),
                    )
                )
    return instances


# ---------------------------------------------------------------------------
# Split and selector evaluation
# ---------------------------------------------------------------------------


def leave_one_family_out_splits(families: np.ndarray) -> list[tuple[str, np.ndarray, np.ndarray]]:
    splits = []
    for family in sorted(str(x) for x in set(families)):
        test_idx = np.flatnonzero(families == family)
        train_idx = np.flatnonzero(families != family)
        splits.append((family, train_idx, test_idx))
    return splits


def _result_from_choices(
    name: str,
    losses: np.ndarray,
    test_idx: np.ndarray,
    choices: np.ndarray,
    baseline_regret: float,
    solver_names: list[str],
) -> c1.SelectorResult:
    test_losses = losses[test_idx]
    chosen_loss = test_losses[np.arange(len(test_idx)), choices]
    oracle_loss = np.min(test_losses, axis=1)
    regret = float(np.mean(chosen_loss - oracle_loss))
    reduction = (baseline_regret - regret) / baseline_regret if baseline_regret > 1e-12 else 0.0
    return c1.SelectorResult(
        name=name,
        mean_loss=float(np.mean(chosen_loss)),
        mean_oracle_loss=float(np.mean(oracle_loss)),
        mean_regret=regret,
        relative_regret_reduction=float(reduction),
        chosen_solver_names=[solver_names[int(i)] for i in choices],
    )


def evaluate_cost_sensitive_regressor(
    features: np.ndarray,
    losses: np.ndarray,
    train_idx: np.ndarray,
    test_idx: np.ndarray,
    baseline_regret: float,
    seed: int,
    feature_names: list[str],
    solver_names: list[str],
) -> tuple[c1.SelectorResult, list[tuple[str, float]]]:
    scaler = StandardScaler()
    x_train = scaler.fit_transform(features[train_idx])
    x_test = scaler.transform(features[test_idx])
    model = RandomForestRegressor(
        n_estimators=180,
        max_depth=7,
        min_samples_leaf=2,
        random_state=seed,
    )
    model.fit(x_train, losses[train_idx])
    predicted = model.predict(x_test)
    choices = np.argmin(predicted, axis=1).astype(int)
    importances = sorted(
        zip(feature_names, (float(x) for x in model.feature_importances_), strict=True),
        key=lambda item: item[1],
        reverse=True,
    )
    return (
        _result_from_choices(
            "cost_sensitive",
            losses,
            test_idx,
            choices,
            baseline_regret,
            solver_names,
        ),
        importances,
    )


def _build_features_and_losses(
    instances: list[c1.RouteInstance],
    effort_penalty: float,
    verbose: bool = False,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    features = np.vstack([c1.extract_features(inst.coords) for inst in instances])
    families = np.array([inst.family for inst in instances])
    losses = []
    for idx, inst in enumerate(instances, start=1):
        runs = c1.run_solvers(inst.coords)
        losses.append(c1.solver_losses(runs, inst.n, effort_penalty))
        if verbose and (idx == 1 or idx % 50 == 0 or idx == len(instances)):
            best = min(runs, key=lambda r: r.length)
            print(
                f"  [{idx:3d}/{len(instances)}] {inst.family:16s} n={inst.n:2d} "
                f"best_length={best.name:18s} len={best.length:.3f}"
            )
    return features, np.vstack(losses), families


def _evaluate_split(
    name: str,
    train_idx: np.ndarray,
    test_idx: np.ndarray,
    features: np.ndarray,
    losses: np.ndarray,
    families: np.ndarray,
    clusters: int,
    seed: int,
) -> RobustSplitResult:
    single = c1.evaluate_single_best(losses, train_idx, test_idx, solver_names=c1.SOLVER_NAMES)
    direct = c1.evaluate_direct_supervised(
        features,
        losses,
        train_idx,
        test_idx,
        single.mean_regret,
        seed,
        c1.SOLVER_NAMES,
    )
    n_clusters = min(clusters, max(2, len(train_idx) // 8))
    density_model = c1.fit_density_peak(features[train_idx], n_clusters=n_clusters)
    density = c1.evaluate_cluster_selector(
        losses,
        train_idx,
        test_idx,
        density_model.labels,
        c1.predict_density_peak(density_model, features[test_idx]),
        baseline_regret=single.mean_regret,
        solver_names=c1.SOLVER_NAMES,
    )
    cost, importances = evaluate_cost_sensitive_regressor(
        features,
        losses,
        train_idx,
        test_idx,
        single.mean_regret,
        seed,
        c1.FEATURE_NAMES,
        c1.SOLVER_NAMES,
    )
    return RobustSplitResult(
        name=name,
        train_families=tuple(sorted(str(x) for x in set(families[train_idx]))),
        test_families=tuple(sorted(str(x) for x in set(families[test_idx]))),
        single_best=single,
        direct_supervised=direct,
        density_cluster=density,
        cost_sensitive=cost,
        feature_importances=importances,
    )


def _mean_split_metric(splits: list[RobustSplitResult], selector: str, metric: str) -> float:
    return float(np.mean([getattr(getattr(split, selector), metric) for split in splits]))


def _aggregate_feature_importances(
    splits: list[RobustSplitResult],
    top_k: int = 8,
) -> list[tuple[str, float]]:
    totals = {name: 0.0 for name in c1.FEATURE_NAMES}
    for split in splits:
        for name, value in split.feature_importances:
            totals[name] += value
    denom = max(1, len(splits))
    ranked = sorted(((name, value / denom) for name, value in totals.items()), key=lambda x: x[1], reverse=True)
    return ranked[:top_k]


def summarize(
    splits: list[RobustSplitResult],
    improvement_margin: float,
) -> tuple[dict[str, float | str], str, list[tuple[str, float]]]:
    loo_splits = [split for split in splits if split.name.startswith("leave_out:")]
    hybrid_splits = [split for split in splits if split.name == "hybrid_holdout"]
    summary: dict[str, float | str] = {
        "single_best_regret": _mean_split_metric(splits, "single_best", "mean_regret"),
        "direct_supervised_regret": _mean_split_metric(
            splits, "direct_supervised", "mean_regret"
        ),
        "density_cluster_regret": _mean_split_metric(splits, "density_cluster", "mean_regret"),
        "cost_sensitive_regret": _mean_split_metric(splits, "cost_sensitive", "mean_regret"),
        "direct_vs_single_reduction": _mean_split_metric(
            splits, "direct_supervised", "relative_regret_reduction"
        ),
        "density_vs_single_reduction": _mean_split_metric(
            splits, "density_cluster", "relative_regret_reduction"
        ),
        "cost_vs_single_reduction": _mean_split_metric(
            splits, "cost_sensitive", "relative_regret_reduction"
        ),
        "loo_cost_reduction": _mean_split_metric(
            loo_splits, "cost_sensitive", "relative_regret_reduction"
        )
        if loo_splits
        else 0.0,
        "hybrid_cost_reduction": _mean_split_metric(
            hybrid_splits, "cost_sensitive", "relative_regret_reduction"
        )
        if hybrid_splits
        else 0.0,
    }
    cost_gain = float(summary["cost_vs_single_reduction"])
    loo_gain = float(summary["loo_cost_reduction"])
    hybrid_gain = float(summary["hybrid_cost_reduction"])
    direct_gain = float(summary["direct_vs_single_reduction"])

    if loo_gain >= improvement_margin and hybrid_gain >= improvement_margin:
        verdict = "clean-positive"
    elif cost_gain >= improvement_margin or direct_gain >= improvement_margin:
        verdict = "partial-positive"
    else:
        verdict = "clean-negative"
    summary["verdict"] = verdict
    return summary, verdict, _aggregate_feature_importances(splits)


def run_benchmark(args: argparse.Namespace) -> RobustReport:
    sizes = args.sizes
    seeds_per_family = args.seeds_per_family
    if args.smoke:
        sizes = [20, 28]
        seeds_per_family = min(seeds_per_family, 2)

    base = c1.generate_instances(
        sizes=sizes,
        seeds_per_family=seeds_per_family,
        base_seed=args.seed,
        families=BASE_FAMILIES,
    )
    hybrid = generate_hybrid_instances(
        sizes=sizes,
        seeds_per_family=seeds_per_family,
        base_seed=args.seed + 99_991,
        families=HYBRID_FAMILIES,
    )
    all_instances = base + hybrid
    features, losses, families = _build_features_and_losses(
        all_instances,
        effort_penalty=args.effort_penalty,
        verbose=args.verbose,
    )
    base_idx = np.arange(len(base), dtype=int)
    hybrid_idx = np.arange(len(base), len(all_instances), dtype=int)
    splits: list[RobustSplitResult] = []

    for held_out, train_local, test_local in leave_one_family_out_splits(families[base_idx]):
        splits.append(
            _evaluate_split(
                f"leave_out:{held_out}",
                base_idx[train_local],
                base_idx[test_local],
                features,
                losses,
                families,
                clusters=args.clusters,
                seed=args.seed + len(splits),
            )
        )

    splits.append(
        _evaluate_split(
            "hybrid_holdout",
            base_idx,
            hybrid_idx,
            features,
            losses,
            families,
            clusters=args.clusters,
            seed=args.seed + 100,
        )
    )
    summary, verdict, top_features = summarize(splits, args.improvement_margin)
    return RobustReport(
        base_instances=len(base),
        hybrid_instances=len(hybrid),
        effort_penalty=args.effort_penalty,
        improvement_margin=args.improvement_margin,
        splits=splits,
        summary=summary,
        top_features=top_features,
        verdict=verdict,
    )


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------


def _fmt_pct(x: float) -> str:
    return f"{x * 100:+.1f}%"


def print_report(report: RobustReport) -> None:
    print("\n=== Solver-portfolio C1b robustness verification ===")
    print(f"compose run: {RUN_ID}, composition: C1 -> C1b follow-up")
    print(f"base instances: {report.base_instances}, hybrid holdout instances: {report.hybrid_instances}")
    print(
        "loss = tour-length gap vs per-instance best + "
        f"{report.effort_penalty:g} * normalized effort"
    )
    print("\n=== held-out-family / hybrid splits ===")
    print("split                 | single-best | direct ML  | density    | cost-sensitive")
    print("-" * 82)
    for split in report.splits:
        print(
            f"{split.name:21s} | "
            f"{split.single_best.mean_regret:11.4f} | "
            f"{split.direct_supervised.mean_regret:10.4f} | "
            f"{split.density_cluster.mean_regret:10.4f} | "
            f"{split.cost_sensitive.mean_regret:14.4f}"
        )

    s = report.summary
    print("\n=== aggregate ===")
    print(f"single-best regret:          {float(s['single_best_regret']):.4f}")
    print(f"direct supervised regret:    {float(s['direct_supervised_regret']):.4f}")
    print(f"density-cluster regret:      {float(s['density_cluster_regret']):.4f}")
    print(f"cost-sensitive regret:       {float(s['cost_sensitive_regret']):.4f}")
    print(
        "cost-sensitive reduction vs single-best: "
        f"{_fmt_pct(float(s['cost_vs_single_reduction']))}"
    )
    print(f"leave-one-family-out cost reduction: {_fmt_pct(float(s['loo_cost_reduction']))}")
    print(f"hybrid-holdout cost reduction:       {_fmt_pct(float(s['hybrid_cost_reduction']))}")

    print("\n=== top structural features ===")
    for name, importance in report.top_features:
        print(f"  {name:22s} {importance:.4f}")
    print("\n=== verdict ===")
    print(report.verdict)


def render_markdown(report: RobustReport) -> str:
    s = report.summary
    lines = [
        "# Solver-Portfolio C1b Robustness Verification",
        "",
        f"- Originating run: `{RUN_ID}`",
        "- Parent composition: `C1`",
        f"- Outcome: **{report.verdict}**",
        f"- Base-family instances: {report.base_instances}",
        f"- Hybrid holdout instances: {report.hybrid_instances}",
        "",
        "## Question",
        "",
        "Did C1 discover a structural route-instance signal, or did it mostly learn "
        "obvious generator labels such as grid, ring, and clustered?",
        "",
        "## Method",
        "",
        "C1b withholds whole families from training. It uses leave-one-family-out "
        "tests on the base generators, then trains on all base generators and tests "
        "on unseen hybrid generators: clustered rings, jittered grids, line-plus-"
        "cluster mixtures, anisotropic blobs, and compact clusters with outliers.",
        "",
        "The main selector is cost-sensitive: it predicts each solver's expected "
        "loss and chooses the solver with the lowest predicted loss. This is tested "
        "against single-best, direct winner classification, and density-cluster "
        "selection.",
        "",
        "## Results",
        "",
        "| split | single-best regret | direct ML | density cluster | cost-sensitive |",
        "|---|---:|---:|---:|---:|",
    ]
    for split in report.splits:
        lines.append(
            f"| {split.name} | {split.single_best.mean_regret:.4f} | "
            f"{split.direct_supervised.mean_regret:.4f} | "
            f"{split.density_cluster.mean_regret:.4f} | "
            f"{split.cost_sensitive.mean_regret:.4f} |"
        )
    lines.extend(
        [
            "",
            f"- Mean cost-sensitive regret reduction vs single-best: "
            f"{_fmt_pct(float(s['cost_vs_single_reduction']))}",
            f"- Leave-one-family-out cost reduction: {_fmt_pct(float(s['loo_cost_reduction']))}",
            f"- Hybrid-holdout cost reduction: {_fmt_pct(float(s['hybrid_cost_reduction']))}",
            "",
            "## Most Useful Structural Features",
            "",
        ]
    )
    for name, importance in report.top_features:
        lines.append(f"- `{name}`: {importance:.4f}")
    lines.extend(["", "## Interpretation", ""])
    if report.verdict == "clean-positive":
        lines.append(
            "This strengthens C1 substantially. The selector still improves over "
            "single-best when whole known families and unseen hybrid families are "
            "withheld, so the useful signal is not just a memorized generator label."
        )
    elif report.verdict == "partial-positive":
        lines.append(
            "This is a partial robustness result. The feature space still contains "
            "useful solver-selection signal, but it does not clear both the held-out "
            "base-family and unseen-hybrid bars. C1 is still worth pursuing, but the "
            "next version should focus on richer training distributions and regression "
            "calibration rather than unsupervised density clusters."
        )
    else:
        lines.append(
            "This weakens C1. Once whole families or hybrid families are withheld, "
            "the tested structural fingerprints do not improve enough over the "
            "single-best baseline. The first result was likely too dependent on easy "
            "generator-family cues."
        )
    lines.extend(
        [
            "",
            "## Reproduction",
            "",
            "```bash",
            "uv run python -m experiments.solver_portfolio_c1b.bench --write-findings",
            "```",
            "",
        ]
    )
    return "\n".join(lines)


def add_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--sizes", type=int, nargs="+", default=[24, 36, 48, 64])
    parser.add_argument("--seeds-per-family", type=int, default=8)
    parser.add_argument("--clusters", type=int, default=5)
    parser.add_argument("--effort-penalty", type=float, default=0.04)
    parser.add_argument("--improvement-margin", type=float, default=0.05)
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--verbose", action="store_true")
    parser.add_argument(
        "--write-findings",
        action="store_true",
        help="Write findings/solver-portfolio-c1b-robustness-20260525.md.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("findings/solver-portfolio-c1b-robustness-20260525.md"),
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
