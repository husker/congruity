"""C1 verification: no-u-turn restarts for finite-game equilibrium search.

Originating composition: ``findings/compose-20260526-205855.md`` C1.

C1 proposes running no-regret or extragradient dynamics on finite games, then
using a no-u-turn-like trajectory trigger to detect cycling and restart or
regularize from the best residual point. This harness tests a concrete version
of that idea on small generated bimatrix games.

Run:
    uv run python -m experiments.game_theory_c1.bench --smoke
    uv run python -m experiments.game_theory_c1.bench --write-findings
"""

from __future__ import annotations

import argparse
import math
import statistics
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np

RUN_ID = "20260526-205855"
COMPOSITION_ID = "C1"
DEFAULT_FINDINGS_PATH = Path("findings/c1-game-theory-no-u-turn-20260526.md")
VARIANTS = [
    "multiplicative_weights",
    "regret_matching",
    "projected_gradient",
    "extragradient",
    "generic_restart",
    "c1_no_u_turn",
]
BASELINE_NAMES = {
    "multiplicative_weights": "multiplicative weights",
    "regret_matching": "regret matching",
    "projected_gradient": "projected gradient",
    "extragradient": "extragradient",
    "generic_restart": "generic residual restart",
    "c1_no_u_turn": "C1 no-u-turn restart",
}


@dataclass(frozen=True)
class BimatrixGame:
    family: str
    size: int
    seed: int
    a: np.ndarray
    b: np.ndarray


@dataclass(frozen=True)
class ProfileMetrics:
    row_regret: float
    col_regret: float
    nashconv: float
    row_value: float
    col_value: float


@dataclass(frozen=True)
class SolverResult:
    variant: str
    family: str
    instance_id: str
    size: int
    seed: int
    step_size: float
    x: np.ndarray
    y: np.ndarray
    final_x: np.ndarray
    final_y: np.ndarray
    best_nashconv: float
    final_nashconv: float
    average_nashconv: float
    history: list[ProfileMetrics]
    triggers: int
    elapsed_s: float


@dataclass(frozen=True)
class BenchmarkReport:
    families: list[str]
    sizes: list[int]
    seeds: list[int]
    iterations: int
    step_sizes: list[float]
    rows: list[SolverResult]
    summary: dict[str, float | int | str]
    family_summary: dict[str, dict[str, float]]
    verdict: str


def _normalize_payoffs(payoffs: np.ndarray) -> np.ndarray:
    centered = payoffs - float(np.mean(payoffs))
    scale = float(np.max(np.abs(centered)))
    if scale <= 1e-12:
        return centered
    return centered / scale


def project_simplex(values: np.ndarray) -> np.ndarray:
    """Euclidean projection onto the probability simplex."""

    v = np.asarray(values, dtype=float)
    if v.ndim != 1:
        raise ValueError("project_simplex expects a one-dimensional vector")
    if len(v) == 0:
        raise ValueError("cannot project an empty vector")
    if not np.all(np.isfinite(v)):
        return np.full(len(v), 1.0 / len(v))

    u = np.sort(v)[::-1]
    cssv = np.cumsum(u)
    candidates = u * np.arange(1, len(u) + 1) > (cssv - 1.0)
    if not np.any(candidates):
        return np.full(len(v), 1.0 / len(v))
    rho = int(np.nonzero(candidates)[0][-1])
    theta = (cssv[rho] - 1.0) / float(rho + 1)
    projected = np.maximum(v - theta, 0.0)
    total = float(projected.sum())
    if total <= 1e-12:
        return np.full(len(v), 1.0 / len(v))
    return projected / total


def _safe_strategy(values: np.ndarray) -> np.ndarray:
    values = np.maximum(np.asarray(values, dtype=float), 0.0)
    total = float(values.sum())
    if total <= 1e-12 or not np.isfinite(total):
        return np.full(len(values), 1.0 / len(values))
    return values / total


def profile_metrics(game: BimatrixGame, x: np.ndarray, y: np.ndarray) -> ProfileMetrics:
    x = _safe_strategy(x)
    y = _safe_strategy(y)
    row_action_payoffs = game.a @ y
    col_action_payoffs = x @ game.b
    row_value = float(x @ row_action_payoffs)
    col_value = float(col_action_payoffs @ y)
    row_regret = max(0.0, float(np.max(row_action_payoffs) - row_value))
    col_regret = max(0.0, float(np.max(col_action_payoffs) - col_value))
    return ProfileMetrics(
        row_regret=row_regret,
        col_regret=col_regret,
        nashconv=row_regret + col_regret,
        row_value=row_value,
        col_value=col_value,
    )


def matching_pennies_game(size: int = 2, seed: int = 0) -> BimatrixGame:
    if size == 2:
        a = np.array([[1.0, -1.0], [-1.0, 1.0]])
    else:
        rng = np.random.default_rng(seed)
        signs = np.fromfunction(lambda i, j: np.where((i + j) % 2 == 0, 1.0, -1.0), (size, size))
        a = signs + 0.05 * rng.normal(size=(size, size))
    return BimatrixGame("matching_pennies", size, seed, _normalize_payoffs(a), _normalize_payoffs(-a))


def random_general_sum_game(size: int, rng: np.random.Generator, seed: int = 0) -> BimatrixGame:
    return BimatrixGame(
        family="random_general_sum",
        size=size,
        seed=seed,
        a=_normalize_payoffs(rng.normal(size=(size, size))),
        b=_normalize_payoffs(rng.normal(size=(size, size))),
    )


def cyclic_game(size: int, seed: int = 0) -> BimatrixGame:
    rng = np.random.default_rng(seed)
    a = -0.25 * np.ones((size, size))
    b = -0.25 * np.ones((size, size))
    for i in range(size):
        a[(i + 1) % size, i] = 1.0
        b[i, (i + 1) % size] = 1.0
    a += 0.03 * rng.normal(size=(size, size))
    b += 0.03 * rng.normal(size=(size, size))
    return BimatrixGame("cyclic", size, seed, _normalize_payoffs(a), _normalize_payoffs(b))


def coordination_game(size: int, seed: int = 0) -> BimatrixGame:
    rng = np.random.default_rng(seed)
    base = -0.25 * np.ones((size, size))
    np.fill_diagonal(base, 1.0)
    a = base + 0.08 * rng.normal(size=(size, size))
    b = base + 0.08 * rng.normal(size=(size, size))
    return BimatrixGame("coordination", size, seed, _normalize_payoffs(a), _normalize_payoffs(b))


def near_zero_sum_game(size: int, seed: int = 0) -> BimatrixGame:
    rng = np.random.default_rng(seed)
    a = rng.normal(size=(size, size))
    b = -a + 0.25 * rng.normal(size=(size, size))
    return BimatrixGame("near_zero_sum", size, seed, _normalize_payoffs(a), _normalize_payoffs(b))


def make_game(family: str, size: int, seed: int) -> BimatrixGame:
    if family == "matching_pennies":
        return matching_pennies_game(size=size, seed=seed)
    if family == "cyclic":
        return cyclic_game(size=size, seed=seed)
    if family == "coordination":
        return coordination_game(size=size, seed=seed)
    if family == "near_zero_sum":
        return near_zero_sum_game(size=size, seed=seed)
    if family == "random_general_sum":
        return random_general_sum_game(size, np.random.default_rng(seed), seed=seed)
    raise ValueError(f"unknown game family: {family}")


def no_u_turn_trigger(
    positions: list[np.ndarray],
    window: int = 12,
    relative_threshold: float = 0.05,
) -> bool:
    """Detect a trajectory that has started moving back toward its anchor."""

    if len(positions) < window + 1:
        return False
    anchor = positions[-window - 1]
    previous = positions[-2]
    current = positions[-1]
    displacement = current - anchor
    velocity = current - previous
    old_velocity = positions[-window] - positions[-window - 1]

    anchor_denom = float(np.linalg.norm(displacement) * np.linalg.norm(velocity))
    anchor_turn = (
        anchor_denom > 1e-12
        and float(np.dot(displacement, velocity)) < -relative_threshold * anchor_denom
    )
    velocity_denom = float(np.linalg.norm(old_velocity) * np.linalg.norm(velocity))
    velocity_turn = (
        velocity_denom > 1e-12
        and float(np.dot(old_velocity, velocity)) < -relative_threshold * velocity_denom
    )
    return anchor_turn or velocity_turn


def _position(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    return np.concatenate([x, y])


def _multiplicative_step(
    game: BimatrixGame,
    x: np.ndarray,
    y: np.ndarray,
    step_size: float,
) -> tuple[np.ndarray, np.ndarray]:
    row_payoffs = np.clip(game.a @ y, -20.0, 20.0)
    col_payoffs = np.clip(x @ game.b, -20.0, 20.0)
    nx = x * np.exp(step_size * row_payoffs)
    ny = y * np.exp(step_size * col_payoffs)
    return _safe_strategy(nx), _safe_strategy(ny)


def _projected_gradient_step(
    game: BimatrixGame,
    x: np.ndarray,
    y: np.ndarray,
    step_size: float,
) -> tuple[np.ndarray, np.ndarray]:
    return project_simplex(x + step_size * (game.a @ y)), project_simplex(
        y + step_size * (x @ game.b)
    )


def _extragradient_step(
    game: BimatrixGame,
    x: np.ndarray,
    y: np.ndarray,
    step_size: float,
) -> tuple[np.ndarray, np.ndarray]:
    probe_x = project_simplex(x + step_size * (game.a @ y))
    probe_y = project_simplex(y + step_size * (x @ game.b))
    return project_simplex(x + step_size * (game.a @ probe_y)), project_simplex(
        y + step_size * (probe_x @ game.b)
    )


def _regret_matching_strategy(regrets: np.ndarray) -> np.ndarray:
    positive = np.maximum(regrets, 0.0)
    total = float(positive.sum())
    if total <= 1e-12:
        return np.full(len(regrets), 1.0 / len(regrets))
    return positive / total


def _record_best(
    game: BimatrixGame,
    x: np.ndarray,
    y: np.ndarray,
    history: list[ProfileMetrics],
    best: tuple[float, np.ndarray, np.ndarray],
) -> tuple[float, np.ndarray, np.ndarray]:
    metrics = profile_metrics(game, x, y)
    history.append(metrics)
    if metrics.nashconv < best[0]:
        return metrics.nashconv, x.copy(), y.copy()
    return best


def solve_game(
    game: BimatrixGame,
    variant: str,
    iterations: int = 500,
    seed: int = 0,
    step_size: float = 0.05,
    trigger_window: int = 12,
    patience: int = 35,
    regularization: float = 0.08,
) -> SolverResult:
    if variant not in VARIANTS:
        raise ValueError(f"unknown solver variant: {variant}")

    start = time.perf_counter()
    rng = np.random.default_rng(seed)
    size = game.size
    x = _safe_strategy(np.ones(size) + 0.01 * rng.normal(size=size))
    y = _safe_strategy(np.ones(size) + 0.01 * rng.normal(size=size))
    uniform = np.full(size, 1.0 / size)
    history: list[ProfileMetrics] = []
    positions = [_position(x, y)]
    best = _record_best(game, x, y, history, (math.inf, x.copy(), y.copy()))
    average_x = np.zeros(size)
    average_y = np.zeros(size)
    row_regrets = np.zeros(size)
    col_regrets = np.zeros(size)
    triggers = 0
    last_improvement = 0
    active_step = step_size

    for t in range(1, iterations + 1):
        before_best = best[0]
        if variant == "multiplicative_weights":
            x, y = _multiplicative_step(game, x, y, active_step)
        elif variant == "projected_gradient":
            x, y = _projected_gradient_step(game, x, y, active_step)
        elif variant == "extragradient":
            x, y = _extragradient_step(game, x, y, active_step)
        elif variant == "regret_matching":
            x = _regret_matching_strategy(row_regrets)
            y = _regret_matching_strategy(col_regrets)
            row_payoffs = game.a @ y
            col_payoffs = x @ game.b
            row_regrets += row_payoffs - float(x @ row_payoffs)
            col_regrets += col_payoffs - float(col_payoffs @ y)
        elif variant in {"generic_restart", "c1_no_u_turn"}:
            x, y = _projected_gradient_step(game, x, y, active_step)

        average_x += x
        average_y += y
        positions.append(_position(x, y))
        best = _record_best(game, x, y, history, best)
        if best[0] < before_best - 1e-12:
            last_improvement = t

        if variant == "generic_restart" and t - last_improvement >= patience:
            triggers += 1
            x = _safe_strategy((1.0 - regularization) * best[1] + regularization * uniform)
            y = _safe_strategy((1.0 - regularization) * best[2] + regularization * uniform)
            active_step *= 0.85
            last_improvement = t
            positions = [_position(x, y)]
        elif variant == "c1_no_u_turn" and no_u_turn_trigger(positions, window=trigger_window):
            triggers += 1
            x = _safe_strategy((1.0 - regularization) * best[1] + regularization * uniform)
            y = _safe_strategy((1.0 - regularization) * best[2] + regularization * uniform)
            active_step *= 0.9
            last_improvement = t
            positions = [_position(x, y)]

    final_metrics = profile_metrics(game, x, y)
    avg_x = _safe_strategy(average_x)
    avg_y = _safe_strategy(average_y)
    average_metrics = profile_metrics(game, avg_x, avg_y)
    best_nashconv, best_x, best_y = best
    return SolverResult(
        variant=variant,
        family=game.family,
        instance_id=f"{game.family}-n{game.size}-s{game.seed}",
        size=game.size,
        seed=game.seed,
        step_size=step_size,
        x=best_x,
        y=best_y,
        final_x=x,
        final_y=y,
        best_nashconv=best_nashconv,
        final_nashconv=final_metrics.nashconv,
        average_nashconv=average_metrics.nashconv,
        history=history,
        triggers=triggers,
        elapsed_s=time.perf_counter() - start,
    )


def _choose_best_result(results: list[SolverResult]) -> SolverResult:
    return min(results, key=lambda row: (row.best_nashconv, row.final_nashconv, row.elapsed_s))


def run_benchmark(
    families: list[str] | None = None,
    sizes: list[int] | None = None,
    seeds: list[int] | None = None,
    iterations: int = 500,
    step_sizes: list[float] | None = None,
) -> BenchmarkReport:
    families = families or ["matching_pennies", "cyclic", "coordination", "near_zero_sum", "random_general_sum"]
    sizes = sizes or [20, 40]
    seeds = seeds or list(range(6))
    step_sizes = step_sizes or [0.02, 0.05, 0.10]

    rows: list[SolverResult] = []
    for family in families:
        for size in sizes:
            for seed in seeds:
                game = make_game(family, size, seed)
                for variant in VARIANTS:
                    candidates = [
                        solve_game(game, variant, iterations, seed=seed + 991, step_size=step)
                        for step in step_sizes
                    ]
                    rows.append(_choose_best_result(candidates))

    summary = _summarize(rows)
    family_summary = _summarize_by_family(rows)
    verdict = _verdict(summary, family_summary)
    summary["verdict"] = verdict
    return BenchmarkReport(
        families=families,
        sizes=sizes,
        seeds=seeds,
        iterations=iterations,
        step_sizes=step_sizes,
        rows=rows,
        summary=summary,
        family_summary=family_summary,
        verdict=verdict,
    )


def _mean(values: list[float]) -> float:
    return float(statistics.fmean(values)) if values else math.nan


def _median(values: list[float]) -> float:
    return float(statistics.median(values)) if values else math.nan


def _ratio(value: float, baseline: float) -> float:
    if abs(value) <= 1e-12 and abs(baseline) <= 1e-12:
        return 1.0
    return value / max(baseline, 1e-12)


def _summarize(rows: list[SolverResult]) -> dict[str, float | int | str]:
    by_variant = {variant: [row for row in rows if row.variant == variant] for variant in VARIANTS}
    summary: dict[str, float | int | str] = {"n_instances": len(rows) // len(VARIANTS)}
    for variant, variant_rows in by_variant.items():
        summary[f"{variant}_mean_best"] = _mean([row.best_nashconv for row in variant_rows])
        summary[f"{variant}_median_best"] = _median([row.best_nashconv for row in variant_rows])
        summary[f"{variant}_mean_final"] = _mean([row.final_nashconv for row in variant_rows])
        summary[f"{variant}_mean_triggers"] = _mean([float(row.triggers) for row in variant_rows])
        summary[f"{variant}_elapsed_s"] = _mean([row.elapsed_s for row in variant_rows])

    baseline_variants = [variant for variant in VARIANTS if variant != "c1_no_u_turn"]
    baseline_means = {variant: float(summary[f"{variant}_mean_best"]) for variant in baseline_variants}
    baseline_medians = {variant: float(summary[f"{variant}_median_best"]) for variant in baseline_variants}
    best_baseline = min(baseline_means, key=baseline_means.get)
    best_median_baseline = min(baseline_medians, key=baseline_medians.get)
    c1_mean = float(summary["c1_no_u_turn_mean_best"])
    c1_median = float(summary["c1_no_u_turn_median_best"])
    summary["best_baseline"] = best_baseline
    summary["best_baseline_mean_best"] = baseline_means[best_baseline]
    summary["best_median_baseline"] = best_median_baseline
    summary["best_baseline_median_best"] = baseline_medians[best_median_baseline]
    summary["c1_mean_ratio_vs_best_baseline"] = _ratio(c1_mean, baseline_means[best_baseline])
    summary["c1_median_ratio_vs_best_baseline"] = _ratio(
        c1_median, baseline_medians[best_median_baseline]
    )
    summary["c1_instance_win_rate"] = _instance_win_rate(rows)
    return summary


def _summarize_by_family(rows: list[SolverResult]) -> dict[str, dict[str, float]]:
    families = sorted({row.family for row in rows})
    out: dict[str, dict[str, float]] = {}
    for family in families:
        family_rows = [row for row in rows if row.family == family]
        stats = {}
        for variant in VARIANTS:
            stats[f"{variant}_mean_best"] = _mean(
                [row.best_nashconv for row in family_rows if row.variant == variant]
            )
        baseline = min(
            (variant for variant in VARIANTS if variant != "c1_no_u_turn"),
            key=lambda variant: stats[f"{variant}_mean_best"],
        )
        stats["best_baseline_mean_best"] = stats[f"{baseline}_mean_best"]
        stats["c1_ratio_vs_best_baseline"] = _ratio(
            stats["c1_no_u_turn_mean_best"], stats["best_baseline_mean_best"]
        )
        out[family] = stats
    return out


def _instance_win_rate(rows: list[SolverResult]) -> float:
    instance_ids = sorted({row.instance_id for row in rows})
    wins = 0
    for instance_id in instance_ids:
        instance_rows = [row for row in rows if row.instance_id == instance_id]
        best = min(row.best_nashconv for row in instance_rows)
        c1 = next(row for row in instance_rows if row.variant == "c1_no_u_turn")
        if c1.best_nashconv <= best + 1e-12:
            wins += 1
    return wins / max(len(instance_ids), 1)


def _verdict(summary: dict[str, float | int | str], family_summary: dict[str, dict[str, float]]) -> str:
    mean_ratio = float(summary["c1_mean_ratio_vs_best_baseline"])
    median_ratio = float(summary["c1_median_ratio_vs_best_baseline"])
    win_rate = float(summary["c1_instance_win_rate"])
    family_wins = sum(1 for stats in family_summary.values() if stats["c1_ratio_vs_best_baseline"] <= 0.95)
    if mean_ratio <= 0.80 and median_ratio <= 0.80 and win_rate >= 0.50 and family_wins >= 3:
        return "clean-positive"
    if mean_ratio < 1.0 and (win_rate >= 0.35 or family_wins >= 2):
        return "partial-positive"
    if mean_ratio <= 1.10 or (mean_ratio <= 1.25 and win_rate >= 0.20):
        return "partial-negative"
    return "clean-negative"


def _markdown_table(headers: list[str], rows: list[list[str]]) -> list[str]:
    lines = ["| " + " | ".join(headers) + " |"]
    lines.append("|" + "|".join("---" for _ in headers) + "|")
    lines.extend("| " + " | ".join(row) + " |" for row in rows)
    return lines


def format_findings(report: BenchmarkReport) -> str:
    summary = report.summary
    lines: list[str] = [
        "# C1 game-theory verification: no-u-turn restart dynamics",
        "",
        f"Originating compose run: `{RUN_ID}` `{COMPOSITION_ID}`.",
        "",
        "## Verdict",
        "",
        f"**{report.verdict}**.",
        "",
        "C1 predicted that a finite-game no-u-turn/cycle trigger would improve ",
        "no-regret or extragradient dynamics by restarting or regularizing from ",
        "the best Nash residual point. This harness implements the cheap local ",
        "version: all solvers receive the same game instances, iteration budget, ",
        "and step-size sweep; C1 must beat the strongest tuned baseline on ",
        "NashConv rather than just beat an undertuned fixed run.",
        "",
        "## Benchmark",
        "",
        f"- Families: `{', '.join(report.families)}`",
        f"- Sizes: `{report.sizes}`",
        f"- Seeds per size/family: `{report.seeds}`",
        f"- Iterations per solver: `{report.iterations}`",
        f"- Step sizes swept: `{report.step_sizes}`",
        f"- Instances: `{summary['n_instances']}`",
        "- Primary metric: best NashConv/exploitability observed during the run",
        "- Baselines: multiplicative weights, regret matching, projected gradient, ",
        "  extragradient, and generic residual restart",
        "",
        "## Overall Results",
        "",
    ]

    rows = []
    for variant in VARIANTS:
        rows.append(
            [
                BASELINE_NAMES[variant],
                f"{float(summary[f'{variant}_mean_best']):.4f}",
                f"{float(summary[f'{variant}_median_best']):.4f}",
                f"{float(summary[f'{variant}_mean_final']):.4f}",
                f"{float(summary[f'{variant}_mean_triggers']):.2f}",
                f"{float(summary[f'{variant}_elapsed_s']):.4f}",
            ]
        )
    lines.extend(
        _markdown_table(
            ["solver", "mean best", "median best", "mean final", "triggers", "seconds"],
            rows,
        )
    )
    lines.extend(
        [
            "",
            f"Best baseline by mean best NashConv: `{summary['best_baseline']}` ",
            f"at `{float(summary['best_baseline_mean_best']):.4f}`.",
            "C1 mean ratio vs best baseline: ",
            f"`{float(summary['c1_mean_ratio_vs_best_baseline']):.3f}`.",
            f"C1 instance win rate: `{float(summary['c1_instance_win_rate']):.3f}`.",
            "",
            "## Family Breakdown",
            "",
        ]
    )
    family_rows = []
    for family, stats in sorted(report.family_summary.items()):
        family_rows.append(
            [
                family,
                f"{stats['c1_no_u_turn_mean_best']:.4f}",
                f"{stats['best_baseline_mean_best']:.4f}",
                f"{stats['c1_ratio_vs_best_baseline']:.3f}",
            ]
        )
    lines.extend(
        _markdown_table(
            ["family", "C1 mean best", "best baseline", "C1 ratio"],
            family_rows,
        )
    )

    interpretation = _interpretation(report)
    lines.extend(["", "## Interpretation", "", interpretation, ""])
    lines.extend(
        [
            "## Reproducibility",
            "",
            "```bash",
            "uv run python -m experiments.game_theory_c1.bench --write-findings",
            "uv run pytest tests/test_game_theory_c1.py -q",
            "```",
            "",
        ]
    )
    return "\n".join(lines)


def _interpretation(report: BenchmarkReport) -> str:
    summary = report.summary
    best_baseline = str(summary["best_baseline"])
    ratio = float(summary["c1_mean_ratio_vs_best_baseline"])
    if report.verdict in {"clean-positive", "partial-positive"}:
        return (
            "The C1 trigger produced a measurable advantage over the strongest tuned "
            f"baseline (`{best_baseline}`), with a mean-best-NashConv ratio of "
            f"{ratio:.3f}. That supports the composition as a locally testable "
            "algorithmic direction, though the L3 prior-art concern still requires "
            "a literature audit and stronger hard-instance benchmark."
        )
    if report.verdict == "partial-negative":
        return (
            "The no-u-turn trigger was close enough to the strongest baseline to avoid "
            "a clean rejection, but it did not deliver the requested 20% median or "
            "mean improvement. This weakens C1: the trigger may be mostly another "
            "adaptive-restart heuristic unless richer geometry or a repair step is added."
        )
    return (
        "The C1 trigger lost clearly to the strongest tuned baseline. This falsifies "
        "the cheap version of the composition: on these local generated games, "
        "trajectory reversal detection did not add useful equilibrium-search signal "
        "beyond standard first-order/no-regret dynamics and generic restart."
    )


def main() -> int:
    parser = argparse.ArgumentParser(description="Verify game-theory C1 no-u-turn restarts")
    parser.add_argument("--smoke", action="store_true", help="Run a fast smoke benchmark")
    parser.add_argument("--write-findings", action="store_true", help="Write markdown findings")
    parser.add_argument("--iterations", type=int, default=None)
    parser.add_argument("--output", type=Path, default=DEFAULT_FINDINGS_PATH)
    args = parser.parse_args()

    if args.smoke:
        report = run_benchmark(
            families=["matching_pennies", "cyclic"],
            sizes=[8],
            seeds=[0, 1],
            iterations=args.iterations or 120,
            step_sizes=[0.03, 0.08],
        )
    else:
        report = run_benchmark(iterations=args.iterations or 450)

    findings = format_findings(report)
    if args.write_findings:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(findings)
        print(f"wrote {args.output}")
    print(findings)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
