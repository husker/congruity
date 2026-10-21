"""C2 verification: novelty/metadynamic MCTS for snake-in-the-box search.

Originating composition: ``findings/compose-20260526-043036.md`` C2.

The claim is that MCTS plus novelty-search descriptors and metadynamic-style
penalties can help escape deceptive partial-snake basins. This harness tests the
core algorithmic question locally: under matched CPU budgets, does the combined
variant find longer exact induced paths in small hypercubes than plain random
rollout MCTS and a strong cheap greedy-restart baseline?

Run:
    uv run python -m experiments.snake_mcts_c2.bench --smoke
    uv run python -m experiments.snake_mcts_c2.bench --write-findings
"""

from __future__ import annotations

import argparse
import math
import random
import statistics
import time
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path

RUN_ID = "20260526-043036"
COMPOSITION_ID = "C2"
DEFAULT_FINDINGS_PATH = Path("findings/snake-mcts-c2-verification-20260526.md")


@dataclass
class SearchNode:
    path: tuple[int, ...]
    untried: list[int]
    parent: SearchNode | None = None
    children: dict[int, SearchNode] = field(default_factory=dict)
    visits: int = 0
    reward_sum: float = 0.0


@dataclass(frozen=True)
class SearchResult:
    variant: str
    dimension: int
    seed: int
    best_vertices: int
    best_path: tuple[int, ...]
    iterations: int
    rollouts: int
    descriptor_bins_seen: int
    elapsed_s: float


@dataclass(frozen=True)
class BenchmarkReport:
    dimensions: list[int]
    seeds_per_dimension: int
    iterations: int
    rollout_limit: int
    greedy_restarts: int
    rows: list[SearchResult]
    summary: dict[str, float | str]
    verdict: str


def hamming_distance(a: int, b: int) -> int:
    return (a ^ b).bit_count()


def cube_neighbors(vertex: int, dimension: int) -> list[int]:
    return [vertex ^ (1 << bit) for bit in range(dimension)]


def is_induced_path(path: list[int] | tuple[int, ...], dimension: int) -> bool:
    if any(v < 0 or v >= (1 << dimension) for v in path):
        return False
    if len(set(path)) != len(path):
        return False
    for i in range(len(path) - 1):
        if hamming_distance(path[i], path[i + 1]) != 1:
            return False
    for i in range(len(path)):
        for j in range(i + 2, len(path)):
            if hamming_distance(path[i], path[j]) == 1:
                return False
    return True


def legal_extensions(path: list[int] | tuple[int, ...], dimension: int) -> list[int]:
    if not path:
        return [0]
    visited = set(path)
    forbidden_anchors = path[:-1]
    out = []
    for candidate in cube_neighbors(path[-1], dimension):
        if candidate in visited:
            continue
        if any(hamming_distance(candidate, prior) == 1 for prior in forbidden_anchors):
            continue
        out.append(candidate)
    return sorted(out)


def forbidden_vertex_count(path: tuple[int, ...], dimension: int) -> int:
    if not path:
        return 0
    visited = set(path)
    forbidden: set[int] = set()
    for vertex in path[:-1]:
        forbidden.update(cube_neighbors(vertex, dimension))
    forbidden.difference_update(visited)
    return len(forbidden)


def descriptor_bin(path: tuple[int, ...], dimension: int) -> tuple[int, int, int]:
    """Low-dimensional trap descriptor used by the C2 variant.

    The bins deliberately mirror the L3 critique: endpoint mobility and
    forbidden-neighborhood density are coarse descriptors, so the experiment can
    tell us whether they are useful or too lossy.
    """
    legal_count = len(legal_extensions(path, dimension))
    occupied_bucket = min(9, int(10 * len(path) / (1 << dimension)))
    forbidden_bucket = min(9, int(10 * forbidden_vertex_count(path, dimension) / (1 << dimension)))
    return (legal_count, occupied_bucket, forbidden_bucket)


def _weighted_choice(options: list[int], weights: list[float], rng: random.Random) -> int:
    total = sum(weights)
    if total <= 0:
        return rng.choice(options)
    threshold = rng.random() * total
    acc = 0.0
    for option, weight in zip(options, weights):
        acc += weight
        if acc >= threshold:
            return option
    return options[-1]


def _mobility_weighted_extension(
    path: tuple[int, ...],
    dimension: int,
    rng: random.Random,
) -> int | None:
    options = legal_extensions(path, dimension)
    if not options:
        return None
    weights = []
    for option in options:
        nxt = (*path, option)
        weights.append((len(legal_extensions(nxt, dimension)) + 1.0) ** 2)
    return _weighted_choice(options, weights, rng)


def greedy_randomized_search(
    dimension: int,
    restarts: int,
    seed: int,
) -> SearchResult:
    rng = random.Random(seed)
    start = time.perf_counter()
    best: tuple[int, ...] = (0,)
    for _ in range(restarts):
        path = (0,)
        while True:
            nxt = _mobility_weighted_extension(path, dimension, rng)
            if nxt is None:
                break
            path = (*path, nxt)
        if len(path) > len(best):
            best = path
    return SearchResult(
        variant="greedy",
        dimension=dimension,
        seed=seed,
        best_vertices=len(best),
        best_path=best,
        iterations=restarts,
        rollouts=restarts,
        descriptor_bins_seen=0,
        elapsed_s=time.perf_counter() - start,
    )


def _new_node(path: tuple[int, ...], dimension: int, parent: SearchNode | None = None) -> SearchNode:
    return SearchNode(path=path, untried=legal_extensions(path, dimension), parent=parent)


def _ucb_child(node: SearchNode, exploration: float) -> SearchNode:
    log_parent = math.log(max(1, node.visits))
    return max(
        node.children.values(),
        key=lambda child: (
            child.reward_sum / max(1, child.visits)
            + exploration * math.sqrt(log_parent / max(1, child.visits))
        ),
    )


def _novelty_weights(
    path: tuple[int, ...],
    options: list[int],
    dimension: int,
    archive: Counter[tuple[int, int, int]],
    novelty_weight: float,
    metadynamic_weight: float,
) -> list[float]:
    weights = []
    for option in options:
        nxt = (*path, option)
        desc = descriptor_bin(nxt, dimension)
        mobility = len(legal_extensions(nxt, dimension)) + 1.0
        novelty_bonus = novelty_weight / (1.0 + archive[desc])
        metadynamic_penalty = 1.0 / (1.0 + metadynamic_weight * archive[desc])
        weights.append(max(1e-9, mobility * (1.0 + novelty_bonus) * metadynamic_penalty))
    return weights


def _choose_expansion(
    node: SearchNode,
    dimension: int,
    rng: random.Random,
    variant: str,
    archive: Counter[tuple[int, int, int]],
    novelty_weight: float,
    metadynamic_weight: float,
) -> int:
    if variant == "plain_mcts":
        return rng.choice(node.untried)
    weights = _novelty_weights(
        node.path,
        node.untried,
        dimension,
        archive,
        novelty_weight,
        metadynamic_weight,
    )
    return _weighted_choice(node.untried, weights, rng)


def _rollout(
    path: tuple[int, ...],
    dimension: int,
    rng: random.Random,
    rollout_limit: int,
    variant: str,
    archive: Counter[tuple[int, int, int]],
    novelty_weight: float,
    metadynamic_weight: float,
) -> tuple[tuple[int, ...], list[tuple[int, int, int]]]:
    out = path
    descriptors: list[tuple[int, int, int]] = []
    while len(out) < rollout_limit:
        options = legal_extensions(out, dimension)
        if not options:
            break
        if variant == "plain_mcts":
            nxt = rng.choice(options)
        else:
            weights = _novelty_weights(
                out,
                options,
                dimension,
                archive,
                novelty_weight,
                metadynamic_weight,
            )
            nxt = _weighted_choice(options, weights, rng)
        out = (*out, nxt)
        descriptors.append(descriptor_bin(out, dimension))
    return out, descriptors


def mcts_search(
    dimension: int,
    iterations: int,
    rollout_limit: int,
    seed: int,
    *,
    variant: str,
    exploration: float = 1.4,
    novelty_weight: float = 2.0,
    metadynamic_weight: float = 0.7,
) -> SearchResult:
    if variant not in {"plain_mcts", "novelty_metadynamic_mcts"}:
        raise ValueError(f"unknown MCTS variant {variant!r}")

    rng = random.Random(seed)
    start = time.perf_counter()
    root = _new_node((0,), dimension)
    archive: Counter[tuple[int, int, int]] = Counter()
    best: tuple[int, ...] = (0,)

    for _ in range(iterations):
        node = root
        while not node.untried and node.children:
            node = _ucb_child(node, exploration)

        if node.untried:
            action = _choose_expansion(
                node,
                dimension,
                rng,
                variant,
                archive,
                novelty_weight,
                metadynamic_weight,
            )
            node.untried.remove(action)
            child_path = (*node.path, action)
            child = _new_node(child_path, dimension, parent=node)
            node.children[action] = child
            node = child

        final_path, descriptors = _rollout(
            node.path,
            dimension,
            rng,
            rollout_limit,
            variant,
            archive,
            novelty_weight,
            metadynamic_weight,
        )
        if len(final_path) > len(best):
            best = final_path
        if variant == "novelty_metadynamic_mcts":
            archive.update(descriptors)

        reward = float(len(final_path))
        while node is not None:
            node.visits += 1
            node.reward_sum += reward
            node = node.parent

    return SearchResult(
        variant=variant,
        dimension=dimension,
        seed=seed,
        best_vertices=len(best),
        best_path=best,
        iterations=iterations,
        rollouts=iterations,
        descriptor_bins_seen=len(archive),
        elapsed_s=time.perf_counter() - start,
    )


def _mean(rows: list[float]) -> float:
    return float(statistics.fmean(rows)) if rows else 0.0


def _median(rows: list[float]) -> float:
    return float(statistics.median(rows)) if rows else 0.0


def summarize(rows: list[SearchResult]) -> tuple[dict[str, float | str], str]:
    by_variant: dict[str, list[SearchResult]] = {}
    for row in rows:
        by_variant.setdefault(row.variant, []).append(row)

    summary: dict[str, float | str] = {}
    for variant, variant_rows in sorted(by_variant.items()):
        values = [float(row.best_vertices) for row in variant_rows]
        summary[f"{variant}_mean_vertices"] = _mean(values)
        summary[f"{variant}_median_vertices"] = _median(values)
        summary[f"{variant}_max_vertices"] = float(max(values))

    paired_plain = {
        (row.dimension, row.seed): row.best_vertices
        for row in by_variant.get("plain_mcts", [])
    }
    paired_greedy = {
        (row.dimension, row.seed): row.best_vertices
        for row in by_variant.get("greedy", [])
    }
    novelty_rows = by_variant.get("novelty_metadynamic_mcts", [])
    novelty_deltas_plain = []
    novelty_deltas_best_baseline = []
    novelty_wins_plain = 0
    novelty_wins_best_baseline = 0
    comparable = 0
    for row in novelty_rows:
        key = (row.dimension, row.seed)
        if key not in paired_plain or key not in paired_greedy:
            continue
        comparable += 1
        baseline_best = max(paired_plain[key], paired_greedy[key])
        novelty_deltas_plain.append(float(row.best_vertices - paired_plain[key]))
        novelty_deltas_best_baseline.append(float(row.best_vertices - baseline_best))
        novelty_wins_plain += int(row.best_vertices > paired_plain[key])
        novelty_wins_best_baseline += int(row.best_vertices > baseline_best)

    summary["novelty_delta_vs_plain_mean"] = _mean(novelty_deltas_plain)
    summary["novelty_delta_vs_best_baseline_mean"] = _mean(novelty_deltas_best_baseline)
    summary["novelty_win_rate_vs_plain"] = novelty_wins_plain / comparable if comparable else 0.0
    summary["novelty_win_rate_vs_best_baseline"] = (
        novelty_wins_best_baseline / comparable if comparable else 0.0
    )

    if (
        summary["novelty_delta_vs_best_baseline_mean"] >= 1.0
        and summary["novelty_win_rate_vs_best_baseline"] >= 0.60
    ):
        verdict = "positive"
    elif summary["novelty_delta_vs_plain_mean"] > 0.0 and summary["novelty_win_rate_vs_plain"] >= 0.50:
        verdict = "partial-positive"
    else:
        verdict = "clean-negative"
    return summary, verdict


def run_benchmark(
    dimensions: list[int],
    seeds_per_dimension: int,
    iterations: int,
    rollout_limit: int,
    greedy_restarts: int,
    seed: int,
) -> BenchmarkReport:
    rows: list[SearchResult] = []
    for dimension in dimensions:
        for local_seed in range(seeds_per_dimension):
            mixed_seed = seed + 10_007 * dimension + 1_009 * local_seed
            rows.append(greedy_randomized_search(dimension, greedy_restarts, mixed_seed))
            rows.append(
                mcts_search(
                    dimension,
                    iterations,
                    rollout_limit,
                    mixed_seed,
                    variant="plain_mcts",
                )
            )
            rows.append(
                mcts_search(
                    dimension,
                    iterations,
                    rollout_limit,
                    mixed_seed,
                    variant="novelty_metadynamic_mcts",
                )
            )
    summary, verdict = summarize(rows)
    return BenchmarkReport(
        dimensions=dimensions,
        seeds_per_dimension=seeds_per_dimension,
        iterations=iterations,
        rollout_limit=rollout_limit,
        greedy_restarts=greedy_restarts,
        rows=rows,
        summary=summary,
        verdict=verdict,
    )


def _format_float(value: float | str) -> str:
    if isinstance(value, str):
        return value
    return f"{value:.3f}"


def render_findings(report: BenchmarkReport) -> str:
    variants = ["greedy", "plain_mcts", "novelty_metadynamic_mcts"]
    lines = [
        "# Snake-in-the-Box C2 Verification",
        "",
        f"- Originating run: `{RUN_ID}`",
        f"- Composition: `{COMPOSITION_ID}`",
        f"- Outcome: **{report.verdict}**",
        f"- Dimensions: {', '.join(str(d) for d in report.dimensions)}",
        f"- Seeds per dimension: {report.seeds_per_dimension}",
        "",
        "## Question",
        "",
        "Does a novelty/metadynamic MCTS policy find longer exact snake paths than plain "
        "MCTS or greedy randomized restarts under matched local compute?",
        "",
        "## Method",
        "",
        "The harness builds partial paths in hypercube graphs from vertex `0`. Every "
        "candidate is checked as an induced path: consecutive vertices must differ in "
        "one bit, vertices may not repeat, and nonconsecutive vertices may not be "
        "hypercube neighbors. The C2 variant uses endpoint mobility, occupied fraction, "
        "and forbidden-neighborhood density as coarse descriptor bins; repeated bins "
        "receive a metadynamic penalty while rare bins receive a novelty bonus.",
        "",
        "Compared variants:",
        "",
        "- greedy randomized restarts with onward-mobility weighting",
        "- vanilla UCT MCTS with random valid rollouts",
        "- novelty/metadynamic MCTS with the same iteration budget",
        "",
        "## Results",
        "",
        "| variant | mean best vertices | median | max |",
        "|---|---:|---:|---:|",
    ]
    for variant in variants:
        lines.append(
            "| "
            f"{variant} | "
            f"{_format_float(report.summary.get(f'{variant}_mean_vertices', 0.0))} | "
            f"{_format_float(report.summary.get(f'{variant}_median_vertices', 0.0))} | "
            f"{_format_float(report.summary.get(f'{variant}_max_vertices', 0.0))} |"
        )
    lines.extend(
        [
            "",
            "- Mean novelty delta vs plain MCTS: "
            f"{_format_float(report.summary['novelty_delta_vs_plain_mean'])} vertices",
            "- Mean novelty delta vs best baseline per seed: "
            f"{_format_float(report.summary['novelty_delta_vs_best_baseline_mean'])} vertices",
            "- Novelty win rate vs plain MCTS: "
            f"{_format_float(report.summary['novelty_win_rate_vs_plain'])}",
            "- Novelty win rate vs best baseline: "
            f"{_format_float(report.summary['novelty_win_rate_vs_best_baseline'])}",
            "",
            "## Interpretation",
            "",
            _interpretation(report),
            "",
            "## Reproduction",
            "",
            "```bash",
            "uv run python -m experiments.snake_mcts_c2.bench --write-findings",
            "```",
            "",
        ]
    )
    return "\n".join(lines)


def _interpretation(report: BenchmarkReport) -> str:
    if report.verdict == "positive":
        return (
            "This is a positive local result: the descriptor-biased MCTS beat both "
            "plain MCTS and greedy randomized restarts often enough to justify a "
            "larger benchmark and deeper ablations."
        )
    if report.verdict == "partial-positive":
        return (
            "This is a partial result. The novelty/metadynamic mechanism improved on "
            "plain MCTS, but it did not clearly beat the strongest cheap baseline. "
            "C2 remains interesting only if the next pass adds stronger descriptors "
            "or domain propagation."
        )
    return (
        "This is a clean negative for the simple C2 mechanism tested here. The "
        "coarse endpoint-saturation and forbidden-density descriptors did not turn "
        "plain MCTS into a stronger local snake constructor under matched compute. "
        "That supports the L3 concern that the descriptors may be too lossy."
    )


def write_findings(report: BenchmarkReport, path: Path = DEFAULT_FINDINGS_PATH) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(render_findings(report))
    return path


def _parse_dimensions(raw: str) -> list[int]:
    return [int(part) for part in raw.split(",") if part.strip()]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dimensions", default="6,7,8")
    parser.add_argument("--seeds-per-dimension", type=int, default=8)
    parser.add_argument("--iterations", type=int, default=900)
    parser.add_argument("--rollout-limit", type=int, default=220)
    parser.add_argument("--greedy-restarts", type=int, default=900)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--write-findings", action="store_true")
    args = parser.parse_args(argv)

    dimensions = _parse_dimensions(args.dimensions)
    if args.smoke:
        dimensions = [5]
        args.seeds_per_dimension = 2
        args.iterations = 120
        args.rollout_limit = 40
        args.greedy_restarts = 120

    report = run_benchmark(
        dimensions=dimensions,
        seeds_per_dimension=args.seeds_per_dimension,
        iterations=args.iterations,
        rollout_limit=args.rollout_limit,
        greedy_restarts=args.greedy_restarts,
        seed=args.seed,
    )
    print(render_findings(report))
    if args.write_findings:
        path = write_findings(report)
        print(f"\nWrote {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
