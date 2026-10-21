"""C4 verification: landmark-biased MCTS for Černý reset-word search.

Originating composition: ``findings/compose-20260526-062120.md`` C4.

C4 proposed Monte Carlo tree search with planning-style landmarks plus bounded
SAT checks. This local harness tests the algorithmic core without adding a SAT
dependency: pair-merge landmarks guide rollouts, while an exact bounded BFS
oracle provides the same kind of optimality certificate for these small
automata.

Run:
    uv run python -m experiments.cerny_c4.bench --smoke
    uv run python -m experiments.cerny_c4.bench --write-findings
"""

from __future__ import annotations

import argparse
import math
import random
import statistics
import time
from collections import deque
from dataclasses import dataclass, field
from pathlib import Path

from experiments.cerny_c3 import bench as c3
from experiments.cerny_c3b import bench as c3b
from experiments.cerny_c3c import bench as c3c

RUN_ID = "20260526-062120"
COMPOSITION_ID = "C4"
DEFAULT_FINDINGS_PATH = Path("findings/c4-cerny-landmark-mcts-20260526.md")


@dataclass
class SearchNode:
    mask: int
    word: tuple[int, ...]
    parent: SearchNode | None = None
    children: dict[int, SearchNode] = field(default_factory=dict)
    untried: list[int] = field(default_factory=list)
    visits: int = 0
    reward_sum: float = 0.0


@dataclass(frozen=True)
class SolverResult:
    variant: str
    dfa: c3.DFA
    instance_id: str
    reset_word: tuple[int, ...] | None
    found_length: int
    iterations: int
    expanded_nodes: int
    elapsed_s: float
    landmark_guided_choices: int = 0
    optimal_length: int | None = None
    optimality_gap: int | None = None
    certified_optimal: bool = False


@dataclass(frozen=True)
class BenchmarkReport:
    n_values: list[int]
    instances_per_n: int
    iterations: int
    rollout_depth_factor: int
    rows: list[SolverResult]
    summary: dict[str, float | int | str]
    verdict: str


def _pair(a: int, b: int) -> tuple[int, int]:
    return (a, b) if a <= b else (b, a)


def _apply_mask(dfa: c3.DFA, mask: int, letter: int) -> int:
    out = 0
    for state, row in enumerate(dfa.transitions):
        if mask & (1 << state):
            out |= 1 << row[letter]
    return out


def _is_singleton(mask: int) -> bool:
    return mask != 0 and (mask & (mask - 1)) == 0


def unresolved_pair_count(mask: int) -> int:
    size = mask.bit_count()
    return size * (size - 1) // 2


def pair_merge_distances(dfa: c3.DFA) -> dict[tuple[int, int], int]:
    """Shortest letter count needed to merge each unordered state pair."""

    dfa.validate()
    pairs = [_pair(a, b) for a in range(dfa.n_states) for b in range(a, dfa.n_states)]
    reverse: dict[tuple[int, int], list[tuple[int, int]]] = {pair: [] for pair in pairs}
    for pair in pairs:
        a, b = pair
        for source_a in range(dfa.n_states):
            for source_b in range(source_a, dfa.n_states):
                for letter in range(dfa.alphabet_size):
                    target = _pair(
                        dfa.transitions[source_a][letter],
                        dfa.transitions[source_b][letter],
                    )
                    if target == pair:
                        reverse[pair].append((source_a, source_b))

    distances: dict[tuple[int, int], int] = {}
    queue: deque[tuple[int, int]] = deque()
    for state in range(dfa.n_states):
        pair = (state, state)
        distances[pair] = 0
        queue.append(pair)

    while queue:
        pair = queue.popleft()
        for pred in reverse[pair]:
            if pred not in distances:
                distances[pred] = distances[pair] + 1
                queue.append(pred)
    return distances


def landmark_distance_sum(
    dfa: c3.DFA,
    mask: int,
    distances: dict[tuple[int, int], int],
) -> int:
    states = [state for state in range(dfa.n_states) if mask & (1 << state)]
    total = 0
    penalty = dfa.n_states * dfa.n_states
    for idx, a in enumerate(states):
        for b in states[idx + 1 :]:
            total += distances.get(_pair(a, b), penalty)
    return total


def _action_features(
    dfa: c3.DFA,
    mask: int,
    letter: int,
    distances: dict[tuple[int, int], int],
) -> tuple[int, int, int]:
    nxt = _apply_mask(dfa, mask, letter)
    return (
        nxt.bit_count(),
        landmark_distance_sum(dfa, nxt, distances),
        letter,
    )


def _choose_landmark_action(
    dfa: c3.DFA,
    mask: int,
    distances: dict[tuple[int, int], int],
    rng: random.Random | None = None,
    temperature: float = 0.0,
) -> int:
    features = [
        (letter, _action_features(dfa, mask, letter, distances))
        for letter in range(dfa.alphabet_size)
    ]
    if temperature <= 0:
        return min(features, key=lambda item: item[1])[0]
    weights = []
    for _, feature in features:
        size, dist, _ = feature
        weights.append(math.exp(-(size * 10.0 + dist) / temperature))
    return _weighted_choice([letter for letter, _ in features], weights, rng or random.Random())


def landmark_greedy_solver(dfa: c3.DFA, max_steps: int | None = None) -> SolverResult:
    start = time.perf_counter()
    distances = pair_merge_distances(dfa)
    mask = (1 << dfa.n_states) - 1
    max_steps = max_steps or max(4, dfa.n_states * dfa.n_states * 2)
    word: list[int] = []
    seen_steps = 0
    seen_masks = {mask}
    while not _is_singleton(mask) and len(word) < max_steps:
        ranked = []
        for letter in range(dfa.alphabet_size):
            nxt = _apply_mask(dfa, mask, letter)
            no_progress = nxt == mask
            revisits = nxt in seen_masks
            ranked.append((no_progress, revisits, _action_features(dfa, mask, letter, distances), letter))
        letter = min(ranked)[-1]
        word.append(letter)
        mask = _apply_mask(dfa, mask, letter)
        seen_masks.add(mask)
        seen_steps += 1
    reset_word = tuple(word) if _is_singleton(mask) else None
    return SolverResult(
        variant="greedy_pair",
        dfa=dfa,
        instance_id="single",
        reset_word=reset_word,
        found_length=len(reset_word) if reset_word is not None else max_steps + 1,
        iterations=seen_steps,
        expanded_nodes=seen_steps,
        elapsed_s=time.perf_counter() - start,
        landmark_guided_choices=seen_steps,
    )


def certify_result(dfa: c3.DFA, result: SolverResult) -> SolverResult:
    optimal, _, _ = c3.shortest_reset_word(dfa)
    gap = None
    certified = False
    if optimal is not None:
        gap = result.found_length - optimal
        certified = result.reset_word is not None and gap == 0
    return SolverResult(
        variant=result.variant,
        dfa=dfa,
        instance_id=result.instance_id,
        reset_word=result.reset_word,
        found_length=result.found_length,
        iterations=result.iterations,
        expanded_nodes=result.expanded_nodes,
        elapsed_s=result.elapsed_s,
        landmark_guided_choices=result.landmark_guided_choices,
        optimal_length=optimal,
        optimality_gap=gap,
        certified_optimal=certified,
    )


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


def _new_node(dfa: c3.DFA, mask: int, word: tuple[int, ...], parent: SearchNode | None) -> SearchNode:
    return SearchNode(
        mask=mask,
        word=word,
        parent=parent,
        untried=list(range(dfa.alphabet_size)),
    )


def _ucb_child(node: SearchNode, exploration: float) -> SearchNode:
    log_parent = math.log(max(1, node.visits))
    return max(
        node.children.values(),
        key=lambda child: (
            child.reward_sum / max(1, child.visits)
            + exploration * math.sqrt(log_parent / max(1, child.visits))
        ),
    )


def _rollout(
    dfa: c3.DFA,
    mask: int,
    prefix: tuple[int, ...],
    distances: dict[tuple[int, int], int],
    rng: random.Random,
    max_depth: int,
    variant: str,
) -> tuple[tuple[int, ...] | None, float, int]:
    word = list(prefix)
    guided = 0
    current = mask
    best_size = current.bit_count()
    while not _is_singleton(current) and len(word) < max_depth:
        if variant == "landmark_mcts":
            if rng.random() < 0.82:
                letter = _choose_landmark_action(dfa, current, distances, rng, temperature=6.0)
                guided += 1
            else:
                letter = rng.randrange(dfa.alphabet_size)
        else:
            letter = rng.randrange(dfa.alphabet_size)
        word.append(letter)
        current = _apply_mask(dfa, current, letter)
        best_size = min(best_size, current.bit_count())

    if _is_singleton(current):
        reward = 2.0 + (max_depth - len(word)) / max_depth
        return tuple(word), reward, guided
    unresolved = unresolved_pair_count(current)
    landmark_sum = landmark_distance_sum(dfa, current, distances)
    reward = (
        1.0 / max(1, best_size)
        + 1.0 / (1.0 + unresolved)
        + 1.0 / (1.0 + landmark_sum)
    )
    return None, reward, guided


def mcts_solver(
    dfa: c3.DFA,
    variant: str,
    iterations: int,
    rollout_depth_factor: int,
    seed: int,
    instance_id: str = "single",
) -> SolverResult:
    start = time.perf_counter()
    rng = random.Random(seed)
    distances = pair_merge_distances(dfa)
    full = (1 << dfa.n_states) - 1
    max_depth = max(4, rollout_depth_factor * dfa.n_states, dfa.n_states * dfa.n_states + 2)
    root = _new_node(dfa, full, (), None)
    best_word: tuple[int, ...] | None = None
    expanded = 0
    guided = 0

    for _ in range(iterations):
        node = root
        path = [node]
        while not node.untried and node.children and not _is_singleton(node.mask):
            node = _ucb_child(node, exploration=1.2)
            path.append(node)

        if node.untried and not _is_singleton(node.mask):
            if variant == "landmark_mcts":
                ranked = sorted(
                    node.untried,
                    key=lambda letter: _action_features(dfa, node.mask, letter, distances),
                )
                letter = ranked[0] if rng.random() < 0.75 else rng.choice(node.untried)
                guided += 1
            else:
                letter = rng.choice(node.untried)
            node.untried.remove(letter)
            nxt = _apply_mask(dfa, node.mask, letter)
            child = _new_node(dfa, nxt, (*node.word, letter), node)
            node.children[letter] = child
            node = child
            path.append(node)
            expanded += 1

        rollout_word, reward, rollout_guided = _rollout(
            dfa,
            node.mask,
            node.word,
            distances,
            rng,
            max_depth,
            variant,
        )
        guided += rollout_guided
        if rollout_word is not None and (best_word is None or len(rollout_word) < len(best_word)):
            best_word = rollout_word
            reward += 1.0

        for visited in path:
            visited.visits += 1
            visited.reward_sum += reward

    return SolverResult(
        variant=variant,
        dfa=dfa,
        instance_id=instance_id,
        reset_word=best_word,
        found_length=len(best_word) if best_word is not None else max_depth + 1,
        iterations=iterations,
        expanded_nodes=expanded,
        elapsed_s=time.perf_counter() - start,
        landmark_guided_choices=guided,
    )


def _instance_suite(n_values: list[int], instances_per_n: int, seed: int) -> list[tuple[str, c3.DFA]]:
    rng = random.Random(seed)
    out: list[tuple[str, c3.DFA]] = []

    def add_if_sync(instance_id: str, dfa: c3.DFA) -> None:
        if c3.shortest_reset_word(dfa)[0] is not None:
            out.append((instance_id, dfa))

    for n in n_values:
        add_if_sync(f"cerny-{n}", c3.cerny_automaton(n))
        add_if_sync(f"far-seed-{n}", c3c.far_cycle_seed(n, min(6, n - 1)))
        idx = 0
        attempts = 0
        while idx < max(0, instances_per_n - 2) and attempts < 1000:
            attempts += 1
            dfa = c3b.random_cycle_dfa(n, rng)
            before = len(out)
            add_if_sync(f"random-cycle-{n}-{idx}", dfa)
            if len(out) > before:
                idx += 1
    return out


def run_benchmark(
    n_values: list[int] | None = None,
    instances_per_n: int = 4,
    iterations: int = 350,
    rollout_depth_factor: int = 4,
    seed: int = 20260526,
) -> BenchmarkReport:
    n_values = n_values or [7, 8, 9, 10]
    rows: list[SolverResult] = []
    for instance_index, (instance_id, dfa) in enumerate(_instance_suite(n_values, instances_per_n, seed)):
        greedy = landmark_greedy_solver(dfa)
        rows.append(certify_result(dfa, _with_instance(greedy, instance_id)))
        rows.append(
            certify_result(
                dfa,
                mcts_solver(
                    dfa,
                    "plain_mcts",
                    iterations,
                    rollout_depth_factor,
                    seed + instance_index * 1009 + 1,
                    instance_id,
                ),
            )
        )
        rows.append(
            certify_result(
                dfa,
                mcts_solver(
                    dfa,
                    "landmark_mcts",
                    iterations,
                    rollout_depth_factor,
                    seed + instance_index * 1009 + 2,
                    instance_id,
                ),
            )
        )

    summary = summarize(rows)
    return BenchmarkReport(
        n_values=n_values,
        instances_per_n=instances_per_n,
        iterations=iterations,
        rollout_depth_factor=rollout_depth_factor,
        rows=rows,
        summary=summary,
        verdict=str(summary["verdict"]),
    )


def _with_instance(result: SolverResult, instance_id: str) -> SolverResult:
    return SolverResult(
        variant=result.variant,
        dfa=result.dfa,
        instance_id=instance_id,
        reset_word=result.reset_word,
        found_length=result.found_length,
        iterations=result.iterations,
        expanded_nodes=result.expanded_nodes,
        elapsed_s=result.elapsed_s,
        landmark_guided_choices=result.landmark_guided_choices,
        optimal_length=result.optimal_length,
        optimality_gap=result.optimality_gap,
        certified_optimal=result.certified_optimal,
    )


def summarize(rows: list[SolverResult]) -> dict[str, float | int | str]:
    by_variant: dict[str, list[SolverResult]] = {}
    for row in rows:
        by_variant.setdefault(row.variant, []).append(row)

    def mean_gap(variant: str) -> float:
        gaps = [float(row.optimality_gap or 0) for row in by_variant[variant]]
        return statistics.mean(gaps)

    def optimal_rate(variant: str) -> float:
        return statistics.mean(1.0 if row.certified_optimal else 0.0 for row in by_variant[variant])

    landmark_gap = mean_gap("landmark_mcts")
    plain_gap = mean_gap("plain_mcts")
    greedy_gap = mean_gap("greedy_pair")
    if landmark_gap + 0.05 < min(plain_gap, greedy_gap):
        verdict = "partial-positive"
    elif landmark_gap <= plain_gap + 0.05:
        verdict = "partial-negative"
    else:
        verdict = "clean-negative"
    return {
        "greedy_mean_gap": greedy_gap,
        "plain_mcts_mean_gap": plain_gap,
        "landmark_mcts_mean_gap": landmark_gap,
        "greedy_optimal_rate": optimal_rate("greedy_pair"),
        "plain_mcts_optimal_rate": optimal_rate("plain_mcts"),
        "landmark_mcts_optimal_rate": optimal_rate("landmark_mcts"),
        "n_instances": len({row.instance_id for row in rows}),
        "verdict": verdict,
    }


def _variant_table(report: BenchmarkReport) -> list[str]:
    lines = [
        "| variant | mean optimality gap | optimal rate | mean elapsed s |",
        "|---|---:|---:|---:|",
    ]
    for variant in ["greedy_pair", "plain_mcts", "landmark_mcts"]:
        rows = [row for row in report.rows if row.variant == variant]
        mean_gap = statistics.mean(float(row.optimality_gap or 0) for row in rows)
        opt_rate = statistics.mean(1.0 if row.certified_optimal else 0.0 for row in rows)
        elapsed = statistics.mean(row.elapsed_s for row in rows)
        lines.append(f"| {variant} | {mean_gap:.3f} | {opt_rate:.3f} | {elapsed:.3f} |")
    return lines


def _hard_cases(report: BenchmarkReport, limit: int = 8) -> list[str]:
    lines = [
        "| instance | optimal | greedy | plain MCTS | landmark MCTS |",
        "|---|---:|---:|---:|---:|",
    ]
    by_instance: dict[str, dict[str, SolverResult]] = {}
    for row in report.rows:
        by_instance.setdefault(row.instance_id, {})[row.variant] = row
    ranked = sorted(
        by_instance.items(),
        key=lambda item: max(row.found_length for row in item[1].values()),
        reverse=True,
    )
    for instance_id, variants in ranked[:limit]:
        optimal = next(iter(variants.values())).optimal_length
        lines.append(
            "| "
            f"{instance_id} | {optimal} | {variants['greedy_pair'].found_length} | "
            f"{variants['plain_mcts'].found_length} | {variants['landmark_mcts'].found_length} |"
        )
    return lines


def format_findings(report: BenchmarkReport) -> str:
    summary = report.summary
    if report.verdict == "partial-positive":
        result = (
            "Landmark MCTS found shorter reset words than both plain MCTS and the "
            "greedy pair-compression baseline on average."
        )
    elif report.verdict == "partial-negative":
        result = (
            "Landmark MCTS was competitive with plain MCTS, but it did not beat the "
            "strong greedy pair-compression baseline."
        )
    else:
        result = (
            "Landmark MCTS did not improve over plain MCTS or greedy pair compression "
            "under this local benchmark."
        )

    lines = [
        "# Černý C4 verification — landmark-biased MCTS reset solver",
        "",
        f"- Originating run: `{RUN_ID}`",
        f"- Composition: `{COMPOSITION_ID}`",
        f"- Outcome: **{report.verdict}**",
        f"- n values: `{report.n_values}`",
        f"- instances per n: `{report.instances_per_n}`",
        f"- MCTS iterations: `{report.iterations}`",
        f"- rollout depth factor: `{report.rollout_depth_factor}`",
        "",
        "## Claim Tested",
        "",
        (
            "C4 claimed that MCTS plus DFA-specific planning landmarks and bounded "
            "certification could find short reset words better than unguided search. "
            "This harness uses pair-merge landmarks as the planning facts and a "
            "bounded exact-BFS certificate as the local substitute for SAT checks."
        ),
        "",
        "## Result",
        "",
        result,
        "",
        *_variant_table(report),
        "",
        "## Key Numbers",
        "",
        f"- Greedy mean optimality gap: `{summary['greedy_mean_gap']:.3f}`",
        f"- Plain MCTS mean optimality gap: `{summary['plain_mcts_mean_gap']:.3f}`",
        f"- Landmark MCTS mean optimality gap: `{summary['landmark_mcts_mean_gap']:.3f}`",
        f"- Landmark MCTS optimal rate: `{summary['landmark_mcts_optimal_rate']:.3f}`",
        f"- Instances tested: `{summary['n_instances']}`",
        "",
        "## Representative Cases",
        "",
        *_hard_cases(report),
        "",
        "## L3 Attacks Checked",
        "",
        "| L3 attack | Verdict from this run |",
        "|---|---|",
        (
            "| Landmarks may be vacuous | Pair-merge landmarks are necessary reset "
            "obligations, but their guidance largely reduces to pair-compression progress. |"
        ),
        (
            "| MCTS may be mismatched to deterministic shortest-word search | Tested "
            "directly against greedy and plain MCTS under fixed rollout budgets. |"
        ),
        (
            "| SAT checks may only certify | The bounded exact-BFS certificate validated "
            "optimality gaps; it did not feed back into MCTS discovery. |"
        ),
        "",
        "## Verdict",
        "",
        (
            "This is a local algorithmic benchmark, not a theorem about reset-word "
            "search. A stronger C4 would need landmarks that add information beyond "
            "ordinary pair-compression and a genuine SAT/UNSAT feedback path."
        ),
        "",
        "## Artifacts",
        "",
        "- `experiments/cerny_c4/bench.py` — landmark extraction, solvers, exact certificates",
        "- `tests/test_cerny_c4.py` — regression tests for landmarks and benchmark validation",
    ]
    return "\n".join(lines) + "\n"


def write_findings(report: BenchmarkReport, path: Path = DEFAULT_FINDINGS_PATH) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(format_findings(report), encoding="utf-8")
    return path


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--write-findings", action="store_true")
    parser.add_argument("--iterations", type=int, default=350)
    parser.add_argument("--rollout-depth-factor", type=int, default=4)
    parser.add_argument("--instances-per-n", type=int, default=4)
    parser.add_argument("--n-values", default="7,8,9,10")
    parser.add_argument("--seed", type=int, default=20260526)
    parser.add_argument("--findings-path", type=Path, default=DEFAULT_FINDINGS_PATH)
    args = parser.parse_args(argv)

    n_values = [int(part.strip()) for part in args.n_values.split(",") if part.strip()]
    iterations = 80 if args.smoke else args.iterations
    instances = 2 if args.smoke else args.instances_per_n
    depth = 3 if args.smoke else args.rollout_depth_factor
    if args.smoke:
        n_values = [6]

    report = run_benchmark(
        n_values=n_values,
        instances_per_n=instances,
        iterations=iterations,
        rollout_depth_factor=depth,
        seed=args.seed,
    )
    print(format_findings(report))
    if args.write_findings:
        path = write_findings(report, args.findings_path)
        print(f"\nwrote {path}")


if __name__ == "__main__":
    main()
