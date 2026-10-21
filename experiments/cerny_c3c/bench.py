"""C3c verification: distance-stress test for Černý hard automata.

C3b showed a strong strict result, but the best automaton was still only two
transition edits from the formula Černý family. C3c asks whether that advantage
survives when candidates must be farther from Černý, using thresholds such as
distance >= 4 and distance >= 6.

Run:
    uv run python -m experiments.cerny_c3c.bench --smoke
    uv run python -m experiments.cerny_c3c.bench --write-findings
"""

from __future__ import annotations

import argparse
import random
import statistics
import time
from dataclasses import dataclass
from pathlib import Path

from experiments.cerny_c3 import bench as c3
from experiments.cerny_c3b import bench as c3b

RUN_ID = "20260526-062120"
COMPOSITION_ID = "C3c"
DEFAULT_FINDINGS_PATH = Path("findings/c3c-cerny-distance-stress-20260526.md")


@dataclass(frozen=True)
class DistanceFamilyResult:
    family: str
    n_states: int
    min_distance: int
    seed: int
    budget: int
    attempts: int
    accepted: int
    unique_scored: int
    best: c3b.StrictScore
    elapsed_s: float


@dataclass(frozen=True)
class BenchmarkReport:
    n_values: list[int]
    distance_thresholds: list[int]
    trials_per_family: int
    budget: int
    rows: list[DistanceFamilyResult]
    summary: dict[str, float | int | str]
    verdict: str


def passes_distance_gate(score: c3b.StrictScore, min_distance: int) -> bool:
    return score.strictly_novel and score.distance_from_cerny >= min_distance


def far_cycle_seed(n_states: int, min_distance: int) -> c3.DFA:
    """Deterministically perturb the Černý compression letter at distinct states."""

    if min_distance >= n_states:
        raise ValueError("min_distance must be smaller than n_states for this seed")
    transitions = [list(row) for row in c3.cerny_automaton(n_states).transitions]
    for offset, state in enumerate(range(1, min_distance + 1)):
        old = transitions[state][1]
        transitions[state][1] = (old + offset + 2) % n_states
    return c3.DFA(tuple(tuple(row) for row in transitions))


def mutate_far_cycle(dfa: c3.DFA, rng: random.Random, min_edits: int = 1) -> c3.DFA:
    out = dfa
    edits = min_edits
    if rng.random() < 0.35:
        edits += 1
    for _ in range(edits):
        out = c3b.mutate_cycle_letter(out, rng)
    return out


def _best(rows: list[c3b.StrictScore]) -> c3b.StrictScore:
    return max(rows, key=lambda row: row.objective_tuple)


def _sample_far_cycle(
    n_states: int,
    min_distance: int,
    rng: random.Random,
    label: str,
    max_attempts: int = 4000,
) -> tuple[c3b.StrictScore, int]:
    reference = c3.cerny_automaton(n_states)
    for attempt in range(1, max_attempts + 1):
        dfa = c3b.random_cycle_dfa(n_states, rng)
        score = c3b.score_strict(dfa, f"{label}_{attempt}", reference)
        if passes_distance_gate(score, min_distance):
            return score, attempt
    fallback = c3b.score_strict(far_cycle_seed(n_states, min_distance), f"{label}_fallback")
    return fallback, max_attempts


def _sample_far_seed_mutation(
    n_states: int,
    min_distance: int,
    rng: random.Random,
    label: str,
    max_attempts: int = 4000,
) -> tuple[c3b.StrictScore, int]:
    reference = c3.cerny_automaton(n_states)
    base = far_cycle_seed(n_states, min_distance)
    for attempt in range(1, max_attempts + 1):
        dfa = mutate_far_cycle(base, rng, min_edits=1)
        score = c3b.score_strict(dfa, f"{label}_{attempt}", reference)
        if passes_distance_gate(score, min_distance):
            return score, attempt
    fallback = c3b.score_strict(base, f"{label}_fallback", reference)
    return fallback, max_attempts


def run_far_random(
    n_states: int,
    min_distance: int,
    budget: int,
    seed: int,
) -> DistanceFamilyResult:
    rng = random.Random(seed)
    start = time.perf_counter()
    rows: list[c3b.StrictScore] = []
    attempts = 0
    for idx in range(budget):
        score, used = _sample_far_cycle(n_states, min_distance, rng, f"far_random_{idx}")
        attempts += used
        if passes_distance_gate(score, min_distance):
            rows.append(score)
    if not rows:
        score, used = _sample_far_seed_mutation(n_states, min_distance, rng, "far_random_recovery")
        attempts += used
        rows.append(score)
    return DistanceFamilyResult(
        family="far_random",
        n_states=n_states,
        min_distance=min_distance,
        seed=seed,
        budget=budget,
        attempts=attempts,
        accepted=len(rows),
        unique_scored=len({row.signature for row in rows}),
        best=_best(rows),
        elapsed_s=time.perf_counter() - start,
    )


def run_far_hill_climb(
    n_states: int,
    min_distance: int,
    budget: int,
    seed: int,
) -> DistanceFamilyResult:
    rng = random.Random(seed)
    reference = c3.cerny_automaton(n_states)
    start = time.perf_counter()
    current, attempts = _sample_far_cycle(n_states, min_distance, rng, "hill_start")
    rows = [current]
    for idx in range(budget):
        attempts += 1
        score = c3b.score_strict(
            mutate_far_cycle(current.dfa, rng),
            f"far_hill_{idx}",
            reference,
        )
        if not passes_distance_gate(score, min_distance):
            continue
        rows.append(score)
        if score.objective_tuple >= current.objective_tuple:
            current = score
    return DistanceFamilyResult(
        family="far_hill_climb",
        n_states=n_states,
        min_distance=min_distance,
        seed=seed,
        budget=budget,
        attempts=attempts,
        accepted=len(rows),
        unique_scored=len({row.signature for row in rows}),
        best=_best(rows),
        elapsed_s=time.perf_counter() - start,
    )


def run_far_seeded_hill_climb(
    n_states: int,
    min_distance: int,
    budget: int,
    seed: int,
) -> DistanceFamilyResult:
    rng = random.Random(seed)
    reference = c3.cerny_automaton(n_states)
    start = time.perf_counter()
    current, attempts = _sample_far_seed_mutation(
        n_states,
        min_distance,
        rng,
        "seeded_hill_start",
    )
    rows = [current]
    for idx in range(budget):
        attempts += 1
        score = c3b.score_strict(
            mutate_far_cycle(current.dfa, rng),
            f"far_seeded_hill_{idx}",
            reference,
        )
        if not passes_distance_gate(score, min_distance):
            continue
        rows.append(score)
        if score.objective_tuple >= current.objective_tuple:
            current = score
    return DistanceFamilyResult(
        family="far_seeded_hill_climb",
        n_states=n_states,
        min_distance=min_distance,
        seed=seed,
        budget=budget,
        attempts=attempts,
        accepted=len(rows),
        unique_scored=len({row.signature for row in rows}),
        best=_best(rows),
        elapsed_s=time.perf_counter() - start,
    )


def run_far_cycle_evolution(
    n_states: int,
    min_distance: int,
    budget: int,
    seed: int,
    population_limit: int = 24,
) -> DistanceFamilyResult:
    rng = random.Random(seed)
    reference = c3.cerny_automaton(n_states)
    start = time.perf_counter()
    rows: list[c3b.StrictScore] = []
    attempts = 0
    parent_pool = [far_cycle_seed(n_states, min_distance), reference]

    while len(rows) < 6 and attempts < 5000:
        attempts += 1
        parent = rng.choice(parent_pool)
        score = c3b.score_strict(
            mutate_far_cycle(parent, rng, min_edits=1),
            f"far_evo_seed_{attempts}",
            reference,
        )
        parent_pool.append(score.dfa)
        if passes_distance_gate(score, min_distance):
            rows.append(score)

    if not rows:
        score, used = _sample_far_seed_mutation(n_states, min_distance, rng, "far_evo_recovery")
        attempts += used
        rows.append(score)

    population = _front(rows, population_limit, min_distance)
    for idx in range(budget):
        attempts += 1
        parent = rng.choice(population).dfa if population else rng.choice(parent_pool)
        score = c3b.score_strict(
            mutate_far_cycle(parent, rng),
            f"far_evo_{idx}",
            reference,
        )
        parent_pool.append(score.dfa)
        if not passes_distance_gate(score, min_distance):
            continue
        rows.append(score)
        population = _front(population + [score], population_limit, min_distance)

    return DistanceFamilyResult(
        family="far_cycle_evolution",
        n_states=n_states,
        min_distance=min_distance,
        seed=seed,
        budget=budget,
        attempts=attempts,
        accepted=len(rows),
        unique_scored=len({row.signature for row in rows}),
        best=_best(rows),
        elapsed_s=time.perf_counter() - start,
    )


def _front(
    rows: list[c3b.StrictScore],
    limit: int,
    min_distance: int,
) -> list[c3b.StrictScore]:
    by_signature: dict[str, c3b.StrictScore] = {}
    for row in rows:
        if not passes_distance_gate(row, min_distance):
            continue
        old = by_signature.get(row.signature)
        if old is None or row.objective_tuple > old.objective_tuple:
            by_signature[row.signature] = row
    front = list(by_signature.values())
    front.sort(key=lambda row: row.objective_tuple, reverse=True)
    return front[:limit]


def run_benchmark(
    n_values: list[int] | None = None,
    distance_thresholds: list[int] | None = None,
    trials_per_family: int = 2,
    budget: int = 420,
    seed: int = 20260526,
) -> BenchmarkReport:
    n_values = n_values or [8, 9, 10]
    distance_thresholds = distance_thresholds or [4, 6]
    rows: list[DistanceFamilyResult] = []
    for n_states in n_values:
        for min_distance in distance_thresholds:
            if min_distance >= n_states:
                continue
            for trial in range(trials_per_family):
                base_seed = seed + n_states * 10_000 + min_distance * 701 + trial * 313
                rows.append(run_far_random(n_states, min_distance, budget, base_seed + 1))
                rows.append(run_far_hill_climb(n_states, min_distance, budget, base_seed + 2))
                rows.append(
                    run_far_seeded_hill_climb(n_states, min_distance, budget, base_seed + 3)
                )
                rows.append(run_far_cycle_evolution(n_states, min_distance, budget, base_seed + 4))
    summary = summarize(rows)
    return BenchmarkReport(
        n_values=n_values,
        distance_thresholds=distance_thresholds,
        trials_per_family=trials_per_family,
        budget=budget,
        rows=rows,
        summary=summary,
        verdict=str(summary["verdict"]),
    )


def summarize(rows: list[DistanceFamilyResult]) -> dict[str, float | int | str]:
    threshold_deltas: dict[int, int] = {}
    threshold_best_evo: dict[int, int] = {}
    threshold_best_baseline: dict[int, int] = {}
    for threshold in sorted({row.min_distance for row in rows}):
        group = [row for row in rows if row.min_distance == threshold]
        evo = max(
            int(row.best.reset_length or -1)
            for row in group
            if row.family == "far_cycle_evolution"
        )
        baseline = max(
            int(row.best.reset_length or -1)
            for row in group
            if row.family in {"far_random", "far_hill_climb", "far_seeded_hill_climb"}
        )
        threshold_deltas[threshold] = evo - baseline
        threshold_best_evo[threshold] = evo
        threshold_best_baseline[threshold] = baseline

    min_delta = min(threshold_deltas.values()) if threshold_deltas else -999
    if min_delta >= 2:
        verdict = "partial-positive"
    elif min_delta >= 0:
        verdict = "partial-negative"
    else:
        verdict = "clean-negative"
    yields = [row.accepted / max(1, row.attempts) for row in rows]
    return {
        "min_delta_across_thresholds": min_delta,
        "best_evolution_reset_length": max(threshold_best_evo.values()) if threshold_best_evo else -1,
        "best_baseline_reset_length": max(threshold_best_baseline.values()) if threshold_best_baseline else -1,
        "mean_acceptance_yield": statistics.mean(yields) if yields else 0.0,
        "threshold_deltas": ", ".join(
            f"d>={threshold}:{delta}" for threshold, delta in threshold_deltas.items()
        ),
        "verdict": verdict,
    }


def _best_result(report: BenchmarkReport) -> tuple[DistanceFamilyResult, c3b.StrictScore]:
    row = max(report.rows, key=lambda item: item.best.objective_tuple)
    return row, row.best


def _threshold_table(report: BenchmarkReport) -> list[str]:
    lines = [
        "| threshold | family | best reset length | mean acceptance yield | mean elapsed s |",
        "|---:|---|---:|---:|---:|",
    ]
    for threshold in sorted({row.min_distance for row in report.rows}):
        for family in [
            "far_random",
            "far_hill_climb",
            "far_seeded_hill_climb",
            "far_cycle_evolution",
        ]:
            rows = [
                row
                for row in report.rows
                if row.min_distance == threshold and row.family == family
            ]
            best = max(int(row.best.reset_length or -1) for row in rows)
            mean_yield = statistics.mean(row.accepted / max(1, row.attempts) for row in rows)
            mean_elapsed = statistics.mean(row.elapsed_s for row in rows)
            lines.append(
                f"| distance >= {threshold} | {family} | {best} | "
                f"{mean_yield:.3f} | {mean_elapsed:.3f} |"
            )
    return lines


def _transition_table(dfa: c3.DFA) -> str:
    lines = ["| state | letter 0 | letter 1 |", "|---:|---:|---:|"]
    for state, row in enumerate(dfa.transitions):
        lines.append(f"| {state} | {row[0]} | {row[1]} |")
    return "\n".join(lines)


def _word_string(word: tuple[int, ...] | None, max_chars: int = 100) -> str:
    if word is None:
        return "n/a"
    out = "".join(str(letter) for letter in word)
    return out if len(out) <= max_chars else out[: max_chars - 3] + "..."


def format_findings(report: BenchmarkReport) -> str:
    summary = report.summary
    best_row, best = _best_result(report)
    if report.verdict == "partial-positive":
        result = (
            "The near-Černý advantage survives this distance stress test: far cycle "
            "evolution beat the best far random, random-start hill, and seeded-hill "
            "baseline at every tested distance threshold."
        )
    elif report.verdict == "partial-negative":
        result = (
            "The near-Černý advantage weakens under distance stress: far cycle evolution "
            "ties or narrowly beats baselines, but without a meaningful margin."
        )
    else:
        result = (
            "The near-Černý advantage collapses under distance stress: once candidates "
            "must be farther from Černý, far cycle evolution no longer beats the baselines."
        )

    lines = [
        "# Černý C3c verification — distance stress test",
        "",
        f"- Originating run: `{RUN_ID}`",
        f"- Composition: `{COMPOSITION_ID}` follow-up",
        f"- Outcome: **{report.verdict}**",
        f"- n values: `{report.n_values}`",
        f"- distance thresholds: `{report.distance_thresholds}`",
        f"- trials per family: `{report.trials_per_family}`",
        f"- budget per family/trial: `{report.budget}` exact evaluations",
        "",
        "## Claim Tested",
        "",
        (
            "C3c asks whether the C3b signal survives after forcing candidates out "
            "of the immediate Černý neighborhood. Candidates must still be strongly "
            "connected, synchronizing, and free of closed-core attachments."
        ),
        "",
        "## Result",
        "",
        result,
        "",
        *_threshold_table(report),
        "",
        "## Key Numbers",
        "",
        f"- Best far evolutionary reset length: `{summary['best_evolution_reset_length']}`",
        f"- Best far baseline reset length: `{summary['best_baseline_reset_length']}`",
        f"- Minimum evolutionary delta across thresholds: `{summary['min_delta_across_thresholds']}`",
        f"- Threshold deltas: `{summary['threshold_deltas']}`",
        f"- Mean acceptance yield: `{summary['mean_acceptance_yield']:.3f}`",
        "",
        "## Best Far Candidate",
        "",
        f"- Family: `{best_row.family}`",
        f"- n states: `{best_row.n_states}`",
        f"- distance threshold: `distance >= {best_row.min_distance}`",
        f"- seed: `{best_row.seed}`",
        f"- exact reset length: `{best.reset_length}`",
        f"- exact BFS expanded subsets: `{best.expansions}`",
        f"- transition edits from formula Černý table: `{best.distance_from_cerny}`",
        f"- reset witness: `{_word_string(best.reset_word)}`",
        "",
        "The transition table uses rows as states and columns as input letters:",
        "",
        _transition_table(best.dfa),
        "",
        "## Verdict",
        "",
        (
            "C3c is not a proof of a new Černý-family construction. It tests whether "
            "the generator's value remains after a simple structural novelty bar. A "
            "next step would need an interpretable parametrized family or a much stronger "
            "distance-from-Černý requirement."
        ),
        "",
        "## Artifacts",
        "",
        "- `experiments/cerny_c3c/bench.py` — distance-threshold benchmark",
        "- `tests/test_cerny_c3c.py` — regression tests for the distance gate",
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
    parser.add_argument("--budget", type=int, default=420)
    parser.add_argument("--trials-per-family", type=int, default=2)
    parser.add_argument("--n-values", default="8,9,10")
    parser.add_argument("--distance-thresholds", default="4,6")
    parser.add_argument("--seed", type=int, default=20260526)
    parser.add_argument("--findings-path", type=Path, default=DEFAULT_FINDINGS_PATH)
    args = parser.parse_args(argv)

    n_values = [int(part.strip()) for part in args.n_values.split(",") if part.strip()]
    thresholds = [
        int(part.strip()) for part in args.distance_thresholds.split(",") if part.strip()
    ]
    budget = 50 if args.smoke else args.budget
    trials = 1 if args.smoke else args.trials_per_family
    if args.smoke:
        n_values = [7]
        thresholds = [4, 6]

    report = run_benchmark(
        n_values=n_values,
        distance_thresholds=thresholds,
        trials_per_family=trials,
        budget=budget,
        seed=args.seed,
    )
    print(format_findings(report))
    if args.write_findings:
        path = write_findings(report, args.findings_path)
        print(f"\nwrote {path}")


if __name__ == "__main__":
    main()
