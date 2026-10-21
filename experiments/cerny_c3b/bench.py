"""C3b verification: strict evolutionary hard-automata generator.

This is the follow-up to ``experiments.cerny_c3``. The first C3 pass found a
hard non-isomorphic n=8 automaton, but the best candidate was not strongly
connected and contained a closed 7-state Černý-like core. C3b tightens the
definition of discovery:

* candidates must be synchronizing,
* the transition digraph must be strongly connected,
* no proper closed subautomaton may carry the same reset threshold,
* exact Černý relabelings are excluded from the novel-candidate pool.

Run:
    uv run python -m experiments.cerny_c3b.bench --smoke
    uv run python -m experiments.cerny_c3b.bench --write-findings
"""

from __future__ import annotations

import argparse
import random
import statistics
import time
from collections import deque
from dataclasses import dataclass
from pathlib import Path

from experiments.cerny_c3 import bench as c3

RUN_ID = "20260526-062120"
COMPOSITION_ID = "C3b"
DEFAULT_FINDINGS_PATH = Path("findings/c3b-cerny-strict-hard-automata-20260526.md")


@dataclass(frozen=True)
class StrictScore:
    label: str
    dfa: c3.DFA
    synchronizing: bool
    reset_length: int | None
    expansions: int
    reset_word: tuple[int, ...] | None
    signature: str
    distance_from_cerny: int
    strongly_connected: bool
    closed_core_size: int | None
    closed_core_reset_length: int | None
    strictly_novel: bool

    @property
    def objective_tuple(self) -> tuple[int, int, int]:
        return (
            self.reset_length if self.reset_length is not None else -1,
            self.expansions,
            self.distance_from_cerny,
        )


@dataclass(frozen=True)
class FamilyResult:
    family: str
    n_states: int
    seed: int
    budget: int
    attempts: int
    strict_scored: int
    unique_scored: int
    best: StrictScore
    elapsed_s: float


@dataclass(frozen=True)
class BenchmarkReport:
    n_values: list[int]
    trials_per_family: int
    budget: int
    rows: list[FamilyResult]
    summary: dict[str, float | int | str]
    verdict: str


def is_strongly_connected(dfa: c3.DFA) -> bool:
    dfa.validate()
    n = dfa.n_states

    def reachable(start: int, reverse: bool = False) -> set[int]:
        graph = [[] for _ in range(n)]
        for state, row in enumerate(dfa.transitions):
            for target in row:
                if reverse:
                    graph[target].append(state)
                else:
                    graph[state].append(target)
        seen = {start}
        queue: deque[int] = deque([start])
        while queue:
            state = queue.popleft()
            for nxt in graph[state]:
                if nxt not in seen:
                    seen.add(nxt)
                    queue.append(nxt)
        return seen

    return len(reachable(0)) == n and len(reachable(0, reverse=True)) == n


def _is_closed_subset(dfa: c3.DFA, mask: int) -> bool:
    for state, row in enumerate(dfa.transitions):
        if not (mask & (1 << state)):
            continue
        for target in row:
            if not (mask & (1 << target)):
                return False
    return True


def _subdfa_from_mask(dfa: c3.DFA, mask: int) -> c3.DFA:
    states = [state for state in range(dfa.n_states) if mask & (1 << state)]
    relabel = {old: new for new, old in enumerate(states)}
    return c3.DFA(
        tuple(tuple(relabel[dfa.transitions[state][letter]] for letter in range(dfa.alphabet_size))
              for state in states)
    )


def closed_core_with_same_threshold(
    dfa: c3.DFA,
    reset_length: int | None,
) -> tuple[int, int] | None:
    if reset_length is None:
        return None
    full = (1 << dfa.n_states) - 1
    best: tuple[int, int] | None = None
    for mask in range(1, full):
        size = mask.bit_count()
        if size < 2 or not _is_closed_subset(dfa, mask):
            continue
        sub_length, _, _ = c3.shortest_reset_word(_subdfa_from_mask(dfa, mask))
        if sub_length == reset_length:
            if best is None or size > best[0]:
                best = (size, sub_length)
    return best


def score_strict(
    dfa: c3.DFA,
    label: str,
    reference: c3.DFA | None = None,
) -> StrictScore:
    reference = reference or c3.cerny_automaton(dfa.n_states)
    base = c3.score_dfa(dfa, label=label, reference=reference)
    strong = is_strongly_connected(dfa)
    core = None if strong else closed_core_with_same_threshold(dfa, base.reset_length)
    strictly_novel = (
        base.synchronizing
        and strong
        and core is None
        and base.signature != c3.canonical_signature(reference)
        and base.distance_from_cerny >= 2
    )
    return StrictScore(
        label=label,
        dfa=dfa,
        synchronizing=base.synchronizing,
        reset_length=base.reset_length,
        expansions=base.expansions,
        reset_word=base.reset_word,
        signature=base.signature,
        distance_from_cerny=base.distance_from_cerny,
        strongly_connected=strong,
        closed_core_size=core[0] if core else None,
        closed_core_reset_length=core[1] if core else None,
        strictly_novel=strictly_novel,
    )


def mutate_cycle_letter(dfa: c3.DFA, rng: random.Random) -> c3.DFA:
    """Mutate only the non-cycle letter.

    The Černý family uses letter 0 as a Hamiltonian directed cycle. Preserving
    that letter keeps the transition digraph strongly connected while allowing
    letter 1 to explore alternative compression dynamics.
    """

    if dfa.alphabet_size != 2:
        raise ValueError("C3b cycle-letter mutation expects a binary alphabet")
    transitions = [list(row) for row in dfa.transitions]
    edits = 1 if rng.random() < 0.8 else 2
    for _ in range(edits):
        state = rng.randrange(dfa.n_states)
        old = transitions[state][1]
        choices = [target for target in range(dfa.n_states) if target != old]
        transitions[state][1] = rng.choice(choices)
    return c3.DFA(tuple(tuple(row) for row in transitions))


def random_cycle_dfa(n_states: int, rng: random.Random) -> c3.DFA:
    return c3.DFA(
        tuple(((state + 1) % n_states, rng.randrange(n_states)) for state in range(n_states))
    )


def _best(rows: list[StrictScore]) -> StrictScore:
    return max(rows, key=lambda row: row.objective_tuple)


def _strict_front(rows: list[StrictScore], limit: int) -> list[StrictScore]:
    best_by_signature: dict[str, StrictScore] = {}
    for row in rows:
        if not row.strictly_novel:
            continue
        old = best_by_signature.get(row.signature)
        if old is None or row.objective_tuple > old.objective_tuple:
            best_by_signature[row.signature] = row
    front = list(best_by_signature.values())
    front.sort(key=lambda row: row.objective_tuple, reverse=True)
    return front[:limit]


def _sample_strict_random(
    n_states: int,
    rng: random.Random,
    label: str,
    max_attempts: int = 2000,
) -> tuple[StrictScore, int]:
    reference = c3.cerny_automaton(n_states)
    for attempt in range(1, max_attempts + 1):
        dfa = c3.random_dfa(n_states, 2, rng)
        score = score_strict(dfa, f"{label}_{attempt}", reference)
        if score.strictly_novel:
            return score, attempt
    # Fallback keeps the harness total, but the row will make failure obvious.
    fallback = score_strict(random_cycle_dfa(n_states, rng), f"{label}_fallback", reference)
    return fallback, max_attempts


def run_cerny_reference(n_states: int, seed: int, budget: int) -> FamilyResult:
    start = time.perf_counter()
    score = score_strict(c3.cerny_automaton(n_states), "cerny_reference")
    return FamilyResult(
        family="cerny_reference",
        n_states=n_states,
        seed=seed,
        budget=budget,
        attempts=1,
        strict_scored=1,
        unique_scored=1,
        best=score,
        elapsed_s=time.perf_counter() - start,
    )


def run_strict_random(n_states: int, budget: int, seed: int) -> FamilyResult:
    rng = random.Random(seed)
    start = time.perf_counter()
    rows: list[StrictScore] = []
    attempts = 0
    for idx in range(budget):
        score, used = _sample_strict_random(n_states, rng, f"strict_random_{idx}")
        attempts += used
        if score.strictly_novel:
            rows.append(score)
    if not rows:
        score, used = _sample_strict_random(n_states, rng, "strict_random_recovery")
        attempts += used
        rows.append(score)
    signatures = {row.signature for row in rows}
    return FamilyResult(
        family="strict_random",
        n_states=n_states,
        seed=seed,
        budget=budget,
        attempts=attempts,
        strict_scored=len(rows),
        unique_scored=len(signatures),
        best=_best(rows),
        elapsed_s=time.perf_counter() - start,
    )


def run_strict_hill_climb(n_states: int, budget: int, seed: int) -> FamilyResult:
    rng = random.Random(seed)
    reference = c3.cerny_automaton(n_states)
    start = time.perf_counter()
    current, attempts = _sample_strict_random(n_states, rng, "hill_start")
    rows = [current]
    for idx in range(budget):
        attempts += 1
        candidate = score_strict(mutate_cycle_letter(current.dfa, rng), f"hill_{idx}", reference)
        if not candidate.strictly_novel:
            continue
        rows.append(candidate)
        if candidate.objective_tuple >= current.objective_tuple:
            current = candidate
    return FamilyResult(
        family="strict_hill_climb",
        n_states=n_states,
        seed=seed,
        budget=budget,
        attempts=attempts,
        strict_scored=len(rows),
        unique_scored=len({row.signature for row in rows}),
        best=_best(rows),
        elapsed_s=time.perf_counter() - start,
    )


def run_strict_cycle_evolution(
    n_states: int,
    budget: int,
    seed: int,
    population_limit: int = 24,
) -> FamilyResult:
    rng = random.Random(seed)
    reference = c3.cerny_automaton(n_states)
    start = time.perf_counter()
    rows: list[StrictScore] = []
    attempts = 0

    # Start near the known hard family, but do not allow the exact reference into
    # the strict novel pool.
    parent_pool = [reference]
    while len(rows) < 6:
        attempts += 1
        parent = rng.choice(parent_pool)
        candidate = score_strict(mutate_cycle_letter(parent, rng), f"evo_seed_{attempts}", reference)
        parent_pool.append(candidate.dfa)
        if candidate.strictly_novel:
            rows.append(candidate)

    population = _strict_front(rows, population_limit)
    for idx in range(budget):
        attempts += 1
        parent = rng.choice(population).dfa if population else rng.choice(parent_pool)
        candidate = score_strict(mutate_cycle_letter(parent, rng), f"evo_{idx}", reference)
        parent_pool.append(candidate.dfa)
        if not candidate.strictly_novel:
            continue
        rows.append(candidate)
        population = _strict_front(population + [candidate], population_limit)

    return FamilyResult(
        family="strict_cycle_evolution",
        n_states=n_states,
        seed=seed,
        budget=budget,
        attempts=attempts,
        strict_scored=len(rows),
        unique_scored=len({row.signature for row in rows}),
        best=_best(rows),
        elapsed_s=time.perf_counter() - start,
    )


def run_benchmark(
    n_values: list[int] | None = None,
    trials_per_family: int = 3,
    budget: int = 360,
    seed: int = 20260526,
) -> BenchmarkReport:
    n_values = n_values or [8, 9, 10]
    rows: list[FamilyResult] = []
    for n_states in n_values:
        for trial in range(trials_per_family):
            base_seed = seed + n_states * 10_000 + trial * 313
            rows.append(run_cerny_reference(n_states, base_seed, budget))
            rows.append(run_strict_random(n_states, budget, base_seed + 1))
            rows.append(run_strict_hill_climb(n_states, budget, base_seed + 2))
            rows.append(run_strict_cycle_evolution(n_states, budget, base_seed + 3))
    summary = summarize(rows)
    return BenchmarkReport(
        n_values=n_values,
        trials_per_family=trials_per_family,
        budget=budget,
        rows=rows,
        summary=summary,
        verdict=str(summary["verdict"]),
    )


def summarize(rows: list[FamilyResult]) -> dict[str, float | int | str]:
    by_family: dict[str, list[FamilyResult]] = {}
    for row in rows:
        by_family.setdefault(row.family, []).append(row)

    def best_length(family: str) -> int:
        return max(int(row.best.reset_length or -1) for row in by_family.get(family, []))

    random_best = best_length("strict_random")
    hill_best = best_length("strict_hill_climb")
    evo_best = best_length("strict_cycle_evolution")
    cerny_best = best_length("cerny_reference")
    baseline_best = max(random_best, hill_best)
    delta = evo_best - baseline_best
    if delta >= 2:
        verdict = "partial-positive"
    elif delta >= 0:
        verdict = "partial-negative"
    else:
        verdict = "clean-negative"
    strict_yields = [
        row.strict_scored / max(1, row.attempts)
        for row in rows
        if row.family != "cerny_reference"
    ]
    return {
        "random_best_reset_length": random_best,
        "hill_best_reset_length": hill_best,
        "evolution_best_reset_length": evo_best,
        "cerny_best_reset_length": cerny_best,
        "evolution_vs_best_strict_baseline_delta": delta,
        "mean_strict_yield": statistics.mean(strict_yields) if strict_yields else 0.0,
        "verdict": verdict,
    }


def _best_strict_result(report: BenchmarkReport) -> tuple[FamilyResult, StrictScore]:
    search_rows = [row for row in report.rows if row.family != "cerny_reference"]
    row = max(search_rows, key=lambda item: item.best.objective_tuple)
    return row, row.best


def _family_summary_rows(report: BenchmarkReport) -> list[str]:
    lines = [
        "| family | best strict reset length | mean strict yield | mean elapsed s |",
        "|---|---:|---:|---:|",
    ]
    for family in ["cerny_reference", "strict_random", "strict_hill_climb", "strict_cycle_evolution"]:
        rows = [row for row in report.rows if row.family == family]
        best = max(int(row.best.reset_length or -1) for row in rows)
        mean_yield = statistics.mean(row.strict_scored / max(1, row.attempts) for row in rows)
        mean_elapsed = statistics.mean(row.elapsed_s for row in rows)
        lines.append(f"| {family} | {best} | {mean_yield:.3f} | {mean_elapsed:.3f} |")
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
    best_row, best = _best_strict_result(report)
    verdict = report.verdict
    if verdict == "partial-positive":
        result = (
            "The strict evolutionary generator beat both strict random sampling and "
            "strict hill climbing after excluding closed-core attachments."
        )
    elif verdict == "partial-negative":
        result = (
            "The strict evolutionary generator matched the best strict baseline but "
            "did not create a meaningful margin."
        )
    else:
        result = (
            "The strict evolutionary generator did not beat the strict random or "
            "hill-climb baselines once attached-core candidates were excluded."
        )

    lines = [
        "# Černý C3b verification — strict hard-automata generator",
        "",
        f"- Originating run: `{RUN_ID}`",
        f"- Composition: `{COMPOSITION_ID}` follow-up",
        f"- Outcome: **{verdict}**",
        f"- n values: `{report.n_values}`",
        f"- trials per family: `{report.trials_per_family}`",
        f"- budget per stochastic family/trial: `{report.budget}` exact evaluations",
        "",
        "## Claim Tested",
        "",
        (
            "C3b asks whether the C3 evolutionary idea still works after rejecting "
            "closed-core attachments and requiring strong connectivity. This is a "
            "stricter discovery test than the original C3 run."
        ),
        "",
        "## Result",
        "",
        result,
        "",
        *_family_summary_rows(report),
        "",
        "## Key Numbers",
        "",
        f"- Best strict random reset length: `{summary['random_best_reset_length']}`",
        f"- Best strict hill-climb reset length: `{summary['hill_best_reset_length']}`",
        f"- Best strict evolutionary reset length: `{summary['evolution_best_reset_length']}`",
        f"- Best Černý reference reset length: `{summary['cerny_best_reset_length']}`",
        (
            "- Strict evolutionary delta vs best strict baseline: "
            f"`{summary['evolution_vs_best_strict_baseline_delta']}`"
        ),
        f"- Mean strict candidate yield: `{summary['mean_strict_yield']:.3f}`",
        "",
        "## Best Strict Novel Candidate",
        "",
        f"- Family: `{best_row.family}`",
        f"- n states: `{best_row.n_states}`",
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
        (
            "Caveat: this is a strict candidate, but it is still near-Černý: the "
            "best table differs from the formula Černý table by only "
            f"`{best.distance_from_cerny}` transition(s). This supports a stronger "
            "hard-instance generator claim, not yet a new structural family claim."
        ),
        "",
        "## Structural Filters",
        "",
        "- Strong connectivity is required, so one-way attached states are rejected.",
        "- Proper closed subautomata with the same reset length are rejected as closed-core attachments.",
        "- Exact state/letter relabelings of the formula Černý automaton are not counted as novel.",
        "",
        "## Verdict",
        "",
        (
            "C3b is a stricter local test, not a proof about the full conjecture. A "
            "compelling next result would show a repeatable family of strongly connected "
            "non-Černý automata whose reset ratios scale better than random or local "
            "mutation baselines."
        ),
        "",
        "## Artifacts",
        "",
        "- `experiments/cerny_c3b/bench.py` — strict structural filters and benchmark",
        "- `tests/test_cerny_c3b.py` — regression tests for the stricter novelty definition",
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
    parser.add_argument("--budget", type=int, default=360)
    parser.add_argument("--trials-per-family", type=int, default=3)
    parser.add_argument("--n-values", default="8,9,10")
    parser.add_argument("--seed", type=int, default=20260526)
    parser.add_argument("--findings-path", type=Path, default=DEFAULT_FINDINGS_PATH)
    args = parser.parse_args(argv)

    n_values = [int(part.strip()) for part in args.n_values.split(",") if part.strip()]
    budget = 40 if args.smoke else args.budget
    trials = 2 if args.smoke else args.trials_per_family
    if args.smoke:
        n_values = [6]

    report = run_benchmark(
        n_values=n_values,
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
