"""C3 verification: evolutionary hard-automata generator.

Originating composition: ``findings/compose-20260526-062120.md`` C3.

The claim is that mutation-testing discipline plus multi-objective evolutionary
selection can discover synchronizing finite automata that are harder than
ordinary random or local-mutation baselines. The local test here is deliberately
bounded: for small n, every proposed automaton is scored with an exact
power-automaton BFS oracle, so reset lengths and witnesses are real.

Run:
    uv run python -m experiments.cerny_c3.bench --smoke
    uv run python -m experiments.cerny_c3.bench --write-findings
"""

from __future__ import annotations

import argparse
import itertools
import random
import statistics
import time
from collections import deque
from dataclasses import dataclass
from pathlib import Path

RUN_ID = "20260526-062120"
COMPOSITION_ID = "C3"
DEFAULT_FINDINGS_PATH = Path("findings/c3-cerny-hard-automata-20260526.md")
_SIGNATURE_CACHE: dict[DFA, str] = {}


@dataclass(frozen=True)
class DFA:
    """Complete deterministic finite automaton.

    ``transitions[state][letter]`` returns the next state. The automata here do
    not need accepting states because synchronization depends only on the
    transition monoid.
    """

    transitions: tuple[tuple[int, ...], ...]

    @property
    def n_states(self) -> int:
        return len(self.transitions)

    @property
    def alphabet_size(self) -> int:
        return len(self.transitions[0]) if self.transitions else 0

    def validate(self) -> None:
        n = self.n_states
        if n == 0:
            raise ValueError("DFA must have at least one state")
        k = self.alphabet_size
        if k == 0:
            raise ValueError("DFA must have at least one letter")
        for row in self.transitions:
            if len(row) != k:
                raise ValueError("all transition rows must have the same alphabet size")
            if any(target < 0 or target >= n for target in row):
                raise ValueError("transition target outside state range")


@dataclass(frozen=True)
class CandidateScore:
    label: str
    dfa: DFA
    synchronizing: bool
    reset_length: int | None
    expansions: int
    reset_word: tuple[int, ...] | None
    signature: str
    distance_from_cerny: int

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
    alphabet_size: int
    seed: int
    budget: int
    attempts: int
    synchronizing_scored: int
    unique_scored: int
    best: CandidateScore
    best_novel: CandidateScore | None
    elapsed_s: float


@dataclass(frozen=True)
class BenchmarkReport:
    n_values: list[int]
    alphabet_size: int
    trials_per_family: int
    budget: int
    rows: list[FamilyResult]
    summary: dict[str, float | int | str]
    verdict: str


def cerny_automaton(n_states: int) -> DFA:
    """Return the standard two-letter Černý automaton C_n.

    Letter 0 is an n-cycle. Letter 1 maps state 0 to 1 and fixes every other
    state. The shortest reset word has length ``(n - 1)^2``.
    """

    if n_states < 2:
        raise ValueError("Černý automata require at least two states")
    transitions = []
    for state in range(n_states):
        cycle = (state + 1) % n_states
        merge = 1 if state == 0 else state
        transitions.append((cycle, merge))
    return DFA(tuple(transitions))


def sink_automaton(n_states: int, alphabet_size: int) -> DFA:
    transitions = tuple(tuple(0 for _ in range(alphabet_size)) for _ in range(n_states))
    return DFA(transitions)


def random_dfa(n_states: int, alphabet_size: int, rng: random.Random) -> DFA:
    transitions = tuple(
        tuple(rng.randrange(n_states) for _ in range(alphabet_size)) for _ in range(n_states)
    )
    return DFA(transitions)


def _image_mask(dfa: DFA, mask: int, letter: int) -> int:
    out = 0
    for state, row in enumerate(dfa.transitions):
        if mask & (1 << state):
            out |= 1 << row[letter]
    return out


def is_reset_word(dfa: DFA, word: tuple[int, ...] | list[int]) -> bool:
    dfa.validate()
    mask = (1 << dfa.n_states) - 1
    for letter in word:
        if letter < 0 or letter >= dfa.alphabet_size:
            return False
        mask = _image_mask(dfa, mask, letter)
    return mask != 0 and (mask & (mask - 1)) == 0


def shortest_reset_word(dfa: DFA) -> tuple[int | None, int, tuple[int, ...] | None]:
    """Exact BFS over the subset automaton.

    Returns ``(length, expanded_subset_count, witness_word)``. The expansion
    count is a cheap, deterministic proxy for exact-solver effort.
    """

    dfa.validate()
    full = (1 << dfa.n_states) - 1
    parents: dict[int, tuple[int, int] | None] = {full: None}
    depth = {full: 0}
    queue: deque[int] = deque([full])
    expansions = 0

    while queue:
        mask = queue.popleft()
        expansions += 1
        if mask & (mask - 1) == 0:
            word: list[int] = []
            cursor = mask
            while parents[cursor] is not None:
                prev, letter = parents[cursor]
                word.append(letter)
                cursor = prev
            word.reverse()
            return depth[mask], expansions, tuple(word)

        for letter in range(dfa.alphabet_size):
            nxt = _image_mask(dfa, mask, letter)
            if nxt not in parents:
                parents[nxt] = (mask, letter)
                depth[nxt] = depth[mask] + 1
                queue.append(nxt)

    return None, expansions, None


def transition_distance(left: DFA, right: DFA) -> int:
    if left.n_states != right.n_states or left.alphabet_size != right.alphabet_size:
        return max(left.n_states * left.alphabet_size, right.n_states * right.alphabet_size)
    return sum(
        1
        for state in range(left.n_states)
        for letter in range(left.alphabet_size)
        if left.transitions[state][letter] != right.transitions[state][letter]
    )


def canonical_signature(dfa: DFA) -> str:
    """Cheap state/letter relabeling signature for duplicate suppression.

    This is not a full nauty/traces replacement, but it is invariant to common
    state renumberings by rebuilding labels from every start state and alphabet
    permutation, then choosing the lexicographically smallest encoding.
    """

    dfa.validate()
    letter_orders = itertools.permutations(range(dfa.alphabet_size))
    encodings = []
    for letters in letter_orders:
        for start in range(dfa.n_states):
            mapping: dict[int, int] = {}
            order: list[int] = []

            def add_state(state: int) -> None:
                if state not in mapping:
                    mapping[state] = len(mapping)
                    order.append(state)

            add_state(start)
            cursor = 0
            while cursor < len(order):
                state = order[cursor]
                for letter in letters:
                    add_state(dfa.transitions[state][letter])
                cursor += 1
            for state in range(dfa.n_states):
                add_state(state)
            encoded = []
            for state in order:
                encoded.append(tuple(mapping[dfa.transitions[state][letter]] for letter in letters))
            encodings.append(repr(tuple(encoded)))
    return min(encodings)


def _cached_signature(dfa: DFA) -> str:
    signature = _SIGNATURE_CACHE.get(dfa)
    if signature is None:
        signature = canonical_signature(dfa)
        _SIGNATURE_CACHE[dfa] = signature
    return signature


def score_dfa(
    dfa: DFA,
    label: str = "score",
    reference: DFA | None = None,
) -> CandidateScore:
    length, expansions, word = shortest_reset_word(dfa)
    signature = _cached_signature(dfa)
    if reference is None:
        distance = 0
    elif signature == _cached_signature(reference):
        distance = 0
    else:
        distance = transition_distance(dfa, reference)
    return CandidateScore(
        label=label,
        dfa=dfa,
        synchronizing=length is not None,
        reset_length=length,
        expansions=expansions,
        reset_word=word,
        signature=signature,
        distance_from_cerny=distance,
    )


def _dominates(left: CandidateScore, right: CandidateScore) -> bool:
    l_obj = left.objective_tuple
    r_obj = right.objective_tuple
    return all(a >= b for a, b in zip(l_obj, r_obj)) and any(
        a > b for a, b in zip(l_obj, r_obj)
    )


def pareto_front(rows: list[CandidateScore], limit: int | None = None) -> list[CandidateScore]:
    best_by_signature: dict[str, CandidateScore] = {}
    for row in rows:
        old = best_by_signature.get(row.signature)
        if old is None or row.objective_tuple > old.objective_tuple:
            best_by_signature[row.signature] = row

    unique = list(best_by_signature.values())
    front = [
        row for row in unique if not any(other is not row and _dominates(other, row) for other in unique)
    ]
    front.sort(key=lambda row: row.objective_tuple, reverse=True)
    return front[:limit] if limit is not None else front


def _best(rows: list[CandidateScore]) -> CandidateScore:
    return max(rows, key=lambda row: row.objective_tuple)


def _best_novel(rows: list[CandidateScore]) -> CandidateScore | None:
    novel = [row for row in rows if row.synchronizing and row.distance_from_cerny >= 2]
    return _best(novel) if novel else None


def _mutate_transition(dfa: DFA, rng: random.Random) -> DFA:
    transitions = [list(row) for row in dfa.transitions]
    state = rng.randrange(dfa.n_states)
    letter = rng.randrange(dfa.alphabet_size)
    old = transitions[state][letter]
    choices = [target for target in range(dfa.n_states) if target != old]
    transitions[state][letter] = rng.choice(choices)
    return DFA(tuple(tuple(row) for row in transitions))


def _mutate_merge_bias(dfa: DFA, rng: random.Random) -> DFA:
    transitions = [list(row) for row in dfa.transitions]
    letter = rng.randrange(dfa.alphabet_size)
    source = rng.randrange(dfa.n_states)
    donor = rng.randrange(dfa.n_states)
    transitions[source][letter] = transitions[donor][letter]
    return DFA(tuple(tuple(row) for row in transitions))


def _mutate_two_edges(dfa: DFA, rng: random.Random) -> DFA:
    out = dfa
    for _ in range(2):
        out = _mutate_transition(out, rng)
    return out


def mutate_dfa(dfa: DFA, rng: random.Random) -> DFA:
    operator = rng.random()
    if operator < 0.55:
        return _mutate_transition(dfa, rng)
    if operator < 0.80:
        return _mutate_merge_bias(dfa, rng)
    return _mutate_two_edges(dfa, rng)


def _random_sync_score(
    n_states: int,
    alphabet_size: int,
    rng: random.Random,
    reference: DFA,
    label: str,
    max_attempts: int = 200,
) -> tuple[CandidateScore, int]:
    for attempt in range(1, max_attempts + 1):
        score = score_dfa(random_dfa(n_states, alphabet_size, rng), label=label, reference=reference)
        if score.synchronizing:
            return score, attempt
    fallback = score_dfa(sink_automaton(n_states, alphabet_size), label=label, reference=reference)
    return fallback, max_attempts


def run_cerny_seed(n_states: int, alphabet_size: int, seed: int, budget: int) -> FamilyResult:
    start = time.perf_counter()
    if alphabet_size != 2:
        best = score_dfa(sink_automaton(n_states, alphabet_size), label="cerny_seed")
    else:
        reference = cerny_automaton(n_states)
        best = score_dfa(reference, label="cerny_seed", reference=reference)
    return FamilyResult(
        family="cerny_seed",
        n_states=n_states,
        alphabet_size=alphabet_size,
        seed=seed,
        budget=budget,
        attempts=1,
        synchronizing_scored=1 if best.synchronizing else 0,
        unique_scored=1,
        best=best,
        best_novel=None,
        elapsed_s=time.perf_counter() - start,
    )


def run_random_sampling(
    n_states: int,
    alphabet_size: int,
    budget: int,
    seed: int,
) -> FamilyResult:
    rng = random.Random(seed)
    reference = cerny_automaton(n_states) if alphabet_size == 2 else sink_automaton(n_states, alphabet_size)
    start = time.perf_counter()
    rows: list[CandidateScore] = []
    signatures: set[str] = set()
    attempts = 0
    for idx in range(budget):
        attempts += 1
        score = score_dfa(
            random_dfa(n_states, alphabet_size, rng),
            label=f"random_{idx}",
            reference=reference,
        )
        if score.synchronizing:
            rows.append(score)
            signatures.add(score.signature)
    if not rows:
        fallback, extra = _random_sync_score(
            n_states, alphabet_size, rng, reference, label="random_fallback"
        )
        attempts += extra
        rows.append(fallback)
        signatures.add(fallback.signature)
    return FamilyResult(
        family="random_sampling",
        n_states=n_states,
        alphabet_size=alphabet_size,
        seed=seed,
        budget=budget,
        attempts=attempts,
        synchronizing_scored=len(rows),
        unique_scored=len(signatures),
        best=_best(rows),
        best_novel=_best_novel(rows),
        elapsed_s=time.perf_counter() - start,
    )


def run_hill_climb(
    n_states: int,
    alphabet_size: int,
    budget: int,
    seed: int,
) -> FamilyResult:
    rng = random.Random(seed)
    reference = cerny_automaton(n_states) if alphabet_size == 2 else sink_automaton(n_states, alphabet_size)
    start = time.perf_counter()
    current, attempts = _random_sync_score(
        n_states, alphabet_size, rng, reference, label="hill_start"
    )
    rows = [current]
    signatures = {current.signature}
    for idx in range(budget):
        attempts += 1
        score = score_dfa(mutate_dfa(current.dfa, rng), label=f"hill_{idx}", reference=reference)
        if not score.synchronizing:
            continue
        rows.append(score)
        signatures.add(score.signature)
        if score.objective_tuple >= current.objective_tuple:
            current = score
    return FamilyResult(
        family="hill_climb",
        n_states=n_states,
        alphabet_size=alphabet_size,
        seed=seed,
        budget=budget,
        attempts=attempts,
        synchronizing_scored=len(rows),
        unique_scored=len(signatures),
        best=_best(rows),
        best_novel=_best_novel(rows),
        elapsed_s=time.perf_counter() - start,
    )


def run_pareto_evolution(
    n_states: int,
    alphabet_size: int,
    budget: int,
    seed: int,
    population_limit: int = 24,
) -> FamilyResult:
    rng = random.Random(seed)
    reference = cerny_automaton(n_states) if alphabet_size == 2 else sink_automaton(n_states, alphabet_size)
    start = time.perf_counter()
    attempts = 0
    rows: list[CandidateScore] = []
    signatures: set[str] = set()

    if alphabet_size == 2:
        cerny = score_dfa(reference, label="evo_cerny_seed", reference=reference)
        rows.append(cerny)
        signatures.add(cerny.signature)
    while len(rows) < max(4, min(8, population_limit // 2)):
        seed_score, extra = _random_sync_score(
            n_states, alphabet_size, rng, reference, label=f"evo_random_seed_{len(rows)}"
        )
        attempts += extra
        if seed_score.signature not in signatures:
            rows.append(seed_score)
            signatures.add(seed_score.signature)

    population = pareto_front(rows, limit=population_limit)
    for idx in range(budget):
        attempts += 1
        parent = rng.choice(population if population else rows)
        score = score_dfa(mutate_dfa(parent.dfa, rng), label=f"evo_{idx}", reference=reference)
        if not score.synchronizing or score.signature in signatures:
            continue
        rows.append(score)
        signatures.add(score.signature)
        population = pareto_front(population + [score], limit=population_limit)

    return FamilyResult(
        family="pareto_evolution",
        n_states=n_states,
        alphabet_size=alphabet_size,
        seed=seed,
        budget=budget,
        attempts=attempts,
        synchronizing_scored=len(rows),
        unique_scored=len(signatures),
        best=_best(rows),
        best_novel=_best_novel(rows),
        elapsed_s=time.perf_counter() - start,
    )


def run_benchmark(
    n_values: list[int] | None = None,
    alphabet_size: int = 2,
    trials_per_family: int = 4,
    budget: int = 220,
    seed: int = 20260526,
) -> BenchmarkReport:
    n_values = n_values or [6, 7, 8]
    rows: list[FamilyResult] = []
    for n_states in n_values:
        for trial in range(trials_per_family):
            base_seed = seed + n_states * 10_000 + trial * 101
            rows.append(run_cerny_seed(n_states, alphabet_size, base_seed, budget))
            rows.append(run_random_sampling(n_states, alphabet_size, budget, base_seed + 1))
            rows.append(run_hill_climb(n_states, alphabet_size, budget, base_seed + 2))
            rows.append(run_pareto_evolution(n_states, alphabet_size, budget, base_seed + 3))

    summary = summarize(rows)
    return BenchmarkReport(
        n_values=n_values,
        alphabet_size=alphabet_size,
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

    def best_novel_length(family: str) -> int:
        novel_lengths = [
            int(row.best_novel.reset_length)
            for row in by_family.get(family, [])
            if row.best_novel and row.best_novel.reset_length is not None
        ]
        return max(novel_lengths) if novel_lengths else -1

    random_best = best_length("random_sampling")
    hill_best = best_length("hill_climb")
    cerny_best = best_length("cerny_seed")
    evo_best = best_length("pareto_evolution")
    evo_novel_best = best_novel_length("pareto_evolution")
    non_cerny_baseline_best = max(random_best, hill_best)
    sync_yields = [
        row.synchronizing_scored / max(1, row.attempts)
        for row in rows
        if row.family in {"random_sampling", "hill_climb", "pareto_evolution"}
    ]
    best_family_vs_random_delta = evo_best - random_best
    novel_vs_baseline_delta = evo_novel_best - non_cerny_baseline_best
    if evo_novel_best >= non_cerny_baseline_best + 1:
        verdict = "partial-positive"
    elif evo_best >= non_cerny_baseline_best + 1 and evo_best == cerny_best:
        verdict = "partial-negative"
    else:
        verdict = "clean-negative"
    return {
        "random_best_reset_length": random_best,
        "hill_best_reset_length": hill_best,
        "cerny_best_reset_length": cerny_best,
        "evolution_best_reset_length": evo_best,
        "evolution_best_novel_reset_length": evo_novel_best,
        "best_family_vs_random_delta": best_family_vs_random_delta,
        "novel_vs_non_cerny_baseline_delta": novel_vs_baseline_delta,
        "mean_sync_yield": statistics.mean(sync_yields) if sync_yields else 0.0,
        "total_rows": len(rows),
        "verdict": verdict,
    }


def _family_summary_rows(report: BenchmarkReport) -> list[str]:
    lines = [
        "| family | best reset length | best novel reset length | mean sync yield | mean elapsed s |",
        "|---|---:|---:|---:|---:|",
    ]
    families = ["cerny_seed", "random_sampling", "hill_climb", "pareto_evolution"]
    for family in families:
        rows = [row for row in report.rows if row.family == family]
        best = max(int(row.best.reset_length or -1) for row in rows)
        novel = [
            int(row.best_novel.reset_length)
            for row in rows
            if row.best_novel and row.best_novel.reset_length is not None
        ]
        mean_yield = statistics.mean(row.synchronizing_scored / max(1, row.attempts) for row in rows)
        mean_elapsed = statistics.mean(row.elapsed_s for row in rows)
        lines.append(
            f"| {family} | {best} | {max(novel) if novel else 'n/a'} | "
            f"{mean_yield:.3f} | {mean_elapsed:.3f} |"
        )
    return lines


def _best_novel_result(report: BenchmarkReport) -> tuple[FamilyResult, CandidateScore] | None:
    candidates = [
        (row, row.best_novel)
        for row in report.rows
        if row.best_novel is not None and row.best_novel.reset_length is not None
    ]
    if not candidates:
        return None
    return max(candidates, key=lambda pair: pair[1].objective_tuple)


def _format_transition_table(dfa: DFA) -> str:
    lines = ["| state | letter 0 | letter 1 |", "|---:|---:|---:|"]
    for state, row in enumerate(dfa.transitions):
        if len(row) == 2:
            lines.append(f"| {state} | {row[0]} | {row[1]} |")
        else:
            targets = " | ".join(str(target) for target in row)
            lines.append(f"| {state} | {targets} | |")
    return "\n".join(lines)


def _word_string(word: tuple[int, ...] | None, max_chars: int = 96) -> str:
    if word is None:
        return "n/a"
    out = "".join(str(letter) for letter in word)
    if len(out) <= max_chars:
        return out
    return out[: max_chars - 3] + "..."


def format_findings(report: BenchmarkReport) -> str:
    summary = report.summary
    verdict = report.verdict
    best_novel = _best_novel_result(report)
    if verdict == "partial-positive":
        verdict_text = (
            "The evolutionary generator found a non-Cerny candidate that beat the best "
            "random or hill-climb baseline by at least one reset step."
        )
    elif verdict == "partial-negative":
        verdict_text = (
            "The evolutionary loop's best score came from the known Černý seed rather "
            "than a new hard automaton, so it did not yet demonstrate discovery."
        )
    else:
        verdict_text = (
            "The evolutionary generator did not beat random or hill-climb baselines with "
            "a novel synchronizing automaton under this local budget."
        )

    lines = [
        "# Černý C3 verification — evolutionary hard-automata generator",
        "",
        f"- Originating run: `{RUN_ID}`",
        f"- Composition: `{COMPOSITION_ID}`",
        f"- Outcome: **{verdict}**",
        f"- n values: `{report.n_values}`",
        f"- alphabet size: `{report.alphabet_size}`",
        f"- trials per family: `{report.trials_per_family}`",
        f"- budget per stochastic family/trial: `{report.budget}` exact evaluations",
        "",
        "## Claim Tested",
        "",
        (
            "C3 claimed that mutation-style generation plus multi-objective evolutionary "
            "selection could produce harder synchronizing DFAs than ordinary random "
            "sampling or simple local mutation, when every candidate is scored with an "
            "exact reset-word oracle."
        ),
        "",
        "## Result",
        "",
        verdict_text,
        "",
        *_family_summary_rows(report),
        "",
        "## Key Numbers",
        "",
        f"- Best random reset length: `{summary['random_best_reset_length']}`",
        f"- Best hill-climb reset length: `{summary['hill_best_reset_length']}`",
        f"- Best known Černý-family reset length in this benchmark: `{summary['cerny_best_reset_length']}`",
        f"- Best evolutionary reset length including seeds: `{summary['evolution_best_reset_length']}`",
        (
            "- Best evolutionary novel/non-Černý reset length: "
            f"`{summary['evolution_best_novel_reset_length']}`"
        ),
        (
            "- Novel evolutionary delta vs best non-Černý baseline: "
            f"`{summary['novel_vs_non_cerny_baseline_delta']}`"
        ),
        f"- Mean synchronizing yield: `{summary['mean_sync_yield']:.3f}`",
        "",
        "## Best Novel Candidate",
        "",
    ]
    if best_novel is None:
        lines.extend(
            [
                "No non-Černý synchronizing candidate survived the novelty filter in this run.",
                "",
            ]
        )
    else:
        row, candidate = best_novel
        lines.extend(
            [
                f"- Family: `{row.family}`",
                f"- n states: `{row.n_states}`",
                f"- seed: `{row.seed}`",
                f"- exact reset length: `{candidate.reset_length}`",
                f"- exact BFS expanded subsets: `{candidate.expansions}`",
                f"- transition edits from formula Černý table: `{candidate.distance_from_cerny}`",
                f"- reset witness: `{_word_string(candidate.reset_word)}`",
                "",
                "The transition table uses rows as states and columns as input letters:",
                "",
                _format_transition_table(candidate.dfa),
                "",
            ]
        )
        if candidate.reset_length == (row.n_states - 2) ** 2:
            lines.extend(
                [
                    "Caveat: this length equals the Černý threshold for one fewer state. "
                    "The transition table should be treated as a useful hard test case, "
                    "but it may be a lower-dimensional Černý-like core with an extra "
                    "state attached rather than a genuinely new extremal construction.",
                    "",
                ]
            )
    lines.extend(
        [
        "## L3 Attacks Checked",
        "",
        "| L3 attack | Verdict from this run |",
        "|---|---|",
        (
            "| Trivial recombination | Tested directly against random sampling and "
            "single-mutation hill climbing. |"
        ),
        (
            "| Exact scoring bottleneck | Exact BFS was cheap at this scale, but that "
            "does not remove the scaling risk for n=10-12. |"
        ),
        (
            "| Mutation operators underspecified | The harness used repair-by-rejection, "
            "merge-biased edits, duplicate suppression, and a novelty objective. |"
        ),
        (
            "| Risk of rediscovering Černý | Tracked separately with distance from the "
            "formula-generated Černý automaton and a novel-candidate metric. |"
        ),
        "",
        "## Verdict",
        "",
        (
            "This is a bounded local test, not a proof about the full conjecture. A true "
            "discovery signal would be a reproducible non-isomorphic automaton family "
            "with unusually high exact reset thresholds, or a generator that reliably "
            "beats random/local baselines without leaning on the known Černý seed."
        ),
        "",
        "## Artifacts",
        "",
        "- `experiments/cerny_c3/bench.py` — exact oracle, generators, benchmark, findings writer",
        "- `tests/test_cerny_c3.py` — oracle and benchmark regression tests",
        ]
    )
    return "\n".join(lines) + "\n"


def write_findings(report: BenchmarkReport, path: Path = DEFAULT_FINDINGS_PATH) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(format_findings(report), encoding="utf-8")
    return path


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--smoke", action="store_true", help="Run a tiny fast benchmark")
    parser.add_argument("--write-findings", action="store_true")
    parser.add_argument("--budget", type=int, default=220)
    parser.add_argument("--trials-per-family", type=int, default=4)
    parser.add_argument("--n-values", default="6,7,8")
    parser.add_argument("--seed", type=int, default=20260526)
    parser.add_argument("--findings-path", type=Path, default=DEFAULT_FINDINGS_PATH)
    args = parser.parse_args(argv)

    n_values = [int(part.strip()) for part in args.n_values.split(",") if part.strip()]
    budget = 20 if args.smoke else args.budget
    trials = 2 if args.smoke else args.trials_per_family
    if args.smoke:
        n_values = [5]

    report = run_benchmark(
        n_values=n_values,
        alphabet_size=2,
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
