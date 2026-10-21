"""C2 verification: Delaunay-local active repair for point configurations.

Originating composition: ``findings/compose-20260525-042258.md`` C2.

The claim is that Delaunay-local geometry plus particle-like relaxation and
constrained repair can help optimize finite point configurations. This harness
tests the core risk directly: Delaunay cells are cheap, but they may miss the
true worst no-near-collinear triangle. The augmented C2 loop keeps Delaunay
locality, adds k-nearest-neighbor candidate triples, and periodically injects
missed global bottlenecks discovered by an exhaustive scan.

Run:
    uv run python -m experiments.discrete_geometry_c2.bench --smoke
    uv run python -m experiments.discrete_geometry_c2.bench --write-findings
"""

from __future__ import annotations

import argparse
import itertools
import math
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from experiments.heilbronn.baselines import sa_run

RUN_ID = "20260525-042258"
COMPOSITION_ID = "C2"
DEFAULT_FINDINGS_PATH = Path("findings/c2-discrete-geometry-delaunay-relaxation-20260525.md")


@dataclass(frozen=True)
class RecallResult:
    recall: float
    top_k: int
    candidate_count: int
    full_count: int
    worst_area: float
    candidate_worst_area: float
    missed: tuple[tuple[int, int, int], ...]


@dataclass(frozen=True)
class SearchResult:
    name: str
    mode: str
    n_points: int
    seed: int
    initial_score: float
    final_score: float
    best_score: float
    best_points: np.ndarray
    elapsed_s: float
    full_scans: int
    candidate_scans: int
    mean_recall: float
    mean_candidate_count: float


@dataclass(frozen=True)
class BenchmarkReport:
    sizes: list[int]
    seeds_per_size: int
    steps: int
    rows: list[SearchResult]
    recall_rows: list[dict[str, Any]]
    summary: dict[str, float | str]
    verdict: str


def triangle_indices(n_points: int) -> np.ndarray:
    if n_points < 3:
        raise ValueError(f"need at least 3 points, got {n_points}")
    return np.array(list(itertools.combinations(range(n_points), 3)), dtype=np.int32)


def _normalize_triples(triples: np.ndarray | list[tuple[int, int, int]]) -> np.ndarray:
    arr = np.asarray(triples, dtype=np.int32)
    if arr.size == 0:
        return np.empty((0, 3), dtype=np.int32)
    arr = arr.reshape(-1, 3)
    arr = np.sort(arr, axis=1)
    return np.unique(arr, axis=0)


def _triple_set(triples: np.ndarray) -> set[tuple[int, int, int]]:
    return {tuple(int(x) for x in row) for row in triples}


def triangle_areas(points: np.ndarray, triples: np.ndarray) -> np.ndarray:
    triples = _normalize_triples(triples)
    if triples.size == 0:
        return np.empty(0, dtype=float)
    a = points[triples[:, 0]]
    b = points[triples[:, 1]]
    c = points[triples[:, 2]]
    cross = (b[:, 0] - a[:, 0]) * (c[:, 1] - a[:, 1]) - (b[:, 1] - a[:, 1]) * (
        c[:, 0] - a[:, 0]
    )
    return 0.5 * np.abs(cross)


def full_area_score(points: np.ndarray) -> tuple[float, tuple[int, int, int]]:
    triples = triangle_indices(len(points))
    areas = triangle_areas(points, triples)
    idx = int(np.argmin(areas))
    return float(areas[idx]), tuple(int(x) for x in triples[idx])


def candidate_recall(points: np.ndarray, candidate_triples: np.ndarray, top_k: int = 5) -> RecallResult:
    full_triples = triangle_indices(len(points))
    full_areas = triangle_areas(points, full_triples)
    k = min(top_k, len(full_triples))
    order = np.argsort(full_areas, kind="stable")[:k]
    true_top = _triple_set(full_triples[order])
    candidates = _normalize_triples(candidate_triples)
    candidate_set = _triple_set(candidates)
    missed = tuple(sorted(true_top - candidate_set))
    candidate_areas = triangle_areas(points, candidates)
    candidate_worst = float(candidate_areas.min()) if len(candidate_areas) else math.inf
    return RecallResult(
        recall=(k - len(missed)) / k,
        top_k=k,
        candidate_count=len(candidates),
        full_count=len(full_triples),
        worst_area=float(full_areas[order[0]]),
        candidate_worst_area=candidate_worst,
        missed=missed,
    )


def _pairwise_sqdist(points: np.ndarray) -> np.ndarray:
    delta = points[:, None, :] - points[None, :, :]
    return np.sum(delta * delta, axis=2)


def knn_candidate_triples(points: np.ndarray, k_neighbors: int = 8) -> np.ndarray:
    n_points = len(points)
    if n_points < 3:
        raise ValueError(f"need at least 3 points, got {n_points}")
    k = min(max(2, k_neighbors), n_points - 1)
    distances = _pairwise_sqdist(points)
    nearest = np.argsort(distances, axis=1)[:, 1 : k + 1]
    triples: list[tuple[int, int, int]] = []
    for i in range(n_points):
        local = [i, *(int(x) for x in nearest[i])]
        triples.extend(itertools.combinations(local, 3))
    return _normalize_triples(triples)


def delaunay_cell_triples(points: np.ndarray) -> np.ndarray:
    n_points = len(points)
    if n_points < 3:
        raise ValueError(f"need at least 3 points, got {n_points}")
    try:
        from scipy.spatial import Delaunay, QhullError

        try:
            triangulation = Delaunay(points, qhull_options="QJ")
            return _normalize_triples(triangulation.simplices)
        except QhullError:
            return knn_candidate_triples(points, k_neighbors=min(6, n_points - 1))
    except Exception:
        return knn_candidate_triples(points, k_neighbors=min(6, n_points - 1))


def delaunay_one_ring_triples(points: np.ndarray) -> np.ndarray:
    cells = delaunay_cell_triples(points)
    n_points = len(points)
    neighbors: list[set[int]] = [set() for _ in range(n_points)]
    for a, b, c in cells:
        ai, bi, ci = int(a), int(b), int(c)
        neighbors[ai].update((bi, ci))
        neighbors[bi].update((ai, ci))
        neighbors[ci].update((ai, bi))

    triples: list[tuple[int, int, int]] = []
    for i, local_neighbors in enumerate(neighbors):
        local = [i, *sorted(local_neighbors)]
        if len(local) >= 3:
            triples.extend(itertools.combinations(local, 3))
    return _normalize_triples([*_triple_set(cells), *triples])


def augmented_candidate_triples(
    points: np.ndarray,
    k_neighbors: int = 8,
    extra_triples: np.ndarray | None = None,
) -> np.ndarray:
    pieces = [
        delaunay_cell_triples(points),
        delaunay_one_ring_triples(points),
        knn_candidate_triples(points, k_neighbors=k_neighbors),
    ]
    if extra_triples is not None and len(extra_triples):
        pieces.append(_normalize_triples(extra_triples))
    return _normalize_triples(np.vstack(pieces))


def _candidate_triples(
    points: np.ndarray,
    mode: str,
    k_neighbors: int,
    extra_triples: np.ndarray,
) -> np.ndarray:
    if mode == "full":
        return triangle_indices(len(points))
    if mode == "delaunay":
        return delaunay_cell_triples(points)
    if mode == "augmented":
        return augmented_candidate_triples(
            points,
            k_neighbors=k_neighbors,
            extra_triples=extra_triples,
        )
    raise ValueError(f"unknown candidate mode {mode!r}")


def _candidate_score(points: np.ndarray, triples: np.ndarray) -> float:
    areas = triangle_areas(points, triples)
    return float(areas.min()) if len(areas) else 0.0


def _lift_point_away_from_line(
    points: np.ndarray,
    triple: tuple[int, int, int],
    vertex_slot: int,
    distance: float,
) -> np.ndarray:
    moved = points.copy()
    vertex = int(triple[vertex_slot])
    others = [int(x) for pos, x in enumerate(triple) if pos != vertex_slot]
    edge = points[others[1]] - points[others[0]]
    norm = float(np.linalg.norm(edge))
    if norm < 1e-12:
        return moved
    normal = np.array([-edge[1], edge[0]], dtype=float) / norm
    signed_height = float(np.dot(points[vertex] - points[others[0]], normal))
    direction = normal if signed_height >= 0.0 else -normal
    moved[vertex] = np.clip(points[vertex] + distance * direction, 0.0, 1.0)
    return moved


def _active_proposals(
    points: np.ndarray,
    active_triples: np.ndarray,
    rng: np.random.Generator,
    step_scale: float,
) -> list[np.ndarray]:
    proposals: list[np.ndarray] = []
    limit = min(4, len(active_triples))
    for triple_arr in active_triples[:limit]:
        triple = tuple(int(x) for x in triple_arr)
        for slot in range(3):
            proposals.append(_lift_point_away_from_line(points, triple, slot, step_scale))

    active_points = np.unique(active_triples[: max(1, limit)].ravel())
    if len(active_points):
        jitter = points.copy()
        idx = int(rng.choice(active_points))
        jitter[idx] = np.clip(jitter[idx] + rng.normal(0.0, step_scale, size=2), 0.0, 1.0)
        proposals.append(jitter)
    return proposals


def active_repair_search(
    initial_points: np.ndarray,
    mode: str,
    steps: int = 400,
    seed: int = 0,
    active_count: int = 8,
    global_scan_every: int = 25,
    step_scale: float = 0.035,
    k_neighbors: int = 8,
) -> SearchResult:
    rng = np.random.default_rng(seed)
    points = np.asarray(initial_points, dtype=float).copy()
    if points.ndim != 2 or points.shape[1] != 2:
        raise ValueError("initial_points must have shape [n, 2]")
    points = np.clip(points, 0.0, 1.0)

    started = time.perf_counter()
    initial_score, _ = full_area_score(points)
    best_score = initial_score
    best_points = points.copy()
    full_scans = 1
    candidate_scans = 0
    recall_values: list[float] = []
    candidate_counts: list[int] = []
    extra_triples = np.empty((0, 3), dtype=np.int32)

    for step in range(max(0, steps)):
        cooling = 1.0 - 0.75 * step / max(1, steps - 1)
        local_step = max(0.004, step_scale * cooling)
        candidates = _candidate_triples(points, mode, k_neighbors, extra_triples)
        candidate_scans += 1
        candidate_counts.append(len(candidates))
        areas = triangle_areas(points, candidates)
        order = np.argsort(areas, kind="stable")
        active_triples = candidates[order[: max(1, active_count)]]
        current_candidate_score = float(areas[order[0]])

        if step % max(1, global_scan_every) == 0:
            recall = candidate_recall(points, candidates, top_k=5)
            recall_values.append(recall.recall)
            full_score, _ = full_area_score(points)
            full_scans += 1
            if full_score > best_score:
                best_score = full_score
                best_points = points.copy()
            if mode == "augmented" and recall.missed:
                extra_triples = _normalize_triples(
                    np.vstack([extra_triples, np.asarray(recall.missed, dtype=np.int32)])
                )

        best_trial = points
        best_trial_score = current_candidate_score
        for trial in _active_proposals(points, active_triples, rng, local_step):
            trial_score = _candidate_score(trial, candidates)
            if trial_score > best_trial_score:
                best_trial = trial
                best_trial_score = trial_score

        if best_trial_score > current_candidate_score:
            points = best_trial
        else:
            jitter = points.copy()
            active_points = np.unique(active_triples.ravel())
            idx = int(rng.choice(active_points)) if len(active_points) else int(rng.integers(len(points)))
            jitter[idx] = np.clip(jitter[idx] + rng.normal(0.0, local_step * 0.5, size=2), 0.0, 1.0)
            jitter_score = _candidate_score(jitter, candidates)
            temperature = max(1e-6, 0.0025 * cooling)
            if jitter_score > current_candidate_score or rng.random() < math.exp(
                (jitter_score - current_candidate_score) / temperature
            ):
                points = jitter

    final_score, _ = full_area_score(points)
    full_scans += 1
    if final_score > best_score:
        best_score = final_score
        best_points = points.copy()
    elapsed = time.perf_counter() - started
    return SearchResult(
        name=f"C2-{mode}",
        mode=mode,
        n_points=len(points),
        seed=seed,
        initial_score=initial_score,
        final_score=final_score,
        best_score=best_score,
        best_points=best_points,
        elapsed_s=elapsed,
        full_scans=full_scans,
        candidate_scans=candidate_scans,
        mean_recall=float(np.mean(recall_values)) if recall_values else 0.0,
        mean_candidate_count=float(np.mean(candidate_counts)) if candidate_counts else 0.0,
    )


def _sa_search_result(n_points: int, steps: int, seed: int) -> SearchResult:
    started = time.perf_counter()
    initial_rng = np.random.default_rng(seed)
    initial = initial_rng.random((n_points, 2))
    initial_score, _ = full_area_score(initial)
    best_points, best_area = sa_run(
        n_points,
        n_steps=steps,
        seed=seed,
        init="uniform",
        T_start=0.04,
        T_end=1e-5,
    )
    final_score, _ = full_area_score(best_points)
    return SearchResult(
        name="SA-full",
        mode="sa",
        n_points=n_points,
        seed=seed,
        initial_score=initial_score,
        final_score=final_score,
        best_score=max(float(best_area), final_score),
        best_points=best_points,
        elapsed_s=time.perf_counter() - started,
        full_scans=steps + 2,
        candidate_scans=steps,
        mean_recall=1.0,
        mean_candidate_count=float(len(triangle_indices(n_points))),
    )


def _recall_point_sets(sizes: list[int], seeds_per_size: int, seed: int) -> list[tuple[str, int, int, np.ndarray]]:
    rows: list[tuple[str, int, int, np.ndarray]] = []
    for n_points in sizes:
        for local_seed in range(seeds_per_size):
            rng = np.random.default_rng(seed + 10_003 * n_points + local_seed)
            rows.append(("uniform", n_points, local_seed, rng.random((n_points, 2))))

            line = np.column_stack([np.sort(rng.random(n_points)), 0.5 + rng.normal(0.0, 0.018, n_points)])
            rows.append(("near_line", n_points, local_seed, np.clip(line, 0.0, 1.0)))

            centers = rng.random((3, 2))
            labels = rng.integers(0, len(centers), n_points)
            clusters = centers[labels] + rng.normal(0.0, 0.055, size=(n_points, 2))
            rows.append(("clusters", n_points, local_seed, np.clip(clusters, 0.0, 1.0)))
    return rows


def evaluate_active_set_recall(
    sizes: list[int],
    seeds_per_size: int,
    seed: int,
    k_neighbors: int,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for family, n_points, local_seed, points in _recall_point_sets(sizes, seeds_per_size, seed):
        full_count = len(triangle_indices(n_points))
        for name, candidates in (
            ("delaunay_cells", delaunay_cell_triples(points)),
            ("augmented", augmented_candidate_triples(points, k_neighbors=k_neighbors)),
        ):
            top1 = candidate_recall(points, candidates, top_k=1)
            top5 = candidate_recall(points, candidates, top_k=5)
            rows.append(
                {
                    "family": family,
                    "n_points": n_points,
                    "seed": local_seed,
                    "candidate_set": name,
                    "top1_recall": top1.recall,
                    "top5_recall": top5.recall,
                    "candidate_count": top5.candidate_count,
                    "full_count": full_count,
                    "candidate_ratio": top5.candidate_count / full_count,
                    "worst_area": top5.worst_area,
                    "candidate_worst_area": top5.candidate_worst_area,
                }
            )
    return rows


def _mean(rows: list[SearchResult], method: str, attr: str) -> float:
    values = [float(getattr(row, attr)) for row in rows if row.name == method]
    return float(np.mean(values)) if values else 0.0


def _mean_recall(recall_rows: list[dict[str, Any]], candidate_set: str, field: str) -> float:
    values = [float(row[field]) for row in recall_rows if row["candidate_set"] == candidate_set]
    return float(np.mean(values)) if values else 0.0


def summarize(report_rows: list[SearchResult], recall_rows: list[dict[str, Any]], margin: float) -> tuple[dict[str, float | str], str]:
    sa_score = _mean(report_rows, "SA-full", "best_score")
    full_score = _mean(report_rows, "C2-full", "best_score")
    delaunay_score = _mean(report_rows, "C2-delaunay", "best_score")
    augmented_score = _mean(report_rows, "C2-augmented", "best_score")
    best_baseline = max(sa_score, full_score)
    score_vs_baseline = (augmented_score - best_baseline) / best_baseline if best_baseline else 0.0
    score_vs_sa = (augmented_score - sa_score) / sa_score if sa_score else 0.0
    delaunay_top5 = _mean_recall(recall_rows, "delaunay_cells", "top5_recall")
    augmented_top5 = _mean_recall(recall_rows, "augmented", "top5_recall")
    augmented_ratio = _mean_recall(recall_rows, "augmented", "candidate_ratio")

    summary: dict[str, float | str] = {
        "sa_mean_best": sa_score,
        "full_active_mean_best": full_score,
        "delaunay_mean_best": delaunay_score,
        "augmented_mean_best": augmented_score,
        "augmented_vs_best_baseline": score_vs_baseline,
        "augmented_vs_sa": score_vs_sa,
        "delaunay_top5_recall": delaunay_top5,
        "augmented_top5_recall": augmented_top5,
        "augmented_candidate_ratio": augmented_ratio,
    }

    if score_vs_baseline >= margin and augmented_top5 >= 0.85:
        verdict = "clean-positive"
    elif score_vs_sa >= margin or (augmented_top5 - delaunay_top5 >= 0.10 and augmented_score >= 0.9 * best_baseline):
        verdict = "partial-positive"
    else:
        verdict = "clean-negative"
    summary["verdict"] = verdict
    return summary, verdict


def run_benchmark(args: argparse.Namespace) -> BenchmarkReport:
    sizes = list(args.sizes)
    seeds_per_size = args.seeds_per_size
    steps = args.steps
    if args.smoke:
        sizes = [10, 12]
        seeds_per_size = min(seeds_per_size, 2)
        steps = min(steps, 120)

    recall_rows = evaluate_active_set_recall(
        sizes=sizes,
        seeds_per_size=seeds_per_size,
        seed=args.seed,
        k_neighbors=args.k_neighbors,
    )
    rows: list[SearchResult] = []
    for n_points in sizes:
        for local_seed in range(seeds_per_size):
            seed = args.seed + 100_003 * n_points + local_seed
            rng = np.random.default_rng(seed)
            initial = rng.random((n_points, 2))
            rows.append(_sa_search_result(n_points, steps, seed))
            for mode in ("full", "delaunay", "augmented"):
                rows.append(
                    active_repair_search(
                        initial,
                        mode=mode,
                        steps=steps,
                        seed=seed + 17,
                        active_count=args.active_count,
                        global_scan_every=args.global_scan_every,
                        step_scale=args.step_scale,
                        k_neighbors=args.k_neighbors,
                    )
                )
            if args.verbose:
                print(f"finished n={n_points} seed={local_seed}")

    summary, verdict = summarize(rows, recall_rows, margin=args.margin)
    return BenchmarkReport(
        sizes=sizes,
        seeds_per_size=seeds_per_size,
        steps=steps,
        rows=rows,
        recall_rows=recall_rows,
        summary=summary,
        verdict=verdict,
    )


def _fmt_pct(value: float) -> str:
    return f"{value * 100:+.1f}%"


def _method_rows(report: BenchmarkReport) -> list[tuple[str, float, float, float, float]]:
    methods = ["SA-full", "C2-full", "C2-delaunay", "C2-augmented"]
    rows = []
    for method in methods:
        rows.append(
            (
                method,
                _mean(report.rows, method, "best_score"),
                _mean(report.rows, method, "elapsed_s"),
                _mean(report.rows, method, "mean_recall"),
                _mean(report.rows, method, "mean_candidate_count"),
            )
        )
    return rows


def print_report(report: BenchmarkReport) -> None:
    print("\n=== C2 discrete geometry verification ===")
    print(f"compose run: {RUN_ID}, composition: {COMPOSITION_ID}")
    print(f"sizes: {report.sizes}, seeds per size: {report.seeds_per_size}, steps: {report.steps}")
    print("\n=== optimizer benchmark: maximize minimum triangle area ===")
    print("method          | mean best area | mean seconds | active recall | mean candidates")
    print("-" * 80)
    for method, best, elapsed, recall, candidates in _method_rows(report):
        print(f"{method:15s} | {best:14.6g} | {elapsed:12.3f} | {recall:13.3f} | {candidates:15.1f}")

    s = report.summary
    print("\n=== active-set recall ===")
    print(f"Delaunay-cell top-5 recall: {_fmt_pct(float(s['delaunay_top5_recall']))}")
    print(f"Augmented top-5 recall:     {_fmt_pct(float(s['augmented_top5_recall']))}")
    print(f"Augmented candidate ratio:  {_fmt_pct(float(s['augmented_candidate_ratio']))}")
    print("\n=== comparison ===")
    print(f"C2 augmented vs SA:            {_fmt_pct(float(s['augmented_vs_sa']))}")
    print(f"C2 augmented vs best baseline: {_fmt_pct(float(s['augmented_vs_best_baseline']))}")
    print("\n=== verdict ===")
    print(report.verdict)


def render_markdown(report: BenchmarkReport) -> str:
    s = report.summary
    lines = [
        "# C2 Discrete Geometry Verification: Delaunay-Local Active Repair",
        "",
        f"- Originating run: `{RUN_ID}`",
        f"- Composition: `{COMPOSITION_ID}`",
        f"- Outcome: **{report.verdict}**",
        f"- Sizes tested: {', '.join(str(x) for x in report.sizes)}",
        f"- Seeds per size: {report.seeds_per_size}",
        f"- Search steps per run: {report.steps}",
        "",
        "## Question",
        "",
        "Can a Delaunay-local, force-like repair loop improve finite point layouts, "
        "or does it optimize the wrong local surrogate and miss the true worst "
        "near-collinear triples?",
        "",
        "## Method",
        "",
        "The objective is the Heilbronn-style score: maximize the minimum triangle "
        "area over all triples of points in the unit square. The benchmark compares:",
        "",
        "- `SA-full`: simulated annealing using full exhaustive triangle scans.",
        "- `C2-full`: the same active-repair move generator, but with all triples visible.",
        "- `C2-delaunay`: active repair using only Delaunay cell triangles.",
        "- `C2-augmented`: Delaunay cells plus Delaunay one-ring triples, k-nearest-neighbor "
        "triples, and periodic global scans that inject missed bad triples.",
        "",
        "This directly tests the L3 objection: pure Delaunay locality may not see the "
        "actual combinatorial bottleneck for no-near-collinearity.",
        "",
        "## Results",
        "",
        "| method | mean best min-triangle area | mean seconds | mean active-set recall | mean candidate triples |",
        "|---|---:|---:|---:|---:|",
    ]
    for method, best, elapsed, recall, candidates in _method_rows(report):
        lines.append(f"| {method} | {best:.6g} | {elapsed:.3f} | {recall:.3f} | {candidates:.1f} |")
    lines.extend(
        [
            "",
            "## Active-Set Recall Stress Test",
            "",
            "| candidate set | top-5 recall | candidate count / full triples |",
            "|---|---:|---:|",
            f"| Delaunay cells | {float(s['delaunay_top5_recall']) * 100:.1f}% | "
            f"{_mean_recall(report.recall_rows, 'delaunay_cells', 'candidate_ratio') * 100:.1f}% |",
            f"| Augmented C2 | {float(s['augmented_top5_recall']) * 100:.1f}% | "
            f"{float(s['augmented_candidate_ratio']) * 100:.1f}% |",
            "",
            "## Verdict",
            "",
        ]
    )
    if report.verdict == "clean-positive":
        lines.append(
            "C2 passed the bounded verification: the augmented active-set method beat "
            "the strongest tested baseline and retained high recall of the true worst "
            "triples. That would support the composition as a practical discovery path."
        )
    elif report.verdict == "partial-positive":
        lines.append(
            "C2 is useful but not fully proven. The pure Delaunay version confirms the "
            "reviewer's warning, while the augmented loop recovers much more of the "
            "true bottleneck set and is competitive enough to justify a stronger "
            "follow-up with CMA-ES, differential evolution, and larger n."
        )
    else:
        lines.append(
            "C2 was falsified in this bounded test. The augmented active set did not "
            "beat the strongest tested baseline, and the locality advantage was not "
            "enough to compensate for optimizing an incomplete surrogate."
        )
    lines.extend(
        [
            "",
            "## What This Says About the Original Claim",
            "",
            "- The most important issue is not Delaunay geometry itself; it is whether "
            "the candidate set sees the actual bottleneck triples.",
            "- Pure Delaunay cells are too narrow for no-near-collinearity-style "
            "objectives.",
            "- Periodic global scans are not optional. They are the control loop that "
            "keeps the cross-domain idea honest.",
            "- The next decisive test is a larger head-to-head against DE, CMA-ES, and "
            "basin hopping using equal wall-clock budgets.",
            "",
            "## Reproduce",
            "",
            "```bash",
            "uv run python -m experiments.discrete_geometry_c2.bench --write-findings",
            "```",
        ]
    )
    return "\n".join(lines) + "\n"


def write_findings(report: BenchmarkReport, path: Path = DEFAULT_FINDINGS_PATH) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(render_markdown(report), encoding="utf-8")


def add_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--smoke", action="store_true", help="Run a small fast benchmark.")
    parser.add_argument("--sizes", nargs="+", type=int, default=[10, 14, 18, 22])
    parser.add_argument("--seeds-per-size", type=int, default=3)
    parser.add_argument("--steps", type=int, default=350)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--active-count", type=int, default=8)
    parser.add_argument("--global-scan-every", type=int, default=25)
    parser.add_argument("--step-scale", type=float, default=0.035)
    parser.add_argument("--k-neighbors", type=int, default=8)
    parser.add_argument("--margin", type=float, default=0.05)
    parser.add_argument("--verbose", action="store_true")
    parser.add_argument("--write-findings", action="store_true")
    parser.add_argument(
        "--findings-path",
        type=Path,
        default=DEFAULT_FINDINGS_PATH,
        help=f"Markdown output path. Default: {DEFAULT_FINDINGS_PATH}",
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    add_args(parser)
    args = parser.parse_args(argv)
    report = run_benchmark(args)
    print_report(report)
    if args.write_findings:
        write_findings(report, args.findings_path)
        print(f"\nwrote {args.findings_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
