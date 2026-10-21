from __future__ import annotations

import numpy as np

from experiments.discrete_geometry_c2 import bench as c2


def _triple_set(triples: np.ndarray) -> set[tuple[int, int, int]]:
    return {tuple(int(x) for x in triple) for triple in triples}


def test_candidate_recall_detects_missing_worst_triangle():
    points = np.array(
        [
            [0.0, 0.0],
            [1.0, 0.0],
            [0.5, 0.0001],
            [0.0, 1.0],
        ],
        dtype=float,
    )
    worst_only = np.array([[0, 1, 2]], dtype=np.int32)
    missing_worst = np.array([[0, 1, 3], [0, 2, 3], [1, 2, 3]], dtype=np.int32)

    all_scores = c2.triangle_areas(points, c2.triangle_indices(len(points)))
    assert np.isclose(float(all_scores.min()), 0.00005)

    assert c2.candidate_recall(points, worst_only, top_k=1).recall == 1.0
    assert c2.candidate_recall(points, missing_worst, top_k=1).recall == 0.0


def test_augmented_candidates_include_delaunay_cell_candidates():
    rng = np.random.default_rng(12)
    points = rng.random((18, 2))

    delaunay_cells = _triple_set(c2.delaunay_cell_triples(points))
    augmented = _triple_set(c2.augmented_candidate_triples(points, k_neighbors=6))

    assert delaunay_cells
    assert delaunay_cells.issubset(augmented)


def test_active_repair_search_preserves_best_full_objective():
    rng = np.random.default_rng(5)
    points = rng.random((12, 2))

    result = c2.active_repair_search(
        points,
        mode="augmented",
        steps=80,
        seed=7,
        active_count=5,
        global_scan_every=10,
        step_scale=0.05,
    )

    assert result.best_points.shape == points.shape
    assert np.all(result.best_points >= 0.0)
    assert np.all(result.best_points <= 1.0)
    assert result.best_score >= result.initial_score - 1e-12
    assert result.candidate_scans > 0
    assert result.full_scans >= 2
    assert 0.0 <= result.mean_recall <= 1.0
