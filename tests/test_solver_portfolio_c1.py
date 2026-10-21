from __future__ import annotations

import numpy as np

from experiments.solver_portfolio_c1 import bench


def test_tour_heuristics_return_valid_tours_and_two_opt_improves():
    coords = np.array(
        [
            [0.0, 0.0],
            [1.0, 0.0],
            [1.0, 1.0],
            [0.0, 1.0],
            [0.45, 0.45],
        ],
        dtype=float,
    )
    dist = bench.distance_matrix(coords)
    nn = bench.nearest_neighbor_tour(dist)
    improved = bench.two_opt(nn, dist, max_passes=10)

    assert sorted(nn) == list(range(len(coords)))
    assert sorted(improved) == list(range(len(coords)))
    assert bench.tour_length(improved, dist) <= bench.tour_length(nn, dist) + 1e-12


def test_density_peak_clustering_separates_two_blobs():
    features = np.array(
        [
            [0.0, 0.1],
            [0.2, -0.1],
            [-0.1, 0.0],
            [10.0, 10.1],
            [9.8, 10.2],
            [10.2, 9.9],
        ],
        dtype=float,
    )
    model = bench.fit_density_peak(features, n_clusters=2, dc_percentile=35.0)
    labels = bench.predict_density_peak(model, features)

    assert len(set(labels[:3])) == 1
    assert len(set(labels[3:])) == 1
    assert labels[0] != labels[3]


def test_cluster_selector_can_beat_single_best_on_regime_data():
    losses = np.array(
        [
            [0.02, 0.30],
            [0.01, 0.32],
            [0.03, 0.28],
            [0.27, 0.02],
            [0.30, 0.01],
            [0.29, 0.03],
        ],
        dtype=float,
    )
    train_idx = np.array([0, 1, 3, 4])
    test_idx = np.array([2, 5])
    train_clusters = np.array([0, 0, 1, 1])
    test_clusters = np.array([0, 1])

    single = bench.evaluate_single_best(losses, train_idx, test_idx)
    clustered = bench.evaluate_cluster_selector(
        losses, train_idx, test_idx, train_clusters, test_clusters
    )

    assert clustered.mean_regret < single.mean_regret
    assert clustered.chosen_solver_names == ["solver_0", "solver_1"]
