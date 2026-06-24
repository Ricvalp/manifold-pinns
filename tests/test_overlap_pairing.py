import numpy as np

from manifold_pinns.geometry.overlaps import (
    build_overlap_pairs_from_point_ids,
    build_overlap_pairs_kdtree,
    interface_loss,
    sample_overlap_pairs,
)


def test_overlap_pairs_map_to_same_ambient_points_with_point_ids():
    z_a = np.array([[1.0, 0.0], [1.5, 0.5], [2.0, 1.0]])
    z_b = np.array([[0.0, 0.0], [0.5, 0.5], [1.0, 1.0]])
    x = np.column_stack([z_a[:, 0], z_a[:, 1], np.zeros(len(z_a))])
    pair = build_overlap_pairs_from_point_ids(
        0,
        1,
        coords_by_chart={0: z_a, 1: z_b},
        point_ids_by_chart={0: np.array([10, 11, 12]), 1: np.array([10, 11, 12])},
        ambient_by_chart={0: x, 1: x},
    )
    sample_a, sample_b = sample_overlap_pairs(pair, batch_size=5, rng=np.random.default_rng(0))
    x_a = np.column_stack([sample_a[:, 0], sample_a[:, 1], np.zeros(len(sample_a))])
    x_b = np.column_stack([sample_b[:, 0] + 1.0, sample_b[:, 1], np.zeros(len(sample_b))])
    np.testing.assert_allclose(x_a, x_b)


def test_interface_loss_zero_for_same_analytic_function_in_two_charts():
    z_a = np.array([[1.0, 0.0], [1.5, 0.5], [2.0, 1.0]])
    z_b = np.array([[0.0, 0.0], [0.5, 0.5], [1.0, 1.0]])
    pair = build_overlap_pairs_from_point_ids(
        0,
        1,
        coords_by_chart={0: z_a, 1: z_b},
        point_ids_by_chart={0: np.array([1, 2, 3]), 1: np.array([1, 2, 3])},
    )
    u_a = lambda z: z[:, 0] + 2.0 * z[:, 1]
    u_b = lambda z: (z[:, 0] + 1.0) + 2.0 * z[:, 1]
    assert interface_loss(u_a, u_b, pair) < 1e-12


def test_kdtree_overlap_fallback_handles_noisy_points():
    z_a = np.array([[0.0, 0.0], [0.5, 0.5], [1.0, 1.0]])
    z_b = z_a.copy()
    x_a = np.column_stack([z_a, np.zeros(len(z_a))])
    x_b = x_a + 1e-7
    pair = build_overlap_pairs_kdtree(
        0,
        1,
        coords_by_chart={0: z_a, 1: z_b},
        ambient_by_chart={0: x_a, 1: x_b},
        tolerance=1e-6,
    )
    assert pair.size == 3
