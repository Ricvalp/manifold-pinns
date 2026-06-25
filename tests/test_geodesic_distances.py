import numpy as np
import networkx as nx

from pinns.eikonal.get_dataset import select_eikonal_point_ids


def test_weighted_dijkstra_differs_from_hop_count_on_unequal_weights():
    graph = nx.Graph()
    graph.add_edge(0, 1, weight=1.0)
    graph.add_edge(1, 2, weight=10.0)
    graph.add_edge(0, 2, weight=2.0)

    hop_distance = dict(nx.all_pairs_shortest_path_length(graph))[0][2]
    weighted_distance = dict(nx.all_pairs_dijkstra_path_length(graph, weight="weight"))[0][2]

    assert hop_distance == 1
    np.testing.assert_allclose(weighted_distance, 2.0)


def test_explicit_eikonal_ids_are_preserved_and_source_is_injected():
    ids, strategy = select_eikonal_point_ids(
        n_nodes=10,
        N=3,
        seed=0,
        idxs=[4, 5, 6],
        distances=np.arange(10, dtype=float),
        enforce_source_bc=True,
        source_idx=0,
    )

    assert strategy == "explicit"
    assert ids.tolist() == [0, 5, 6]


def test_random_eikonal_sampling_includes_source():
    ids, strategy = select_eikonal_point_ids(
        n_nodes=20,
        N=5,
        seed=1,
        idxs=None,
        distances=np.arange(20, dtype=float),
        enforce_source_bc=True,
        source_idx=0,
        sampling_strategy="random",
    )

    assert strategy == "random"
    assert len(ids) == 5
    assert 0 in ids


def test_stratified_geodesic_sampling_covers_tail():
    ids, strategy = select_eikonal_point_ids(
        n_nodes=101,
        N=6,
        seed=2,
        idxs=None,
        distances=np.arange(101, dtype=float),
        enforce_source_bc=True,
        source_idx=0,
        sampling_strategy="stratified_geodesic",
    )

    assert strategy == "stratified_geodesic"
    assert len(ids) == 6
    assert 0 in ids
    assert np.max(ids) >= 80
