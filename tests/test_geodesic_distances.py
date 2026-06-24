import numpy as np
import networkx as nx


def test_weighted_dijkstra_differs_from_hop_count_on_unequal_weights():
    graph = nx.Graph()
    graph.add_edge(0, 1, weight=1.0)
    graph.add_edge(1, 2, weight=10.0)
    graph.add_edge(0, 2, weight=2.0)

    hop_distance = dict(nx.all_pairs_shortest_path_length(graph))[0][2]
    weighted_distance = dict(nx.all_pairs_dijkstra_path_length(graph, weight="weight"))[0][2]

    assert hop_distance == 1
    np.testing.assert_allclose(weighted_distance, 2.0)
