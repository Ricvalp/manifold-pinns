import numpy as np

import logging
import networkx as nx
import jax
import flax
from typing import Any, Dict
from copy import deepcopy
import multiprocessing as mp
from functools import partial

from charts.get_charts import create_graph


def numpy_collate(batch):
    if isinstance(batch[0], np.ndarray):
        return np.stack(batch)
    elif isinstance(batch[0], (tuple, list)):
        transposed = zip(*batch)
        return [numpy_collate(samples) for samples in transposed]
    else:
        return np.array(batch)


def calculate_distance_matrix_single_process(chart_data, nearest_neighbors):
    pts, chart_id = chart_data
    logging.info(f"Calculating distances for chart {chart_id}")
    G = create_graph(pts=pts, nearest_neighbors=nearest_neighbors)
    # Check that graph is a single connected component
    if not nx.is_connected(G):
        raise ValueError(
            f"Graph for chart {chart_id} is not a single connected component"
        )
    distances = dict(nx.all_pairs_shortest_path_length(G, cutoff=None))
    distances_matrix = np.zeros((len(pts), len(pts)))
    for j in range(len(pts)):
        for k in range(len(pts)):
            distances_matrix[j, k] = distances[j][k]
    logging.info(f"Finished calculating distances for chart {chart_id}")
    return chart_id, distances_matrix


def compute_distance_matrix(charts, nearest_neighbors):
    """
    Compute the distance matrices for each chart.
    """
    distance_matrix = {}
    chart_data = [(charts[i], i) for i in charts.keys()]

    # Create a pool of workers
    num_processes = mp.cpu_count() - 1  # Leave one CPU free
    with mp.Pool(processes=num_processes) as pool:
        logging.info(f"Calculating distances using {num_processes} processes")
        results = pool.map(
            partial(
                calculate_distance_matrix_single_process,
                nearest_neighbors=nearest_neighbors,
            ),
            chart_data,
        )

    for chart_id, matrix in results:
        distance_matrix[chart_id] = matrix

    return distance_matrix


def get_model(cfg: Dict[str, Any]) -> flax.linen.Module:
    """Returns the model for the given config.

    Args:
        nef_cfg (Dict[str, Any]): The model config.

    Returns:
        flax.linen.Module: The model.

    """

    model_cfg = deepcopy(cfg).unlock()

    if model_cfg.name not in dir(models):
        raise NotImplementedError(
            f"Model {model_cfg['name']} not implemented. Available are: {dir(models)}"
        )
    else:
        model = getattr(models, model_cfg.name)
        return model(n_hidden=model_cfg.n_hidden, rff_dim=model_cfg.rff_dim)
