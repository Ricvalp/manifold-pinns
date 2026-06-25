import json
import numpy as np
from pathlib import Path
from jax import random
import igl
import jax.numpy as jnp
import logging
from flax.training import checkpoints
from charts.get_charts import load_charts, find_verts_in_charts


def _exact_geodesic_from_source(verts, connectivity, source_idx):
    n_vertices = verts.shape[0]
    if not 0 <= source_idx < n_vertices:
        raise ValueError(
            f"source_idx={source_idx} is outside the mesh vertex range "
            f"[0, {n_vertices})."
        )

    distances = igl.exact_geodesic(
        verts,
        connectivity,
        VS=np.array([source_idx], dtype=np.int64),
        VT=np.arange(n_vertices, dtype=np.int64),
    )
    if distances.shape[0] != n_vertices:
        raise RuntimeError(
            "libigl exact_geodesic returned "
            f"{distances.shape[0]} distances for {n_vertices} mesh vertices."
        )
    return distances


def get_dataset(
    charts_path,
    N=100,
    idxs=None,
    seed=42,
    *,
    enforce_source_bc=True,
    source_idx=0,
    sampling_strategy="random",
    sampling_num_bins=None,
    save_point_ids_path=None,
):

    (
        loaded_charts3d,
        loaded_charts_idxs,
        loaded_boundaries,
        loaded_boundary_indices,
        loaded_charts2d,
    ) = load_charts(charts_path)

    x = {}
    y = {}

    for chart_key in loaded_charts2d.keys():
        x[chart_key] = loaded_charts2d[chart_key][:, 0]
        y[chart_key] = loaded_charts2d[chart_key][:, 1]

    boundaries_x = {}
    boundaries_y = {}

    for key in loaded_boundary_indices.keys():
        start_boundary_indices = np.array(loaded_boundary_indices[key])

        starting_chart = key[0]
        starting_chart_points = loaded_charts2d[starting_chart][start_boundary_indices]
        ending_chart = key[1]

        if starting_chart not in boundaries_x:
            boundaries_x[starting_chart] = {}
        if starting_chart not in boundaries_y:
            boundaries_y[starting_chart] = {}

        boundaries_x[starting_chart][ending_chart] = starting_chart_points[:, 0]
        boundaries_y[starting_chart][ending_chart] = starting_chart_points[:, 1]

    bcs_x, bcs_y, bcs = get_eikonal_bcs(
        charts_path=charts_path,
        x=x,
        y=y,
        charts3d=loaded_charts3d,
        N=N,
        seed=seed,
        idxs=idxs,
        enforce_source_bc=enforce_source_bc,
        source_idx=source_idx,
        sampling_strategy=sampling_strategy,
        sampling_num_bins=sampling_num_bins,
        save_point_ids_path=save_point_ids_path,
    )

    return x, y, boundaries_x, boundaries_y, bcs_x, bcs_y, bcs, loaded_charts3d


def get_eikonal_bcs(
    charts_path,
    x,
    y,
    charts3d,
    N=50,
    seed=42,
    idxs=None,
    *,
    enforce_source_bc=True,
    source_idx=0,
    sampling_strategy="random",
    sampling_num_bins=None,
    save_point_ids_path=None,
):

    verts = np.load(charts_path + "/verts.pkl", allow_pickle=True)
    connectivity = np.load(charts_path + "/connectivity.pkl", allow_pickle=True)

    if N > verts.shape[0]:
        N = verts.shape[0]

    Y_eg = _exact_geodesic_from_source(verts, connectivity, source_idx)
    n_nodes = verts.shape[0]

    idx_train, resolved_strategy = select_eikonal_point_ids(
        n_nodes=n_nodes,
        N=N,
        seed=seed,
        idxs=idxs,
        distances=Y_eg,
        enforce_source_bc=enforce_source_bc,
        source_idx=source_idx,
        sampling_strategy=sampling_strategy,
        sampling_num_bins=sampling_num_bins,
    )
    if save_point_ids_path is not None:
        _save_point_ids(
            save_point_ids_path,
            idx_train,
            Y_eg,
            strategy=resolved_strategy,
            seed=seed,
            source_idx=source_idx,
            enforce_source_bc=enforce_source_bc,
            sampling_num_bins=sampling_num_bins,
        )
    Y = Y_eg[idx_train]
    bcs_points = verts[idx_train]

    bcs_x = {}
    bcs_y = {}
    bcs = {}

    chart_in_mesh_indices, mesh_in_chart_indices = find_verts_in_charts(
        charts3d, bcs_points
    )

    logging.info("gt solution indices: %s", idx_train)

    for chart_key in mesh_in_chart_indices.keys():
        if (
            len(mesh_in_chart_indices[chart_key]) > 0
            and len(chart_in_mesh_indices[chart_key]) > 0
        ):
            bcs_x[chart_key] = x[chart_key][mesh_in_chart_indices[chart_key]]
            bcs_y[chart_key] = y[chart_key][mesh_in_chart_indices[chart_key]]
            bcs[chart_key] = Y[chart_in_mesh_indices[chart_key]]

    return bcs_x, bcs_y, bcs


def select_eikonal_point_ids(
    *,
    n_nodes,
    N,
    seed,
    idxs,
    distances,
    enforce_source_bc=True,
    source_idx=0,
    sampling_strategy="random",
    sampling_num_bins=None,
):
    """Select sparse Eikonal supervision vertices."""

    if N <= 0:
        return np.array([], dtype=np.int64), "empty"
    N = min(int(N), int(n_nodes))

    if idxs is not None:
        selected = np.asarray(idxs[-N:], dtype=np.int64)
        if enforce_source_bc and source_idx not in selected:
            if selected.size == 0:
                selected = np.asarray([source_idx], dtype=np.int64)
            else:
                selected[0] = source_idx
        return np.sort(selected).astype(np.int64), "explicit"

    strategy = (sampling_strategy or "random").lower()
    if strategy in {"random", "uniform", "uniform_random"}:
        selected = _sample_random_point_ids(
            n_nodes=n_nodes,
            N=N,
            seed=seed,
            enforce_source_bc=enforce_source_bc,
            source_idx=source_idx,
        )
    elif strategy in {
        "stratified",
        "stratified_geodesic",
        "geodesic_stratified",
        "distance_stratified",
    }:
        selected = _sample_stratified_geodesic_point_ids(
            distances=np.asarray(distances),
            N=N,
            seed=seed,
            enforce_source_bc=enforce_source_bc,
            source_idx=source_idx,
            sampling_num_bins=sampling_num_bins,
        )
        strategy = "stratified_geodesic"
    else:
        raise ValueError(
            "Unknown sparse point sampling strategy "
            f"'{sampling_strategy}'. Expected 'random' or 'stratified_geodesic'."
        )

    return np.sort(selected).astype(np.int64), strategy


def _sample_random_point_ids(
    *,
    n_nodes,
    N,
    seed,
    enforce_source_bc,
    source_idx,
):
    if enforce_source_bc:
        rng = np.random.default_rng(seed)
        candidates = np.setdiff1d(np.arange(n_nodes), np.array([source_idx]))
        sampled = rng.choice(candidates, size=max(N - 1, 0), replace=False)
        return np.concatenate([np.array([source_idx]), sampled])

    key = random.PRNGKey(seed)
    return np.asarray(random.choice(key, n_nodes, (N,), replace=False))


def _sample_stratified_geodesic_point_ids(
    *,
    distances,
    N,
    seed,
    enforce_source_bc,
    source_idx,
    sampling_num_bins,
):
    rng = np.random.default_rng(seed)
    selected = [int(source_idx)] if enforce_source_bc and N > 0 else []
    remaining = N - len(selected)
    if remaining <= 0:
        return np.asarray(selected, dtype=np.int64)

    candidates = np.arange(distances.shape[0], dtype=np.int64)
    if enforce_source_bc:
        candidates = candidates[candidates != source_idx]
    candidates = candidates[np.isfinite(distances[candidates])]
    if candidates.size == 0:
        return np.asarray(selected, dtype=np.int64)

    ordered = candidates[np.argsort(distances[candidates])]
    num_bins = int(sampling_num_bins or remaining)
    num_bins = max(1, min(num_bins, remaining, ordered.size))
    bins = [
        bin_ids for bin_ids in np.array_split(ordered, num_bins) if bin_ids.size > 0
    ]

    per_bin_used = [set() for _ in bins]
    while len(selected) < N and any(
        len(used) < len(bin_ids) for used, bin_ids in zip(per_bin_used, bins)
    ):
        for bin_idx, bin_ids in enumerate(bins):
            if len(selected) >= N:
                break
            used = per_bin_used[bin_idx]
            available = np.asarray([idx for idx in bin_ids if int(idx) not in used])
            if available.size == 0:
                continue
            picked = int(rng.choice(available))
            used.add(picked)
            selected.append(picked)

    if len(selected) < N:
        fallback = np.setdiff1d(candidates, np.asarray(selected, dtype=np.int64))
        if fallback.size > 0:
            extra = rng.choice(
                fallback,
                size=min(N - len(selected), fallback.size),
                replace=False,
            )
            selected.extend(int(idx) for idx in extra)

    return np.asarray(selected, dtype=np.int64)


def _save_point_ids(
    save_point_ids_path,
    idx_train,
    distances,
    *,
    strategy,
    seed,
    source_idx,
    enforce_source_bc,
    sampling_num_bins,
):
    save_path = Path(save_point_ids_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    ids = np.asarray(idx_train, dtype=np.int64)
    np.save(save_path, ids)
    metadata = {
        "ids": ids.tolist(),
        "geodesic_values": [float(distances[idx]) for idx in ids],
        "strategy": strategy,
        "seed": int(seed),
        "source_idx": int(source_idx),
        "enforce_source_bc": bool(enforce_source_bc),
        "sampling_num_bins": (
            None if sampling_num_bins is None else int(sampling_num_bins)
        ),
    }
    save_path.with_suffix(".json").write_text(
        json.dumps(metadata, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def get_eikonal_gt_solution(charts_path):

    verts = np.load(charts_path + "/verts.pkl", allow_pickle=True)
    connectivity = np.load(charts_path + "/connectivity.pkl", allow_pickle=True)

    Y_eg = _exact_geodesic_from_source(verts, connectivity, source_idx=0)

    return verts, Y_eg
