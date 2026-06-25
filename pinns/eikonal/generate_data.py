from pathlib import Path

import ml_collections
from tqdm import tqdm
from pinns.eikonal.samplers import (
    UniformBCSampler,
    UniformSampler,
    UniformBoundarySampler,
)

from pinns.eikonal.get_dataset import get_dataset
from pinns.eikonal.chart_geometry import (
    chart_backend,
    ensure_chart_coordinates,
    prepare_chart_geometry,
)
from pinns.eikonal.plot import (
    plot_charts_solution,
    plot_charts_with_supernodes,
    plot_domains,
    plot_domains_3d,
    plot_domains_3d_html,
    plot_domains_with_metric,
    plot_combined_3d_with_metric,
)

import numpy as np


def _sparse_point_ids_path(config: ml_collections.ConfigDict) -> str | None:
    sparse_cfg = getattr(config, "sparse_points", None)
    sparse_root = getattr(sparse_cfg, "path", None) if sparse_cfg is not None else None
    if not sparse_root:
        return None
    return str(Path(sparse_root) / f"N{config.N}_seed{config.bcs_seed}.npy")


def _sparse_sampling_kwargs(config: ml_collections.ConfigDict) -> dict:
    sparse_cfg = getattr(config, "sparse_points", None)
    if sparse_cfg is None:
        return {}
    return {
        "sampling_strategy": getattr(sparse_cfg, "strategy", "random"),
        "sampling_num_bins": getattr(sparse_cfg, "num_bins", None),
    }


def generate_data(config: ml_collections.ConfigDict):
    """Generate cached training batches for the eikonal PINN."""

    loaded_charts3d = None
    charts_mu = None
    charts_std = None
    sqrt_det_g = None
    decoder = None
    conditionings = None
    d_params = None

    if config.plot and chart_backend(config) == "uae":
        (
            loaded_charts3d,
            charts_mu,
            charts_std,
            _,
            sqrt_det_g,
            decoder,
            conditionings,
            d_params,
        ) = prepare_chart_geometry(config)
    else:
        ensure_chart_coordinates(config)

    x, y, boundaries_x, boundaries_y, bcs_x, bcs_y, bcs, charts3d = get_dataset(
        charts_path=config.dataset.charts_path,
        N=config.N,
        idxs=config.idxs,
        seed=config.bcs_seed,
        enforce_source_bc=getattr(getattr(config, "eikonal", None), "enforce_source_bc", True),
        source_idx=getattr(getattr(config, "eikonal", None), "source_idx", 0),
        **_sparse_sampling_kwargs(config),
        save_point_ids_path=_sparse_point_ids_path(config),
    )

    Path(config.figure_path).mkdir(parents=True, exist_ok=True)
    Path(config.training.batches_path).mkdir(parents=True, exist_ok=True)

    if config.plot and chart_backend(config) == "uae":

        plot_charts_solution(
            bcs_x,
            bcs_y,
            bcs,
            name=config.figure_path + "/generated_eikonal_train_bcs.png",
            vmin=0.0,
            vmax=1.5,
        )

        plot_charts_with_supernodes(
            loaded_charts3d,
            np.random.randint(0, len(loaded_charts3d), 64),
            name=Path(config.figure_path) / "charts_with_supernodes.png",
        )

        plot_domains(
            x,
            y,
            boundaries_x,
            boundaries_y,
            bcs_x=bcs_x,
            bcs_y=bcs_y,
            bcs=bcs,
            name=Path(config.figure_path) / "domains.png",
        )

        plot_domains_3d(
            x,
            y,
            bcs_x=bcs_x,
            bcs_y=bcs_y,
            bcs=bcs,
            decoder=decoder,
            conditionings=conditionings,
            d_params=d_params,
            charts_mu=charts_mu,
            charts_std=charts_std,
            name=Path(config.figure_path) / "domains_3d.png",
        )

        plot_domains_3d_html(
            x,
            y,
            bcs_x=bcs_x,
            bcs_y=bcs_y,
            bcs=bcs,
            decoder=decoder,
            conditionings=conditionings,
            d_params=d_params,
            charts_mu=charts_mu,
            charts_std=charts_std,
            name=Path(config.figure_path) / "domains_3d.html",
        )

        plot_domains_with_metric(
            x,
            y,
            sqrt_det_g,
            conditionings=conditionings,
            name=Path(config.figure_path) / "domains_with_metric.png",
        )

        plot_combined_3d_with_metric(
            x,
            y,
            decoder=decoder,
            sqrt_det_g=sqrt_det_g,
            conditionings=conditionings,
            d_params=d_params,
            charts_mu=charts_mu,
            charts_std=charts_std,
            name=Path(config.figure_path) / "combined_3d_with_metric.png",
        )
    elif config.plot:
        print(
            "Skipping decoder-dependent diagnostic plots for "
            f"chart.backend={chart_backend(config)}."
        )

    bcs_sampler = iter(
        UniformBCSampler(
            bcs_x=bcs_x,
            bcs_y=bcs_y,
            bcs=bcs,
            num_charts=len(x),
            batch_size=config.training.batch_size,
            load_existing_batches=False,
        )
    )

    res_sampler = iter(
        UniformSampler(
            x=x,
            y=y,
            sigma=0.02,
            batch_size=config.training.batch_size,
        )
    )

    boundary_sampler = iter(
        UniformBoundarySampler(
            boundaries_x=boundaries_x,
            boundaries_y=boundaries_y,
            batch_size=config.training.batch_size,
            load_existing_batches=False,
        )
    )

    res_batches = []
    boundary_batches = []
    boundary_pairs_idxs = []
    bcs_batches = []
    bcs_values = []

    num_boundary_batches = getattr(config.training, "num_boundary_batches", 500)
    for step in tqdm(range(num_boundary_batches), desc="Generating batches"):

        # batch = next(res_sampler), next(boundary_sampler), next(bcs_sampler)
        # res_batches.append(batch[0])

        batch = None, next(boundary_sampler)  # next(bcs_sampler)

        boundary_batches.append(batch[1][0])
        boundary_pairs_idxs.append(batch[1][1])
        # bcs_batches.append(batch[2][0])
        # bcs_values.append(batch[2][1])

    # res_batches_array = np.array(res_batches)
    boundary_batches_array = np.array(boundary_batches)
    boundary_pairs_idxs_array = np.array(boundary_pairs_idxs)
    # bcs_batches_array = np.array(bcs_batches)
    # bcs_values_array = np.array(bcs_values)

    # np.save(config.training.batches_path + "res_batches.npy", res_batches_array)
    np.save(
        Path(config.training.batches_path) / "boundary_batches.npy",
        boundary_batches_array,
    )
    np.save(
        Path(config.training.batches_path) / "boundary_pairs_idxs.npy",
        boundary_pairs_idxs_array,
    )

    # np.save(config.training.batches_path + "bcs_batches.npy", bcs_batches_array)
    # np.save(config.training.values_path + "bcs_values.npy", bcs_values_array)

    # print("Size of res_batches in MB: ", res_batches_array.nbytes / 1024 / 1024)
    print(
        "Size of boundary_batches in MB: ",
        boundary_batches_array.nbytes / 1024 / 1024,
    )
    print(
        "Size of boundary_pairs_idxs in MB: ",
        boundary_pairs_idxs_array.nbytes / 1024 / 1024,
    )
    # print("Size of bcs_batches in MB: ", bcs_batches_array.nbytes / 1024 / 1024)
    # print("Size of bcs_values in MB: ", bcs_values_array.nbytes / 1024 / 1024)

    # if step % 100 == 0:
    #     res_batches_array = np.array(res_batches)
    #     boundary_batches_array = np.array(boundary_batches)
    #     boundary_pairs_idxs_array = np.array(boundary_pairs_idxs)
    #     bcs_batches_array = np.array(bcs_batches)
    #     bcs_values_array = np.array(bcs_values)

    #     np.save(config.training.res_batches_path, res_batches_array)
    #     np.save(config.training.boundary_batches_path, boundary_batches_array)
    #     np.save(config.training.boundary_pairs_idxs_path, boundary_pairs_idxs_array)
    #     np.save(config.training.bcs_batches_path, bcs_batches_array)
    #     np.save(config.training.bcs_values_path, bcs_values_array)

    #     print("Size of res_batches in MB: ", res_batches_array.nbytes / 1024 / 1024)
    #     print(
    #         "Size of boundary_batches in MB: ",
    #         boundary_batches_array.nbytes / 1024 / 1024,
    #     )
    #     print(
    #         "Size of boundary_pairs_idxs in MB: ",
    #         boundary_pairs_idxs_array.nbytes / 1024 / 1024,
    #     )
    #     print("Size of bcs_batches in MB: ", bcs_batches_array.nbytes / 1024 / 1024)
    #     print("Size of bcs_values in MB: ", bcs_values_array.nbytes / 1024 / 1024)
