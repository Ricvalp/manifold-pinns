from absl import app, logging
from ml_collections import config_flags
import numpy as np
import scipy.linalg
from pathlib import Path
from datasets import get_dataset
from charts import (
    plot_3d_charts,
    plot_html_3d_charts,
    plot_html_3d_boundaries,
    get_charts,
    find_verts_in_charts,
    plot_3d_points,
    plot_3d_chart,
)
from charts import (
    save_charts,
)


_TASK_FILE = config_flags.DEFINE_config_file(
    "config", default="charts/config/make_charts_coil.py"
)


def main(_):
    cfg = load_cfgs(_TASK_FILE)
    Path(cfg.figure_path).mkdir(parents=True, exist_ok=True)

    train_data = get_dataset(cfg.dataset)
    verts, connectivity = train_data.verts, train_data.connectivity

    plot_3d_points(
        points=train_data.data,
        name=Path(cfg.figure_path) / f"{cfg.dataset.name}_points.png",
    )

    logging.info(
        f"Loaded {cfg.dataset.name} dataset. Got {len(train_data.data)} points"
    )

    charts, charts_idxs, boundaries, boundary_indices, sampled_points = get_charts(
        points=train_data.data,
        charts_config=cfg.charts,
    )

    M = np.random.normal(0, 1, (3, 3))
    TrM = np.trace(M)
    M = M - ( TrM * np.eye(3) / 3)

    charts = {
        key: (scipy.linalg.expm(cfg.charts.deformation_magnitude * M) @ chart.T).T for key, chart in charts.items()
        }
    boundaries = {
        key: (scipy.linalg.expm(cfg.charts.deformation_magnitude * M) @ boundary.T).T for key, boundary in boundaries.items()
    }

    verts = (scipy.linalg.expm(cfg.charts.deformation_magnitude * M) @ verts.T).T

    chart_in_mesh_indices, mesh_in_chart_indices = find_verts_in_charts(charts, verts)

    save_charts(
        charts_path=cfg.dataset.charts_path,
        charts=charts,
        charts_idxs=charts_idxs,
        boundaries=boundaries,
        boundary_indices=boundary_indices,
        chart_in_mesh_indices=chart_in_mesh_indices,
        mesh_in_chart_indices=mesh_in_chart_indices,
        verts=verts,
        connectivity=connectivity,
    )

    logging.info(f"Got {len(charts)} charts. Saved charts to {cfg.dataset.charts_path}")

    plot_3d_chart(
        chart=charts[0],
        name=Path(cfg.figure_path) / f"{cfg.dataset.name}_chart_0.png",
    )
    plot_html_3d_charts(
        charts=charts,
        sampled_points=sampled_points,
        name=Path(cfg.figure_path) / f"{cfg.dataset.name}_charts.html",
    )
    plot_html_3d_boundaries(
        boundaries=boundaries,
        name=Path(cfg.figure_path) / f"{cfg.dataset.name}_boundaries.html",
    )
    plot_3d_charts(
        charts=charts,
        gt_charts=None,
        name=Path(cfg.figure_path) / f"{cfg.dataset.name}_charts.png",
    )

def load_cfgs(
    _TASK_FILE,
):
    cfg = _TASK_FILE.value

    return cfg

def deform_points(points, M, t=0.0):
    """
    Apply a smooth, invertible deformation (diffeomorphism) to the chart points.
    
    Args:
        points: points to transform
        deformation_magnitude: controls the strength of deformation (0.0 = no deformation)
    
    Returns:
        Transformed points
    """
    
    
    TrM = np.trace(M)
    M = M - ( TrM * np.eye(3) / 3)
    
    points = (scipy.linalg.expm(t * M) @ points.T).T
    
    return points

if __name__ == "__main__":
    app.run(main)
