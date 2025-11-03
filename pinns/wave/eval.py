from pathlib import Path
import numpy as np
from datetime import datetime

import ml_collections
import models

from pinns.wave.plot import plot_solutions, plot_2d_solutions
from jaxpi.utils import restore_checkpoint, load_config



def evaluate(config: ml_collections.ConfigDict):
    """Evaluate a trained wave PINN checkpoint."""

    figure_path = "./figures/" + str(datetime.now().strftime("%Y%m%d-%H%M%S"))

    model_config = load_config(
        Path(config.eval.checkpoint_dir) / "cfg.json",
    )

    charts3d = np.load(Path(model_config.saving.checkpoint_dir) / "charts_sequence.npy")
    ts = np.load(Path(model_config.saving.checkpoint_dir) / "ts.npy")

    model = models.WaveTime(
        model_config,
        inv_metric_tensor=None,
        sqrt_det_g=None,
        conditionings=None,
        ts=None,
        ics=(None, None, None),
    )

    model.state = restore_checkpoint(model.state, Path(config.eval.checkpoint_dir).absolute(), step=config.eval.step)

    sol = []
    for chart, t in zip(charts3d, ts):
        x, y, z = chart[:, 0], chart[:, 1], chart[:, 2]
        z = model.u_pred_fn(model.state.params, x, y, t)
        sol.append(z)

    plot_solutions(charts3d, ts=ts, values=sol, view_angle=(30, 100), save_dir=Path(figure_path), every_n=10, s=20, show_time=False, vmin=-0.1, vmax=1, cmap='magma')
    solutions = np.array(sol).reshape(len(sol), model_config.grid_size, model_config.grid_size)
    plot_2d_solutions(solutions=solutions, ts=ts, save_dir=Path(figure_path), every_n=10, show_time=False, vmin=-0.1, vmax=1, cmap='magma')
