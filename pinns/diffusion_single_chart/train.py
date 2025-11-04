import os
import time
from pathlib import Path
import seaborn as sns
import matplotlib.pyplot as plt

import json


import jax
import jax.numpy as jnp
import ml_collections
from . import models
from tqdm import tqdm
from jax.tree_util import tree_map
from .samplers import (
    UniformICSampler,
    UniformSampler,
)

from charts import (
    get_metric_tensor_and_sqrt_det_g_grid_universal_autodecoder,
)


from .plot import (
    plot_domains,
    plot_domains_3d,
    plot_domains_with_metric,
    plot_combined_3d_with_metric,
    plot_u0,
    plot_charts_sequence,
    plot_charts_sequence_with_solution,
    plot_charts_individually,
    plot_results,
)

import wandb
from jaxpi.utils import save_checkpoint, load_config

from .utils import set_profiler

import matplotlib.pyplot as plt
import numpy as np


def train_and_evaluate(config: ml_collections.ConfigDict):
    """Train the diffusion PINN and evaluate checkpoints during training."""

    wandb_config = config.wandb
    run = wandb.init(
        project=wandb_config.project,
        name=wandb_config.name,
        config=config,
    )

    Path(config.figure_path).mkdir(parents=True, exist_ok=True)

    autoencoder_config = load_config(
        Path(config.autoencoder_checkpoint.checkpoint_path) / "cfg.json",
    )

    checkpoint_root = Path(config.saving.checkpoint_dir)
    run_checkpoint_dir = checkpoint_root / run.id
    run_checkpoint_dir.mkdir(parents=True, exist_ok=True)

    # Persist the updated saving path so downstream utilities pick the run-specific directory.
    config.saving.checkpoint_dir = str(run_checkpoint_dir)

    with open(run_checkpoint_dir / "cfg.json", "w") as f:
        json.dump(config.to_dict(), f, indent=4)

    X = np.linspace(0, 1, 50)
    Y = np.linspace(0, 1, 50)
    XX, YY = np.meshgrid(X, Y)
    coords = np.zeros((XX.size, 3))
    coords[:, 0] = XX.flatten()
    coords[:, 1] = YY.flatten()

    boundaries_x = np.zeros((XX.size, 3))
    boundaries_x[:, 0] = XX.flatten()
    boundaries_x[:, 1] = YY.flatten()

    boundaries_y = np.zeros((XX.size, 3))
    boundaries_y[:, 0] = XX.flatten()
    boundaries_y[:, 1] = YY.flatten()
    
    charts3d = []
    ts = np.linspace(0.0, 0.8, 200)
    for t in ts:
        charts3d.append(get_deformed_points(coords, t))

    plot_charts_sequence(charts3d, ts, name=Path(config.figure_path) / "charts_sequence.png")
    
    (
        inv_metric_tensor,
        sqrt_det_g,
        decoder,
    ), (conditionings, d_params) = get_metric_tensor_and_sqrt_det_g_grid_universal_autodecoder(
        autoencoder_cfg=autoencoder_config,
        cfg=config,
        charts=charts3d,
        coords=coords[:, :2],
        inverse=True,
    )

    x = coords[:, 0]
    y = coords[:, 1]

    def initial_conditions_spike(x, y, x0=0.5, y0=0.5, sigma=.1, amplitude=30.0):
        return amplitude * np.exp(-((x - x0) ** 2 + (y - y0) ** 2) / sigma**2)

    u0 = initial_conditions_spike(x, y)


    plot_u0(x, y, u0, name=Path(config.figure_path) / "u0.png")

    plot_domains_with_metric(x, y, sqrt_det_g, conditionings, name=Path(config.figure_path) / "sequence.png")

    ics_sampler = iter(
        UniformICSampler(
            x=x,
            y=y,
            u0=u0,
            batch_size=config.training.batch_size,
        )
    )

    res_sampler = iter(
        UniformSampler(
            x=x,
            y=y,
            ts=ts,
            sigma=0.01,
            batch_size=config.training.batch_size,
        )
    )

    model = models.DiffusionTime(
        config,
        inv_metric_tensor=inv_metric_tensor,
        sqrt_det_g=sqrt_det_g,
        conditionings=conditionings,
        ts=ts,
        ics=(x, y, u0),
    )

    print("Waiting for JIT...")

    debug_logged = False
    step = 0

    for step in tqdm(range(1, config.training.max_steps + 1), desc="Training"):

        # set_profiler(config.profiler, step, config.profiler.log_dir)

        batch = next(res_sampler), next(ics_sampler)
        loss, losses, model.state = model.step(model.state, batch)

        if step % config.wandb.log_every_steps == 0:
            wandb.log({"loss": loss}, step)
        
        if config.weighting.scheme in ["grad_norm", "ntk"]:
            if step % config.weighting.update_every_steps == 0:
                model.state = model.update_weights(model.state, batch)

        # Saving
        if config.saving.save_every_steps is not None:
            if (step + 1) % config.saving.save_every_steps == 0 or (
                step + 1
            ) == config.training.max_steps:
                save_checkpoint(
                    model.state,
                    config.saving.checkpoint_dir,
                    keep=config.saving.num_keep_ckpts,
                )

    params = model.state.params
    x_jnp = jnp.asarray(x)
    y_jnp = jnp.asarray(y)
    ts_jnp = jnp.asarray(ts)

    @jax.jit
    def predict_all(times):
        return jax.vmap(lambda t: model.u_pred_fn(params, x_jnp, y_jnp, t))(times)

    u_preds = np.asarray(predict_all(ts_jnp))

    plot_results(
        x,
        y,
        ts,
        u_preds,
        save_dir=config.figure_path,
        prefix="train_solution",
        num_snapshots=10,
        log_to_wandb=True,
        wandb_key="solution_snapshots",
        wandb_step=step,
    )

    return model




def get_deformed_points(grid, t):
    """
    Apply a smooth deformation in the z-direction to the chart points.
    
    Args:
        chart_id: index of the chart to transform
        t: controls the strength of deformation (0.0 = no deformation)
    
    Returns:
        Transformed points
    """
    # Get the base points for this chart
    points = grid.copy()  # Make a copy to avoid modifying original data
    
    # Get x and y coordinates
    x = points[:, 0]
    y = points[:, 1]
    
    # Generate random frequencies (but still zero at boundaries)
    freq_x = np.array([1, 2])  # Random integer between 1 and 3
    freq_y = np.array([1, 2])
    
    # Create a 2D sine wave deformation in z-direction that is zero at the boundaries
    # Sum over various frequencies for a more complex deformation pattern
    deformation_z = np.zeros_like(x)
    for i in range(len(freq_x)):
        # Each term is zero when x=0, x=1, y=0, or y=1
        deformation_z += t * np.sin(freq_x[i] * np.pi * x) * np.sin(freq_y[i] * np.pi * y)
    
    # Apply the deformation to z coordinate
    points[:, 2] = deformation_z
    
    return points
