from pathlib import Path

import jax
import jax.numpy as jnp
import ml_collections
import numpy as np

from jaxpi.utils import restore_checkpoint

from . import models
from .plot import plot_results
from .utils import get_last_checkpoint_dir


def _initial_conditions_spike(
    x: np.ndarray,
    y: np.ndarray,
    *,
    x0: float = 0.5,
    y0: float = 0.5,
    sigma: float = 0.1,
    amplitude: float = 30.0,
) -> np.ndarray:
    """Gaussian spike initial condition centered in the square domain."""
    return amplitude * np.exp(-((x - x0) ** 2 + (y - y0) ** 2) / sigma**2)


def evaluate(config: ml_collections.ConfigDict) -> None:
    """Evaluate a trained diffusion PINN checkpoint and plot predictions."""

    figure_path = Path(config.figure_path).expanduser().resolve()
    figure_path.mkdir(parents=True, exist_ok=True)

    # Recreate the reference grid used during training.
    x_lin = np.linspace(0.0, 1.0, 50)
    y_lin = np.linspace(0.0, 1.0, 50)
    xx, yy = np.meshgrid(x_lin, y_lin)

    coords = np.zeros((xx.size, 3))
    coords[:, 0] = xx.ravel()
    coords[:, 1] = yy.ravel()

    ts = np.linspace(0.0, 0.8, 200)
    x = coords[:, 0]
    y = coords[:, 1]
    u0 = _initial_conditions_spike(x, y)

    def _identity_metric(_, points: jnp.ndarray) -> jnp.ndarray:
        batch = points.shape[0]
        eye = jnp.eye(2)
        return jnp.broadcast_to(eye, (batch, 2, 2))

    def _unit_volume(_, points: jnp.ndarray) -> jnp.ndarray:
        return jnp.ones((points.shape[0],))

    conditionings = jnp.zeros((len(ts), 1))

    model = models.DiffusionTime(
        config,
        inv_metric_tensor=_identity_metric,
        sqrt_det_g=_unit_volume,
        conditionings=conditionings,
        ts=ts,
        ics=(x, y, u0),
    )

    # Locate the checkpoint to restore.
    if getattr(config.eval, "eval_with_last_ckpt", True):
        run_dir = get_last_checkpoint_dir(config.eval.checkpoint_dir)
        ckpt_path = (
            Path(config.eval.checkpoint_dir).expanduser() / run_dir
        ).resolve()
        restore_step = None
    else:
        ckpt_path = Path(config.eval.eval_checkpoint_dir).expanduser().resolve()
        restore_step = getattr(config.eval, "step", None)

    model.state = restore_checkpoint(model.state, ckpt_path, step=restore_step)

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
        save_dir=figure_path,
        prefix="eval_solution",
        num_snapshots=10,
        log_to_wandb=False,
    )
