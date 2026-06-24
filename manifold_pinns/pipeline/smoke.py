"""Fast smoke checks used by the CLI and Makefile."""

from __future__ import annotations

import json
import time

import jax
import jax.numpy as jnp
import ml_collections
import optax
from flax.training import train_state

from manifold_pinns.geometry.operators import eikonal_residual
from universal_autoencoder.losses import make_loss_fn
from universal_autoencoder.monge import fit_pca_monge_chart


def _synthetic_monge_points(n: int = 9) -> jnp.ndarray:
    x = jnp.linspace(-0.5, 0.5, n)
    y = jnp.linspace(-0.5, 0.5, n)
    xx, yy = jnp.meshgrid(x, y)
    zz = 0.2 * xx**2 - 0.1 * yy**2
    return jnp.stack([xx.ravel(), yy.ravel(), zz.ravel()], axis=-1)


def run_uae_smoke(dataset_name: str, overrides=None) -> ml_collections.ConfigDict:
    """Fit a tiny PCA/Monge chart and verify UAE loss dispatch."""

    cfg = ml_collections.ConfigDict()
    cfg.dataset_name = dataset_name
    cfg.uae = ml_collections.ConfigDict()
    cfg.uae.architecture = "pca_monge"
    cfg.train = ml_collections.ConfigDict()
    cfg.train.reg = "geo+riemannian"
    cfg.train.loss_weights = ml_collections.ConfigDict()
    cfg.train.loss_weights.reconstruction = 1.0
    cfg.train.loss_weights.geodesic = 0.25
    cfg.train.loss_weights.riemannian = 0.5

    points = _synthetic_monge_points()
    chart = fit_pca_monge_chart(points)
    reconstruction_mse = chart.reconstruction_mse(points)

    loss_fn = make_loss_fn(
        cfg,
        reconstruction_loss=lambda params, batch, key: jnp.array(reconstruction_mse),
        geodesic_loss=lambda params, batch, key: jnp.array(0.25),
        riemannian_loss=lambda params, batch, key: jnp.array(0.5),
    )
    loss, aux = loss_fn(None, None, jax.random.PRNGKey(0))
    summary = {
        "smoke": "uae",
        "dataset": dataset_name,
        "architecture": cfg.uae.architecture,
        "loss": float(loss),
        "reconstruction_mse": float(aux["reconstruction"]),
        "geodesic": float(aux["geodesic"]),
        "riemannian": float(aux["riemannian"]),
    }
    print(json.dumps(summary, sort_keys=True))
    return cfg


def _linear_apply(params, z):
    return jnp.dot(params["w"], z) + params["b"]


def run_eikonal_smoke(config_name: str, overrides=None) -> ml_collections.ConfigDict:
    """Run one optimizer step for a synthetic flat-chart Eikonal problem."""

    cfg = ml_collections.ConfigDict()
    cfg.experiment = "eikonal"
    cfg.config_name = config_name
    cfg.runtime = ml_collections.ConfigDict()
    cfg.runtime.enable_x64 = False
    cfg.eikonal = ml_collections.ConfigDict()
    cfg.eikonal.enforce_source_bc = True
    cfg.eikonal.source_bc_weight = 1.0

    decoder = lambda z: jnp.array([z[0], z[1], 0.0])
    z_batch = jnp.stack(
        [
            jnp.linspace(0.0, 1.0, 16),
            jnp.linspace(0.0, 0.5, 16),
        ],
        axis=-1,
    )

    params = {"w": jnp.array([0.8, 0.2]), "b": jnp.array(0.1)}
    tx = optax.adam(1e-2)
    state = train_state.TrainState.create(
        apply_fn=lambda variables, z: _linear_apply(variables["params"], z),
        params=params,
        tx=tx,
    )

    def loss_fn(params):
        u_fn = lambda z: _linear_apply(params, z)
        residual = eikonal_residual(decoder, u_fn, z_batch)
        res_loss = jnp.mean(residual**2)
        source_loss = u_fn(jnp.array([0.0, 0.0])) ** 2
        total = res_loss + cfg.eikonal.source_bc_weight * source_loss
        return total, {"res": res_loss, "source": source_loss}

    step_start = time.perf_counter()
    (loss, aux), grads = jax.value_and_grad(loss_fn, has_aux=True)(state.params)
    state = state.apply_gradients(grads=grads)
    jax.tree.map(
        lambda x: x.block_until_ready() if hasattr(x, "block_until_ready") else x,
        state.params,
    )
    step_time_s = time.perf_counter() - step_start
    summary = {
        "smoke": "pinn",
        "experiment": "eikonal",
        "config": config_name,
        "loss": float(loss),
        "res_loss": float(aux["res"]),
        "source_bc_loss": float(aux["source"]),
        "step_time_s": step_time_s,
    }
    print(json.dumps(summary, sort_keys=True))
    return cfg


def run_pinn_smoke(experiment: str, config_name: str, overrides=None) -> ml_collections.ConfigDict:
    if experiment != "eikonal":
        raise ValueError("Smoke mode currently supports the eikonal PINN path.")
    return run_eikonal_smoke(config_name, overrides=overrides)
