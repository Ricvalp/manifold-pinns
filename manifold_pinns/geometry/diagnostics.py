"""Reusable chart quality diagnostics."""

from __future__ import annotations

from typing import Callable

import jax.numpy as jnp

from .metrics import metric_batch


def chart_diagnostics(
    decoder: Callable[[jnp.ndarray], jnp.ndarray],
    z_batch: jnp.ndarray,
    x_target: jnp.ndarray | None = None,
) -> dict[str, jnp.ndarray]:
    """Return metric and optional reconstruction diagnostics for one chart."""

    metrics = metric_batch(decoder, z_batch)
    out = {
        "metric_cond_mean": jnp.mean(metrics.cond),
        "metric_cond_max": jnp.max(metrics.cond),
        "sqrt_det_g_min": jnp.min(metrics.sqrt_det_g),
        "sqrt_det_g_max": jnp.max(metrics.sqrt_det_g),
    }
    if x_target is not None:
        x_pred = jnp.stack([decoder(z) for z in z_batch])
        out["reconstruction_mse"] = jnp.mean((x_pred - x_target) ** 2)
    return out
