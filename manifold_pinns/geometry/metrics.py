"""Metric utilities for two-dimensional surface charts."""

from __future__ import annotations

from typing import Callable, NamedTuple

import jax
import jax.numpy as jnp


class MetricBatch(NamedTuple):
    """Batched induced metric quantities for a chart decoder."""

    J: jnp.ndarray
    g: jnp.ndarray
    ginv: jnp.ndarray
    sqrt_det_g: jnp.ndarray
    det_g: jnp.ndarray
    trace_g: jnp.ndarray
    cond: jnp.ndarray


def inv_2x2_spd(g: jnp.ndarray, eps: float = 1e-8) -> jnp.ndarray:
    """Analytic inverse for batches of symmetric positive 2x2 matrices.

    Args:
        g: Array with trailing shape ``(2, 2)``.
        eps: Lower determinant clamp used to avoid NaNs for nearly singular
            learned chart metrics.

    Returns:
        Array with the same leading shape as ``g`` and trailing shape ``(2, 2)``.
    """

    g = jnp.asarray(g)
    a = g[..., 0, 0]
    b = 0.5 * (g[..., 0, 1] + g[..., 1, 0])
    c = g[..., 1, 1]
    det = jnp.maximum(a * c - b * b, eps)
    row0 = jnp.stack([c / det, -b / det], axis=-1)
    row1 = jnp.stack([-b / det, a / det], axis=-1)
    return jnp.stack([row0, row1], axis=-2)


def _metric_diagnostics(g: jnp.ndarray, eps: float) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    a = g[..., 0, 0]
    b = 0.5 * (g[..., 0, 1] + g[..., 1, 0])
    c = g[..., 1, 1]
    trace = a + c
    det = jnp.maximum(a * c - b * b, eps)
    disc = jnp.sqrt(jnp.maximum((a - c) ** 2 + 4.0 * b * b, 0.0))
    eig_min = jnp.maximum(0.5 * (trace - disc), eps)
    eig_max = jnp.maximum(0.5 * (trace + disc), eps)
    cond = eig_max / eig_min
    return det, trace, cond


def metric_batch(
    decoder: Callable[[jnp.ndarray], jnp.ndarray],
    z_batch: jnp.ndarray,
    *,
    eps: float = 1e-8,
) -> MetricBatch:
    """Compute decoder Jacobians and induced metric quantities once.

    The decoder must map a single chart coordinate ``z`` with shape ``(2,)`` to
    one ambient point with shape ``(3,)``. If a decoder needs conditioning,
    close over that conditioning before calling this function.
    """

    z_batch = jnp.asarray(z_batch)
    jac_single = jax.jacfwd(decoder)
    J = jax.vmap(jac_single)(z_batch)
    g = jnp.einsum("...ai,...aj->...ij", J, J)
    ginv = inv_2x2_spd(g, eps=eps)
    det_g, trace_g, cond = _metric_diagnostics(g, eps=eps)
    sqrt_det = jnp.sqrt(det_g)
    return MetricBatch(
        J=J,
        g=g,
        ginv=ginv,
        sqrt_det_g=sqrt_det,
        det_g=det_g,
        trace_g=trace_g,
        cond=cond,
    )
