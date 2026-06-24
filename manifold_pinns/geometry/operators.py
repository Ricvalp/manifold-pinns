"""Intrinsic differential operators for chart coordinates."""

from __future__ import annotations

from typing import Callable

import jax
import jax.numpy as jnp

from .metrics import inv_2x2_spd, metric_batch


def ambient_surface_gradient(
    J: jnp.ndarray,
    ginv: jnp.ndarray,
    grad_z_u: jnp.ndarray,
) -> jnp.ndarray:
    """Convert chart-coordinate gradients to ambient tangent gradients."""

    tangent_coeffs = jnp.einsum("...ij,...j->...i", ginv, grad_z_u)
    return jnp.einsum("...ai,...i->...a", J, tangent_coeffs)


def eikonal_residual_from_grad(
    grad_z_u: jnp.ndarray,
    ginv: jnp.ndarray,
    *,
    target_norm: float | jnp.ndarray = 1.0,
) -> jnp.ndarray:
    """Return ``||grad_M u||^2 - target_norm^2`` for batched gradients."""

    sqnorm = jnp.einsum("...i,...ij,...j->...", grad_z_u, ginv, grad_z_u)
    return sqnorm - jnp.asarray(target_norm) ** 2


def eikonal_residual(
    decoder: Callable[[jnp.ndarray], jnp.ndarray],
    u_fn: Callable[[jnp.ndarray], jnp.ndarray],
    z_batch: jnp.ndarray,
    *,
    target_norm: float | jnp.ndarray = 1.0,
    eps: float = 1e-8,
) -> jnp.ndarray:
    """Evaluate the Eikonal residual on one chart."""

    metrics = metric_batch(decoder, z_batch, eps=eps)
    grad_u = jax.vmap(jax.grad(u_fn))(z_batch)
    return eikonal_residual_from_grad(
        grad_u,
        metrics.ginv,
        target_norm=target_norm,
    )


def _metric_terms_single(
    decoder: Callable[[jnp.ndarray], jnp.ndarray],
    z: jnp.ndarray,
    *,
    eps: float,
) -> tuple[jnp.ndarray, jnp.ndarray]:
    J = jax.jacfwd(decoder)(z)
    g = J.T @ J
    ginv = inv_2x2_spd(g, eps=eps)
    det = jnp.maximum(g[0, 0] * g[1, 1] - g[0, 1] * g[1, 0], eps)
    return ginv, jnp.sqrt(det)


def laplace_beltrami_single(
    decoder: Callable[[jnp.ndarray], jnp.ndarray],
    u_fn: Callable[[jnp.ndarray], jnp.ndarray],
    z: jnp.ndarray,
    *,
    eps: float = 1e-8,
) -> jnp.ndarray:
    """Compute the scalar Laplace-Beltrami operator at one chart point."""

    def flux_component(zz: jnp.ndarray, component: int) -> jnp.ndarray:
        ginv, sqrt_det = _metric_terms_single(decoder, zz, eps=eps)
        grad_u = jax.grad(u_fn)(zz)
        flux = sqrt_det * (ginv @ grad_u)
        return flux[component]

    div0 = jax.grad(lambda zz: flux_component(zz, 0))(z)[0]
    div1 = jax.grad(lambda zz: flux_component(zz, 1))(z)[1]
    _, sqrt_det = _metric_terms_single(decoder, z, eps=eps)
    return (div0 + div1) / sqrt_det


def laplace_beltrami(
    decoder: Callable[[jnp.ndarray], jnp.ndarray],
    u_fn: Callable[[jnp.ndarray], jnp.ndarray],
    z_batch: jnp.ndarray,
    *,
    eps: float = 1e-8,
) -> jnp.ndarray:
    """Vectorized Laplace-Beltrami operator for one chart."""

    return jax.vmap(lambda z: laplace_beltrami_single(decoder, u_fn, z, eps=eps))(
        z_batch
    )
