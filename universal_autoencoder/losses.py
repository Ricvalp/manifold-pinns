"""Loss dispatch and chart-quality regularizers for UAE training."""

from __future__ import annotations

from typing import Callable

import jax
import jax.numpy as jnp

from manifold_pinns.geometry.metrics import metric_batch


ScalarLoss = Callable[[object, object, object | None], jnp.ndarray]


def _zero_like(value: jnp.ndarray) -> jnp.ndarray:
    return jnp.zeros((), dtype=jnp.asarray(value).dtype)


def make_loss_fn(
    cfg,
    *,
    reconstruction_loss: ScalarLoss,
    geodesic_loss: ScalarLoss | None = None,
    riemannian_loss: ScalarLoss | None = None,
    pde_chart_loss: ScalarLoss | None = None,
) -> Callable[[object, object, object | None], tuple[jnp.ndarray, dict[str, jnp.ndarray]]]:
    """Create a UAE objective from ``cfg.train.reg``.

    Component functions receive ``(params, batch, key)`` and return scalar JAX
    arrays. The returned aux dictionary uses stable component names for logging
    and tests.
    """

    reg = getattr(cfg.train, "reg", None)
    weights = getattr(cfg.train, "loss_weights", None)

    def weight(name: str, default: float) -> float:
        if weights is None:
            return default
        return float(getattr(weights, name, default))

    def loss_fn(params, batch, key=None):
        recon = reconstruction_loss(params, batch, key)
        geo = _zero_like(recon)
        riemann = _zero_like(recon)
        pde = _zero_like(recon)

        if reg in (None, "none", "reconstruction"):
            total = weight("reconstruction", 1.0) * recon
        elif reg == "geodesic_preservation":
            if geodesic_loss is None:
                raise ValueError("geodesic_preservation requires geodesic_loss")
            geo = geodesic_loss(params, batch, key)
            total = weight("reconstruction", 1.0) * recon + weight("geodesic", 3.0) * geo
        elif reg == "geo+riemannian":
            if geodesic_loss is None or riemannian_loss is None:
                raise ValueError("geo+riemannian requires geodesic and riemannian losses")
            geo = geodesic_loss(params, batch, key)
            riemann = riemannian_loss(params, batch, key)
            total = (
                weight("reconstruction", 1.0) * recon
                + weight("geodesic", 1.0) * geo
                + weight("riemannian", 1.0) * riemann
            )
        elif reg == "pde_chart":
            if pde_chart_loss is None:
                raise ValueError("pde_chart requires pde_chart_loss")
            pde = pde_chart_loss(params, batch, key)
            total = weight("reconstruction", 1.0) * recon + weight("pde_chart", 1.0) * pde
        else:
            raise ValueError(f"Unknown UAE regularizer: {reg}")

        aux = {
            "reconstruction": recon,
            "geodesic": geo,
            "riemannian": riemann,
            "pde_chart": pde,
            "total": total,
        }
        return total, aux

    return loss_fn


def sampled_geodesic_loss(
    dist_matrix: jnp.ndarray,
    z: jnp.ndarray,
    key: jax.Array,
    *,
    num_pairs: int = 4096,
) -> jnp.ndarray:
    """Pair-sampled geodesic preservation loss.

    Args:
        dist_matrix: Batched graph-geodesic distance matrices, ``[B, N, N]``.
        z: Batched chart coordinates, ``[B, N, 2]``.
        key: JAX PRNG key.
        num_pairs: Number of point pairs per chart.
    """

    batch_size, num_points, _ = z.shape
    key_i, key_j = jax.random.split(key)
    i = jax.random.randint(key_i, (batch_size, num_pairs), 0, num_points)
    j = jax.random.randint(key_j, (batch_size, num_pairs), 0, num_points)

    zi = jnp.take_along_axis(z, i[..., None], axis=1)
    zj = jnp.take_along_axis(z, j[..., None], axis=1)
    dz = jnp.linalg.norm(zi - zj, axis=-1)

    batch_idx = jnp.arange(batch_size)[:, None]
    dg = dist_matrix[batch_idx, i, j]

    dz = dz / (jnp.mean(dz, axis=-1, keepdims=True) + 1e-8)
    dg = dg / (jnp.mean(dg, axis=-1, keepdims=True) + 1e-8)
    return jnp.mean((dz - dg) ** 2)


def immersion_loss(
    decoder: Callable[[jnp.ndarray], jnp.ndarray],
    z_batch: jnp.ndarray,
    *,
    sigma_min_floor: float = 1e-3,
) -> jnp.ndarray:
    """Penalize decoder Jacobians whose smallest singular value collapses."""

    metrics = metric_batch(decoder, z_batch)
    singular_values = jnp.linalg.svd(metrics.J, compute_uv=False)
    sigma_min = jnp.min(singular_values, axis=-1)
    return jnp.mean(jnp.maximum(0.0, sigma_min_floor - sigma_min) ** 2)


def condition_loss(
    decoder: Callable[[jnp.ndarray], jnp.ndarray],
    z_batch: jnp.ndarray,
    *,
    eps: float = 1e-8,
) -> jnp.ndarray:
    """Log-condition penalty for induced chart metrics."""

    metrics = metric_batch(decoder, z_batch, eps=eps)
    return jnp.mean(jnp.log(metrics.cond + eps) ** 2)


def smoothness_loss(
    decoder: Callable[[jnp.ndarray], jnp.ndarray],
    z_batch: jnp.ndarray,
) -> jnp.ndarray:
    """Squared Frobenius norm of decoder Hessians."""

    hessian = jax.vmap(jax.hessian(decoder))(z_batch)
    return jnp.mean(jnp.sum(hessian**2, axis=(-3, -2, -1)))
