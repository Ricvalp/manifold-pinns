"""JAX-native fixed-shape samplers for training hot paths."""

from __future__ import annotations

import functools

import jax
import jax.numpy as jnp


@functools.partial(jax.jit, static_argnames=("batch_size",))
def sample_residual_batch(
    key: jax.Array,
    coords: jnp.ndarray,
    *,
    batch_size: int,
    noise_std: float = 0.0,
) -> jnp.ndarray:
    """Sample residual coordinates from padded chart arrays.

    Args:
        key: JAX PRNG key.
        coords: Coordinate array with shape ``[C, N, D]``.
        batch_size: Static number of points per chart.
        noise_std: Optional Gaussian coordinate noise.
    """

    num_charts, num_points, _ = coords.shape
    key_idx, key_noise = jax.random.split(key)
    idx = jax.random.randint(
        key_idx,
        (num_charts, batch_size),
        minval=0,
        maxval=num_points,
    )
    batch = jnp.take_along_axis(coords, idx[..., None], axis=1)
    return batch + noise_std * jax.random.normal(key_noise, batch.shape)


@functools.partial(jax.jit, static_argnames=("batch_size",))
def sample_overlap_batch(
    key: jax.Array,
    z_src_all: jnp.ndarray,
    z_dst_all: jnp.ndarray,
    *,
    batch_size: int,
) -> tuple[jnp.ndarray, jnp.ndarray]:
    """Sample paired overlap coordinates with one index vector per edge."""

    if z_src_all.shape != z_dst_all.shape:
        raise ValueError("z_src_all and z_dst_all must have identical padded shapes")
    num_edges, num_points, _ = z_src_all.shape
    idx = jax.random.randint(
        key,
        (num_edges, batch_size),
        minval=0,
        maxval=num_points,
    )
    z_src = jnp.take_along_axis(z_src_all, idx[..., None], axis=1)
    z_dst = jnp.take_along_axis(z_dst_all, idx[..., None], axis=1)
    return z_src, z_dst
