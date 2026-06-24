import jax.numpy as jnp
import numpy as np

from manifold_pinns.geometry.metrics import inv_2x2_spd, metric_batch
from manifold_pinns.geometry.operators import laplace_beltrami


def test_inv_2x2_spd_matches_linalg_inv():
    mats = jnp.array(
        [
            [[2.0, 0.2], [0.2, 1.0]],
            [[4.0, -0.5], [-0.5, 3.0]],
        ]
    )
    np.testing.assert_allclose(inv_2x2_spd(mats), jnp.linalg.inv(mats), rtol=1e-6)


def test_flat_chart_metric():
    decoder = lambda z: jnp.array([z[0], z[1], 0.0])
    z = jnp.array([[0.0, 0.0], [1.0, -2.0]])
    metrics = metric_batch(decoder, z)
    expected_j = jnp.array([[[1.0, 0.0], [0.0, 1.0], [0.0, 0.0]]] * 2)
    np.testing.assert_allclose(metrics.J, expected_j, rtol=1e-6)
    np.testing.assert_allclose(metrics.g, jnp.tile(jnp.eye(2), (2, 1, 1)), rtol=1e-6)
    np.testing.assert_allclose(metrics.ginv, jnp.tile(jnp.eye(2), (2, 1, 1)), rtol=1e-6)
    np.testing.assert_allclose(metrics.sqrt_det_g, jnp.ones(2), rtol=1e-6)


def test_flat_laplacian_polynomial_and_sine():
    decoder = lambda z: jnp.array([z[0], z[1], 0.0])
    z = jnp.array([[0.2, 0.3], [0.4, 0.5]])

    poly = lambda zz: zz[0] ** 2 + zz[1] ** 2
    np.testing.assert_allclose(laplace_beltrami(decoder, poly, z), 4.0, rtol=1e-5)

    sine = lambda zz: jnp.sin(jnp.pi * zz[0]) * jnp.sin(jnp.pi * zz[1])
    expected = -2.0 * jnp.pi**2 * jnp.array([sine(zz) for zz in z])
    np.testing.assert_allclose(laplace_beltrami(decoder, sine, z), expected, rtol=1e-5)


def test_monge_patch_metric():
    a = 0.3
    b = -0.2
    decoder = lambda z: jnp.array([z[0], z[1], a * z[0] ** 2 + b * z[1] ** 2])
    z = jnp.array([[0.5, -0.25], [-0.1, 0.3]])
    metrics = metric_batch(decoder, z)
    hz1 = 2.0 * a * z[:, 0]
    hz2 = 2.0 * b * z[:, 1]
    expected = jnp.stack(
        [
            jnp.stack([1.0 + hz1**2, hz1 * hz2], axis=-1),
            jnp.stack([hz1 * hz2, 1.0 + hz2**2], axis=-1),
        ],
        axis=-2,
    )
    np.testing.assert_allclose(metrics.g, expected, rtol=1e-6)
