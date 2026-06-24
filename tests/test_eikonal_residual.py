import jax.numpy as jnp
import numpy as np

from manifold_pinns.geometry.operators import eikonal_residual


def test_flat_eikonal_residual_for_linear_solution_is_zero():
    decoder = lambda z: jnp.array([z[0], z[1], 0.0])
    u_fn = lambda z: z[0]
    z = jnp.array([[0.0, 0.0], [0.2, 0.7], [1.0, -1.0]])
    residual = eikonal_residual(decoder, u_fn, z, target_norm=1.0)
    np.testing.assert_allclose(residual, jnp.zeros(z.shape[0]), atol=1e-6)
