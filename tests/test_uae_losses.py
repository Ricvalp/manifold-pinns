import jax
import jax.numpy as jnp
import ml_collections

from universal_autoencoder.losses import make_loss_fn, sampled_geodesic_loss


def _cfg(reg):
    cfg = ml_collections.ConfigDict()
    cfg.train = ml_collections.ConfigDict()
    cfg.train.reg = reg
    cfg.train.loss_weights = ml_collections.ConfigDict()
    cfg.train.loss_weights.reconstruction = 1.0
    cfg.train.loss_weights.geodesic = 2.0
    cfg.train.loss_weights.riemannian = 3.0
    return cfg


def test_uae_loss_dispatch_changes_objective_and_aux_keys():
    components = {
        "reconstruction_loss": lambda params, batch, key: jnp.array(1.0),
        "geodesic_loss": lambda params, batch, key: jnp.array(2.0),
        "riemannian_loss": lambda params, batch, key: jnp.array(3.0),
    }
    recon_fn = make_loss_fn(_cfg("reconstruction"), **components)
    geo_fn = make_loss_fn(_cfg("geodesic_preservation"), **components)
    riemann_fn = make_loss_fn(_cfg("geo+riemannian"), **components)
    recon_loss, recon_aux = recon_fn(None, None, None)
    geo_loss, geo_aux = geo_fn(None, None, None)
    riemann_loss, riemann_aux = riemann_fn(None, None, None)

    assert float(recon_loss) == 1.0
    assert float(geo_loss) == 5.0
    assert float(riemann_loss) == 14.0
    assert {"reconstruction", "geodesic", "riemannian", "total"} <= set(riemann_aux)
    assert float(recon_aux["geodesic"]) == 0.0
    assert float(geo_aux["geodesic"]) == 2.0


def test_sampled_geodesic_loss_runs_on_tiny_batch():
    z = jnp.array([[[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]]])
    dist = jnp.array([[[0.0, 1.0, 1.0], [1.0, 0.0, 1.4], [1.0, 1.4, 0.0]]])
    loss = sampled_geodesic_loss(dist, z, jax.random.PRNGKey(0), num_pairs=8)
    assert jnp.isfinite(loss)
