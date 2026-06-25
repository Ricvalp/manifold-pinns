"""Chart-geometry backends for Eikonal M-PINN experiments."""

from __future__ import annotations

import json
import pickle
from pathlib import Path
from typing import Any

import jax
import jax.numpy as jnp
import ml_collections
import numpy as np

from charts import get_metric_tensor_and_sqrt_det_g_universal_autodecoder, load_charts3d
from manifold_pinns.geometry.metrics import metric_batch
from universal_autoencoder.monge import fit_pca_monge_chart
from jaxpi.utils import load_config


def chart_backend(config: ml_collections.ConfigDict) -> str:
    chart_cfg = getattr(config, "chart", None)
    return getattr(chart_cfg, "backend", "uae") if chart_cfg is not None else "uae"


def prepare_chart_geometry(config: ml_collections.ConfigDict):
    """Load charts and construct metric functions for the requested backend."""

    backend = chart_backend(config)
    charts3d = _load_normalized_charts(config.dataset.charts_path)
    loaded_charts3d = charts3d["charts"]
    charts_mu = charts3d["mu"]
    charts_std = charts3d["std"]

    if backend == "uae":
        autoencoder_config = load_config(
            Path(config.autoencoder_checkpoint.checkpoint_path) / "cfg.json",
        )
        (
            inv_metric_tensor,
            sqrt_det_g,
            decoder,
        ), (conditionings, d_params) = get_metric_tensor_and_sqrt_det_g_universal_autodecoder(
            autoencoder_cfg=autoencoder_config,
            cfg=config,
            charts=loaded_charts3d,
            inverse=True,
        )
        _write_charts2d_metadata(Path(config.dataset.charts_path), backend="uae")
    elif backend == "pca_monge":
        (
            inv_metric_tensor,
            sqrt_det_g,
            decoder,
        ), (conditionings, d_params) = get_metric_tensor_and_sqrt_det_g_pca_monge(
            cfg=config,
            charts=loaded_charts3d,
            inverse=True,
        )
    else:
        raise ValueError(f"Unknown Eikonal chart backend '{backend}'.")

    return (
        loaded_charts3d,
        charts_mu,
        charts_std,
        inv_metric_tensor,
        sqrt_det_g,
        decoder,
        conditionings,
        d_params,
    )


def ensure_chart_coordinates(
    config: ml_collections.ConfigDict, *, for_plotting: bool = False
) -> None:
    """Ensure ``charts2d.pkl`` matches the configured chart backend."""

    backend = chart_backend(config)
    charts_path = Path(config.dataset.charts_path)
    charts2d_path = charts_path / "charts2d.pkl"
    regenerate = getattr(config.dataset, "regenerate_charts2d", False)
    existing_backend = _charts2d_backend(charts_path)
    has_matching_backend = existing_backend == backend
    is_legacy_uae_cache = backend == "uae" and existing_backend is None

    if (
        charts2d_path.exists()
        and not regenerate
        and (has_matching_backend or is_legacy_uae_cache)
    ):
        print(f"Using existing 2D chart coordinates: {charts2d_path}")
        return

    if (
        not for_plotting
        and backend == "uae"
        and charts2d_path.exists()
        and not regenerate
        and is_legacy_uae_cache
    ):
        print(f"Using existing 2D chart coordinates: {charts2d_path}")
        return

    prepare_chart_geometry(config)


def get_metric_tensor_and_sqrt_det_g_pca_monge(
    cfg: ml_collections.ConfigDict,
    charts: dict[int, np.ndarray],
    inverse: bool = False,
):
    """Fit PCA/Monge charts and return metric functions with the UAE signature."""

    del inverse

    models = {key: fit_pca_monge_chart(chart) for key, chart in charts.items()}
    coords = {
        key: models[key].encode(chart).astype(np.float32)
        for key, chart in charts.items()
    }

    charts_path = Path(cfg.dataset.charts_path)
    charts_path.mkdir(parents=True, exist_ok=True)
    with (charts_path / "charts2d.pkl").open("wb") as f:
        pickle.dump(coords, f)
    _write_charts2d_metadata(charts_path, backend="pca_monge")

    params = {
        "mu": jnp.asarray(
            np.stack([models[key].mu for key in sorted(models)]), dtype=jnp.float32
        ),
        "frame": jnp.asarray(
            np.stack([models[key].frame for key in sorted(models)]), dtype=jnp.float32
        ),
        "coeffs": jnp.asarray(
            np.stack([models[key].coeffs for key in sorted(models)]), dtype=jnp.float32
        ),
    }

    def decode_one(chart_params: dict[str, jnp.ndarray], z: jnp.ndarray) -> jnp.ndarray:
        z = jnp.asarray(z)
        z1 = z[..., 0]
        z2 = z[..., 1]
        features = jnp.stack(
            [
                jnp.ones_like(z1),
                z1,
                z2,
                z1**2,
                z1 * z2,
                z2**2,
            ],
            axis=-1,
        )
        h = features @ chart_params["coeffs"]
        local = jnp.stack([z1, z2, h], axis=-1)
        return chart_params["mu"] + local @ chart_params["frame"].T

    def induced_inverse_riemannian_metric(chart_params: dict[str, jnp.ndarray], z: jnp.ndarray):
        return metric_batch(lambda zz: decode_one(chart_params, zz), z).ginv

    def sqrt_det_g(chart_params: dict[str, jnp.ndarray], z: jnp.ndarray):
        return metric_batch(lambda zz: decode_one(chart_params, zz), z).sqrt_det_g

    return (
        jax.jit(induced_inverse_riemannian_metric),
        jax.jit(sqrt_det_g),
        PcaMongeDecoder(decode_one),
    ), (params, params)


class PcaMongeDecoder:
    """Minimal adapter matching the UAE decoder ``apply`` interface."""

    def __init__(self, decode_one):
        self._decode_one = decode_one

    def apply(self, variables: dict[str, Any], z: jnp.ndarray, chart_params: dict[str, jnp.ndarray]):
        del variables
        return jax.vmap(lambda zz: self._decode_one(chart_params, zz))(z)


def _load_normalized_charts(charts_path: str):
    loaded_charts3d, _, _, _ = load_charts3d(charts_path)
    charts_mu = np.zeros((len(loaded_charts3d.keys()), 3))
    charts_std = np.zeros((len(loaded_charts3d.keys()),))
    normalized = {}
    for key, chart in loaded_charts3d.items():
        chart = np.asarray(chart)
        mu = chart.mean(axis=0)
        std = chart.std()
        charts_mu[key] = mu
        charts_std[key] = std
        normalized[key] = (chart - mu) / std
    return {"charts": normalized, "mu": charts_mu, "std": charts_std}


def _charts2d_backend(charts_path: Path) -> str | None:
    metadata_path = charts_path / "charts2d_metadata.json"
    if not metadata_path.exists():
        return None
    return json.loads(metadata_path.read_text(encoding="utf-8")).get("backend")


def _write_charts2d_metadata(charts_path: Path, *, backend: str) -> None:
    metadata = {"backend": backend}
    (charts_path / "charts2d_metadata.json").write_text(
        json.dumps(metadata, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
