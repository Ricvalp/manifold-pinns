"""Model-selection helpers for UAE architectures."""

from __future__ import annotations

from universal_autoencoder.monge import fit_pca_monge_atlas, fit_pca_monge_chart


def get_uae_architecture(cfg):
    """Return the configured UAE architecture name with legacy defaults."""

    if hasattr(cfg, "uae") and hasattr(cfg.uae, "architecture"):
        return cfg.uae.architecture
    return getattr(cfg, "architecture", "upt_siren")


def build_chart_model(cfg, charts=None):
    """Build a selectable chart model.

    ``upt_siren`` remains handled by the existing experiment code. ``pca_monge``
    returns fitted deterministic chart decoders and therefore requires charts.
    """

    architecture = get_uae_architecture(cfg)
    if architecture == "upt_siren":
        return None
    if architecture == "pca_monge":
        if charts is None:
            raise ValueError("pca_monge requires chart point clouds")
        if isinstance(charts, dict):
            return {key: fit_pca_monge_chart(value) for key, value in charts.items()}
        return fit_pca_monge_atlas(charts)
    raise ValueError(f"Unknown UAE architecture: {architecture}")
