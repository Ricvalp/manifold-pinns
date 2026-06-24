"""Utilities for Step 0 - generating UAE training datasets."""

from importlib import import_module
from typing import Any, Mapping, Optional

import ml_collections

from .utils import apply_overrides

_EXPERIMENTS = {
    "bunny": "universal_autoencoder.experiments.bunny.fit_universal_autoencoder_bunny",
    "coil": "universal_autoencoder.experiments.coil.fit_universal_autoencoder_coil",
    "square": "universal_autoencoder.experiments.square.fit_universal_autoencoder_square",
}


def get_module(dataset_name: str):
    """Return the dataset-specific experiment module."""
    if dataset_name not in _EXPERIMENTS:
        raise ValueError(f"Unknown dataset '{dataset_name}'.")
    return import_module(_EXPERIMENTS[dataset_name])


def build_config(dataset_name: str, overrides: Optional[Mapping[str, Any]] = None) -> ml_collections.ConfigDict:
    """Create a dataset-specific configuration with optional overrides."""
    module = get_module(dataset_name)
    cfg = module.load_cfgs()
    if overrides:
        apply_overrides(cfg, overrides)
    return cfg


def generate_dataset(dataset_name: str, overrides: Optional[Mapping[str, Any]] = None) -> ml_collections.ConfigDict:
    """Generate UAE training patches for the requested dataset.

    Args:
        dataset_name: Dataset identifier (e.g. ``"bunny"``).
        overrides: Optional partial config overrides.

    Returns:
        The configuration used to generate the dataset.
    """
    module = get_module(dataset_name)
    cfg = build_config(dataset_name, overrides)
    cfg.dataset.create_dataset = True
    module.run_experiment(cfg)
    return cfg
