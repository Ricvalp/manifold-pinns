"""Utilities for Step 1 – training the universal autoencoder."""

from typing import Any, Mapping, Optional

import ml_collections

from .step0_dataset import build_config, get_module


def train_autoencoder(
    dataset_name: str,
    overrides: Optional[Mapping[str, Any]] = None,
    *,
    create_dataset: bool = False,
) -> ml_collections.ConfigDict:
    """Train the universal autoencoder for a specific dataset.

    Args:
        dataset_name: Dataset identifier (``"bunny"``, ``"coil"``, ``"square"``).
        overrides: Optional partial config overrides.
        create_dataset: If True, regenerate the dataset before training.

    Returns:
        The configuration that was used for training.
    """
    module = get_module(dataset_name)
    cfg = build_config(dataset_name, overrides)
    cfg.dataset.create_dataset = create_dataset
    module.run_experiment(cfg)
    return cfg
