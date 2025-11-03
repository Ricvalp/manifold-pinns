"""Utilities for Step 2 – training or evaluating PINNs on learned manifolds."""

from importlib import import_module
from typing import Any, Callable, Dict, Mapping, Optional

import ml_collections

from .utils import apply_overrides

_PINN_EXPERIMENTS: Dict[str, Dict[str, Any]] = {
    "eikonal": {
        "train": ("pinns.eikonal.train", "train_and_evaluate"),
        "eval": ("pinns.eikonal.eval", "evaluate"),
        "generate_data": ("pinns.eikonal.generate_data", "generate_data"),
        "configs": "pinns.eikonal.configs",
    },
    "wave": {
        "train": ("pinns.wave.train", "train_and_evaluate"),
        "eval": ("pinns.wave.eval", "evaluate"),
        "generate_data": ("pinns.wave.generate_data", "generate_data"),
        "configs": "pinns.wave.configs",
    },
    "diffusion": {
        "train": ("pinns.diffusion.train", "train_and_evaluate"),
        "eval": ("pinns.diffusion.eval", "evaluate"),
        "generate_data": ("pinns.diffusion.generate_data", "generate_data"),
        "configs": "pinns.diffusion.configs",
    },
}


def _load_callable(experiment: str, mode: str) -> Callable[[ml_collections.ConfigDict], None]:
    """Load the callable implementing the requested experiment/mode."""
    exp = _PINN_EXPERIMENTS.get(experiment)
    if exp is None:
        raise ValueError(f"Unknown PINN experiment '{experiment}'.")
    if mode not in exp:
        raise ValueError(f"Mode '{mode}' not available for experiment '{experiment}'.")
    module_name, attr = exp[mode]
    module = import_module(module_name)
    return getattr(module, attr)


def _load_config(experiment: str, config_name: str) -> ml_collections.ConfigDict:
    """Load the configuration module for a given experiment."""
    exp = _PINN_EXPERIMENTS.get(experiment)
    if exp is None:
        raise ValueError(f"Unknown PINN experiment '{experiment}'.")
    config_module = import_module(f"{exp['configs']}.{config_name}")
    return config_module.get_config()


def run_pinn_experiment(
    experiment: str,
    config_name: str,
    mode: str = "train",
    overrides: Optional[Mapping[str, Any]] = None,
) -> ml_collections.ConfigDict:
    """Run one of the PINN experiments.

    Args:
        experiment: One of ``\"eikonal\"``, ``\"wave\"`` or ``\"diffusion\"``.
        config_name: Name of the config module to load.
        mode: Execution mode (``\"train\"`` | ``\"eval\"`` | ``\"generate_data\"``).
        overrides: Optional partial config overrides.

    Returns:
        The configuration that was used for the run.
    """
    runner = _load_callable(experiment, mode)
    cfg = _load_config(experiment, config_name)
    cfg.mode = mode
    if overrides:
        apply_overrides(cfg, overrides)
    runner(cfg)
    return cfg
