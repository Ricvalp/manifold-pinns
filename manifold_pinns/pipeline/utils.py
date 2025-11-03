"""Shared helpers for configuring manifold PINN experiments."""

from typing import Any, Mapping

import ml_collections


def apply_overrides(config: ml_collections.ConfigDict, overrides: Mapping[str, Any]) -> None:
    """Recursively apply overrides to a ConfigDict.

    Args:
        config: Base configuration to mutate.
        overrides: Nested mapping with new values. Nested dictionaries must
            mirror the structure of ``config``.
    """
    for key, value in overrides.items():
        if isinstance(value, Mapping) and hasattr(config, key):
            apply_overrides(getattr(config, key), value)
        else:
            setattr(config, key, value)
