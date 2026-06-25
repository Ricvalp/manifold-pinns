"""Environment-driven path helpers for portable experiment runs."""

from __future__ import annotations

import os
from pathlib import Path


def repo_root() -> Path:
    """Return the repository root, overridable for unusual launch contexts."""

    default = Path(__file__).resolve().parents[2]
    return _as_path(os.environ.get("MANIFOLD_PINNS_REPO_ROOT", str(default)))


def storage_root() -> Path:
    """Return the root for machine-local heavy experiment artifacts."""

    return _as_path(
        os.environ.get(
            "MANIFOLD_PINNS_STORAGE_ROOT",
            str(repo_root() / ".local_runs"),
        )
    )


def data_root() -> Path:
    return _root_from_env("MANIFOLD_PINNS_DATA_ROOT", "data")


def run_root() -> Path:
    return _root_from_env("MANIFOLD_PINNS_RUN_ROOT", "runs")


def checkpoint_root() -> Path:
    return _root_from_env("MANIFOLD_PINNS_CHECKPOINT_ROOT", "checkpoints")


def figure_root() -> Path:
    return _root_from_env("MANIFOLD_PINNS_FIGURE_ROOT", "figures")


def batch_root() -> Path:
    return _root_from_env("MANIFOLD_PINNS_BATCH_ROOT", "batches")


def profiler_root() -> Path:
    return _root_from_env("MANIFOLD_PINNS_PROFILER_ROOT", "profiler")


def eval_root() -> Path:
    return _root_from_env("MANIFOLD_PINNS_EVAL_ROOT", "eval")


def data_path(*parts: str) -> str:
    return str(data_root().joinpath(*parts))


def run_path(*parts: str) -> str:
    return str(run_root().joinpath(*parts))


def checkpoint_path(*parts: str) -> str:
    return str(checkpoint_root().joinpath(*parts))


def figure_path(*parts: str) -> str:
    return str(figure_root().joinpath(*parts))


def batch_path(*parts: str) -> str:
    return str(batch_root().joinpath(*parts))


def profiler_path(*parts: str) -> str:
    return str(profiler_root().joinpath(*parts))


def eval_path(*parts: str) -> str:
    return str(eval_root().joinpath(*parts))


def env_path(name: str, default: str | Path) -> str:
    """Read a path env var, falling back to a default path."""

    return str(_as_path(os.environ.get(name, str(default))))


def env_bool(name: str, default: bool = False) -> bool:
    raw = os.environ.get(name)
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on"}


def env_int(name: str, default: int) -> int:
    raw = os.environ.get(name)
    return default if raw in (None, "") else int(raw)


def wandb_entity() -> str | None:
    return os.environ.get("WANDB_ENTITY") or None


def wandb_project(default: str) -> str:
    return (
        os.environ.get("MANIFOLD_PINNS_WANDB_PROJECT")
        or os.environ.get("WANDB_PROJECT")
        or default
    )


def _root_from_env(name: str, default_child: str) -> Path:
    return _as_path(os.environ.get(name, str(storage_root() / default_child)))


def _as_path(value: str | Path) -> Path:
    return Path(os.path.expandvars(str(value))).expanduser().resolve()
