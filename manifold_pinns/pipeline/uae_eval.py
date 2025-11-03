"""Evaluation utilities for UAE checkpoints."""

import json
from pathlib import Path
from typing import Any, Mapping, Optional

import jax
import ml_collections
import optax
from flax.training import checkpoints, train_state
from torch.utils.data import DataLoader

from .step0_dataset import build_config, get_module
from .utils import apply_overrides


def _load_cfg(
    dataset_name: str,
    run_id: str,
    overrides: Optional[Mapping[str, Any]] = None,
) -> ml_collections.ConfigDict:
    base_cfg = build_config(dataset_name)
    checkpoint_dir = Path(base_cfg.checkpoint.path) / run_id
    cfg_path = checkpoint_dir / "cfg.json"
    if cfg_path.exists():
        with open(cfg_path, "r") as f:
            cfg_dict = json.load(f)
        cfg = ml_collections.ConfigDict(cfg_dict)
    else:
        cfg = base_cfg
    if overrides:
        apply_overrides(cfg, overrides)
    cfg.dataset.create_dataset = False
    return cfg


def _create_optimizer(cfg: ml_collections.ConfigDict):
    if cfg.train.optimizer == "adam":
        return optax.adam(cfg.train.lr)
    if cfg.train.optimizer == "cosine_decay":
        lr_schedule = optax.warmup_cosine_decay_schedule(
            init_value=0.0,
            peak_value=cfg.train.lr,
            warmup_steps=50000,
            decay_steps=cfg.train.num_steps,
            end_value=cfg.train.lr / 100,
        )
        return optax.adam(lr_schedule)
    raise ValueError(f"Optimizer {cfg.train.optimizer} not supported for evaluation.")


def evaluate_autoencoder(
    dataset_name: str,
    run_id: str,
    overrides: Optional[Mapping[str, Any]] = None,
    *,
    checkpoint_step: Optional[int] = None,
    num_samples: int = 5,
    output_path: Optional[str] = None,
) -> float:
    """Evaluate a trained UAE checkpoint on the validation split.

    Args:
        dataset_name: Experiment identifier (bunny, coil, square).
        run_id: WandB/run folder to load checkpoints from.
        overrides: Optional configuration overrides.
        checkpoint_step: Optional checkpoint step. Defaults to the latest.
        num_samples: Number of samples to visualise.
        output_path: Optional file path for reconstruction plots.

    Returns:
        Reconstruction MSE on the evaluation batch.
    """
    module = get_module(dataset_name)
    cfg = _load_cfg(dataset_name, run_id, overrides)

    checkpoint_dir = (Path(cfg.checkpoint.path) / run_id).resolve()
    if not checkpoint_dir.exists():
        raise FileNotFoundError(
            f"Checkpoint directory {checkpoint_dir} not found. "
            "Verify run_id and dataset."
        )

    collate_fn = getattr(module, "numpy_collate")
    if dataset_name == "square":
        dataset_cls = module.UniversalAESquareDataset
        model_cls = module.UniversalAutoencoderGrid
    else:
        dataset_cls = module.UniversalAEDataset
        model_cls = module.UniversalAutoencoder
    decoder_cls = module.ModulatedSIREN

    val_dataset = dataset_cls(config=cfg, train=False)
    val_loader = DataLoader(
        dataset=val_dataset,
        batch_size=cfg.train.batch_size,
        shuffle=False,
        num_workers=0,
        collate_fn=collate_fn,
    )

    data_iter = iter(val_loader)
    batch = next(data_iter)
    if dataset_name == "square":
        init_points, supernode_idxs = batch
    else:
        init_points, supernode_idxs, _ = batch

    key = jax.random.PRNGKey(cfg.seed)
    key, params_subkey = jax.random.split(key)

    model = model_cls(cfg=cfg)
    decoder = decoder_cls(cfg=cfg)
    decoder_apply_fn = decoder.apply

    if dataset_name == "square":
        params = model.init(
            params_subkey,
            init_points,
            supernode_idxs,
            init_points[..., :2],
        )["params"]
    else:
        params = model.init(params_subkey, init_points, supernode_idxs)["params"]

    optimizer = _create_optimizer(cfg)
    state = train_state.TrainState.create(apply_fn=model.apply, params=params, tx=optimizer)

    state = checkpoints.restore_checkpoint(
        checkpoint_dir,
        target=state,
        step=checkpoint_step,
    )

    if output_path is None:
        eval_dir = Path(cfg.checkpoint.path) / "eval"
        eval_dir.mkdir(parents=True, exist_ok=True)
        output_path = eval_dir / f"{run_id}_reconstruction.png"
    else:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

    mse = module.test_reconstruction(
        state,
        val_loader,
        decoder_apply_fn,
        num_samples=num_samples,
        name=str(output_path),
    )

    print(f"Saved reconstruction plot to {output_path}")
    print(f"Validation reconstruction MSE: {mse:.6f}")
    return mse
