"""
Profile the UAE training step for the square dataset.

Runs a few training steps under JAX profiler in a subprocess
with a guaranteed POSIX locale, producing valid JSON trace files
for TensorBoard or chrome://tracing.
"""

import os
import sys
import locale
import subprocess
from pathlib import Path
from typing import Tuple

import jax
import jax.numpy as jnp
import optax
from flax.training import train_state
from torch.utils.data import DataLoader
from tqdm import tqdm

from datasets.uae_square import UniversalAESquareDataset
from universal_autoencoder.experiments.square.fit_universal_autoencoder_square import (
    count_parameters,
    load_cfgs,
    numpy_collate,
)
from universal_autoencoder.siren import ModulatedSIREN
from universal_autoencoder.upt_autoencoder_grid import UniversalAutoencoderGrid


# ----------------------------------------------------------------------------
# Core training helpers
# ----------------------------------------------------------------------------
def _initialise_state(cfg, model, batch, params_key):
    points, supernode_idxs = batch
    params = model.init(
        params_key,
        points,
        supernode_idxs,
        points[..., :2],
    )["params"]
    print(f"Number of parameters: {count_parameters(params)}")

    if cfg.train.optimizer == "adam":
        optimizer = optax.adam(learning_rate=cfg.train.lr)
    elif cfg.train.optimizer == "cosine_decay":
        lr_schedule = optax.warmup_cosine_decay_schedule(
            init_value=0.0,
            peak_value=cfg.train.lr,
            warmup_steps=10_000,
            decay_steps=cfg.train.num_steps,
            end_value=cfg.train.lr / 100,
        )
        optimizer = optax.adam(lr_schedule)
    else:
        raise ValueError(f"Unsupported optimizer {cfg.train.optimizer}")

    return train_state.TrainState.create(
        apply_fn=model.apply,
        params=params,
        tx=optimizer,
    )


@jax.jit
def _train_step(state, batch):
    points, supernode_idxs = batch
    coords = points[..., :2]

    def loss_fn(params):
        preds, _ = state.apply_fn({"params": params}, points, supernode_idxs, coords)
        recon_loss = jnp.sum((preds - points) ** 2, axis=-1).mean()
        return recon_loss

    loss, grads = jax.value_and_grad(loss_fn)(state.params)
    new_state = state.apply_gradients(grads=grads)
    return new_state, loss


# ----------------------------------------------------------------------------
# Actual profiling logic
# ----------------------------------------------------------------------------
def _run_profiling(profile_steps, trace_dir):
    """Inner function executed in the subprocess with a clean locale."""
    import time

    cfg = load_cfgs()
    cfg.dataset.create_dataset = False
    cfg.wandb.use = False

    dataset = UniversalAESquareDataset(config=cfg, train=True)
    data_loader = DataLoader(
        dataset=dataset,
        batch_size=cfg.train.batch_size,
        shuffle=False,
        num_workers=0,
        collate_fn=numpy_collate,
    )
    data_iter = iter(data_loader)

    model = UniversalAutoencoderGrid(cfg=cfg)

    key = jax.random.PRNGKey(cfg.seed)
    key, params_key = jax.random.split(key)
    init_batch = next(data_iter)
    state = _initialise_state(cfg, model, init_batch, params_key)

    os.makedirs(trace_dir, exist_ok=True)
    print(f"Profiler trace will be saved to {trace_dir}")

    batch = next(data_iter)
    state, loss = _train_step(state, batch)
    
    print("Profiling started...")

    with jax.profiler.trace(str(trace_dir)):

        progress = tqdm(range(profile_steps), desc="Profiling UAE (square)")
        for _ in progress:
            try:
                batch = next(data_iter)
            except StopIteration:
                data_iter = iter(data_loader)
                batch = next(data_iter)

            state, loss = _train_step(state, batch)
            # jax.block_until_ready(loss)
            progress.set_postfix(loss=float(loss))

    print(f"✅ Profiler trace written to {trace_dir}")


# ----------------------------------------------------------------------------
# Subprocess launcher to enforce C locale
# ----------------------------------------------------------------------------
def main(profile_steps: int = 5, trace_dir: Path = Path("profiling/square_uae")):
    trace_dir = trace_dir.resolve()

    # Check if we’re already in the subprocess
    if os.environ.get("MPINN_PROFILING_SUBPROCESS") == "1":
        _run_profiling(profile_steps, trace_dir)
        return

    print("Launching profiling subprocess with LC_ALL=C ...")

    env = os.environ.copy()
    env.update({
        "LC_ALL": "C",
        "LC_NUMERIC": "C",
        "MPINN_PROFILING_SUBPROCESS": "1",
    })

    cmd = [sys.executable, __file__, str(profile_steps), str(trace_dir)]
    subprocess.run(cmd, env=env, check=True)


if __name__ == "__main__":
    import sys

    if len(sys.argv) >= 3:
        steps = int(sys.argv[1])
        path = Path(sys.argv[2])
        _run_profiling(steps, path)
    else:
        main()
