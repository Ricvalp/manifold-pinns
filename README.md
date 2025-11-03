# Manifold PINNs

End-to-end tooling for solving physics-informed neural networks (PINNs) on curved manifolds. The workflow is split into three explicit stages:

0. **Dataset generation** – extract surface patches for the universal autoencoder (UAE).
1. **UAE training** – learn a shared atlas of decoders.
2. **PINN training** – solve downstream PDEs on the learned manifolds.

This repository keeps experiment-specific code paths independent (copy-first philosophy) while providing light-weight orchestration utilities and documentation to glue the stages together.

## Environment

All commands below assume the repository root as the working directory and `PYTHONPATH=.`, e.g.

```bash
export PYTHONPATH=.
```

The project uses `jax`, `flax`, `ml_collections`, `optax`, `torch`, `wandb` and plotting dependencies.

### Dependency Installation

Create a fresh virtual environment (Python 3.10+ recommended) and install dependencies via `pip`:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
```

1. **PyTorch (CPU build)**

   ```bash
   pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
   ```

   This installs the latest stable CPU-only wheels from the official PyTorch repository.

2. **JAX with NVIDIA GPU support**

   Pick the wheel that matches your CUDA/cuDNN stack. For CUDA 12 (recommended for current NVIDIA drivers):

   ```bash
   pip install --upgrade "jax[cuda12_pip]" -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html
   ```

   For CUDA 11.8, swap the extra for `"jax[cuda11_pip]"`. See the [official JAX installation guide](https://github.com/google/jax#installation) for other platforms.

3. **Remaining Python packages**

   ```bash
   pip install absl-py flax ml-collections optax wandb matplotlib tqdm \
       networkx scikit-learn scipy trimesh
   ```

Verify your setup:

```bash
python -c "import jax; print(jax.devices())"
python -c "import torch; print(torch.__version__)"
```

## Quick Start via CLI

A new helper CLI wraps the three stages:

```bash
python -m manifold_pinns.pipeline.cli <command> [...]
```

### Step 0 – dataset generation

```bash
python -m manifold_pinns.pipeline.cli dataset bunny
python -m manifold_pinns.pipeline.cli dataset coil
python -m manifold_pinns.pipeline.cli dataset square
```

Override config values with dot-separated keys if needed:

```bash
python -m manifold_pinns.pipeline.cli dataset bunny \
  --override "dataset.iterations=50,dataset.points_per_unit_area=12"
```

### Step 1 – universal autoencoder training

```bash
python -m manifold_pinns.pipeline.cli uae bunny
python -m manifold_pinns.pipeline.cli uae coil
python -m manifold_pinns.pipeline.cli uae square
```

Pass `--create-dataset` to regenerate charts before training, or reuse the same `--override` flag to tweak hyperparameters:

```bash
python -m manifold_pinns.pipeline.cli uae coil \
  --create-dataset \
  --override "train.lr=5e-5,wandb.use=False"
```

### Step 2 – PINN experiments

```bash
# Eikonal on the coil dataset
python -m manifold_pinns.pipeline.cli pinn eikonal coil --mode train

# Wave equation on the square domain
python -m manifold_pinns.pipeline.cli pinn wave square --mode train

# Diffusion example
python -m manifold_pinns.pipeline.cli pinn diffusion square --mode train
```

Evaluation and data generation reuse the same entry point; just change `--mode` and optionally override paths:

```bash
python -m manifold_pinns.pipeline.cli pinn eikonal coil \
  --mode eval \
  --override "eval.checkpoint_dir=./pinns/eikonal/coil/checkpoints/latest"
```

## Convenience Scripts

Legacy shell scripts now delegate to the CLI while retaining their original interfaces:

| Stage | Script | Notes |
|-------|--------|-------|
| Step 0 | `./generate_uae_dataset.sh [bunny\|coil\|square]` | Generates UAE patches |
| Step 1 | `./train_uae.sh [bunny\|coil\|square]` | Trains the UAE |
| Step 2 | `./train_eikonal.sh [dataset [mode [chart]]]` | Trains/evaluates eikonal PINNs |
| Step 2 | `./eval_eikonal.sh [dataset [checkpoint [chart]]]` | Evaluates eikonal PINNs |
| Step 2 | `./train_wave.sh` | Trains the wave PINN |
| Step 2 | `./eval_wave.sh [checkpoint [step]]` | Evaluates the wave PINN |

The scripts remain copy-oriented; each experiment keeps bespoke configs, samplers and plotting utilities.

## Repository Layout

- `datasets/`: mesh utilities, dataset generators and analytic datasets. Added docstrings highlight entry points for patch creation.
- `universal_autoencoder/experiments/<dataset>/`: per-dataset UAE configurations with `run_experiment` helpers used by the CLI.
- `pinns/<experiment>/`: experiment-specific PINN stacks (diffusion, eikonal, wave) with independent configs and trainers.
- `manifold_pinns/pipeline/`: new orchestration helpers and CLI for the three-stage workflow.

## Tips

- Each stage logs to Weights & Biases when enabled. Disable with `--override "wandb.use=False"` or by editing the relevant config.
- Generated datasets live in `./datasets/<name>/`. Check the configs for exact filenames (e.g., `charts_1`, `uae_dataset`).
- Autoencoder checkpoints are stored under `universal_autoencoder/experiments/<dataset>/checkpoints/`.
- PINN checkpoints and figures are saved inside the corresponding `pinns/<experiment>/<dataset>/` folders.

## Sanity Check

To verify the repository after changes:

```bash
python -m compileall manifold_pinns pinns universal_autoencoder datasets
```

This ensures all Python modules import without syntax errors (run it inside your virtual environment).
