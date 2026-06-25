.PHONY: install lock test smoke-uae smoke-eikonal profile-eikonal benchmark-metric

install:
	@uv sync

lock:
	@uv lock

test:
	@uv run pytest -q
	@uv run python -m compileall manifold_pinns pinns universal_autoencoder datasets jaxpi

smoke-uae:
	@uv run python -m manifold_pinns.pipeline.cli uae coil --smoke --override "wandb.use=False"

smoke-eikonal:
	@uv run python -m manifold_pinns.pipeline.cli pinn eikonal coil --mode smoke --override "wandb.use=False"

profile-eikonal:
	@uv run python scripts/profile_eikonal_step.py

benchmark-metric:
	@uv run python scripts/benchmark_metric_batch.py
