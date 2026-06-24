.PHONY: install test smoke-uae smoke-eikonal profile-eikonal benchmark-metric

install:
	pip install -e ./jaxpi
	pip install -e .

test:
	pytest -q
	python -m compileall manifold_pinns pinns universal_autoencoder datasets jaxpi

smoke-uae:
	python -m manifold_pinns.pipeline.cli uae coil --smoke --override "wandb.use=False"

smoke-eikonal:
	python -m manifold_pinns.pipeline.cli pinn eikonal coil --mode smoke --override "wandb.use=False"

profile-eikonal:
	python scripts/profile_eikonal_step.py

benchmark-metric:
	python scripts/benchmark_metric_batch.py
