# RECAP.md

Last updated: 2026-06-25

## State
Refactor workflow for M-PINNs experiments. Supported entry point:

```bash
uv run python -m manifold_pinns.pipeline.cli ...
```

Near-term plan is `EXPERIMENTS.md`: Day 0 smoke/profile, Day 1 geometry/overlap checks, Day 2 coil UAE checkpoint, Day 3 chart diagnostics, Day 4 Eikonal data dry run, Days 5-8 sparse coil Eikonal pilot.

## Recent Progress
- Set up `uv` workflow: `.python-version`, `uv.lock`, editable local `jaxpi`, `Makefile` via `uv run`.
- Added env-driven artifact roots in `manifold_pinns/pipeline/env.py`, plus `env.sh` and `env_snellius.sh`.
- Wired UAE/Eikonal configs to env roots for data, checkpoints, figures, batches, profiler, eval, W&B, caches.
- Fixed Eikonal `wandb.use=False`, stale eval plotting import, sparse point-ID saving, list-valued CLI overrides, and no-arg `wandb.run.save()` in UAE trainers.
- Added scripts: `check_geometry_analytic.py`, `check_overlap_pairs.py`, `run_chart_diagnostics.py`, `create_eikonal_atlas.py`.
- Coil UAE checkpoints: 50k `/mnt/external_storage/manifold-pinns/checkpoints/uae/coil/no_wandb_0`, step `49993`; 100k W&B `/mnt/external_storage/manifold-pinns/checkpoints/uae/coil/vqwx2biw`, step `99985`.
- Day 3 `pca_monge` chart diagnostics passed on 64 charts; max metric condition about `6.14`.
- Fixed Eikonal libigl call to use explicit `VS`/`VT`; generated Day 4 coil data now succeeds with cached `charts2d.pkl`, sparse IDs, and overlap batches.
- `generate_data` reuses matching `charts2d.pkl`; force refresh with `dataset.regenerate_charts2d=True`. Cached overlap batch count is `training.num_boundary_batches` default `500`.
- Sparse coil Eikonal pilots use exact IDs `[0,475,871,3122,3563,5914,7370,9841]`; Day 4 15k old-UAE eval corr `0.950`, RMSE `44.05`, rel L2 `0.195`, tail RMSE `113.6`.
- Added `chart.backend={uae,pca_monge}` for Eikonal train/eval/generate_data. `pca_monge` fits normalized chart PCA/Monge decoders, writes backend metadata, uses separate atlas/batch dirs, and supports exact sparse `idxs`.
- Correlation JSON now includes RMSE, relative L2, correlation, and top-10% tail metrics; added `scripts/compare_eikonal_metrics.py`.
- 100k-UAE + 50k Eikonal seed0: random N=8 RMSE `43.86`, tail `110.4`; stratified N=8 RMSE `28.87`, tail `69.69`; N=16 RMSE `11.12`, tail `4.05`; N=32 RMSE `11.57`, tail `5.97`.

## Verified
Passing:

```bash
uv run pytest -q
uv run python -m compileall manifold_pinns pinns universal_autoencoder datasets jaxpi scripts
uv run python scripts/check_geometry_analytic.py
uv run python scripts/check_overlap_pairs.py
uv run python -m manifold_pinns.pipeline.cli uae coil --smoke --override "wandb.use=False"
uv run python -m manifold_pinns.pipeline.cli pinn eikonal coil --mode smoke --override "wandb.use=False"
```

Result: `19 passed`; compileall passed. PCA backend verified with temp data; full 15k and 50k evaluations completed. Day 4 data shape remains `(500, 92, 3, 2, 128, 2)`.

## Resume
Local: `export MANIFOLD_PINNS_STORAGE_ROOT=/path/to/mount/manifold-pinns && source env.sh && uv sync`
Snellius: `export SNELLIUS_SCRATCH=/scratch-shared/$USER && source env_snellius.sh && uv sync --extra cuda`
Then set `PRELIM_RUN="$MANIFOLD_PINNS_RUN_ROOT/prelim_coil_refactor_20260624"`, `MANIFOLD_PINNS_UAE_COIL_CHECKPOINT="$MANIFOLD_PINNS_CHECKPOINT_ROOT/uae/coil/vqwx2biw"`, and `MANIFOLD_PINNS_UAE_COIL_STEP=99985`.

## Caveats / Next
- `pca_monge` is useful as a baseline but currently not competitive with the neural UAE for sparse coil Eikonal.
- Added sparse point strategies `random` and `stratified_geodesic` with saved `.npy` IDs plus JSON geodesic metadata. Next: multi-seed N=16, random-vs-stratified N=16, then Snellius-scale runs.
