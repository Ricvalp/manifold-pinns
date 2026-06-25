# AGENTS.md

## Operating Notes

- Before substantial work, read `RECAP.md`, then inspect `git status --short` and the relevant files before editing.
- After any substantial change or progress, update `RECAP.md` in the same turn. Include what changed, verification run, known caveats, and the next concrete step.
- Keep `RECAP.md` under 50 lines; compress older details instead of appending indefinitely.
- Use the refactored CLI path as the supported workflow: `uv run python -m manifold_pinns.pipeline.cli ...`.
- Keep heavy/generated artifacts out of the repo. Use `env.sh` locally and `env_snellius.sh` on Snellius so data, checkpoints, runs, figures, W&B files, caches, profiler traces, and eval outputs go through `MANIFOLD_PINNS_*` environment roots.
- Prefer `uv` for environment management. Run `uv sync` locally and `uv sync --extra cuda` on GPU/Snellius when CUDA JAX is needed.
- Preserve user work. Do not revert unrelated changes or use destructive git commands unless explicitly requested.
- For major code changes, run at least `uv run pytest -q` and `uv run python -m compileall manifold_pinns pinns universal_autoencoder datasets jaxpi scripts`.

## Project Direction

The implementation is being prepared for credible M-PINNs experiments: reliable chart/atlas geometry, stable metric/PDE operators, paired overlap losses, reproducible sparse Eikonal coil pilots, profiling, and eventual Snellius-scale runs. Favor correctness, reproducibility, and clear experiment artifacts over ad hoc script changes.
