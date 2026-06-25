# EXPERIMENTS.md

## Purpose

This document defines the first short experimental funnel after the refactor of `Ricvalp/manifold-pinns` on the `refactor` branch.

The goal is **not** to run the final paper-scale experiments yet. The goal is to test whether the refactored implementation and the new chart/atlas architecture are reliable enough to justify moving to the expensive paper experiments.

The target method is M-PINNs: partition a point-cloud surface into overlapping charts, construct local chart decoders, compute the induced metric

\[
g_i(z) = J_{d_i}(z)^\top J_{d_i}(z),
\]

and train local PINNs with intrinsic PDE residuals and overlap consistency. The paper-relevant preliminary experiment is the sparse Eikonal problem on the coil, because it directly tests the regime where the method is supposed to be useful: sparse observations, complex surface geometry, and small local PINNs.

This plan covers **days 0 to 8**:

1. environment and smoke checks;
2. analytic geometry/operator correctness tests;
3. overlap-pairing tests;
4. metric/runtime profiling;
5. short UAE/chart-diagnostic runs;
6. sparse Eikonal coil pilot.

Do not implement cardiac, biological membrane, moving-surface, deformed bunny, or full baseline experiments at this stage.

---

## Repository assumptions

Use the current refactor branch:

```bash
git checkout refactor
git pull
```

The refactored repository exposes the supported three-stage workflow through:

```bash
python -m manifold_pinns.pipeline.cli <command> ...
```

The stages are:

```text
0. dataset generation
1. UAE / chart model training
2. PINN training
```

The current README also exposes useful make targets:

```bash
make install
make test
make smoke-uae
make smoke-eikonal
make profile-eikonal
make benchmark-metric
```

Install from the repository root:

```bash
make install
```

or explicitly:

```bash
pip install -e ./jaxpi
pip install -e .
```

Disable Weights & Biases in all preliminary runs unless actively debugging online logging:

```bash
--override "wandb.use=False"
```

Keep the full run artifacts local and structured. Recommended root:

```text
runs/prelim_coil_refactor_<YYYYMMDD>/
```

Every experiment should save a machine-readable summary:

```text
metrics.json
config_resolved.json
stdout.log
stderr.log
```

For sweep experiments, also save one aggregate table:

```text
results.csv
```

Minimum metadata for every row in `results.csv`:

```text
run_id
experiment_name
dataset
architecture
seed
bcs_seed
num_train_points
num_charts
num_residual_points
max_steps
checkpoint_path
checkpoint_step
device
dtype
compile_time_s
train_wall_time_s
eval_wall_time_s
status
failure_reason
```

---

## Global rules for these preliminary experiments

### Use the refactored code path only

Do not run or compare against old scripts that bypass the current `manifold_pinns.pipeline.cli` entry point unless a current script explicitly delegates to that CLI.

### Keep the runs short

These are not final paper runs. Prefer:

```text
1 to 3 seeds for correctness/architecture checks
3 seeds for the sparse Eikonal pilot
10k to 50k UAE/PINN steps unless the run is obviously undertrained
```

Do not tune aggressively. These experiments should reveal gross failures, architectural ranking, and rough Eikonal behavior.

### Use consistent sparse data across methods

For the Eikonal pilot, every architecture must see the same sparse training points for a given `(N, bcs_seed)` pair. Save the exact global point IDs used for the sparse observations.

### Enforce the Eikonal source condition

The preliminary Eikonal experiment should be well-posed. Use:

```text
eikonal.enforce_source_bc=True
```

Run hard source ansatz only as an ablation after the soft source version works.

### Do not add external baselines yet

For days 0 to 8, the main comparison is among chart/atlas variants within the refactored M-PINN stack. Do not spend time integrating Delta-PINNs, PINNsur, graph methods, cardiac data, or external meshfree solvers yet.

A minimal global-coordinate PINN baseline can be included only if it is already implemented cleanly in the refactored codebase. Otherwise defer it.

---

# Day 0 — Installation, import, smoke, and profiling gate

## Goal

Verify that the refactored repository is runnable and that the core JAX geometry/PINN utilities compile. This is a bug gate, not an experiment for the paper.

## Commands

From the repository root:

```bash
make install
make test
```

Then run the fast smoke checks:

```bash
make smoke-uae
make smoke-eikonal
```

Run the synthetic profiling utilities:

```bash
make profile-eikonal | tee runs/day0_profile_eikonal.json
make benchmark-metric | tee runs/day0_benchmark_metric.json
```

Equivalent direct commands:

```bash
python scripts/profile_eikonal_step.py | tee runs/day0_profile_eikonal.json
python scripts/benchmark_metric_batch.py | tee runs/day0_benchmark_metric.json
```

## What to record

Create:

```text
runs/prelim_coil_refactor_<YYYYMMDD>/day0/
  test.log
  smoke_uae.log
  smoke_eikonal.log
  profile_eikonal.json
  benchmark_metric.json
  environment.txt
```

`environment.txt` should include:

```bash
python --version
pip freeze
git rev-parse HEAD
git status --short
python - <<'PY'
import jax
print("jax", jax.__version__)
print("devices", jax.devices())
PY
```

## Pass criteria

Day 0 passes only if:

```text
pytest passes
compileall passes
smoke UAE runs without NaNs
smoke Eikonal runs without NaNs
profile_eikonal_step.py emits valid JSON
benchmark_metric_batch.py emits valid JSON
runtime.enable_x64 is False unless explicitly overridden
no repeated JAX recompilation is visible in the smoke logs
```

If this fails, stop. Do not run chart diagnostics or Eikonal pilots before fixing it.

---

# Day 0–1 — Analytic geometry/operator correctness tests

## Goal

Test the geometric core independently of the UAE and PINN training code.

The M-PINN residuals depend on the decoder Jacobian and the pullback metric. If `metric_batch`, `eikonal_residual`, or `laplace_beltrami` is wrong, the downstream Eikonal experiment is not interpretable.

These tests should live as either pytest tests or small JSON-emitting scripts. Prefer both if time permits.

Recommended locations:

```text
tests/test_geometry_analytic.py
scripts/check_geometry_analytic.py
```

## Test A: flat plane metric

Use decoder:

\[
d(z_1,z_2) = (z_1,z_2,0).
\]

Expected quantities:

\[
J = \begin{bmatrix}1 & 0 \\ 0 & 1 \\ 0 & 0\end{bmatrix},
\quad
J^\top J = I_2,
\quad
\sqrt{\det g}=1,
\quad
\kappa(g)=1.
\]

Use a batch of random points:

```python
z = uniform([-1, 1]^2, shape=[4096, 2])
```

Measure:

```text
max_abs_g_minus_I
max_abs_ginv_minus_I
max_abs_sqrt_det_minus_1
max_abs_cond_minus_1
```

Pass thresholds:

```text
max_abs_g_minus_I < 1e-6
max_abs_ginv_minus_I < 1e-6
max_abs_sqrt_det_minus_1 < 1e-6
max_abs_cond_minus_1 < 1e-6
```

If running with non-float32 or on an unusual backend, relax to `1e-5`, but record the reason.

## Test B: flat plane Laplacian

Use the same plane decoder and scalar field:

\[
u(z_1,z_2) = z_1^2 + z_2^2.
\]

Expected:

\[
\Delta_{\mathcal M} u = 4.
\]

Measure:

```text
mean_abs_laplace_error
max_abs_laplace_error
```

Pass thresholds:

```text
mean_abs_laplace_error < 1e-5
max_abs_laplace_error < 1e-4
```

## Test C: flat plane Eikonal residual

Use scalar field:

\[
u(z_1,z_2) = z_1.
\]

Expected:

\[
\|\nabla_{\mathcal M} u\|^2 - 1 = 0.
\]

Measure:

```text
eikonal_residual_mean_abs
eikonal_residual_max_abs
```

Pass thresholds:

```text
eikonal_residual_mean_abs < 1e-6
eikonal_residual_max_abs < 1e-6
```

## Test D: analytic quadratic Monge patch metric

Use decoder:

\[
d(z_1,z_2) = (z_1,z_2,h(z_1,z_2)),
\]

with:

\[
h(z_1,z_2)=a z_1^2 + b z_1z_2 + c z_2^2.
\]

Suggested coefficients:

```text
a = 0.15
b = -0.07
c = 0.11
```

Analytic derivatives:

\[
h_{z_1}=2az_1+bz_2,
\quad
h_{z_2}=bz_1+2cz_2.
\]

Expected metric:

\[
g =
\begin{bmatrix}
1+h_{z_1}^2 & h_{z_1}h_{z_2} \\
h_{z_1}h_{z_2} & 1+h_{z_2}^2
\end{bmatrix}.
\]

Measure:

```text
mean_abs_metric_error
max_abs_metric_error
mean_abs_ginv_error
max_abs_ginv_error
sqrt_det_g_min
sqrt_det_g_max
cond_mean
cond_p95
cond_max
```

Pass thresholds:

```text
mean_abs_metric_error < 1e-6
max_abs_metric_error < 1e-5
mean_abs_ginv_error < 1e-5
max_abs_ginv_error < 1e-4
sqrt_det_g_min > 0
cond_max < 10
```

The condition-number threshold should be easy for this mild synthetic patch. If it fails, inspect `inv_2x2_spd` and `metric_batch` before proceeding.

## Output schema

Save one JSON file:

```text
runs/prelim_coil_refactor_<YYYYMMDD>/day1_geometry/geometry_analytic.json
```

Suggested structure:

```json
{
  "flat_metric": {
    "max_abs_g_minus_I": 0.0,
    "max_abs_ginv_minus_I": 0.0,
    "max_abs_sqrt_det_minus_1": 0.0,
    "max_abs_cond_minus_1": 0.0,
    "passed": true
  },
  "flat_laplacian": {
    "mean_abs_laplace_error": 0.0,
    "max_abs_laplace_error": 0.0,
    "passed": true
  },
  "flat_eikonal": {
    "eikonal_residual_mean_abs": 0.0,
    "eikonal_residual_max_abs": 0.0,
    "passed": true
  },
  "monge_metric": {
    "mean_abs_metric_error": 0.0,
    "max_abs_metric_error": 0.0,
    "mean_abs_ginv_error": 0.0,
    "max_abs_ginv_error": 0.0,
    "sqrt_det_g_min": 1.0,
    "sqrt_det_g_max": 1.1,
    "cond_mean": 1.0,
    "cond_p95": 1.1,
    "cond_max": 1.2,
    "passed": true
  }
}
```

---

# Day 1 — Overlap-pairing and interface-loss correctness

## Goal

Verify that interface losses compare the **same physical point** represented in two different chart coordinate systems.

This is essential. If overlap pairs are not aligned by global point ID or ambient nearest-neighbor matching, the interface loss may force unrelated points to have equal solution values.

The refactored code includes paired-overlap utilities under `manifold_pinns.geometry.overlaps`. Use them directly.

Recommended locations:

```text
tests/test_overlap_pairs.py
scripts/check_overlap_pairs.py
```

## Synthetic overlap fixture

Construct one simple ambient surface:

\[
x=(s,t,0).
\]

Create two chart coordinate systems over overlapping rectangles:

```text
chart A local coordinates: z_A = (s, t)
chart B local coordinates: z_B = R(theta) @ (s, t) + shift
```

Use a shared set of global point IDs for points in the overlap.

Expected:

```text
point_ids_A intersect point_ids_B gives paired rows
x_ambient_A[pair_idx] == x_ambient_B[pair_idx]
sample_overlap_pairs uses one shared index vector
```

## Interface value test

Define a scalar function in ambient coordinates:

\[
u(x,y,z)=2x-y+0.3.
\]

Define chart-local evaluators:

```python
u_A(z_A) = u(d_A(z_A))
u_B(z_B) = u(d_B(z_B))
```

On paired overlap points:

```text
interface_loss(u_A, u_B, pairs) should be approximately zero
```

Pass threshold:

```text
interface_loss < 1e-10 for NumPy float64 fixture
interface_loss < 1e-6 for JAX/float32 fixture
```

## Negative control

Shuffle `z_dst` while keeping `z_src` fixed. The interface loss should increase.

Save:

```text
paired_interface_loss
shuffled_interface_loss
ratio = shuffled_interface_loss / paired_interface_loss
```

Pass threshold:

```text
paired_interface_loss < 1e-6
shuffled_interface_loss > 1e-3
```

The exact shuffled threshold can vary with fixture scale. The important property is that the paired version is near zero and the shuffled version is not.

## Output

Save:

```text
runs/prelim_coil_refactor_<YYYYMMDD>/day1_overlap/overlap_check.json
```

Suggested JSON:

```json
{
  "num_pairs": 512,
  "max_ambient_pair_error": 0.0,
  "paired_interface_loss": 0.0,
  "shuffled_interface_loss": 0.01,
  "passed": true
}
```

Stop if this fails. Do not train sparse Eikonal until overlap pairing is correct.

---

# Day 1–2 — Metric and Eikonal step profiling

## Goal

Establish basic performance numbers before running longer pilots. These are not final speed claims; they are regression guards.

Use the existing scripts:

```bash
python scripts/profile_eikonal_step.py \
  | tee runs/prelim_coil_refactor_<YYYYMMDD>/day2_profile/profile_eikonal_step.json

python scripts/benchmark_metric_batch.py \
  | tee runs/prelim_coil_refactor_<YYYYMMDD>/day2_profile/benchmark_metric_batch.json
```

## Additional recommended profiling script

Add a small script if not already present:

```text
scripts/benchmark_chart_architectures.py
```

This should compare the candidate chart decoders on identical synthetic batches:

```text
pca_monge
current neural UAE architecture
any new architecture added by the refactor
```

Metrics:

```text
architecture
num_charts
num_points_per_chart
compile_time_s
decode_time_ms_mean
decode_time_ms_std
metric_time_ms_mean
metric_time_ms_std
memory_peak_mb, if easy
metric_cond_mean
metric_cond_p95
metric_cond_max
sqrt_det_g_min
sqrt_det_g_max
```

Use a small batch:

```text
num_charts = 16
num_points_per_chart = 256 or 512
```

This should finish in minutes.

## Pass criteria

No hard timing threshold yet, because hardware varies. The pass criteria are:

```text
valid JSON output
no NaNs
metric_cond_max finite
sqrt_det_g_min positive
steady-state step time substantially lower than compile time
no obvious recompilation per step
```

If `metric_cond_max` is infinite or extremely large on simple synthetic charts, inspect chart Jacobians and decoder regularization before proceeding.

---

# Day 2 — Small coil dataset and short chart-model training

## Goal

Produce a small coil dataset and a short trained chart model sufficient for diagnostics. This is not the final UAE training.

## Generate or refresh a small coil UAE dataset

Use the CLI from the repository root.

Suggested command:

```bash
python -m manifold_pinns.pipeline.cli dataset coil \
  --override "dataset.iterations=20,dataset.points_per_unit_area=4,dataset.num_points=512,dataset.num_files=1,dataset.save_charts_every=10"
```

If the dataset CLI uses slightly different config keys, keep the intent:

```text
small number of generated charts
512 points per chart if possible
low points-per-unit-area
single generated file
fast enough to regenerate in under one hour
```

## Run the UAE smoke check

```bash
python -m manifold_pinns.pipeline.cli uae coil \
  --smoke \
  --override "wandb.use=False"
```

## Short training for the current neural chart architecture

Run a short chart-model training. Suggested first run:

```bash
python -m manifold_pinns.pipeline.cli uae coil \
  --override "wandb.use=False,train.num_steps=10000,train.batch_size=16,checkpoint.save_every=10000,dataset.num_points=512,encoder_supernodes_cfg.max_degree=32"
```

If 10k steps is too short to see meaningful reconstruction, run 25k or 50k. Do not run the 3M-step setting yet.

Save the resulting checkpoint path and step in:

```text
runs/prelim_coil_refactor_<YYYYMMDD>/day2_uae/uae_short_checkpoint.txt
```

## Optional short training with derivative-aware losses

If the refactor exposes loss weights for immersion/conditioning/smoothness, run one short variant:

```bash
python -m manifold_pinns.pipeline.cli uae coil \
  --override "wandb.use=False,train.num_steps=10000,train.batch_size=16,checkpoint.save_every=10000,dataset.num_points=512,encoder_supernodes_cfg.max_degree=32,train.loss_weights.immersion=0.01,train.loss_weights.condition=0.001,train.loss_weights.smooth=0.0001"
```

Only run this if the losses are implemented and tested. Otherwise skip.

## Output

Save:

```text
runs/prelim_coil_refactor_<YYYYMMDD>/day2_uae/
  dataset_generation.log
  smoke_uae.log
  uae_short_train.log
  uae_short_checkpoint.txt
  resolved_config.json
```

---

# Day 2–4 — Chart architecture diagnostics on the coil

## Goal

Rank the available chart architectures by geometric quality and speed before using them inside Eikonal PINNs.

The winning architecture is not necessarily the one with the best reconstruction error. The important downstream quantity is whether the decoder produces a stable metric for PDE residuals.

## Architectures to include

Include only architectures available in the refactored codebase.

Recommended minimum:

```text
pca_monge
current neural UAE architecture
```

If the refactor added a new architecture, include it under its actual config name.

Use the config key exposed by the refactor. The current coil UAE config uses:

```text
uae.architecture
```

Expected names may be similar to:

```text
upt_siren
pca_monge
```

If the exact names differ, use the implementation names and record them in `config_resolved.json`.

## Diagnostic script

Create or use:

```text
scripts/run_chart_diagnostics.py
```

Suggested command shape:

```bash
python scripts/run_chart_diagnostics.py \
  --dataset coil \
  --architecture pca_monge \
  --charts-path ./datasets/coil/uae_dataset \
  --max-charts 64 \
  --num-points 512 \
  --output runs/prelim_coil_refactor_<YYYYMMDD>/day3_chart_diagnostics/pca_monge.json
```

For neural UAE checkpoints:

```bash
python scripts/run_chart_diagnostics.py \
  --dataset coil \
  --architecture <neural_architecture_name> \
  --charts-path ./datasets/coil/uae_dataset \
  --checkpoint <PATH_FROM_DAY2> \
  --checkpoint-step <STEP_FROM_DAY2> \
  --max-charts 64 \
  --num-points 512 \
  --output runs/prelim_coil_refactor_<YYYYMMDD>/day3_chart_diagnostics/<arch>.json
```

If this script does not exist yet, it should be small. It should load a set of chart point clouds, construct or load the decoder for each chart, sample/evaluate local coordinates, and call the shared geometry diagnostics.

## Required metrics

For each chart and each architecture, compute:

```text
reconstruction_mse
reconstruction_rmse
metric_cond_mean
metric_cond_median
metric_cond_p95
metric_cond_max
sqrt_det_g_min
sqrt_det_g_median
sqrt_det_g_max
min_singular_value_J_min
min_singular_value_J_median
min_singular_value_J_p05
max_singular_value_J_p95
decode_time_ms_mean
metric_time_ms_mean
failure_flag
failure_reason
```

If normals are available:

```text
normal_angle_error_mean_deg
normal_angle_error_median_deg
normal_angle_error_p95_deg
```

If paired overlaps are available:

```text
overlap_reconstruction_error_mean
overlap_reconstruction_error_p95
```

## Recommended aggregation

Save both per-chart and aggregate outputs:

```text
runs/prelim_coil_refactor_<YYYYMMDD>/day3_chart_diagnostics/
  chart_metrics_<architecture>.csv
  aggregate_chart_metrics.csv
  <architecture>.json
```

`aggregate_chart_metrics.csv` should contain one row per architecture:

```text
architecture
num_charts
num_failures
reconstruction_rmse_mean
reconstruction_rmse_std
metric_cond_median
metric_cond_p95
metric_cond_max
sqrt_det_g_min
sqrt_det_g_max
min_singular_value_J_p05
decode_time_ms_mean
metric_time_ms_mean
```

## Suggested plots

Create:

```text
metric_cond_boxplot.png
reconstruction_vs_cond_scatter.png
sqrt_det_g_histograms.png
chart_reconstruction_examples_<architecture>.png
```

Do not over-polish. These are internal decision plots.

## Decision criteria

Prefer an architecture for the Eikonal pilot if:

```text
no chart failures on the 64-chart diagnostic subset
metric_cond_p95 < 1e4, preferably < 1e3
sqrt_det_g_min > 1e-6
min_singular_value_J_p05 > 1e-4
reconstruction_rmse is not obviously worse than other contenders
metric_time_ms_mean is acceptable
```

If `pca_monge` has much better metric conditioning and only moderately worse reconstruction, use it in the Eikonal pilot. That is a valid and useful result.

If the neural UAE reconstructs well but has many high-condition-number charts, do not use it in sparse Eikonal until regularization or coordinate scaling is fixed.

---

# Day 4 — Short end-to-end dry run on coil Eikonal

## Goal

Run one complete but short M-PINN Eikonal training on the coil using the best chart architecture from diagnostics.

This is a dry run to verify data generation, checkpoint loading, sparse observations, source boundary condition, residual sampling, overlap losses, evaluation, and output writing.

## Generate Eikonal data if needed

Use the current CLI:

```bash
python -m manifold_pinns.pipeline.cli pinn eikonal coil \
  --mode generate_data \
  --override "wandb.use=False,N=8,bcs_seed=0,eikonal.enforce_source_bc=True"
```

If `generate_data` does not depend on `N`, still pass it for reproducibility and record the resolved config.

## Dry-run training command

Use a short number of steps:

```bash
python -m manifold_pinns.pipeline.cli pinn eikonal coil \
  --mode train \
  --override "wandb.use=False,N=8,seed=0,bcs_seed=0,runtime.enable_x64=False,eikonal.enforce_source_bc=True,eikonal.hard_source_ansatz=False,training.max_steps=5000,training.batch_size=128,logging.eval_every_steps=1000,saving.save_every_steps=5000"
```

If the run uses a neural UAE checkpoint, include:

```text
autoencoder_checkpoint.checkpoint_path=<PATH>
autoencoder_checkpoint.step=<STEP>
```

If the run uses `pca_monge`, include the implementation-specific architecture override, for example:

```text
uae.architecture=pca_monge
```

or the actual refactored config key for the chart decoder.

## Required checks

The dry run passes if:

```text
training starts without recompilation every iteration
loss is finite for all logged steps
source boundary loss is finite and decreases or remains controlled
residual loss is finite
interface loss is finite
checkpoint is saved
evaluation runs on the saved checkpoint
metrics are written to JSON/CSV
```

Do not judge final accuracy from this 5k-step run.

## Output

Save:

```text
runs/prelim_coil_refactor_<YYYYMMDD>/day4_eikonal_dryrun/
  train.log
  eval.log
  metrics.json
  resolved_config.json
  checkpoint.txt
```

---

# Days 5–8 — Sparse Eikonal coil pilot

## Goal

Run the first paper-shaped pilot experiment.

This tests whether the refactored M-PINN stack can solve the Eikonal equation on the coil in the sparse-data regime and whether the selected chart architecture behaves well downstream.

The equation is:

\[
\|\nabla_{\mathcal M} u\| = 1,
\quad
u(x_b)=0.
\]

The solution is the geodesic distance from the source/base point.

Unlike the old paper setup, this pilot should enforce the source condition. Sparse observations are used to anchor the solution and test data efficiency, not to compensate for an intentionally omitted boundary/source condition.

## Experimental matrix

Use a small but informative grid.

### Sparse data counts

```text
N_train_points ∈ {4, 8, 16, 32}
```

### Seeds

Use three seeds:

```text
seed ∈ {0, 1, 2}
bcs_seed ∈ {0, 1, 2}
```

Recommended pairing:

```text
seed = bcs_seed
```

This keeps the first pilot simple. Later, separate initialization variability from sparse-point variability.

### Architectures

Include the best one or two architectures from Day 2–4 diagnostics.

Recommended:

```text
best_chart_architecture
second_best_chart_architecture, only if it is not too expensive
```

For example:

```text
pca_monge
neural_uae_short
```

Do not run more than two architectures in this pilot.

### Chart counts

Start with the current default chart count from the coil dataset/config. If chart count is easily configurable and already stable, run:

```text
num_charts ∈ {20, 40}
```

If changing chart count requires regenerating large datasets or touching code, use only the current default and record it.

### Training steps

Use:

```text
training.max_steps = 50000
```

If training is clearly underconverged but stable, allow one extension to:

```text
training.max_steps = 100000
```

Do not exceed 100k steps in this preliminary pilot.

### Batch size

Start with:

```text
training.batch_size = 128
```

If the run is too slow and memory is available, try:

```text
training.batch_size = 256
```

Keep the same batch size across architecture variants unless one architecture fails due to memory.

## Sparse point selection

For each `(N_train_points, bcs_seed)` pair, save the sparse observation point IDs:

```text
runs/prelim_coil_refactor_<YYYYMMDD>/day5_8_sparse_eikonal/data_points/N<N>_seed<S>.npy
```

Sampling rule:

1. Always include the source point only as a source/boundary condition, not as a normal sparse observation unless the code requires it.
2. Sample the remaining observation points from non-source points.
3. Prefer stratified sampling by ground-truth geodesic distance quantiles if easy.
4. If stratified sampling is not implemented, use uniform random sampling with fixed `bcs_seed`.
5. Use the same sparse point IDs for every architecture.

Recommended stratified scheme:

```text
Divide non-source points into N quantile bins by geodesic distance.
Sample one point from each bin.
```

For `N=4`, use four broad quantiles. For `N=32`, use 32 quantiles or 8 quantiles with 4 samples per bin.

If the current code already has a sampling convention, use it for this pilot but save the sampled IDs.

## Ground truth

Use the existing coil ground-truth geodesic solution if available.

Record:

```text
ground_truth_path
ground_truth_type
source_idx
source_ambient_coordinate
num_eval_points
```

For evaluation, use all available coil points if feasible. Otherwise use a fixed evaluation subset:

```text
num_eval_points = 5000 or 10000
```

The same evaluation points must be used for every architecture and seed.

## Training command template

For each architecture, `N`, and seed:

```bash
python -m manifold_pinns.pipeline.cli pinn eikonal coil \
  --mode train \
  --override "wandb.use=False,N=<N>,seed=<SEED>,bcs_seed=<SEED>,runtime.enable_x64=False,eikonal.enforce_source_bc=True,eikonal.hard_source_ansatz=False,training.max_steps=50000,training.batch_size=128,logging.eval_every_steps=1000,saving.save_every_steps=50000,logging.log_errors=True,logging.log_losses=True"
```

Add architecture-specific overrides.

For a neural UAE checkpoint:

```text
autoencoder_checkpoint.checkpoint_path=<PATH_FROM_DAY2_OR_DAY3>
autoencoder_checkpoint.step=<STEP>
```

For PCA/Monge:

```text
<chart_architecture_key>=pca_monge
```

Use the exact config key implemented in the refactor.

Recommended run ID format:

```text
eikonal_coil__arch_<ARCH>__N_<N>__seed_<SEED>
```

## Evaluation command template

After each run, evaluate the checkpoint:

```bash
python -m manifold_pinns.pipeline.cli pinn eikonal coil \
  --mode eval \
  --override "wandb.use=False,N=<N>,seed=<SEED>,bcs_seed=<SEED>,eval.eval_with_last_ckpt=True,eval.N=5000,eval.use_existing_solution=False,eval.plot_everything=False"
```

If evaluation requires an explicit checkpoint directory:

```bash
python -m manifold_pinns.pipeline.cli pinn eikonal coil \
  --mode eval \
  --override "wandb.use=False,N=<N>,seed=<SEED>,bcs_seed=<SEED>,eval.checkpoint_dir=<CHECKPOINT_DIR>,eval.step=<STEP>,eval.N=5000,eval.plot_everything=False"
```

## Metrics to report

For each run, compute and save:

```text
rmse
mae
mse
relative_l2
correlation
source_bc_abs_error
source_bc_squared_error
pde_residual_mse
pde_residual_mae
interface_value_mse
interface_gradient_mse, if implemented
train_data_mse
heldout_data_mse
metric_cond_median_on_residual_points
metric_cond_p95_on_residual_points
metric_cond_max_on_residual_points
sqrt_det_g_min_on_residual_points
sqrt_det_g_max_on_residual_points
compile_time_s
train_wall_time_s
eval_wall_time_s
step_time_ms_mean
memory_peak_mb, if easy
num_nan_steps
status
failure_reason
```

At minimum, the pilot needs:

```text
rmse
relative_l2
correlation
source_bc_abs_error
pde_residual_mse
interface_value_mse
train_wall_time_s
```

## Aggregated outputs

Create:

```text
runs/prelim_coil_refactor_<YYYYMMDD>/day5_8_sparse_eikonal/
  results_raw.csv
  results_aggregate.csv
  rmse_vs_N.png
  corr_vs_N.png
  rel_l2_vs_N.png
  time_vs_N.png
  pred_vs_gt_examples.png
  residual_hist_examples.png
```

`results_raw.csv` has one row per run.

`results_aggregate.csv` has one row per `(architecture, N_train_points)`:

```text
architecture
num_train_points
num_runs
rmse_mean
rmse_std
rmse_sem
relative_l2_mean
relative_l2_std
correlation_mean
correlation_std
source_bc_abs_error_mean
pde_residual_mse_mean
interface_value_mse_mean
train_wall_time_s_mean
num_failures
```

Use standard deviation for error bars in preliminary plots. Later, for the paper, decide whether to show standard deviation, standard error, or confidence intervals.

## Plot requirements

### Plot 1: RMSE vs number of sparse points

```text
x-axis: N_train_points
 y-axis: RMSE
line: architecture
error bars: std over seeds
```

Use log scale for RMSE if values span orders of magnitude.

### Plot 2: correlation vs number of sparse points

```text
x-axis: N_train_points
 y-axis: correlation with ground truth
line: architecture
error bars: std over seeds
```

### Plot 3: relative L2 vs number of sparse points

Same structure.

### Plot 4: prediction vs ground truth scatter

For one representative seed per architecture, plot predicted distance against ground truth distance for:

```text
N=8
N=16
N=32
```

Mark sparse training points in a different marker.

### Plot 5: residual histogram

For the same representative runs, show the distribution of Eikonal residuals on evaluation points.

This helps detect the failure mode where PDE residual is low but the solution is globally wrong.

## Pass criteria

The sparse Eikonal coil pilot is successful if:

```text
all runs complete without NaNs for N >= 8
correlation generally increases with N
RMSE or relative L2 generally decreases with N
correlation at N=16 is preferably > 0.90 for the best architecture
correlation at N=32 is preferably > 0.95 for the best architecture
source_bc_abs_error is small relative to the solution scale
interface_value_mse is not dominating the loss
metric_cond_p95_on_residual_points remains finite and preferably < 1e4
```

Do not overinterpret `N=4`. It is allowed to be noisy.

The pilot fails if:

```text
predictions are almost constant
PDE residual is low but correlation is poor for N=16 and N=32
source condition is not satisfied
many charts have singular or near-singular metrics
training repeatedly produces NaNs
architectures need completely different hand-tuned hyperparameters to run
```

## Immediate diagnosis if the pilot fails

### Case A: source condition fails

Try:

```text
eikonal.source_bc_weight = 10 or 100
eikonal.hard_source_ansatz = True, if implemented
```

Do not change the chart architecture first.

### Case B: PDE residual low but solution globally wrong

Likely causes:

```text
sparse observations insufficient or badly sampled
overlap consistency too weak
source condition too weak
loss balancing issue
PINN too small
```

First try:

```text
stratified sparse-point sampling
larger source/data weight
larger interface weight
slightly larger local PINN hidden_dim, e.g. 32 instead of 16
```

### Case C: metric condition numbers explode

Likely causes:

```text
chart decoder has folds or near-singular Jacobians
coordinates are badly scaled
neural UAE checkpoint is undertrained
metric regularization is too weak
```

Try:

```text
pca_monge chart decoder
coordinate normalization
immersion/conditioning regularization
remove problematic charts from diagnostic subset and inspect them visually
```

### Case D: training is slow

First inspect:

```text
metric_time_ms_mean
step_time_ms_mean
batch size
JAX compilation count
host-to-device transfer
```

Then try:

```text
larger batch size
fewer eval calls
lower logging frequency
keep runtime.enable_x64=False
avoid plotting during training
```

---

# Expected day-by-day schedule

## Day 0

Run:

```text
make install
make test
make smoke-uae
make smoke-eikonal
make profile-eikonal
make benchmark-metric
```

Deliverable:

```text
Day 0 pass/fail log and JSON profiles.
```

## Day 1

Run analytic geometry/operator tests and overlap-pairing tests.

Deliverable:

```text
geometry_analytic.json
overlap_check.json
```

Stop if either fails.

## Day 2

Generate a small coil chart dataset, run UAE smoke, and run one short chart-model training.

Deliverable:

```text
short chart-model checkpoint
training log
resolved config
```

## Day 3

Run chart diagnostics for `pca_monge` and the current neural UAE architecture.

Deliverable:

```text
aggregate_chart_metrics.csv
metric/reconstruction plots
architecture ranking
```

## Day 4

Run one short end-to-end coil Eikonal dry run with the best architecture.

Deliverable:

```text
one successful train/eval cycle
checkpoint
dry-run metrics
```

## Days 5–8

Run sparse Eikonal coil pilot:

```text
N ∈ {4, 8, 16, 32}
seeds ∈ {0, 1, 2}
architectures = best one or two
```

Deliverable:

```text
results_raw.csv
results_aggregate.csv
RMSE vs N
correlation vs N
relative L2 vs N
prediction-vs-ground-truth examples
residual histograms
```

---

# Decision after Day 8

## If the sparse Eikonal pilot is strong

Proceed to paper-scale implementation of the first real experiment:

```text
sparse Eikonal inverse problem on anatomical or cardiac-like surfaces
```

Also keep the coil pilot as a development benchmark and possible appendix sanity check.

## If chart diagnostics are strong but Eikonal is weak

Focus next on PINN optimization:

```text
loss weights
source condition enforcement
sparse point sampling
interface gradient consistency
local PINN capacity
residual sampling
```

Do not redesign the chart architecture yet.

## If Eikonal works only with PCA/Monge

This is not a failure. It may indicate that the simpler geometric chart architecture is the stronger method. Continue with PCA/Monge as the main architecture and keep the neural UAE as an ablation.

## If Eikonal works only with the neural UAE

Use the neural UAE as the main architecture and keep PCA/Monge as the deterministic baseline.

## If neither architecture works

Do not proceed to paper-scale experiments. Return to:

```text
metric conditioning
source condition
paired overlaps
data sampling
PINN loss balancing
```

---

# Final checklist before moving beyond Day 8

Before implementing the full paper experiments, the following should be true:

```text
[ ] geometry analytic tests pass
[ ] overlap pairing tests pass
[ ] metric profiling emits stable finite diagnostics
[ ] short UAE/chart-model run completes
[ ] chart diagnostics rank at least one architecture as usable
[ ] coil Eikonal dry run trains and evaluates end-to-end
[ ] sparse Eikonal pilot produces nontrivial RMSE/correlation curves
[ ] all sparse-point IDs and evaluation-point IDs are saved
[ ] all outputs are saved to CSV/JSON, not only W&B
[ ] commands are reproducible from the repository root
```

Only after this checklist passes should you spend time on the final paper experiments.
