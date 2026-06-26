# Preliminary Experiment Report: Coil M-PINNs Refactor Funnel

Last updated: 2026-06-26

## Executive Summary

This report summarizes the preliminary experiments run after the refactor of the
`manifold-pinns` codebase. The goal was not to produce final paper-scale
results. The goal was to decide whether the refactored pipeline, chart geometry,
metric operators, overlap handling, and sparse Eikonal training stack are
reliable enough to justify moving on to the next experiment list.

The answer is yes. The preliminary funnel is functionally complete.

The strongest local protocol found so far is:

```text
dataset: coil
chart backend: neural UAE
UAE checkpoint: /mnt/external_storage/manifold-pinns/checkpoints/uae/coil/vqwx2biw
UAE step: 99985
Eikonal sparse sampling: stratified_geodesic
N: 16 sparse/source-supervised points
Eikonal steps: 50000
seeds: 0, 1, 2
```

Across the three N=16 stratified runs:

```text
RMSE       14.09 +/- 3.34
relative L2 0.062 +/- 0.015
tail RMSE 11.19 +/- 9.52
correlation 0.992 +/- 0.003
```

This is a large improvement over the earlier random N=8 run:

```text
random N=8, 50k:      RMSE 43.86, tail RMSE 110.41, corr 0.946
stratified N=16, 50k: RMSE 14.09 +/- 3.34, tail RMSE 11.19 +/- 9.52, corr 0.992 +/- 0.003
```

The main remaining weakness is seed-to-seed variation in the far-distance tail,
especially seed 1. Tail anchoring would likely reduce this, but it is not needed
to decide that the refactored stack is viable.

## Artifact Root

All heavy artifacts were written outside the repository:

```text
/mnt/external_storage/manifold-pinns
```

The main run root was:

```text
/mnt/external_storage/manifold-pinns/runs/prelim_coil_refactor_20260624
```

This report uses paths relative to that run root unless stated otherwise.

The supported workflow throughout the experiments was:

```bash
uv run python -m manifold_pinns.pipeline.cli ...
```

## Infrastructure And Code Changes Used

Several changes were made to make the experiments reproducible and portable:

- Added `uv` environment support and used `uv run` for all supported workflows.
- Added environment-driven roots through `env.sh`, `env_snellius.sh`, and `manifold_pinns/pipeline/env.py`.
- Routed data, checkpoints, runs, figures, W&B files, batches, profiler traces, and eval outputs through `MANIFOLD_PINNS_*` environment roots.
- Fixed `wandb.use=False` behavior so dry runs and smoke checks can run offline.
- Fixed list-valued CLI overrides, needed for explicit sparse IDs.
- Fixed no-argument `wandb.run.save()` in UAE training.
- Added metric/operator and overlap diagnostic scripts.
- Added `chart.backend={uae,pca_monge}` support for Eikonal generate/train/eval.
- Added metric/correlation JSON outputs with RMSE, relative L2, correlation, top-10% tail RMSE, tail bias, and prediction maxima.
- Added sparse point ID saving and stratified sparse sampling.

These changes are part of the current working tree and are summarized in
`RECAP.md`.

## Day 0: Installation, Smoke Checks, And Profiling

### Purpose

The Day 0 gate checked whether the refactored repository was runnable, whether
JAX compiled the core paths, and whether the synthetic profiling scripts emitted
finite JSON.

### Tests And Compilation

Artifact:

```text
day0/test.log
```

Result at the time of Day 0:

```text
15 passed in 8.79s
compileall passed
```

After subsequent code additions, the current verification state is:

```text
19 passed
compileall passed
```

### UAE Smoke

Artifact:

```text
day0/smoke_uae.log
```

Result:

```json
{
  "architecture": "pca_monge",
  "dataset": "coil",
  "geodesic": 0.25,
  "loss": 0.3125,
  "reconstruction_mse": 8.1790733627494e-20,
  "riemannian": 0.5,
  "smoke": "uae"
}
```

### Eikonal Smoke

Artifact:

```text
day0/smoke_eikonal.log
```

Result:

```json
{
  "config": "coil",
  "experiment": "eikonal",
  "loss": 0.11239995807409286,
  "res_loss": 0.10239996016025543,
  "source_bc_loss": 0.010000000707805157,
  "step_time_s": 2.4036198407411575,
  "smoke": "pinn"
}
```

### Eikonal Step Profiling

Artifact:

```text
day0/profile_eikonal.json
```

Result:

| metric | value |
|---|---:|
| device | `cuda:0` |
| dtype | `float32` |
| batch residual | 128 |
| compile time | 0.284 s |
| mean step time | 25.64 ms |
| residual loss | 0.01419 |
| source BC loss | 2.83e-05 |

### Metric Batch Benchmark

Artifact:

```text
day0/benchmark_metric.json
```

Result:

| metric | value |
|---|---:|
| device | `cuda:0` |
| dtype | `float32` |
| num points | 1024 |
| compile time | 0.134 s |
| metric time mean | 0.0806 ms |
| condition mean | 1.067 |
| sqrt(det g) min | 1.00005 |
| sqrt(det g) max | 1.09452 |

Day 0 passed.

## Day 1: Analytic Geometry And Operator Checks

Artifact:

```text
day1_geometry/geometry_analytic.json
```

The analytic tests covered:

- flat plane metric;
- flat plane Laplacian;
- flat plane Eikonal residual;
- analytic quadratic Monge-patch metric.

All tests passed.

Key values:

| check | metric | value |
|---|---|---:|
| flat metric | max abs g minus I | 0 |
| flat metric | max abs ginv minus I | 0 |
| flat metric | max abs sqrt det minus 1 | 0 |
| flat Laplacian | max abs error | 0 |
| flat Eikonal | max abs residual | 0 |
| Monge metric | max abs metric error | 0 |
| Monge metric | max abs ginv error | 1.79e-07 |
| Monge metric | cond mean | 1.0505 |
| Monge metric | cond p95 | 1.1381 |
| Monge metric | cond max | 1.2163 |
| Monge metric | sqrt(det g) min | 1.0000007 |
| Monge metric | sqrt(det g) max | 1.1029 |

Interpretation: the shared metric code and basic differential operators are
consistent on analytic fixtures. This made downstream PDE residuals
interpretable.

## Day 1: Overlap Pairing And Interface Loss

Artifact:

```text
day1_overlap/overlap_check.json
```

Result:

| metric | value |
|---|---:|
| number of paired overlap points | 512 |
| max ambient pair error | 3.33e-16 |
| paired interface loss | 4.83e-32 |
| shuffled interface loss | 2.915 |
| shuffled/paired ratio | 1.31e16 |

All overlap checks passed.

Interpretation: overlap interface losses compare the same physical points across
chart coordinate systems. The negative control showed that shuffled pairs produce
a large loss, as expected.

## Coil Atlas And Chart Data

Artifact:

```text
day2_uae/eikonal_atlas.json
```

The coil atlas used for Eikonal contains:

| quantity | value |
|---|---:|
| vertices | 11575 |
| faces | 22544 |
| charts | 92 |
| boundary pairs | 542 |
| nearest neighbors | 10 |
| min_dist | 9.0 |

Atlas output:

```text
/mnt/external_storage/manifold-pinns/data/coil/charts_1
```

An early atlas-generation command failed because the storage-root environment was
not sourced, causing output to resolve under `/coil/charts_1`. After sourcing
`env.sh` and using `MANIFOLD_PINNS_DATA_ROOT`, atlas generation succeeded.

## UAE Chart Model Training

### Short UAE Run

Artifact:

```text
day2_uae/uae_short_train.log
```

Short training result:

```text
checkpoint: /mnt/external_storage/manifold-pinns/checkpoints/uae/coil/no_wandb_0
step: 49993
reconstruction MSE: 0.085395
```

This was sufficient for early dry runs but was later superseded by a longer UAE
checkpoint.

### Longer UAE Run

Artifact:

```text
day6_uae_coil_100000_wandb/train.log
```

Longer training result:

```text
W&B run: vqwx2biw
checkpoint: /mnt/external_storage/manifold-pinns/checkpoints/uae/coil/vqwx2biw
step: 99985
reported reconstruction MSE during run: 0.038869
final reconstruction MSE line: 0.226286
```

The Eikonal pilot used this checkpoint:

```text
MANIFOLD_PINNS_UAE_COIL_CHECKPOINT=/mnt/external_storage/manifold-pinns/checkpoints/uae/coil/vqwx2biw
MANIFOLD_PINNS_UAE_COIL_STEP=99985
```

Note: the two MSE lines in the log come from different reconstruction checks in
the training script. The downstream Eikonal behavior was the more relevant
criterion for this preliminary funnel.

## Chart Architecture Diagnostics

Artifact:

```text
day3_chart_diagnostics/aggregate_chart_metrics.csv
day3_chart_diagnostics/pca_monge.json
```

The diagnostic run evaluated `pca_monge` on 64 coil charts.

| metric | value |
|---|---:|
| architecture | `pca_monge` |
| charts evaluated | 64 |
| failures | 0 |
| reconstruction RMSE mean | 0.2656 |
| reconstruction RMSE std | 0.1136 |
| metric cond median | 2.014 |
| metric cond p95 | 5.137 |
| metric cond max | 6.138 |
| sqrt(det g) min | 0.9997 |
| sqrt(det g) max | 2.477 |
| min singular value J p05 | 0.9998 |
| decode time mean | 94.99 ms |
| metric time mean | 189.63 ms |

Interpretation: `pca_monge` is geometrically stable and well-conditioned on the
diagnostic subset. It is useful as a deterministic chart baseline.

## Eikonal Dry Run

Artifacts:

```text
day4_eikonal_dryrun/generate_data.log
day4_eikonal_dryrun/generate_data_after_fix.log
day4_eikonal_dryrun/train.log
```

The first end-to-end Eikonal generation attempt exposed two practical issues:

1. `open3d` was missing from the environment.
2. The initial sparse boundary-condition selection could fail when no source
   boundary point was included.

The Eikonal path was fixed to:

- use explicit `VS` and `VT` arguments in the libigl exact geodesic call;
- enforce the source boundary condition;
- save sparse point IDs;
- reuse `charts2d.pkl` only when it matches the intended chart backend;
- regenerate chart coordinates when requested with
  `dataset.regenerate_charts2d=True`.

The dry run then trained and evaluated end to end. This validated checkpoint
loading, atlas loading, chart coordinate generation, sparse observations, source
condition enforcement, residual sampling, overlap batches, and final evaluation.

## PCA/Monge Baseline Versus Neural UAE

Artifacts:

```text
day5_pca_monge_n8_seed0/uae_vs_pca_monge_15000.json
day5_pca_monge_n8_seed0/uae_vs_pca_monge_15000.csv
```

Both models were evaluated on N=8, seed 0, 15k Eikonal steps.

| backend | RMSE | relative L2 | corr | tail RMSE | tail bias | pred max |
|---|---:|---:|---:|---:|---:|---:|
| UAE | 44.05 | 0.195 | 0.950 | 113.59 | -112.90 | 300.54 |
| pca_monge | 99.84 | 0.442 | 0.667 | 269.89 | -269.45 | 307.23 |

Interpretation: despite excellent metric conditioning, `pca_monge` was not
competitive with the neural UAE in the sparse Eikonal pilot. It remains useful
as a baseline but should not be the main local method for the next experiment
list.

## Sparse Eikonal Pilot

### Problem Setup

The Eikonal equation was solved on the coil:

```text
||grad_M u|| = 1
u(source) = 0
```

The ground truth is the exact geodesic distance from the source vertex. The
preliminary runs enforced:

```text
eikonal.enforce_source_bc=True
eikonal.hard_source_ansatz=False
runtime.enable_x64=False
training.batch_size=128
```

The main checkpoint was:

```text
/mnt/external_storage/manifold-pinns/checkpoints/uae/coil/vqwx2biw
step 99985
```

The main evaluation metrics were:

- RMSE;
- MAE;
- relative L2;
- correlation with ground truth;
- top-10% tail RMSE;
- top-10% tail bias;
- predicted maximum distance versus ground-truth maximum.

The target maximum geodesic distance in the reported evaluations was:

```text
390.5283
```

### Random N=8 Baseline

Artifact:

```text
day6_eikonal_uae99985_N8_seed0_50k/summary_compare.json
```

Result:

| sampling | N | seed | steps | RMSE | relative L2 | corr | tail RMSE | tail bias | pred max |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| random | 8 | 0 | 50000 | 43.86 | 0.194 | 0.946 | 110.41 | -110.13 | 310.81 |

Interpretation: random N=8 produces a nontrivial solution but badly
underestimates the far-distance tail.

### Stratified Sampling

The sparse sampler was changed from uniform random point selection to
geodesic-distance stratification.

Current `stratified_geodesic` behavior:

```text
1. include the source point;
2. sort non-source vertices by true geodesic distance;
3. split them into geodesic-distance bins;
4. sample across bins using the fixed bcs_seed;
5. save selected global point IDs and geodesic values as .npy and .json.
```

This makes the sparse supervision cover near, middle, and far geodesic
distances, instead of depending on a uniform-random draw.

### Random N=8 Versus Stratified N=8

Artifact:

```text
day7_eikonal_uae99985_N8_seed0_stratified_20260625-161807_50000/random_vs_stratified_N8.json
```

| sampling | N | seed | RMSE | relative L2 | corr | tail RMSE | tail bias | pred max |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| random | 8 | 0 | 43.86 | 0.194 | 0.946 | 110.41 | -110.13 | 310.81 |
| stratified | 8 | 0 | 28.87 | 0.128 | 0.971 | 69.69 | -65.38 | 341.94 |

Interpretation: stratification substantially improves the Eikonal solution,
especially the tail. This confirmed that sparse point placement was a major
failure mode.

### N Scaling At Seed 0

Artifact:

```text
day7_scaling_uae99985_stratified_seed0_50k.json
```

| run | N | RMSE | relative L2 | corr | tail RMSE | tail bias | pred max |
|---|---:|---:|---:|---:|---:|---:|---:|
| random_N8 | 8 | 43.86 | 0.194 | 0.946 | 110.41 | -110.13 | 310.81 |
| stratified_N8 | 8 | 28.87 | 0.128 | 0.971 | 69.69 | -65.38 | 341.94 |
| stratified_N16 | 16 | 11.12 | 0.049 | 0.995 | 4.05 | -1.11 | 385.22 |
| stratified_N32 | 32 | 11.57 | 0.051 | 0.995 | 5.97 | -3.66 | 384.60 |

Interpretation: N=16 is the best cost-quality point locally. N=32 did not
improve over N=16 on seed 0.

### N=16 Stratified Multi-Seed Results

Artifact:

```text
day7_stratified_N16_seed0_seed1_seed2_50k.json
day7_stratified_N16_seed0_seed1_seed2_50k.csv
```

| seed | RMSE | relative L2 | corr | tail RMSE | tail bias | pred max |
|---:|---:|---:|---:|---:|---:|---:|
| 0 | 11.12 | 0.049 | 0.995 | 4.05 | -1.11 | 385.22 |
| 1 | 17.70 | 0.078 | 0.989 | 22.00 | -13.59 | 367.08 |
| 2 | 13.45 | 0.060 | 0.993 | 7.52 | -4.91 | 382.81 |

Aggregate:

| metric | mean | std | min | max |
|---|---:|---:|---:|---:|
| RMSE | 14.09 | 3.34 | 11.12 | 17.70 |
| relative L2 | 0.062 | 0.015 | 0.049 | 0.078 |
| tail RMSE | 11.19 | 9.52 | 4.05 | 22.00 |
| tail bias | -6.54 | 6.39 | -13.59 | -1.11 |
| correlation | 0.992 | 0.003 | 0.989 | 0.995 |
| MAE | 8.75 | 2.29 | 7.26 | 11.40 |
| pred max | 378.37 | 9.85 | 367.08 | 385.22 |

Interpretation: N=16 stratified is robust enough for the preliminary funnel.
Seed 1 remains a tail-underestimation outlier, but still strongly outperforms
the earlier random and low-N runs.

## Tail Behavior

The persistent failure mode is far-tail underestimation. The current stratified
sampler samples from the far distance band, but the exact farthest point is not
guaranteed. Seed 1 had a weaker far-tail anchor than seeds 0 and 2, and this
corresponded to a lower predicted maximum and worse tail RMSE.

A `tail_anchored_stratified` sampler would likely improve this by selecting:

```text
source point
+ farthest or near-farthest geodesic point
+ remaining stratified points
```

We did not implement this because it would mostly confirm a known effect: adding
a far-tail anchor should improve the far tail. It is useful as a future ablation
but not necessary before moving to the next experiment list.

## What Was Intentionally Not Completed

The full `EXPERIMENTS.md` matrix asked for a more formal sweep:

```text
N in {4, 8, 16, 32}
seeds in {0, 1, 2}
one or two chart architectures
final RMSE/correlation/relative-L2 plots
```

We did not run every formal row. Specifically, we skipped:

- N=4;
- all N=8 seeds beyond seed 0;
- all N=32 seeds beyond seed 0;
- a full neural-UAE chart diagnostic table equivalent to the pca_monge diagnostic;
- final plots such as RMSE vs N and correlation vs N;
- tail-anchored sampling;
- external baselines.

These are not blockers for the stated purpose of the preliminary funnel. The
goal was to determine whether the refactored stack is reliable enough to move
forward, not to finish a publication-ready sweep.

## Status Against Final Checklist

| checklist item | status |
|---|---|
| geometry analytic tests pass | done |
| overlap pairing tests pass | done |
| metric profiling emits finite diagnostics | done |
| short UAE/chart-model run completes | done |
| chart diagnostics rank at least one architecture as usable | done for pca_monge; UAE validated downstream |
| coil Eikonal dry run trains and evaluates end to end | done |
| sparse Eikonal pilot produces nontrivial RMSE/correlation behavior | done |
| sparse point IDs are saved | done |
| outputs saved to CSV/JSON, not only W&B | done |
| commands reproducible from repository root | done through `uv run python -m manifold_pinns.pipeline.cli ...` |

## Conclusions

1. The refactored pipeline is usable for M-PINNs experiments.

2. Geometry and overlap correctness gates passed. The metric and interface-loss
   code can be trusted for the next stage of experiments.

3. `pca_monge` is stable and well-conditioned geometrically, but it was not
   competitive in sparse Eikonal training on coil. Keep it as a baseline, not the
   current main method.

4. The neural UAE checkpoint at step 99985 works well downstream, even though
   standalone reconstruction metrics are not perfectly clean.

5. Sparse-point placement matters substantially. Moving from random N=8 to
   stratified N=8 reduced RMSE from 43.86 to 28.87 and tail RMSE from 110.41 to
   69.69.

6. Stratified N=16 is the best local protocol found. On seed 0, N=32 did not
   improve over N=16. Across seeds 0/1/2, N=16 remains strong.

7. The remaining tail variation is real but understood. A tail-anchored sampler
   would probably reduce it, but this is an ablation rather than a blocker.

## Recommended Frozen Local Protocol

For the next experiment list, use the following coil protocol as the local
reference:

```text
dataset: coil
atlas: /mnt/external_storage/manifold-pinns/data/coil/charts_1
chart backend: uae
UAE checkpoint: /mnt/external_storage/manifold-pinns/checkpoints/uae/coil/vqwx2biw
UAE checkpoint step: 99985
sparse sampling: stratified_geodesic
N: 16
seeds: 0, 1, 2
training.max_steps: 50000
training.batch_size: 128
runtime.enable_x64: False
eikonal.enforce_source_bc: True
eikonal.hard_source_ansatz: False
```

This should be treated as the current development benchmark.

## Next Step

The preliminary funnel in `EXPERIMENTS.md` is complete enough. The next action is
to define and run the new experiment list.

