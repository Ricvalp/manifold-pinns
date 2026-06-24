# Codex Prompt: Refactor and Improve `manifold-pinns` for a Stronger M-PINNs Paper

You are working on the `manifold-pinns` repository, especially the `refactor` branch:

```bash
git clone https://github.com/Ricvalp/manifold-pinns.git
cd manifold-pinns
git checkout refactor
```

You have full access to the local system, GPU, package manager, profiler, repository history, and the codebase. First inspect the current state of the repository before making changes. Some path names below are based on a static review of the public `refactor` branch and may have changed; verify everything before editing.

The goal is not only to clean the code. The goal is to make the implementation strong enough to support a revised NeurIPS-style paper about M-PINNs: **amortized local-atlas physics-informed learning on sparse and changing point-cloud surfaces**.

The method in the paper is:

1. Partition a point-cloud surface/manifold into overlapping charts.
2. Use a **Universal Autoencoder (UAE)** inspired by the Universal Physics Transformer to produce, zero-shot, chart-specific local embeddings/decoders.
3. For each chart decoder `d_i : R^2 -> R^3`, compute the pullback Riemannian metric

   ```text
   g_i(z) = J_{d_i}(z)^T J_{d_i}(z)
   ```

4. Train local PINNs in the learned chart coordinates using intrinsic operators such as the surface gradient and Laplace-Beltrami operator.
5. Couple local PINNs with domain-decomposition/interface losses across overlapping charts.

The current paper was rejected. The new implementation should support a stronger story: M-PINNs are useful when geometry is complex, measurements are sparse, and the surface may vary across time or subjects. Do not optimize only for the existing toy demos. The code should support credible experiments, reliable baselines, profiling, reproducibility, and architecture ablations.

---

## Top-Level Deliverables

Produce a set of code changes that accomplishes the following, in priority order:

1. **Correctness fixes**: fix likely bugs in imports, train-step APIs, UAE loss dispatch, overlap/interface pairing, geodesic-distance computation, and PDE residual consistency.
2. **JAX performance refactor**: reduce Python/NumPy/Torch host overhead, vectorize over points/charts, use JIT/VMAP/SCAN correctly, avoid unnecessary float64, reduce repeated metric/Jacobian calls, and make single-GPU performance clean before attempting multi-GPU.
3. **Geometry/PDE reliability**: add tests and diagnostics proving that learned charts produce stable metrics and correct intrinsic operators.
4. **Architecture alternatives**: implement at least one simpler and more geometric alternative to the current UPT/Perceiver/SIREN UAE: a PCA/Monge chart decoder. Keep the current UAE as the legacy/default or one selectable option.
5. **Reproducibility**: add a pinned install path, smoke tests, CLI checks, benchmark/profiling scripts, and experiment commands that can be used in the paper appendix.

Do not silently remove old functionality. Add feature flags and configs where needed. Preserve old experiments unless they are demonstrably broken; when replacing behavior, add tests or migration notes.

---

## Working Protocol

Before making large changes:

1. Print repository status:

   ```bash
   git status
   git branch --show-current
   git log --oneline -5
   find . -maxdepth 3 -type f | sort | sed 's#^./##' | head -200
   ```

2. Inspect the CLI and current README:

   ```bash
   sed -n '1,240p' README.md
   sed -n '1,260p' manifold_pinns/pipeline/cli.py
   ```

3. Run the current sanity check from the README:

   ```bash
   python -m compileall manifold_pinns pinns universal_autoencoder datasets jaxpi
   ```

4. Try all CLI entry points in dry/smoke mode if such a mode exists. If it does not exist, add one.

5. Add or update tests before large rewrites. The minimum acceptable test suite should include:

   ```bash
   pytest -q
   python -m compileall manifold_pinns pinns universal_autoencoder datasets jaxpi
   ```

6. After each major change, run tests and at least one very small smoke training step.

---

## Part A — Correctness Fixes

### A1. Fix package/import layout

The `refactor` branch added a new CLI under `manifold_pinns/pipeline`, but some experiment files may still use local imports such as:

```python
import models
from samplers import UniformSampler
```

These are fragile when called from the repo root through:

```bash
python -m manifold_pinns.pipeline.cli ...
```

Replace local imports with package imports. Examples:

```python
from pinns.eikonal import models
from pinns.eikonal.samplers import UniformBCSampler, UniformSampler, UniformBoundarySampler
```

and similarly for `pinns.wave`, `pinns.diffusion`, and UAE experiments.

Also inspect the `jaxpi` packaging. If the repo currently requires `pip install -e ./jaxpi`, make this explicit. Prefer a root `pyproject.toml` that installs the repository and the local `jaxpi` package correctly, or document and script the two editable installs:

```bash
pip install -e ./jaxpi
pip install -e .
```

Acceptance criteria:

```bash
python -m compileall manifold_pinns pinns universal_autoencoder datasets jaxpi
python -m manifold_pinns.pipeline.cli --help
python -m manifold_pinns.pipeline.cli dataset --help
python -m manifold_pinns.pipeline.cli uae --help
python -m manifold_pinns.pipeline.cli pinn --help
```

all run from the repository root without manually changing directories.

---

### A2. Standardize train-step APIs

Inspect `jaxpi/jaxpi/models.py` and all subclasses. There may be inconsistent return signatures between multi-chart and single-chart models. Standardize every model step to return:

```python
loss, aux, state = model.step(...)
```

where:

- `loss` is the scalar optimized loss;
- `aux` is a dictionary or frozen dict containing unweighted/weighted component losses and diagnostic scalars;
- `state` is the updated train state.

Use the same API for all PINNs: eikonal, diffusion, wave, single-chart, and multi-chart.

Recommended structure:

```python
def losses(self, params, batch, *args) -> dict[str, jnp.ndarray]:
    ...


def loss(self, params, weights, batch, *args):
    losses = self.losses(params, batch, *args)
    weighted = jax.tree.map(lambda l, w: l * w, losses, weights)
    total = functools.reduce(operator.add, jax.tree.leaves(weighted))
    aux = {
        "losses": losses,
        "weighted_losses": weighted,
        "total": total,
    }
    return total, aux


@functools.partial(jax.jit, static_argnums=(0,))
def step(self, state, weights, batch, *args):
    (loss, aux), grads = jax.value_and_grad(self.loss, has_aux=True)(
        state.params, weights, batch, *args
    )
    state = state.apply_gradients(grads=grads)
    return loss, aux, state
```

If jitting bound methods causes recompilation or `self` issues, move the jitted functions outside the class and pass pure objects such as `apply_fn`, config constants, and train state explicitly.

Acceptance criteria:

- Every training loop unpacks `loss, aux, model.state` consistently.
- Add a unit/smoke test that constructs each model and performs one training step on tiny synthetic data.

---

### A3. Fix overlap/interface sampling: pair the same physical points

This is a critical correctness issue.

The interface loss must compare predictions from neighboring charts at the **same physical point**. The loss should be:

```text
u_i(z_i(x)) ≈ u_j(z_j(x))
```

for the same ambient point `x` in the chart overlap.

If the current `UniformBoundarySampler` independently samples from chart `i -> j` and chart `j -> i`, it is likely wrong. It can enforce equality between unrelated overlap points.

Refactor overlap data into a paired representation. Suggested data structure:

```python
@dataclass(frozen=True)
class OverlapPairs:
    src_chart: int
    dst_chart: int
    z_src: np.ndarray  # [N_ij, 2], coordinates in source chart
    z_dst: np.ndarray  # [N_ij, 2], coordinates in destination chart
    x_ambient: np.ndarray | None = None  # [N_ij, 3]
    point_ids: np.ndarray | None = None  # [N_ij]
```

Sampling should use one index vector:

```python
idx = rng.integers(0, len(pair.z_src), size=batch_size)
z_src = pair.z_src[idx]
z_dst = pair.z_dst[idx]
```

The interface loss should compare:

```python
u_src = u_net(params[src_chart], z_src)
u_dst = u_net(params[dst_chart], z_dst)
loss_value = mean((u_src - u_dst) ** 2)
```

Add optional intrinsic gradient consistency:

```text
L_interface = mean(|u_i - u_j|^2)
            + lambda_grad * mean(||grad_M u_i - grad_M u_j||^2)
```

For gradient consistency, compare ambient tangent gradients if possible:

```text
grad_ambient u_i = J_i g_i^{-1} grad_z u_i
```

where `J_i = J_{d_i}(z_i)` and `g_i = J_i^T J_i`.

Acceptance criteria:

- Add a test with two synthetic overlapping charts representing the same plane. The interface sampler must return paired points that map to the same ambient coordinates.
- Add a test that interface loss is near zero when two chart networks represent the same analytic function expressed in two coordinate systems.

---

### A4. Do not use floating-point tuple equality for overlaps

Inspect chart construction code, especially anything like:

```python
set(map(tuple, points))
```

or tuple intersections of floating-point coordinates.

Replace this with one of:

1. original point IDs propagated through chart generation;
2. KD-tree matching with a documented tolerance;
3. mesh vertex indices if available.

Preferred: point IDs. Every chart should carry both local coordinates and original/global point IDs. Overlap pairing should be derived from shared global IDs.

Acceptance criteria:

- Chart generation saves `point_ids` for each chart.
- Overlap construction uses `point_ids` where available.
- A noisy-point-cloud test uses KD-tree tolerance and passes.

---

### A5. Fix UAE loss dispatch

Inspect UAE experiment files such as:

```text
universal_autoencoder/experiments/coil/fit_universal_autoencoder_coil.py
universal_autoencoder/experiments/bunny/...
universal_autoencoder/experiments/square/...
```

If config exposes something like:

```python
cfg.train.reg = "geodesic_preservation"  # or "geo+riemannian"
```

but the train step always calls one fixed loss function, fix it.

Implement explicit loss dispatch:

```python
def make_loss_fn(cfg):
    if cfg.train.reg in (None, "none", "reconstruction"):
        return recon_loss_fn
    if cfg.train.reg == "geodesic_preservation":
        return geo_loss_fn
    if cfg.train.reg == "geo+riemannian":
        return geo_riemann_loss_fn
    if cfg.train.reg == "pde_chart":
        return pde_chart_loss_fn
    raise ValueError(f"Unknown UAE regularizer: {cfg.train.reg}")
```

Make aux dictionaries consistent across all losses:

```python
return total, {
    "reconstruction": recon,
    "geodesic": geo,
    "riemannian": riemann,
    "immersion": immersion,
    "condition": cond,
}
```

where unavailable components are either omitted or set to `0.0` explicitly.

Acceptance criteria:

- Changing `cfg.train.reg` changes the optimized objective, not just logging.
- Add a unit test that monkeypatches tiny inputs and verifies different loss choices produce different aux keys.

---

### A6. Verify weighted geodesic distances

Inspect chart/geodesic utilities. If the code uses NetworkX:

```python
nx.all_pairs_shortest_path_length(G)
```

this computes unweighted hop counts, not weighted geodesic distances, even if graph edges have Euclidean weights.

Use weighted Dijkstra:

```python
nx.all_pairs_dijkstra_path_length(G, weight="weight")
```

or preferably use `scipy.sparse.csgraph.dijkstra` on a CSR adjacency matrix for speed.

Acceptance criteria:

- Add a test graph with unequal edge weights where hop-count and weighted-distance answers differ. The code must return weighted distances.
- Document whether distances are graph-geodesic approximations, mesh geodesics, or Euclidean distances.

---

### A7. Fix Eikonal boundary/source condition handling

The paper currently used an ill-posed scarce-data setup by not imposing the source condition. For stronger experiments and fair baselines, the code should support both:

1. legacy ill-posed setup, for reproducing old figures;
2. well-posed sparse inverse setup with source/boundary condition.

For Eikonal:

```text
||grad_M T|| = 1 / c(x)
T(x_source) = 0
```

or for constant speed:

```text
||grad_M T|| = 1
T(x_source) = 0
```

Add config flags:

```python
cfg.eikonal.enforce_source_bc = True
cfg.eikonal.source_bc_weight = ...
cfg.eikonal.hard_source_ansatz = False
```

If feasible, implement a hard ansatz:

```text
T(z) = rho(z, z_source) * softplus(N_theta(z))
```

or another construction ensuring nonnegativity and `T(source)=0`. Keep soft source loss as a simpler option.

Acceptance criteria:

- The Eikonal training loop logs source BC loss.
- Legacy results remain reproducible through a config flag.
- New default should be the well-posed setup.

---

## Part B — JAX Performance Refactor

### B1. Disable global float64 by default

Search for:

```python
jax.config.update("jax_enable_x64", True)
```

Disable global x64 unless there is a documented numerical reason. Use float32 by default. If metric inversion requires extra precision, cast only that small computation locally or, better, regularize chart conditioning.

Recommended default:

```python
jax.config.update("jax_enable_x64", False)
```

Add config:

```python
cfg.runtime.enable_x64 = False
```

Acceptance criteria:

- Default training runs in float32.
- Optional x64 mode still works.
- Benchmark shows wall-clock difference between float32 and x64 on a small run.

---

### B2. Make samplers JAX-native

Current samplers may use Python iterators, NumPy RNG, and Torch DataLoader. This causes host-device transfers and prevents fusing multiple steps.

For fixed datasets, move arrays to device once and sample inside JAX.

Example residual sampler:

```python
@jax.jit
def sample_residual_batch(key, coords, batch_size: int, noise_std: float):
    # coords: [C, N, 2]
    C, N, D = coords.shape
    key_idx, key_noise = jax.random.split(key)
    idx = jax.random.randint(key_idx, (C, batch_size), minval=0, maxval=N)
    batch = jnp.take_along_axis(coords, idx[..., None], axis=1)
    noise = noise_std * jax.random.normal(key_noise, batch.shape)
    return batch + noise
```

Example paired overlap sampler:

```python
@jax.jit
def sample_overlap_batch(key, z_src_all, z_dst_all, batch_size: int):
    # z_src_all, z_dst_all: [E, Nmax, 2]
    E, N, _ = z_src_all.shape
    idx = jax.random.randint(key, (E, batch_size), minval=0, maxval=N)
    z_src = jnp.take_along_axis(z_src_all, idx[..., None], axis=1)
    z_dst = jnp.take_along_axis(z_dst_all, idx[..., None], axis=1)
    return z_src, z_dst
```

If different charts/edges have different lengths, use padding plus masks:

```python
coords: [C, Nmax, 2]
mask:   [C, Nmax]
overlap_z_src: [E, Mmax, 2]
overlap_z_dst: [E, Mmax, 2]
overlap_mask:  [E, Mmax]
edge_index:    [E, 2]
```

Acceptance criteria:

- At least Eikonal residual/data/interface sampling can be done inside a jitted training step.
- Host-side samplers remain only as compatibility wrappers or data-preparation utilities.

---

### B3. Vectorize residuals over points and charts

Avoid Python loops over charts in the training hot path. Use a canonical batched representation:

```python
chart_coords: [C, N, 2]
chart_cond:   [C, cond_dim]
chart_params: pytree with leading axis C
data_z:       [C, B_data, 2]
res_z:        [C, B_res, 2]
```

Use `vmap` over points and charts:

```python
def u_single(params_i, z):
    return apply_fn({"params": params_i}, z)[0]


grad_u_single = jax.grad(u_single, argnums=1)


def residual_one_chart(params_i, cond_i, z_batch, std_i):
    grad_u = jax.vmap(grad_u_single, in_axes=(None, 0))(params_i, z_batch)  # [B, 2]
    ginv = metric_inv_batch(cond_i, z_batch)                                # [B, 2, 2]
    sqnorm = jnp.einsum("bi,bij,bj->b", grad_u, ginv, grad_u)
    return sqnorm - std_i ** 2


residual_all_charts = jax.vmap(
    residual_one_chart,
    in_axes=(0, 0, 0, 0),
)
```

Acceptance criteria:

- No Python chart loop in Eikonal residual loss.
- Residual function is JIT compiled once for fixed shape.
- Log compile time separately from steady-state step time.

---

### B4. Use analytic 2x2 metric inverse in the hot path

For 2D surface charts, the metric is 2x2 SPD. Avoid general `jnp.linalg.inv` in the inner residual loop.

```python
def inv_2x2_spd(g, eps=1e-8):
    # g: [..., 2, 2]
    a = g[..., 0, 0]
    b = 0.5 * (g[..., 0, 1] + g[..., 1, 0])
    c = g[..., 1, 1]
    det = a * c - b * b
    det = jnp.maximum(det, eps)
    return jnp.stack(
        [
            jnp.stack([ c / det, -b / det], axis=-1),
            jnp.stack([-b / det,  a / det], axis=-1),
        ],
        axis=-2,
    )
```

Also return diagnostics:

```python
trace = a + c
det = a * c - b * b
cond_est = ...
```

Acceptance criteria:

- Analytic inverse agrees with `jnp.linalg.inv` on random SPD 2x2 matrices.
- Eikonal and Laplacian residuals use the analytic inverse by default.

---

### B5. Cache or fuse metric/Jacobian computations

The decoder Jacobian is expensive. Avoid repeatedly calling the metric network separately for each derivative term.

Implement a small geometry API:

```python
@dataclass
class MetricBatch:
    J: jnp.ndarray       # [B, 3, 2]
    g: jnp.ndarray       # [B, 2, 2]
    ginv: jnp.ndarray    # [B, 2, 2]
    sqrt_det_g: jnp.ndarray  # [B]
    cond: jnp.ndarray | None
```

```python
def decoder_single(cond_i, z):
    return decoder_apply(cond_i, z)  # [3]


def metric_batch(cond_i, z_batch):
    J = jax.vmap(jax.jacfwd(decoder_single, argnums=1), in_axes=(None, 0))(cond_i, z_batch)
    # Depending on jacobian layout, normalize to [B, 3, 2]
    g = jnp.einsum("bai,baj->bij", J, J)
    ginv = inv_2x2_spd(g)
    sqrt_det = jnp.sqrt(jnp.maximum(g[..., 0, 0] * g[..., 1, 1] - g[..., 0, 1] ** 2, 1e-12))
    return MetricBatch(J=J, g=g, ginv=ginv, sqrt_det_g=sqrt_det, cond=None)
```

Use the same metric batch for all residual terms at those points.

Acceptance criteria:

- Eikonal residual calls `metric_batch` once per chart batch.
- Laplacian residual calls metric/Jacobian in a controlled way and has tests against analytic cases.

---

### B6. Use `lax.scan` for multiple optimizer steps per dispatch

Once sampling is JAX-native, wrap multiple training steps in `lax.scan`:

```python
@functools.partial(jax.jit, static_argnames=("num_steps",))
def train_n_steps(state, key, static_data, num_steps: int):
    keys = jax.random.split(key, num_steps)

    def body(state, key):
        batch = make_batch(key, static_data)
        loss, aux, state = train_step(state, batch)
        return state, {"loss": loss, **aux}

    return jax.lax.scan(body, state, keys)
```

Use this for inner loops, then log every `num_steps` outside JIT.

Acceptance criteria:

- Training has a config `runtime.steps_per_dispatch` or similar.
- Wall-clock benchmark compares old one-step dispatch vs `scan` dispatch.

---

### B7. Avoid recompilation from dynamic shapes and Python objects

Audit `jax.jit` usage. Common issues:

- Passing Python dicts with changing keys into jitted functions.
- Passing different batch sizes each call.
- Passing `self` as a dynamic object.
- Rebuilding lambdas/functions inside training loops.
- Logging inside jitted code.

Fix by:

- padding variable-length arrays and using masks;
- making batch sizes static config values;
- defining pure functions at module scope;
- using `static_argnames` for integer config values;
- not passing large configs into jitted functions.

Add optional `jax_log_compiles` for debugging:

```python
jax.config.update("jax_log_compiles", cfg.runtime.log_compiles)
```

Acceptance criteria:

- A short training run logs one compile for the main train step, not a compile every iteration.

---

### B8. Precompute UAE supernode neighborhoods

Inspect `universal_autoencoder/upt_encoder.py`. If `k_nearest_neighbors` computes full pairwise distances and sorts inside the forward pass, precompute neighborhoods for fixed training charts.

For each chart and each supernode, store:

```python
supernode_neighbor_idx: [num_charts, num_supernodes, k]
```

Then the encoder gathers neighborhoods by index rather than recomputing pairwise distances.

For test-time dynamic charts, use CPU KD-tree once per chart or a JAX approximate method outside the repeated training hot path.

Acceptance criteria:

- UAE forward pass no longer performs an O(N^2) all-pairs distance/sort for fixed training data.
- Add benchmark comparing old and new encoder path on a small batch.

---

### B9. Subsample UAE pairwise geodesic losses

If the geodesic preservation loss compares all `N x N` pairs for `N=1000`, reduce it. For PDE charts, local metric quality is usually more important than all global pairwise distances.

Implement sampled pair loss:

```python
def sampled_geodesic_loss(dist_matrix, z, key, num_pairs=4096):
    B, N, _ = z.shape
    key_i, key_j = jax.random.split(key)
    i = jax.random.randint(key_i, (B, num_pairs), 0, N)
    j = jax.random.randint(key_j, (B, num_pairs), 0, N)

    zi = jnp.take_along_axis(z, i[..., None], axis=1)
    zj = jnp.take_along_axis(z, j[..., None], axis=1)
    dz = jnp.linalg.norm(zi - zj, axis=-1)

    batch_idx = jnp.arange(B)[:, None]
    dg = dist_matrix[batch_idx, i, j]

    dz = dz / (jnp.mean(dz, axis=-1, keepdims=True) + 1e-8)
    dg = dg / (jnp.mean(dg, axis=-1, keepdims=True) + 1e-8)
    return jnp.mean((dz - dg) ** 2)
```

Add a local-pair option using kNN/geodesic-neighborhood pairs only.

Acceptance criteria:

- Config supports `cfg.train.geodesic_num_pairs`.
- Full pairwise mode remains available for small tests.

---

### B10. Multi-GPU only after clean single-GPU batching

Do not start with `pmap` or `pjit`. First make the chart axis clean on one GPU.

After that, add experimental multi-device support where the chart axis is sharded:

```text
charts 0..C-1 split across devices
local residual/data losses computed per device
overlap edges either owned by one device or handled by all-gathering boundary predictions
```

Start with single-node `pmap` if it is simple. Use `pjit`/named sharding only if the codebase is already clean enough.

Acceptance criteria for multi-GPU is optional. Do not destabilize single-GPU code for it.

---

## Part C — Geometry and PDE Operator Tests

Add a test module such as:

```text
tests/test_geometry_operators.py
tests/test_eikonal_residual.py
tests/test_overlap_pairing.py
tests/test_uae_losses.py
tests/test_cli_smoke.py
```

Minimum tests:

### C1. Flat chart metric test

Decoder:

```text
d(z1, z2) = (z1, z2, 0)
```

Expected:

```text
J = [[1, 0], [0, 1], [0, 0]]
g = I_2
g^{-1} = I_2
sqrt(det(g)) = 1
```

### C2. Flat Eikonal residual test

For `u(z) = z1`, on the flat chart:

```text
||grad_M u||^2 - 1 = 0
```

### C3. Flat Laplacian tests

For `u(z) = z1^2 + z2^2`, on the flat chart:

```text
Delta u = 4
```

For `u(z) = sin(pi z1) sin(pi z2)`:

```text
Delta u = -2 pi^2 u
```

### C4. Monge patch metric test

For:

```text
d(z1, z2) = (z1, z2, h(z1,z2))
h(z1,z2) = a z1^2 + b z2^2
```

Expected metric:

```text
g = [[1 + h_z1^2, h_z1 h_z2],
     [h_z1 h_z2, 1 + h_z2^2]]
```

### C5. Overlap pairing test

Create two coordinate charts for the same plane:

```text
chart A: x = (z1, z2, 0)
chart B: x = (z1 + 1, z2, 0)
```

Construct overlap with known shared ambient points. Verify sampler returns paired `z_A`, `z_B` mapping to the same `x`.

### C6. UAE loss dispatch test

Tiny synthetic chart batch. Verify `cfg.train.reg` selects different loss functions and aux keys.

---

## Part D — Architecture / Method Improvements

The current UPT/Perceiver/SIREN UAE may be overbuilt for the real need. The paper needs **stable differentiable charts**, not only point-cloud reconstruction. Implement architecture alternatives behind config flags so the paper can ablate them.

### D1. Keep current UAE as `upt_siren`

Preserve the existing model as:

```python
cfg.uae.architecture = "upt_siren"
```

Do not delete it. Clean its implementation and loss handling, but keep old checkpoints loadable if feasible.

---

### D2. Implement `pca_monge` chart decoder

This is the highest-priority method addition.

For each chart `P_i`:

1. Center points:

   ```text
   x_centered = x - mu_i
   ```

2. Compute local PCA frame:

   ```text
   R_i = [e1, e2, n]
   ```

3. Deterministic chart coordinates:

   ```text
   z(x) = ((x - mu_i)^T e1, (x - mu_i)^T e2)
   ```

4. Learn a conditional height field:

   ```text
   d_theta(z; c_i) = mu_i + R_i @ [z1, z2, h_theta(z; c_i)]
   ```

Optional in-plane residual:

```text
d_theta(z; c_i) = mu_i + R_i @ [z + r_parallel_theta(z; c_i), h_theta(z; c_i)]
```

Implement as a selectable UAE architecture:

```python
cfg.uae.architecture = "pca_monge"
cfg.uae.monge.use_inplane_residual = False
cfg.uae.monge.height_network = "siren"  # or "mlp"
cfg.uae.monge.conditioning = "pointnet" # or reuse UPT chart encoder
```

Suggested module paths:

```text
universal_autoencoder/monge.py
universal_autoencoder/models.py
universal_autoencoder/geometry.py
```

The output API should match the current UAE:

```python
coords, recon, decoder_context = model.apply(...)
metric = metric_from_decoder(decoder_context, z)
```

or create a common `ChartDecoder` protocol.

Why this matters:

- inverse coordinates are deterministic;
- chart gauge freedom is reduced;
- metrics should be better conditioned;
- no Perceiver coordinate encoder is required;
- it is faster and easier to defend in a paper.

Acceptance criteria:

- `pca_monge` can train/reconstruct on a tiny synthetic surface dataset.
- Metric tests pass.
- Eikonal PINN can use `pca_monge` decoders with the same downstream interface as `upt_siren`.

---

### D3. Add derivative-aware chart losses

The UAE should not optimize only reconstruction. The PDE residual depends on derivatives of the decoder. Add losses that regularize the chart as an immersion.

Implement optional losses:

```text
L_immersion = E_z [max(0, eps - sigma_min(J_d(z)))^2]
L_condition = E_z [log(kappa(J_d(z)^T J_d(z)))^2]
L_smooth = E_z ||H_d(z)||_F^2
```

If normals are available:

```text
L_normal = E_z [1 - <n_mesh, n_decoder>^2]
```

If mesh/discrete Laplacian is available, optional operator matching:

```text
L_laplace = sum_k ||Delta_chart f_k - Delta_mesh f_k||^2
```

Config:

```python
cfg.train.loss_weights.reconstruction = 1.0
cfg.train.loss_weights.geodesic = 0.0
cfg.train.loss_weights.immersion = 0.0
cfg.train.loss_weights.condition = 0.0
cfg.train.loss_weights.smooth = 0.0
cfg.train.loss_weights.normal = 0.0
```

Start with conservative defaults, but make all losses available.

Acceptance criteria:

- Losses compute without NaNs on flat and synthetic Monge charts.
- Diagnostics log min singular value, max condition number, mean condition number, and reconstruction error.

---

### D4. Implement optional LoRA/hypernetwork SIREN decoder

The current FiLM-modulated SIREN may be under-expressive. Add an optional decoder mode:

```text
W_l(c) = W_l^0 + A_l(c) B_l(c)
```

Config:

```python
cfg.uae.decoder = "film_siren"       # legacy
cfg.uae.decoder = "lora_siren"       # new
cfg.uae.lora.rank = 4
```

Do not make this the default until tested. It is an ablation candidate.

Acceptance criteria:

- Forward pass works.
- Parameter count is logged.
- Reconstruction and metric diagnostics compare against FiLM-SIREN.

---

### D5. Learn or compute transition maps across chart overlaps

A stronger learned-atlas method should know how charts transition across overlaps.

For paired overlap points, define empirical transitions:

```text
z_j = tau_ij(z_i)
```

At minimum, store paired samples. Optionally fit simple transition maps:

```python
class TransitionMLP(nn.Module):
    ...
```

and losses:

```text
L_transition_geom = ||d_j(tau_ij(z_i)) - d_i(z_i)||^2
L_transition_solution = ||u_j(tau_ij(z_i)) - u_i(z_i)||^2
```

Optional gradient consistency through `D tau_ij`:

```text
grad_i u_i ≈ D tau_ij^T grad_j u_j
```

Do not overcomplicate this initially. The first necessary step is correct paired overlaps.

---

## Part E — Stronger PINN Training Infrastructure

### E1. Add residual-adaptive sampling

Add optional residual-adaptive sampling for PINN residual points:

1. sample candidate points;
2. evaluate residual magnitude;
3. sample future points with probability proportional to residual magnitude plus epsilon.

Config:

```python
cfg.sampling.residual_adaptive.enabled = False
cfg.sampling.residual_adaptive.num_candidates = 8192
cfg.sampling.residual_adaptive.temperature = 1.0
```

Keep uniform sampling as the default until validated.

---

### E2. Add loss balancing

PINNs are sensitive to loss scales. Add configurable loss weighting strategies:

```python
cfg.loss_balancing.method = "fixed"       # default
cfg.loss_balancing.method = "grad_norm"   # optional
cfg.loss_balancing.method = "ntk"         # optional
```

At minimum, log unweighted losses and weighted losses separately.

---

### E3. Add stronger optimizer options

Current code likely uses Adam with warmup/cosine decay. Keep that. Add optional second-stage optimizers where feasible:

```python
cfg.optim.stage1 = "adam"
cfg.optim.stage2 = "lbfgs"  # optional, if practical in JAX
```

If JAX L-BFGS is too much work, add a TODO and keep Adam stable. Do not block correctness work on L-BFGS.

---

### E4. Architectures for local PINNs

Support local PINN architecture options:

```python
cfg.pinn.architecture = "mlp_tanh"
cfg.pinn.architecture = "siren"
cfg.pinn.architecture = "fourier_mlp"
```

For paper experiments, local networks must not be underpowered relative to baselines. Log parameter counts per chart and total.

---

## Part F — Profiling and Benchmarks

Add profiling scripts and instrumentation. Suggested files:

```text
scripts/profile_eikonal_step.py
scripts/profile_uae_forward.py
scripts/benchmark_metric_batch.py
scripts/benchmark_samplers.py
```

Each script should print a compact JSON or CSV summary:

```json
{
  "device": "...",
  "dtype": "float32",
  "num_charts": 40,
  "batch_residual": 1024,
  "compile_time_s": 12.3,
  "step_time_ms_mean": 18.4,
  "step_time_ms_std": 0.9,
  "memory_gb": 4.2
}
```

Separate timings for:

- data loading/sampling;
- UAE forward;
- decoder Jacobian/metric;
- PDE residual;
- interface loss;
- optimizer update;
- evaluation and plotting.

Use `block_until_ready()` when timing JAX code.

Example:

```python
start = time.perf_counter()
loss, aux, state = train_step(...)
jax.tree.map(lambda x: x.block_until_ready() if hasattr(x, "block_until_ready") else x, loss)
elapsed = time.perf_counter() - start
```

Acceptance criteria:

- There is a reproducible benchmark command for one Eikonal step and one UAE forward pass.
- Benchmarks compare old/legacy and new/vectorized code where feasible.

---

## Part G — Reproducibility and Project Hygiene

### G1. Add pinned environment

Add one of:

```text
pyproject.toml
requirements.txt
requirements-lock.txt
environment.yml
```

At minimum pin major packages:

```text
jax
jaxlib
flax
optax
ml-collections
numpy
scipy
networkx
trimesh
torch
matplotlib
wandb
pytest
```

Do not rely on “install latest packages”. JAX/Flax APIs change.

### G2. Replace deprecated JAX APIs

Search for deprecated APIs such as:

```python
jax.tree_map
```

Replace with:

```python
jax.tree.map
```

or:

```python
jax.tree_util.tree_map
```

### G3. Add Makefile or task runner

Add simple commands:

```makefile
install:
	pip install -e ./jaxpi
	pip install -e .

test:
	pytest -q
	python -m compileall manifold_pinns pinns universal_autoencoder datasets jaxpi

smoke-eikonal:
	python -m manifold_pinns.pipeline.cli pinn eikonal coil --mode train --override "train.num_steps=2,wandb.use=False"

profile-eikonal:
	python scripts/profile_eikonal_step.py
```

### G4. Improve CLI with smoke/dry-run mode

Add:

```bash
python -m manifold_pinns.pipeline.cli pinn eikonal coil --mode smoke
python -m manifold_pinns.pipeline.cli uae coil --mode smoke
```

or:

```bash
--smoke
```

Smoke mode should run on tiny synthetic data or a tiny subset and finish quickly.

---

## Part H — Future Experiment Compatibility Only

Do not implement the new paper experiments yet. Do not add cardiac datasets, biological membrane datasets, new external benchmark loaders, or full new experiment scripts. Those will be added later.

However, refactor the code so that the future experiments can be added without rewriting the core method.

The future experiments will likely involve:

1. sparse Eikonal inverse problems on new point-cloud surfaces;
2. PDEs on time-varying surfaces;
3. atlas-quality and geometry-generalization studies across many shapes.

Your task is to make the implementation compatible with these future experiments by exposing clean abstractions.

### H1. Dataset compatibility

Create or clean up generic data structures that can represent:

- ambient points, e.g. `[N, 3]`;
- optional mesh faces;
- optional normals;
- global point IDs;
- chart membership;
- local chart coordinates;
- paired chart-overlap coordinates;
- sparse observation locations and values;
- source/boundary/initial-condition data;
- optional time stamps;
- optional ground-truth solution fields;
- optional reference PDE coefficients.

Do not implement any new real dataset loader now. Use only tiny synthetic fixtures in tests.

### H2. PDE compatibility

Refactor PDE residual code so that new PDEs can be added through a common interface.

At minimum, preserve or support:

- static Eikonal residuals;
- source/boundary-condition losses;
- Laplace-Beltrami-based residuals;
- time-dependent PDE inputs;
- access to metric quantities `J`, `g`, `g_inv`, `sqrt_det_g`.

For moving-surface PDEs, do not implement a full physical model yet. Only expose hooks that would later allow access to:

- `d(z, t)`;
- `partial_t d(z, t)`;
- time-dependent metric terms;
- optional surface velocity.

### H3. Method and baseline compatibility

Create a clean method-selection mechanism so later experiments can compare:

- M-PINN with the legacy UPT/SIREN UAE;
- M-PINN with PCA/Monge charts;
- vanilla coordinate PINNs;
- Delta-PINN-style baselines if already present or easy to wrap.

Do not implement new third-party baselines now. Only make the interface clean enough that they can be added later.

### H4. Metrics compatibility

Create reusable metric functions for:

- MSE;
- RMSE;
- MAE;
- relative L2;
- correlation;
- residual loss;
- source or boundary-condition error;
- wall-clock time;
- preprocessing time;
- memory usage where feasible.

Do not create final paper tables yet. Return metrics as dictionaries, JSON, CSV, or pandas-compatible records.

### H5. Chart diagnostics compatibility

Implement reusable chart-diagnostic functions independent of any specific experiment:

- reconstruction error;
- metric condition number;
- minimum singular value of decoder Jacobian;
- `sqrt(det(g))` range;
- normal error if normals are provided;
- overlap reconstruction consistency;
- optional analytic operator error on synthetic charts.

These diagnostics should work on synthetic tests and on existing toy datasets.

### H6. Acceptance criteria

At the end of this refactor, it should be possible to add future experiments by writing new dataset loaders and config files, not by rewriting core chart, metric, overlap, PDE, or training logic.

Do not add the future experiments themselves.

---

## Part I — Suggested Internal Refactor Structure

Do not rewrite everything in one giant change if avoidable, but aim toward these modules:

```text
manifold_pinns/
  config/
  geometry/
    metrics.py
    operators.py
    overlaps.py
    charts.py
  data/
    datasets.py
    samplers.py
    batching.py
  models/
    pinn.py
    local_networks.py
    train_state.py
  uae/
    base.py
    upt_siren.py
    monge.py
    siren.py
    losses.py
  experiments/
    eikonal.py
    diffusion.py
    wave.py
  pipeline/
    cli.py
```

Given the current copy-first philosophy, do this incrementally. It is acceptable to first add shared utility modules and migrate Eikonal only. Do not break wave/diffusion while refactoring Eikonal.

---

## Part J — Concrete Implementation Checklist

Work through this checklist sequentially.

### Phase 1: Make current branch robust

- [ ] Fix imports so CLI works from repo root.
- [ ] Add install metadata or clear editable install script.
- [ ] Standardize model step return signatures.
- [ ] Fix UAE loss dispatch.
- [ ] Fix weighted geodesic distance computation.
- [ ] Replace float tuple overlap matching with point IDs or KD-tree tolerance.
- [ ] Add minimal tests and compileall.

### Phase 2: Fix interface losses

- [ ] Build paired overlap data structure.
- [ ] Refactor boundary sampler to sample paired coordinates.
- [ ] Refactor interface loss to use paired points.
- [ ] Add optional gradient/flux consistency.
- [ ] Add synthetic overlap tests.

### Phase 3: JAX performance

- [ ] Disable global x64 by default.
- [ ] Convert Eikonal samplers to JAX-native arrays.
- [ ] Vectorize Eikonal residual over points and charts.
- [ ] Add analytic 2x2 metric inverse.
- [ ] Add metric batch API.
- [ ] Add `lax.scan` training dispatch.
- [ ] Add profiling scripts.

### Phase 4: UAE speed and chart quality

- [ ] Precompute supernode neighborhoods.
- [ ] Subsample geodesic pair losses.
- [ ] Add immersion/condition/smoothness losses.
- [ ] Log chart diagnostics.
- [ ] Add chart-quality tests.

### Phase 5: Architecture alternative

- [ ] Implement `pca_monge` UAE architecture.
- [ ] Make downstream PINN consume either `upt_siren` or `pca_monge` decoders through a common interface.
- [ ] Add small training smoke test for `pca_monge`.
- [ ] Add comparison script for `upt_siren` vs `pca_monge` on reconstruction, metric quality, and downstream Eikonal error.

### Phase 6: Future experiment compatibility

- [ ] Add generic dataset containers for surfaces, charts, overlaps, sparse observations, and optional time.
- [ ] Add generic PDE residual interface.
- [ ] Add reusable metrics functions.
- [ ] Add method-selection/config interface.
- [ ] Add chart-diagnostic utilities.
- [ ] Add synthetic tests proving these interfaces work.
- [ ] Do not implement new real experiments or external dataset loaders in this phase.

---

## Part K — Expected Acceptance Criteria

At the end, the repository should satisfy:

```bash
make install
make test
python -m manifold_pinns.pipeline.cli --help
python -m manifold_pinns.pipeline.cli uae coil --smoke --override "wandb.use=False"
python -m manifold_pinns.pipeline.cli pinn eikonal coil --mode smoke --override "wandb.use=False"
python scripts/profile_eikonal_step.py
python scripts/benchmark_metric_batch.py
```

If Makefile is not added, equivalent commands must be documented in README.

Functional expectations:

- Eikonal one-step smoke training runs without import hacks.
- UAE one-step smoke training runs without import hacks.
- Changing `cfg.train.reg` actually changes the UAE optimized loss.
- Interface sampler compares the same physical overlap points.
- Flat metric / Eikonal / Laplacian tests pass.
- Float32 default works.
- Main training step does not recompile every iteration for fixed shapes.
- Profiling reports separate timings for sampler, metric/Jacobian, residual, interface, optimizer.
- `pca_monge` architecture exists and can be selected from config.

---

## Part L — Important Cautions

1. **Do not optimize a mathematically wrong loss.** Fix overlap pairing before spending time on GPU speed.
2. **Do not assume visual reconstruction implies good charts.** Always report metric condition and derivative diagnostics.
3. **Do not rely on global float64 to hide ill-conditioned charts.** Add chart regularization.
4. **Do not compare against weak baselines only.** The code should make it possible to compare against Delta-PINNs and vanilla PINNs at minimum, with matched model capacity and training budgets.
5. **Do not make the new Monge architecture replace the old UAE silently.** It should be a config option and an ablation.
6. **Do not break legacy experiments.** If old behavior is scientifically questionable, preserve it under a legacy flag and make the new default more correct.
7. **Do not log inside JIT.** Return aux values and log outside compiled functions.
8. **Do not pass variable-length Python lists/dicts into JIT hot paths.** Use padded arrays and masks.
9. **Do not attempt multi-GPU before chart-axis batching is clean.** Single-GPU vectorization is the immediate priority.

---

## Part M — Summary of the Desired End State

The final codebase should make this scientific claim testable:

> M-PINNs learn amortized differentiable local atlases for raw point-cloud surfaces. These atlases provide stable intrinsic metrics for local domain-decomposed PINNs, enabling sparse inverse PDE learning and geometry transfer on complex and changing surfaces with lower per-geometry overhead than spectral or mesh-dependent PINN baselines.

The code should reflect that claim. The most important implementation objects are not generic neural-network components; they are:

- reliable chart construction;
- paired overlap maps;
- stable decoder Jacobians;
- correct Riemannian metrics;
- correct intrinsic PDE operators;
- efficient vectorized residual evaluation;
- reproducible experiments and baselines.

Implement changes accordingly.
