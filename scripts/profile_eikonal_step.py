"""Profile one synthetic Eikonal training step."""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import jax
import jax.numpy as jnp
import optax
from flax.training import train_state

from manifold_pinns.geometry.operators import eikonal_residual


def main() -> None:
    decoder = lambda z: jnp.array([z[0], z[1], 0.0])
    z_batch = jax.random.uniform(jax.random.PRNGKey(0), (128, 2))
    params = {"w": jnp.array([0.7, 0.1]), "b": jnp.array(0.05)}
    tx = optax.adam(1e-2)
    state = train_state.TrainState.create(
        apply_fn=lambda variables, z: jnp.dot(variables["params"]["w"], z)
        + variables["params"]["b"],
        params=params,
        tx=tx,
    )

    def loss_fn(params):
        u_fn = lambda z: jnp.dot(params["w"], z) + params["b"]
        residual = eikonal_residual(decoder, u_fn, z_batch)
        res_loss = jnp.mean(residual**2)
        source_loss = u_fn(jnp.array([0.0, 0.0])) ** 2
        return res_loss + source_loss, {"res": res_loss, "source": source_loss}

    @jax.jit
    def step(state):
        (loss, aux), grads = jax.value_and_grad(loss_fn, has_aux=True)(state.params)
        return state.apply_gradients(grads=grads), loss, aux

    compile_start = time.perf_counter()
    state, loss, aux = step(state)
    loss.block_until_ready()
    compile_time_s = time.perf_counter() - compile_start

    times = []
    for _ in range(20):
        start = time.perf_counter()
        state, loss, aux = step(state)
        loss.block_until_ready()
        times.append(time.perf_counter() - start)

    device = str(jax.devices()[0])
    summary = {
        "device": device,
        "dtype": str(z_batch.dtype),
        "num_charts": 1,
        "batch_residual": int(z_batch.shape[0]),
        "compile_time_s": compile_time_s,
        "step_time_ms_mean": float(1000.0 * jnp.mean(jnp.array(times))),
        "step_time_ms_std": float(1000.0 * jnp.std(jnp.array(times))),
        "sampler_time_ms_mean": 0.0,
        "metric_residual_optimizer_time_ms_mean": float(1000.0 * jnp.mean(jnp.array(times))),
        "loss": float(loss),
        "res_loss": float(aux["res"]),
        "source_bc_loss": float(aux["source"]),
    }
    print(json.dumps(summary, sort_keys=True))


if __name__ == "__main__":
    main()
