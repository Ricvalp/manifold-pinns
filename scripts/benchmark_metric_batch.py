"""Benchmark batched induced metric computation on a synthetic Monge patch."""

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

from manifold_pinns.geometry.metrics import metric_batch


def main() -> None:
    z = jax.random.uniform(jax.random.PRNGKey(1), (1024, 2), minval=-1.0, maxval=1.0)

    def decoder(zz):
        return jnp.array([zz[0], zz[1], 0.1 * zz[0] ** 2 - 0.2 * zz[1] ** 2])

    @jax.jit
    def compute(z_batch):
        return metric_batch(decoder, z_batch)

    compile_start = time.perf_counter()
    metrics = compute(z)
    metrics.sqrt_det_g.block_until_ready()
    compile_time_s = time.perf_counter() - compile_start

    times = []
    for _ in range(20):
        start = time.perf_counter()
        metrics = compute(z)
        metrics.sqrt_det_g.block_until_ready()
        times.append(time.perf_counter() - start)

    summary = {
        "device": str(jax.devices()[0]),
        "dtype": str(z.dtype),
        "num_points": int(z.shape[0]),
        "compile_time_s": compile_time_s,
        "metric_time_ms_mean": float(1000.0 * jnp.mean(jnp.array(times))),
        "metric_time_ms_std": float(1000.0 * jnp.std(jnp.array(times))),
        "cond_mean": float(jnp.mean(metrics.cond)),
        "sqrt_det_g_min": float(jnp.min(metrics.sqrt_det_g)),
        "sqrt_det_g_max": float(jnp.max(metrics.sqrt_det_g)),
    }
    print(json.dumps(summary, sort_keys=True))


if __name__ == "__main__":
    main()
