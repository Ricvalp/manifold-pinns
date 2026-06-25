"""Emit analytic geometry/operator correctness diagnostics as JSON."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import jax.numpy as jnp
import numpy as np

from manifold_pinns.geometry.metrics import metric_batch
from manifold_pinns.geometry.operators import eikonal_residual, laplace_beltrami


def _write_json(summary: dict, output: str | None) -> None:
    payload = json.dumps(summary, indent=2, sort_keys=True)
    if output is None:
        print(payload)
        return
    output_path = Path(output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(payload + "\n", encoding="utf-8")


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", help="Path for the JSON summary.")
    args = parser.parse_args(argv)

    rng = np.random.default_rng(0)
    z = jnp.asarray(rng.uniform(-1.0, 1.0, size=(4096, 2)), dtype=jnp.float32)

    plane_decoder = lambda zz: jnp.array([zz[0], zz[1], 0.0])
    flat_metrics = metric_batch(plane_decoder, z)
    eye = jnp.tile(jnp.eye(2), (z.shape[0], 1, 1))
    flat_metric = {
        "max_abs_g_minus_I": float(jnp.max(jnp.abs(flat_metrics.g - eye))),
        "max_abs_ginv_minus_I": float(jnp.max(jnp.abs(flat_metrics.ginv - eye))),
        "max_abs_sqrt_det_minus_1": float(jnp.max(jnp.abs(flat_metrics.sqrt_det_g - 1.0))),
        "max_abs_cond_minus_1": float(jnp.max(jnp.abs(flat_metrics.cond - 1.0))),
    }
    flat_metric["passed"] = (
        flat_metric["max_abs_g_minus_I"] < 1e-6
        and flat_metric["max_abs_ginv_minus_I"] < 1e-6
        and flat_metric["max_abs_sqrt_det_minus_1"] < 1e-6
        and flat_metric["max_abs_cond_minus_1"] < 1e-6
    )

    poly = lambda zz: zz[0] ** 2 + zz[1] ** 2
    lap_error = jnp.abs(laplace_beltrami(plane_decoder, poly, z) - 4.0)
    flat_laplacian = {
        "mean_abs_laplace_error": float(jnp.mean(lap_error)),
        "max_abs_laplace_error": float(jnp.max(lap_error)),
    }
    flat_laplacian["passed"] = (
        flat_laplacian["mean_abs_laplace_error"] < 1e-5
        and flat_laplacian["max_abs_laplace_error"] < 1e-4
    )

    u_linear = lambda zz: zz[0]
    eik_error = jnp.abs(eikonal_residual(plane_decoder, u_linear, z))
    flat_eikonal = {
        "eikonal_residual_mean_abs": float(jnp.mean(eik_error)),
        "eikonal_residual_max_abs": float(jnp.max(eik_error)),
    }
    flat_eikonal["passed"] = (
        flat_eikonal["eikonal_residual_mean_abs"] < 1e-6
        and flat_eikonal["eikonal_residual_max_abs"] < 1e-6
    )

    a, b, c = 0.15, -0.07, 0.11
    monge_decoder = lambda zz: jnp.array(
        [zz[0], zz[1], a * zz[0] ** 2 + b * zz[0] * zz[1] + c * zz[1] ** 2]
    )
    monge_metrics = metric_batch(monge_decoder, z)
    hz1 = 2.0 * a * z[:, 0] + b * z[:, 1]
    hz2 = b * z[:, 0] + 2.0 * c * z[:, 1]
    expected_g = jnp.stack(
        [
            jnp.stack([1.0 + hz1**2, hz1 * hz2], axis=-1),
            jnp.stack([hz1 * hz2, 1.0 + hz2**2], axis=-1),
        ],
        axis=-2,
    )
    expected_ginv = jnp.linalg.inv(expected_g)
    metric_error = jnp.abs(monge_metrics.g - expected_g)
    ginv_error = jnp.abs(monge_metrics.ginv - expected_ginv)
    cond_np = np.asarray(monge_metrics.cond)
    monge_metric = {
        "mean_abs_metric_error": float(jnp.mean(metric_error)),
        "max_abs_metric_error": float(jnp.max(metric_error)),
        "mean_abs_ginv_error": float(jnp.mean(ginv_error)),
        "max_abs_ginv_error": float(jnp.max(ginv_error)),
        "sqrt_det_g_min": float(jnp.min(monge_metrics.sqrt_det_g)),
        "sqrt_det_g_max": float(jnp.max(monge_metrics.sqrt_det_g)),
        "cond_mean": float(np.mean(cond_np)),
        "cond_p95": float(np.percentile(cond_np, 95)),
        "cond_max": float(np.max(cond_np)),
    }
    monge_metric["passed"] = (
        monge_metric["mean_abs_metric_error"] < 1e-6
        and monge_metric["max_abs_metric_error"] < 1e-5
        and monge_metric["mean_abs_ginv_error"] < 1e-5
        and monge_metric["max_abs_ginv_error"] < 1e-4
        and monge_metric["sqrt_det_g_min"] > 0.0
        and monge_metric["cond_max"] < 10.0
    )

    summary = {
        "flat_metric": flat_metric,
        "flat_laplacian": flat_laplacian,
        "flat_eikonal": flat_eikonal,
        "monge_metric": monge_metric,
    }
    summary["passed"] = all(section["passed"] for section in summary.values())
    _write_json(summary, args.output)


if __name__ == "__main__":
    main()
