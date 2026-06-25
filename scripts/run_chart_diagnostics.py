"""Run chart-quality diagnostics for generated UAE chart datasets."""

from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import jax
import jax.numpy as jnp
import numpy as np

from manifold_pinns.geometry.metrics import metric_batch
from universal_autoencoder.monge import fit_pca_monge_chart


def _load_chart_arrays(charts_path: Path, max_charts: int | None) -> list[np.ndarray]:
    chart_files = sorted(charts_path.glob("charts_*.npy"))
    if not chart_files:
        raise FileNotFoundError(f"No charts_*.npy files found in {charts_path}")

    charts: list[np.ndarray] = []
    for chart_file in chart_files:
        arr = np.load(chart_file, allow_pickle=True)
        for chart in arr:
            charts.append(np.asarray(chart, dtype=np.float64))
            if max_charts is not None and len(charts) >= max_charts:
                return charts
    return charts


def _subsample(chart: np.ndarray, num_points: int | None, rng: np.random.Generator) -> np.ndarray:
    if num_points is None or chart.shape[0] <= num_points:
        return chart
    idx = rng.choice(chart.shape[0], size=num_points, replace=False)
    return chart[idx]


def _diagnose_pca_monge(chart: np.ndarray) -> dict[str, float]:
    model = fit_pca_monge_chart(chart)
    z = jnp.asarray(model.encode(chart), dtype=jnp.float32)
    x_target = jnp.asarray(chart, dtype=jnp.float32)
    decoder = lambda zz: model.decode_jax(zz)

    decode_fn = jax.jit(jax.vmap(decoder))
    metric_fn = jax.jit(lambda zz: metric_batch(decoder, zz))

    start = time.perf_counter()
    x_pred = decode_fn(z)
    x_pred.block_until_ready()
    decode_time_ms = 1000.0 * (time.perf_counter() - start)

    start = time.perf_counter()
    metrics = metric_fn(z)
    metrics.sqrt_det_g.block_until_ready()
    metric_time_ms = 1000.0 * (time.perf_counter() - start)

    reconstruction_mse = float(jnp.mean((x_pred - x_target) ** 2))
    singular_values = np.linalg.svd(np.asarray(metrics.J), compute_uv=False)
    min_singular = np.min(singular_values, axis=-1)
    max_singular = np.max(singular_values, axis=-1)
    cond = np.asarray(metrics.cond)
    sqrt_det_g = np.asarray(metrics.sqrt_det_g)

    return {
        "reconstruction_mse": reconstruction_mse,
        "reconstruction_rmse": float(np.sqrt(reconstruction_mse)),
        "metric_cond_mean": float(np.mean(cond)),
        "metric_cond_median": float(np.median(cond)),
        "metric_cond_p95": float(np.percentile(cond, 95)),
        "metric_cond_max": float(np.max(cond)),
        "sqrt_det_g_min": float(np.min(sqrt_det_g)),
        "sqrt_det_g_median": float(np.median(sqrt_det_g)),
        "sqrt_det_g_max": float(np.max(sqrt_det_g)),
        "min_singular_value_J_min": float(np.min(min_singular)),
        "min_singular_value_J_median": float(np.median(min_singular)),
        "min_singular_value_J_p05": float(np.percentile(min_singular, 5)),
        "max_singular_value_J_p95": float(np.percentile(max_singular, 95)),
        "decode_time_ms_mean": decode_time_ms,
        "metric_time_ms_mean": metric_time_ms,
        "failure_flag": False,
        "failure_reason": "",
    }


def _aggregate(rows: list[dict[str, float | int | str | bool]], architecture: str) -> dict[str, float | int | str]:
    failures = [row for row in rows if row["failure_flag"]]
    valid = [row for row in rows if not row["failure_flag"]]
    out: dict[str, float | int | str] = {
        "architecture": architecture,
        "num_charts": len(rows),
        "num_failures": len(failures),
    }
    if not valid:
        return out

    def values(key: str) -> np.ndarray:
        return np.asarray([float(row[key]) for row in valid])

    out.update(
        {
            "reconstruction_rmse_mean": float(np.mean(values("reconstruction_rmse"))),
            "reconstruction_rmse_std": float(np.std(values("reconstruction_rmse"))),
            "metric_cond_median": float(np.median(values("metric_cond_median"))),
            "metric_cond_p95": float(np.percentile(values("metric_cond_p95"), 95)),
            "metric_cond_max": float(np.max(values("metric_cond_max"))),
            "sqrt_det_g_min": float(np.min(values("sqrt_det_g_min"))),
            "sqrt_det_g_max": float(np.max(values("sqrt_det_g_max"))),
            "min_singular_value_J_p05": float(np.percentile(values("min_singular_value_J_p05"), 5)),
            "decode_time_ms_mean": float(np.mean(values("decode_time_ms_mean"))),
            "metric_time_ms_mean": float(np.mean(values("metric_time_ms_mean"))),
        }
    )
    return out


def _write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="coil")
    parser.add_argument("--architecture", required=True)
    parser.add_argument("--charts-path", required=True)
    parser.add_argument("--max-charts", type=int, default=64)
    parser.add_argument("--num-points", type=int, default=512)
    parser.add_argument("--output", required=True, help="Aggregate JSON output path.")
    parser.add_argument("--per-chart-csv", help="Per-chart CSV output path.")
    parser.add_argument("--aggregate-csv", help="Single-row aggregate CSV output path.")
    args = parser.parse_args(argv)

    if args.architecture != "pca_monge":
        raise NotImplementedError(
            "run_chart_diagnostics.py currently supports --architecture pca_monge. "
            "Use the UAE training checkpoint logs for neural architecture diagnostics."
        )

    rng = np.random.default_rng(0)
    charts = _load_chart_arrays(Path(args.charts_path), args.max_charts)
    rows: list[dict] = []
    for chart_id, chart in enumerate(charts):
        row: dict[str, float | int | str | bool] = {
            "dataset": args.dataset,
            "architecture": args.architecture,
            "chart_id": chart_id,
            "num_points": int(min(chart.shape[0], args.num_points)),
        }
        try:
            row.update(_diagnose_pca_monge(_subsample(chart, args.num_points, rng)))
        except Exception as exc:  # pragma: no cover - diagnostic path
            row.update(
                {
                    "failure_flag": True,
                    "failure_reason": repr(exc),
                }
            )
        rows.append(row)

    aggregate = _aggregate(rows, args.architecture)
    summary = {
        "dataset": args.dataset,
        "architecture": args.architecture,
        "charts_path": str(Path(args.charts_path).resolve()),
        "aggregate": aggregate,
        "passed": bool(
            aggregate.get("num_failures", 0) == 0
            and float(aggregate.get("sqrt_det_g_min", 0.0)) > 0.0
            and np.isfinite(float(aggregate.get("metric_cond_max", np.inf)))
        ),
    }

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    per_chart_csv = (
        Path(args.per_chart_csv)
        if args.per_chart_csv
        else output.with_name(f"chart_metrics_{args.architecture}.csv")
    )
    _write_csv(per_chart_csv, rows)

    if args.aggregate_csv:
        _write_csv(Path(args.aggregate_csv), [aggregate])


if __name__ == "__main__":
    main()
