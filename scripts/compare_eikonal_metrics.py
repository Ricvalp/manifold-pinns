"""Compare Eikonal evaluation metrics written by ``plot_correlation``."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path


METRIC_KEYS = [
    "correlation",
    "rmse",
    "relative_l2",
    "mae",
    "tail_rmse",
    "tail_relative_l2",
    "tail_bias",
    "pred_max",
    "target_max",
]


def _read_metrics(label: str, path: Path) -> dict:
    metrics = json.loads(path.read_text(encoding="utf-8"))
    return {
        "label": label,
        "path": str(path),
        **{key: metrics.get(key) for key in METRIC_KEYS},
    }


def _write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = ["label", *METRIC_KEYS, "path"]
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--metric",
        nargs=2,
        action="append",
        metavar=("LABEL", "JSON"),
        required=True,
        help="Metric label and plot_correlation JSON path. Repeat for each run.",
    )
    parser.add_argument("--output-csv", required=True)
    parser.add_argument("--output-json")
    args = parser.parse_args(argv)

    rows = [_read_metrics(label, Path(path)) for label, path in args.metric]
    _write_csv(Path(args.output_csv), rows)
    if args.output_json:
        output_json = Path(args.output_json)
        output_json.parent.mkdir(parents=True, exist_ok=True)
        output_json.write_text(
            json.dumps(rows, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )

    print(json.dumps(rows, sort_keys=True))


if __name__ == "__main__":
    main()
