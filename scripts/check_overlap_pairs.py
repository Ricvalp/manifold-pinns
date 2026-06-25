"""Emit overlap-pairing/interface-loss diagnostics as JSON."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np

from manifold_pinns.geometry.overlaps import (
    OverlapPairs,
    build_overlap_pairs_from_point_ids,
    interface_loss,
)


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
    st = rng.uniform([-1.0, -0.5], [1.0, 0.5], size=(512, 2))
    point_ids = np.arange(10_000, 10_000 + len(st))

    theta = 0.37
    rotation = np.array(
        [
            [np.cos(theta), -np.sin(theta)],
            [np.sin(theta), np.cos(theta)],
        ]
    )
    shift = np.array([0.4, -0.15])

    z_a = st
    z_b = st @ rotation.T + shift
    ambient = np.column_stack([st, np.zeros(len(st))])

    pair = build_overlap_pairs_from_point_ids(
        0,
        1,
        coords_by_chart={0: z_a, 1: z_b},
        point_ids_by_chart={0: point_ids, 1: point_ids},
        ambient_by_chart={0: ambient, 1: ambient},
    )

    def decode_a(z: np.ndarray) -> np.ndarray:
        return z

    def decode_b(z: np.ndarray) -> np.ndarray:
        return (z - shift) @ rotation

    ambient_a = np.column_stack([decode_a(pair.z_src), np.zeros(pair.size)])
    ambient_b = np.column_stack([decode_b(pair.z_dst), np.zeros(pair.size)])
    max_ambient_pair_error = float(np.max(np.abs(ambient_a - ambient_b)))

    u_ambient = lambda xy: 2.0 * xy[:, 0] - xy[:, 1] + 0.3
    u_a = lambda z: u_ambient(decode_a(z))
    u_b = lambda z: u_ambient(decode_b(z))

    paired_interface_loss = interface_loss(u_a, u_b, pair)
    shuffled = OverlapPairs(
        src_chart=pair.src_chart,
        dst_chart=pair.dst_chart,
        z_src=pair.z_src,
        z_dst=pair.z_dst[rng.permutation(pair.size)],
        x_ambient=pair.x_ambient,
        point_ids=pair.point_ids,
    )
    shuffled_interface_loss = interface_loss(u_a, u_b, shuffled)

    summary = {
        "num_pairs": pair.size,
        "max_ambient_pair_error": max_ambient_pair_error,
        "paired_interface_loss": paired_interface_loss,
        "shuffled_interface_loss": shuffled_interface_loss,
        "ratio": float(
            shuffled_interface_loss / max(paired_interface_loss, np.finfo(float).eps)
        ),
    }
    summary["passed"] = (
        pair.size == len(st)
        and max_ambient_pair_error < 1e-12
        and paired_interface_loss < 1e-12
        and shuffled_interface_loss > 1e-3
    )
    _write_json(summary, args.output)


if __name__ == "__main__":
    main()
