"""Create chart-atlas pickle files consumed by the Eikonal PINN pipeline."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import ml_collections

from charts.get_charts import get_charts, save_charts
from datasets.utils import Mesh
from manifold_pinns.pipeline.env import data_path


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="coil", choices=["coil", "bunny", "propeller"])
    parser.add_argument("--mesh", help="Input OBJ mesh path.")
    parser.add_argument("--output", help="Output atlas directory.")
    parser.add_argument("--min-dist", type=float, default=9.0)
    parser.add_argument("--nearest-neighbors", type=int, default=10)
    parser.add_argument("--summary", help="Optional JSON summary path.")
    args = parser.parse_args(argv)

    default_meshes = {
        "coil": ROOT / "datasets" / "obj_files" / "coil.obj",
        "bunny": ROOT / "datasets" / "obj_files" / "stanford_bunny.obj",
        "propeller": ROOT / "datasets" / "obj_files" / "propeller.obj",
    }
    mesh_path = Path(args.mesh) if args.mesh else default_meshes[args.dataset]
    output = Path(args.output) if args.output else Path(data_path(args.dataset, "charts_1"))

    mesh = Mesh(str(mesh_path))
    charts_cfg = ml_collections.ConfigDict()
    charts_cfg.alg = "fast_region_growing"
    charts_cfg.min_dist = args.min_dist
    charts_cfg.nearest_neighbors = args.nearest_neighbors

    charts, charts_idxs, boundaries, boundary_indices, _ = get_charts(
        points=mesh.verts,
        charts_config=charts_cfg,
    )
    save_charts(
        str(output),
        charts,
        charts_idxs,
        boundaries,
        boundary_indices,
        verts=mesh.verts,
        connectivity=mesh.connectivity,
    )

    summary = {
        "dataset": args.dataset,
        "mesh": str(mesh_path.resolve()),
        "output": str(output.resolve()),
        "num_vertices": int(mesh.verts.shape[0]),
        "num_faces": int(mesh.connectivity.shape[0]),
        "num_charts": int(len(charts)),
        "num_boundary_pairs": int(len(boundary_indices)),
        "min_dist": args.min_dist,
        "nearest_neighbors": args.nearest_neighbors,
    }
    payload = json.dumps(summary, indent=2, sort_keys=True)
    if args.summary:
        summary_path = Path(args.summary)
        summary_path.parent.mkdir(parents=True, exist_ok=True)
        summary_path.write_text(payload + "\n", encoding="utf-8")
    print(payload)


if __name__ == "__main__":
    main()
