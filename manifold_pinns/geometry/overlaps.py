"""Paired chart-overlap data structures and losses."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Mapping

import numpy as np
from scipy.spatial import cKDTree


@dataclass(frozen=True)
class OverlapPairs:
    """Coordinates for the same physical points in two overlapping charts."""

    src_chart: int
    dst_chart: int
    z_src: np.ndarray
    z_dst: np.ndarray
    x_ambient: np.ndarray | None = None
    point_ids: np.ndarray | None = None

    def __post_init__(self) -> None:
        if len(self.z_src) != len(self.z_dst):
            raise ValueError("z_src and z_dst must contain paired rows.")
        if self.x_ambient is not None and len(self.x_ambient) != len(self.z_src):
            raise ValueError("x_ambient must have one row per overlap pair.")
        if self.point_ids is not None and len(self.point_ids) != len(self.z_src):
            raise ValueError("point_ids must have one value per overlap pair.")

    @property
    def size(self) -> int:
        return int(len(self.z_src))


def build_overlap_pairs_from_point_ids(
    src_chart: int,
    dst_chart: int,
    coords_by_chart: Mapping[int, np.ndarray],
    point_ids_by_chart: Mapping[int, np.ndarray],
    ambient_by_chart: Mapping[int, np.ndarray] | None = None,
) -> OverlapPairs:
    """Build paired overlap coordinates using shared global point IDs."""

    src_ids = np.asarray(point_ids_by_chart[src_chart])
    dst_ids = np.asarray(point_ids_by_chart[dst_chart])
    point_ids, src_idx, dst_idx = np.intersect1d(
        src_ids,
        dst_ids,
        assume_unique=False,
        return_indices=True,
    )
    x_ambient = None
    if ambient_by_chart is not None and len(point_ids) > 0:
        x_ambient = np.asarray(ambient_by_chart[src_chart])[src_idx]
    return OverlapPairs(
        src_chart=src_chart,
        dst_chart=dst_chart,
        z_src=np.asarray(coords_by_chart[src_chart])[src_idx],
        z_dst=np.asarray(coords_by_chart[dst_chart])[dst_idx],
        x_ambient=x_ambient,
        point_ids=point_ids,
    )


def build_overlap_pairs_kdtree(
    src_chart: int,
    dst_chart: int,
    coords_by_chart: Mapping[int, np.ndarray],
    ambient_by_chart: Mapping[int, np.ndarray],
    *,
    tolerance: float = 1e-8,
) -> OverlapPairs:
    """Build paired overlaps by tolerance matching in ambient space.

    This fallback is for data without stable global point IDs.
    """

    src_x = np.asarray(ambient_by_chart[src_chart])
    dst_x = np.asarray(ambient_by_chart[dst_chart])
    tree = cKDTree(dst_x)
    src_indices = []
    dst_indices = []
    for src_idx, point in enumerate(src_x):
        matches = tree.query_ball_point(point, tolerance)
        for dst_idx in matches:
            src_indices.append(src_idx)
            dst_indices.append(dst_idx)
    src_indices = np.asarray(src_indices, dtype=np.int64)
    dst_indices = np.asarray(dst_indices, dtype=np.int64)
    return OverlapPairs(
        src_chart=src_chart,
        dst_chart=dst_chart,
        z_src=np.asarray(coords_by_chart[src_chart])[src_indices],
        z_dst=np.asarray(coords_by_chart[dst_chart])[dst_indices],
        x_ambient=src_x[src_indices],
        point_ids=None,
    )


def sample_overlap_pairs(
    pair: OverlapPairs,
    batch_size: int,
    rng: np.random.Generator | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Sample paired overlap coordinates with one shared index vector."""

    if pair.size == 0:
        raise ValueError("Cannot sample from an empty overlap.")
    if rng is None:
        rng = np.random.default_rng()
    idx = rng.integers(0, pair.size, size=(batch_size,))
    return pair.z_src[idx], pair.z_dst[idx]


def interface_loss(
    u_src: Callable[[np.ndarray], np.ndarray],
    u_dst: Callable[[np.ndarray], np.ndarray],
    pair: OverlapPairs,
) -> float:
    """Mean squared interface mismatch on paired overlap points."""

    pred_src = np.asarray(u_src(pair.z_src))
    pred_dst = np.asarray(u_dst(pair.z_dst))
    return float(np.mean((pred_src - pred_dst) ** 2))
