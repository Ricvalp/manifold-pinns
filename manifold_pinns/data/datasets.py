"""Generic data containers shared by current and future experiments."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping

import numpy as np

from manifold_pinns.geometry.overlaps import OverlapPairs


@dataclass(frozen=True)
class SurfaceData:
    """A point-cloud or mesh surface with optional geometry annotations."""

    points: np.ndarray
    faces: np.ndarray | None = None
    normals: np.ndarray | None = None
    point_ids: np.ndarray | None = None
    time: float | None = None


@dataclass(frozen=True)
class ChartData:
    """Local chart coordinates and membership for one surface atlas."""

    ambient_points: Mapping[int, np.ndarray]
    local_coords: Mapping[int, np.ndarray]
    point_ids: Mapping[int, np.ndarray]
    chart_metadata: Mapping[int, dict] | None = None


@dataclass(frozen=True)
class OverlapData:
    """Paired chart overlap samples."""

    pairs: Mapping[tuple[int, int], OverlapPairs]


@dataclass(frozen=True)
class SparseObservations:
    """Sparse PDE observations, boundary data, or source constraints."""

    locations: np.ndarray
    values: np.ndarray
    chart_ids: np.ndarray | None = None
    point_ids: np.ndarray | None = None
    time: np.ndarray | None = None
