"""Geometry primitives for chart-based manifold PINNs."""

from .metrics import MetricBatch, inv_2x2_spd, metric_batch
from .operators import (
    ambient_surface_gradient,
    eikonal_residual,
    eikonal_residual_from_grad,
    laplace_beltrami,
    laplace_beltrami_single,
)
from .overlaps import (
    OverlapPairs,
    build_overlap_pairs_from_point_ids,
    build_overlap_pairs_kdtree,
    interface_loss,
    sample_overlap_pairs,
)

__all__ = [
    "MetricBatch",
    "OverlapPairs",
    "ambient_surface_gradient",
    "build_overlap_pairs_from_point_ids",
    "build_overlap_pairs_kdtree",
    "eikonal_residual",
    "eikonal_residual_from_grad",
    "interface_loss",
    "inv_2x2_spd",
    "laplace_beltrami",
    "laplace_beltrami_single",
    "metric_batch",
    "sample_overlap_pairs",
]
