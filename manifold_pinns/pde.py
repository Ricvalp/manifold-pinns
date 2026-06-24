"""Common PDE residual interfaces for chart-based PINNs."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Protocol

import jax.numpy as jnp

from manifold_pinns.geometry.metrics import MetricBatch
from manifold_pinns.geometry.operators import eikonal_residual_from_grad


class ChartPDE(Protocol):
    """Protocol for PDE residuals evaluated in local chart coordinates."""

    def residual(
        self,
        u_fn: Callable[[jnp.ndarray], jnp.ndarray],
        z_batch: jnp.ndarray,
        metrics: MetricBatch,
        *,
        time: jnp.ndarray | None = None,
    ) -> jnp.ndarray:
        ...


@dataclass(frozen=True)
class EikonalPDE:
    """Static Eikonal residual ``||grad_M T|| = 1 / c``."""

    speed: float = 1.0

    def residual_from_grad(self, grad_z_u: jnp.ndarray, metrics: MetricBatch) -> jnp.ndarray:
        return eikonal_residual_from_grad(
            grad_z_u,
            metrics.ginv,
            target_norm=1.0 / self.speed,
        )
