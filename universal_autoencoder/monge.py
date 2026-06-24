"""PCA/Monge chart decoder alternative for stable local atlases."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import jax.numpy as jnp
import numpy as np


def _quadratic_features(z: np.ndarray) -> np.ndarray:
    z = np.asarray(z)
    z1 = z[..., 0]
    z2 = z[..., 1]
    return np.stack(
        [
            np.ones_like(z1),
            z1,
            z2,
            z1**2,
            z1 * z2,
            z2**2,
        ],
        axis=-1,
    )


def _quadratic_features_jax(z: jnp.ndarray) -> jnp.ndarray:
    z = jnp.asarray(z)
    z1 = z[..., 0]
    z2 = z[..., 1]
    return jnp.stack(
        [
            jnp.ones_like(z1),
            z1,
            z2,
            z1**2,
            z1 * z2,
            z2**2,
        ],
        axis=-1,
    )


@dataclass(frozen=True)
class PcaMongeChart:
    """Deterministic local PCA coordinates plus a quadratic Monge height field."""

    mu: np.ndarray
    frame: np.ndarray
    coeffs: np.ndarray
    use_inplane_residual: bool = False

    def encode(self, points: np.ndarray) -> np.ndarray:
        centered = np.asarray(points) - self.mu
        return centered @ self.frame[:, :2]

    def height(self, z: np.ndarray) -> np.ndarray:
        return _quadratic_features(z) @ self.coeffs

    def decode(self, z: np.ndarray) -> np.ndarray:
        z = np.asarray(z)
        h = self.height(z)
        local = np.stack([z[..., 0], z[..., 1], h], axis=-1)
        return self.mu + local @ self.frame.T

    def decode_jax(self, z: jnp.ndarray) -> jnp.ndarray:
        coeffs = jnp.asarray(self.coeffs)
        frame = jnp.asarray(self.frame)
        mu = jnp.asarray(self.mu)
        h = _quadratic_features_jax(z) @ coeffs
        local = jnp.stack([z[..., 0], z[..., 1], h], axis=-1)
        return mu + local @ frame.T

    def reconstruction_mse(self, points: np.ndarray) -> float:
        z = self.encode(points)
        recon = self.decode(z)
        return float(np.mean((recon - points) ** 2))


def fit_pca_monge_chart(points: np.ndarray) -> PcaMongeChart:
    """Fit a deterministic PCA/Monge decoder to one local point cloud chart."""

    points = np.asarray(points, dtype=np.float64)
    if points.ndim != 2 or points.shape[1] != 3:
        raise ValueError("points must have shape [N, 3]")
    mu = points.mean(axis=0)
    centered = points - mu
    cov = centered.T @ centered / max(len(points) - 1, 1)
    eigvals, eigvecs = np.linalg.eigh(cov)
    order = np.argsort(eigvals)[::-1]
    frame = eigvecs[:, order]
    if np.linalg.det(frame) < 0:
        frame[:, -1] *= -1

    local = centered @ frame
    z = local[:, :2]
    height = local[:, 2]
    coeffs, *_ = np.linalg.lstsq(_quadratic_features(z), height, rcond=None)
    return PcaMongeChart(mu=mu, frame=frame, coeffs=coeffs)


def fit_pca_monge_atlas(charts: Iterable[np.ndarray]) -> list[PcaMongeChart]:
    """Fit one PCA/Monge decoder per chart."""

    return [fit_pca_monge_chart(chart) for chart in charts]
