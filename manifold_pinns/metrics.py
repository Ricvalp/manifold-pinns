"""Reusable scalar metrics for experiments and benchmarks."""

from __future__ import annotations

import time
from contextlib import contextmanager
from typing import Iterator

import numpy as np


def mse(pred, target) -> float:
    pred = np.asarray(pred)
    target = np.asarray(target)
    return float(np.mean((pred - target) ** 2))


def rmse(pred, target) -> float:
    return float(np.sqrt(mse(pred, target)))


def mae(pred, target) -> float:
    pred = np.asarray(pred)
    target = np.asarray(target)
    return float(np.mean(np.abs(pred - target)))


def relative_l2(pred, target, eps: float = 1e-12) -> float:
    pred = np.asarray(pred)
    target = np.asarray(target)
    return float(np.linalg.norm(pred - target) / (np.linalg.norm(target) + eps))


def correlation(pred, target) -> float:
    pred = np.asarray(pred).reshape(-1)
    target = np.asarray(target).reshape(-1)
    if pred.size < 2:
        return float("nan")
    return float(np.corrcoef(pred, target)[0, 1])


@contextmanager
def wall_clock_timer(record: dict[str, float], key: str = "wall_time_s") -> Iterator[None]:
    start = time.perf_counter()
    try:
        yield
    finally:
        record[key] = time.perf_counter() - start
