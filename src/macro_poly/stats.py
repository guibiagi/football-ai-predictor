from __future__ import annotations

import numpy as np


def trade_roi_binary(p_entry: float, won: bool, fee_rate: float = 0.02) -> float:
    """ROI sobre capital em risco (preço pago)."""
    if p_entry <= 0 or p_entry >= 1:
        return float("nan")
    if won:
        gross = 1.0 - p_entry
        fee = fee_rate * gross
        return (gross - fee) / p_entry
    return -1.0


def bootstrap_mean_ci(
    rois: list[float],
    n_boot: int = 10_000,
    seed: int = 42,
) -> tuple[float, float, float]:
    arr = np.array([r for r in rois if np.isfinite(r)], dtype=float)
    if len(arr) == 0:
        return float("nan"), float("nan"), float("nan")
    rng = np.random.default_rng(seed)
    means = []
    for _ in range(n_boot):
        sample = rng.choice(arr, size=len(arr), replace=True)
        means.append(sample.mean())
    means = np.array(means)
    return float(arr.mean()), float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))


def bootstrap_p_value_roi(rois: list[float], n_boot: int = 10_000, seed: int = 42) -> float:
    """H0: mean ROI <= 0 (one-sided)."""
    arr = np.array([r for r in rois if np.isfinite(r)], dtype=float)
    if len(arr) < 2:
        return 1.0
    obs = arr.mean()
    rng = np.random.default_rng(seed)
    # permutation-style under H0: center at 0 by shifting
    shifted = arr - obs
    count = 0
    for _ in range(n_boot):
        sample = rng.choice(shifted, size=len(shifted), replace=True)
        if sample.mean() >= obs:
            count += 1
    return (count + 1) / (n_boot + 1)


def bonferroni(p_values: list[float]) -> list[float]:
    m = len(p_values)
    return [min(1.0, p * m) for p in p_values]


def benjamini_hochberg(p_values: list[float]) -> list[float]:
    m = len(p_values)
    if m == 0:
        return []
    indexed = sorted(enumerate(p_values), key=lambda x: x[1])
    adj = [1.0] * m
    prev = 1.0
    for rank, (i, p) in enumerate(reversed(indexed), start=1):
        k = m - rank + 1
        val = min(prev, p * m / k)
        adj[i] = val
        prev = val
    return adj


def binary_log_loss(p: float, y: int) -> float:
    p = np.clip(p, 1e-15, 1 - 1e-15)
    return float(-(y * np.log(p) + (1 - y) * np.log(1 - p)))


def binary_brier(p: float, y: int) -> float:
    return float((p - y) ** 2)
