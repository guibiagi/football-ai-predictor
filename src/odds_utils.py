"""Utilitários de odds e log-loss (sem dependência de scripts de backtest)."""

from __future__ import annotations

import numpy as np


def market_probs(odds: np.ndarray) -> np.ndarray:
    raw = 1.0 / odds
    return raw / raw.sum(axis=1, keepdims=True)


def log_loss_multiclass(probs: np.ndarray, y: np.ndarray) -> float:
    return float(-np.mean(np.log(np.clip(probs[np.arange(len(y)), y], 1e-15, 1))))


def brier_multiclass(probs: np.ndarray, y: np.ndarray) -> float:
    onehot = np.zeros_like(probs)
    onehot[np.arange(len(y)), y] = 1
    return float(np.mean(np.sum((probs - onehot) ** 2, axis=1)))


def log_loss_binary(p: np.ndarray, y: np.ndarray) -> float:
    p = np.clip(p, 1e-12, 1 - 1e-12)
    return float(-np.mean(y * np.log(p) + (1 - y) * np.log(1 - p)))
