"""Correção para múltiplos testes (Bonferroni, BH-FDR, reality check simplificado)."""

from __future__ import annotations

import numpy as np
from scipy import stats


def one_sided_pvalue_mean_gt_zero(returns: np.ndarray) -> float:
    """H0: média <= 0 vs H1: média > 0 (teste t unilateral)."""
    r = np.asarray(returns, dtype=float)
    r = r[~np.isnan(r)]
    if len(r) < 2:
        return 1.0
    t, p_two = stats.ttest_1samp(r, 0.0, alternative="greater")
    if np.isnan(t):
        return 1.0
    return float(p_two)


def bonferroni_adjust(p_values: list[float]) -> list[float]:
    m = max(len(p_values), 1)
    return [min(1.0, p * m) for p in p_values]


def benjamini_hochberg(p_values: list[float]) -> list[float]:
    m = len(p_values)
    if m == 0:
        return []
    order = np.argsort(p_values)
    adj = [1.0] * m
    prev = 1.0
    for rank in range(m - 1, -1, -1):
        i = order[rank]
        q = min(prev, p_values[i] * m / (rank + 1))
        adj[i] = min(1.0, q)
        prev = q
    return adj


def white_reality_check_pvalue(
    strategy_returns: list[np.ndarray],
    n_boot: int = 2000,
    seed: int = 42,
) -> float:
    """
    Reality check de White (bootstrap): sob H0 (média zero em cada estratégia),
    P(max_j mean(R_j) >= max observado).
    """
    clean = [np.asarray(r, dtype=float) for r in strategy_returns if len(r) >= 5]
    if not clean:
        return 1.0
    means = [float(np.mean(r)) for r in clean]
    max_obs = max(means)
    if max_obs <= 0:
        return 1.0
    rng = np.random.default_rng(seed)
    n_strat = len(clean)
    n = min(len(r) for r in clean)
    # Alinha por índice (mesmo tamanho efetivo: subamostra comum)
    stacked = np.column_stack([r[:n] for r in clean])
    centered = stacked - stacked.mean(axis=0, keepdims=True)
    count = 0
    for _ in range(n_boot):
        idx = rng.integers(0, n, size=n)
        boot_max = float(np.max(centered[idx].mean(axis=0)))
        if boot_max >= max_obs:
            count += 1
    return (count + 1) / (n_boot + 1)


def spa_hansen_conservative_p(
    holdout_returns: np.ndarray,
    dev_returns: np.ndarray | None = None,
) -> float:
    """
    Proxy conservador SPA: se dev foi usado para escolher, usa p unilateral no holdout
    (holdout intocado); dev só entra na nota metodológica.
    """
    return one_sided_pvalue_mean_gt_zero(holdout_returns)
