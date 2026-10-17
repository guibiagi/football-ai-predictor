"""Backtest Over/Under 2.5 vs Pinnacle — utilitários reutilizáveis (sem apostas reais)."""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Iterable

import numpy as np
import pandas as pd

from src.clv import clv_probability
from src.odds_utils import log_loss_binary

logger = logging.getLogger(__name__)

RNG = np.random.default_rng(42)


def odds_valid(odds: np.ndarray) -> np.ndarray:
    """Linhas com odds decimais válidas em todas as colunas."""
    return (~np.isnan(odds).any(axis=1)) & (odds > 1.0).all(axis=1)


def devig_binary(odds: np.ndarray, method: str = "multiplicative") -> np.ndarray:
    """Probabilidade justa de over (coluna 0) para mercado binário."""
    if method == "multiplicative":
        inv = 1.0 / odds
        return inv[:, 0] / inv.sum(axis=1)
    if method == "power":
        return _devig_power_binary(odds)
    if method == "shin":
        return _devig_shin_binary(odds)
    raise ValueError(f"devig desconhecido: {method}")


def _devig_power_binary(odds: np.ndarray, tol: float = 1e-9) -> np.ndarray:
    out = np.empty(len(odds))
    for i, row in enumerate(odds):
        lo, hi = 0.5, 3.0
        for _ in range(60):
            mid = (lo + hi) / 2
            s = (row ** (-mid)).sum()
            if s > 1.0:
                lo = mid
            else:
                hi = mid
        k = (lo + hi) / 2
        inv = row ** (-k)
        out[i] = inv[0] / inv.sum()
    return out


def _devig_shin_binary(odds: np.ndarray, tol: float = 1e-8) -> np.ndarray:
    """Shin (1991) aproximado para 2 resultados."""
    out = np.empty(len(odds))
    n = 2
    for i, row in enumerate(odds):
        imp = 1.0 / row
        z = 0.0
        for _ in range(100):
            denom = np.sqrt(1.0 + z) + z * (n - 1) * imp
            p = imp / denom
            s = p.sum()
            z_new = max(0.0, (s - 1.0) / (n - 1) / imp.sum())
            if abs(z_new - z) < tol:
                z = z_new
                break
            z = z_new
        denom = np.sqrt(1.0 + z) + z * (n - 1) * imp
        p = imp / denom
        out[i] = p[0]
    return out


def brier_binary(p: np.ndarray, y: np.ndarray) -> float:
    return float(np.mean((p - y) ** 2))


def bootstrap_roi_ci(
    returns: np.ndarray,
    n_boot: int = 10_000,
    alpha: float = 0.05,
    rng: np.random.Generator | None = None,
) -> tuple[float, float, float]:
    """ROI médio e IC (percentis). returns = lucro por unidade (ex.: odds-1 ou -1)."""
    if len(returns) == 0:
        return float("nan"), float("nan"), float("nan")
    rng = rng or RNG
    roi_obs = float(returns.mean())
    idx = rng.integers(0, len(returns), size=(n_boot, len(returns)))
    boots = returns[idx].mean(axis=1)
    lo, hi = np.percentile(boots, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    return roi_obs, float(lo), float(hi)


def bootstrap_p_roi_nonpos(returns: np.ndarray, n_boot: int = 10_000,
                           rng: np.random.Generator | None = None) -> float:
    """p-value unilateral H0: ROI <= 0 vs Ha: ROI > 0 (bootstrap das médias)."""
    if len(returns) == 0:
        return 1.0
    rng = rng or RNG
    idx = rng.integers(0, len(returns), size=(n_boot, len(returns)))
    boots = returns[idx].mean(axis=1)
    # P(ROI_boot <= 0): alto quando ROI observado <= 0; baixo quando ROI > 0
    return float((boots <= 0).mean())


def bonferroni_adjust(p_values: np.ndarray) -> np.ndarray:
    m = len(p_values)
    return np.minimum(p_values * m, 1.0)


def bh_fdr_adjust(p_values: np.ndarray) -> np.ndarray:
    """Benjamini–Hochberg q-values."""
    m = len(p_values)
    order = np.argsort(p_values)
    ranked = np.empty(m)
    prev = 1.0
    for rank, idx in enumerate(order[::-1], start=1):
        q = p_values[idx] * m / (m - rank + 1)
        prev = min(prev, q)
        ranked[idx] = prev
    return np.minimum(ranked, 1.0)


def bootstrap_best_strategy_null_p(
    strategy_returns: list[np.ndarray],
    n_boot: int = 2000,
    rng: np.random.Generator | None = None,
) -> float:
    """Heurística: melhor ROI observado vs máximo sob reamostragem independente por estratégia.

    Não é o Reality Check de White (1999) nem o SPA de Hansen (2005); não preserva
    correlação entre estratégias. Nome honesto para não overclaim.
    """
    rng = rng or RNG
    rois = [r.mean() if len(r) else -np.inf for r in strategy_returns]
    best_obs = max(rois)
    if not np.isfinite(best_obs):
        return 1.0
    count = 0
    for _ in range(n_boot):
        sim_best = -np.inf
        for rets in strategy_returns:
            if len(rets) == 0:
                continue
            idx = rng.integers(0, len(rets), size=len(rets))
            sim = rets[idx].mean()
            sim_best = max(sim_best, sim)
        if sim_best >= best_obs:
            count += 1
    return (count + 1) / (n_boot + 1)


@dataclass
class BetRow:
    variant_id: str
    split: str  # train | holdout
    returns: np.ndarray
    clv: np.ndarray
    p_model: np.ndarray
    y_over: np.ndarray
    p_close_fair: np.ndarray


@dataclass
class VariantResult:
    variant_id: str
    hypothesis: str
    split: str
    n: int
    roi: float
    roi_ci_lo: float
    roi_ci_hi: float
    clv_mean: float
    clv_pos_pct: float
    p_raw: float
    log_loss_model: float
    log_loss_pin: float
    brier_model: float
    brier_pin: float
    survives: bool
    meta: dict[str, Any] = field(default_factory=dict)


def flat_bet_returns(odds_side: np.ndarray, won: np.ndarray) -> np.ndarray:
    return np.where(won, odds_side - 1.0, -1.0)


def pick_edge_side(
    p_model_over: np.ndarray,
    p_fair_over: np.ndarray,
    odds_open: np.ndarray,
    edge_min: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Retorna máscara de aposta, lado (1=over, 0=under), odds do lado."""
    p_under = 1.0 - p_model_over
    p_fair_under = 1.0 - p_fair_over
    edge_over = p_model_over - p_fair_over
    edge_under = p_under - p_fair_under
    bet_over = edge_over >= edge_min
    bet_under = edge_under >= edge_min
    both = bet_over & bet_under
    # se ambos passam, escolhe maior edge
    choose_over = edge_over >= edge_under
    side_over = np.where(both, choose_over, bet_over)
    mask = bet_over | bet_under
    odds_side = np.where(side_over, odds_open[:, 0], odds_open[:, 1])
    return mask, side_over, odds_side


def summarize_bets(
    variant_id: str,
    hypothesis: str,
    split: str,
    mask: np.ndarray,
    odds_side: np.ndarray,
    side_over: np.ndarray,
    y_over: np.ndarray,
    p_model_over: np.ndarray,
    p_close_fair: np.ndarray,
    p_open_fair: np.ndarray,
) -> VariantResult:
    if mask.sum() == 0:
        return VariantResult(
            variant_id, hypothesis, split, 0, 0.0, np.nan, np.nan,
            np.nan, np.nan, 1.0, np.nan, np.nan, np.nan, np.nan, False,
        )
    won = np.where(side_over[mask], y_over[mask] == 1, y_over[mask] == 0)
    rets = flat_bet_returns(odds_side[mask], won)
    clv = []
    for i, is_over in zip(np.where(mask)[0], side_over[mask]):
        p_entry = p_open_fair[i] if is_over else 1.0 - p_open_fair[i]
        p_close = p_close_fair[i] if is_over else 1.0 - p_close_fair[i]
        clv.append(clv_probability(p_entry, p_close))
    clv_arr = np.array(clv, dtype=float)
    roi, lo, hi = bootstrap_roi_ci(rets)
    p_raw = bootstrap_p_roi_nonpos(rets)
    pm = p_model_over[mask]
    ym = y_over[mask]
    pc = p_close_fair[mask]
    return VariantResult(
        variant_id=variant_id,
        hypothesis=hypothesis,
        split=split,
        n=int(mask.sum()),
        roi=roi,
        roi_ci_lo=lo,
        roi_ci_hi=hi,
        clv_mean=float(np.nanmean(clv_arr)),
        clv_pos_pct=float(np.nanmean(clv_arr > 0) * 100),
        p_raw=p_raw,
        log_loss_model=log_loss_binary(pm, ym),
        log_loss_pin=log_loss_binary(pc, ym),
        brier_model=brier_binary(pm, ym),
        brier_pin=brier_binary(pc, ym),
        survives=False,
        meta={
            "returns": rets.tolist(),
            "clv": clv_arr.tolist(),
            "p_model": pm.tolist(),
            "y": ym.tolist(),
            "p_close": pc.tolist(),
        },
    )


def calibration_bins(p: np.ndarray, y: np.ndarray, n_bins: int = 10) -> pd.DataFrame:
    bins = np.linspace(0, 1, n_bins + 1)
    idx = np.clip(np.digitize(p, bins) - 1, 0, n_bins - 1)
    rows = []
    for b in range(n_bins):
        m = idx == b
        if m.sum() == 0:
            continue
        rows.append({
            "bin_lo": bins[b],
            "bin_hi": bins[b + 1],
            "n": int(m.sum()),
            "pred": float(p[m].mean()),
            "actual": float(y[m].mean()),
        })
    return pd.DataFrame(rows)


def apply_survival_flags(results: list[VariantResult],
                         p_adj_bonf: np.ndarray,
                         q_bh: np.ndarray) -> None:
    for r, pb, qb in zip(results, p_adj_bonf, q_bh):
        r.survives = (
            r.n > 0 and r.roi > 0 and r.p_raw < 0.05
            and (pb < 0.05 or qb < 0.10)
        )


def aggregate_variant_results(results: list[VariantResult]) -> list[VariantResult]:
    """Agrega resultados por liga numa linha por (variant_id, hypothesis, split)."""
    buckets: dict[tuple[str, str, str], list[VariantResult]] = {}
    for r in results:
        key = (r.variant_id, r.hypothesis, r.split)
        buckets.setdefault(key, []).append(r)
    out: list[VariantResult] = []
    for (vid, hyp, split), rs in sorted(buckets.items()):
        rets, clv, pm, ym, pc = [], [], [], [], []
        for r in rs:
            m = r.meta
            if not m.get("returns"):
                continue
            rets.extend(m["returns"])
            clv.extend(m.get("clv", []))
            pm.extend(m.get("p_model", []))
            ym.extend(m.get("y", []))
            pc.extend(m.get("p_close", []))
        if not rets:
            out.append(VariantResult(vid, hyp, split, 0, 0.0, np.nan, np.nan,
                                     np.nan, np.nan, 1.0, np.nan, np.nan, np.nan, np.nan, False))
            continue
        rets_a = np.array(rets, dtype=float)
        clv_a = np.array(clv, dtype=float) if clv else np.array([])
        pm_a = np.array(pm, dtype=float) if pm else np.array([])
        ym_a = np.array(ym, dtype=float) if ym else np.array([])
        pc_a = np.array(pc, dtype=float) if pc else np.array([])
        roi, lo, hi = bootstrap_roi_ci(rets_a)
        ll_m = log_loss_binary(pm_a, ym_a) if len(pm_a) else float("nan")
        ll_p = log_loss_binary(pc_a, ym_a) if len(pc_a) else float("nan")
        br_m = brier_binary(pm_a, ym_a) if len(pm_a) else float("nan")
        br_p = brier_binary(pc_a, ym_a) if len(pc_a) else float("nan")
        out.append(VariantResult(
            variant_id=vid,
            hypothesis=hyp,
            split=split,
            n=len(rets_a),
            roi=roi,
            roi_ci_lo=lo,
            roi_ci_hi=hi,
            clv_mean=float(np.nanmean(clv_a)) if len(clv_a) else float("nan"),
            clv_pos_pct=float(np.nanmean(clv_a > 0) * 100) if len(clv_a) else float("nan"),
            p_raw=bootstrap_p_roi_nonpos(rets_a),
            log_loss_model=ll_m,
            log_loss_pin=ll_p,
            brier_model=br_m,
            brier_pin=br_p,
            survives=False,
            meta={"returns": rets_a.tolist()},
        ))
    return out


def save_variant_log(path: Path, entries: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(entries, f, indent=2, ensure_ascii=False)
