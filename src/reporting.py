"""Métricas reproduzíveis: JSON, bootstrap, calibração."""

from __future__ import annotations

import json
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd

from src.odds_utils import brier_multiclass as brier
from src.odds_utils import log_loss_multiclass as log_loss
from src.odds_utils import market_probs


def bootstrap_ci(
    values: np.ndarray,
    n_boot: int = 2000,
    alpha: float = 0.05,
    seed: int = 42,
) -> tuple[float, float, float]:
    """Média e IC (percentil) via bootstrap."""
    values = np.asarray(values, dtype=float)
    values = values[~np.isnan(values)]
    if len(values) == 0:
        return float("nan"), float("nan"), float("nan")
    rng = np.random.default_rng(seed)
    means = []
    n = len(values)
    for _ in range(n_boot):
        samp = values[rng.integers(0, n, size=n)]
        means.append(float(np.mean(samp)))
    lo, hi = np.percentile(means, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    return float(np.mean(values)), float(lo), float(hi)


def calibration_bins_1x2(p: np.ndarray, y: np.ndarray, n_bins: int = 10) -> list[dict]:
    """Calibração por faixa na probabilidade do resultado que ocorreu."""
    p_pick = p[np.arange(len(y)), y]
    bins = np.linspace(0, 1, n_bins + 1)
    idx = np.digitize(p_pick, bins) - 1
    rows = []
    for b in range(n_bins):
        m = idx == b
        if m.sum() == 0:
            continue
        pred = float(p_pick[m].mean())
        actual = float(m.mean())
        rows.append({
            "bin_lo": float(bins[b]),
            "bin_hi": float(bins[b + 1]),
            "n": int(m.sum()),
            "predicted": round(pred, 4),
            "actual": round(actual, 4),
            "error": round(pred - actual, 4),
        })
    return rows


def summarize_bets_binary(
    returns: np.ndarray,
    clv: np.ndarray,
    p_model: np.ndarray,
    p_close: np.ndarray,
    y: np.ndarray,
) -> dict:
    from src.odds_utils import log_loss_binary

    returns = np.asarray(returns, dtype=float)
    clv = np.asarray(clv, dtype=float)
    n = len(returns)
    if n == 0:
        return {"n_bets": 0}
    roi_mean, roi_lo, roi_hi = bootstrap_ci(returns)
    clv_mean = float(np.nanmean(clv))
    return {
        "n_bets": n,
        "hit_rate": round(float((returns > 0).mean()), 4),
        "roi": round(roi_mean, 4),
        "roi_ci95": [round(roi_lo, 4), round(roi_hi, 4)],
        "clv_mean": round(clv_mean, 4),
        "clv_positive_pct": round(float(np.mean(clv > 0) * 100), 2),
        "log_loss_model": round(log_loss_binary(p_model[:, 0], y), 4),
        "log_loss_close": round(log_loss_binary(p_close[:, 0], y), 4),
        "log_loss_gain_vs_close": round(
            log_loss_binary(p_close[:, 0], y) - log_loss_binary(p_model[:, 0], y), 4
        ),
    }


def summarize_bets(
    returns: np.ndarray,
    clv: np.ndarray,
    p_model: np.ndarray,
    p_close: np.ndarray,
    y: np.ndarray,
) -> dict:
    returns = np.asarray(returns, dtype=float)
    clv = np.asarray(clv, dtype=float)
    n = len(returns)
    if n == 0:
        return {
            "n_bets": 0,
            "hit_rate": None,
            "roi": None,
            "roi_ci95": None,
            "clv_mean": None,
            "clv_positive_pct": None,
            "log_loss_model": None,
            "log_loss_close": None,
            "brier_model": None,
            "brier_close": None,
            "calibration": [],
        }
    roi_mean, roi_lo, roi_hi = bootstrap_ci(returns)
    clv_mean = float(np.nanmean(clv)) if np.isfinite(clv).any() else None
    clv_pos = float(np.nanmean(clv > 0) * 100) if np.isfinite(clv).any() else None
    return {
        "n_bets": n,
        "hit_rate": round(float((returns > 0).mean()), 4),
        "roi": round(roi_mean, 4),
        "roi_ci95": [round(roi_lo, 4), round(roi_hi, 4)],
        "clv_mean": round(clv_mean, 4) if clv_mean is not None else None,
        "clv_positive_pct": round(clv_pos, 2) if clv_pos is not None else None,
        "log_loss_model": round(log_loss(p_model, y), 4),
        "log_loss_close": round(log_loss(p_close, y), 4),
        "log_loss_gain_vs_close": round(log_loss(p_close, y) - log_loss(p_model, y), 4),
        "brier_model": round(brier(p_model, y), 4),
        "brier_close": round(brier(p_close, y), 4),
        "calibration": calibration_bins_1x2(p_model, y),
    }


def write_metrics_report(payload: dict, reports_dir: str | Path = "reports") -> Path:
    reports_dir = Path(reports_dir)
    reports_dir.mkdir(parents=True, exist_ok=True)
    stamp = date.today().strftime("%Y%m%d")
    path = reports_dir / f"metrics_{stamp}.json"
    if path.exists():
        stamp = date.today().strftime("%Y%m%d_%H%M%S")
        path = reports_dir / f"metrics_{stamp}.json"
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    return path


def compare_metrics_reports(current: Path, previous: Path) -> dict:
    cur = json.loads(current.read_text(encoding="utf-8"))
    prev = json.loads(previous.read_text(encoding="utf-8"))
    diff = {"current": str(current), "previous": str(previous), "deltas": {}}
    for key in ("aggregate", "by_league"):
        if key not in cur or key not in prev:
            continue
        if key == "aggregate":
            for m in ("log_loss_model", "roi", "clv_mean"):
                if m in cur["aggregate"] and m in prev["aggregate"]:
                    a, b = cur["aggregate"][m], prev["aggregate"][m]
                    if a is not None and b is not None:
                        diff["deltas"][m] = round(a - b, 4)
    return diff
