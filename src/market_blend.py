"""Shrinkage em direção ao mercado (sem margem) + recalibração isotônica opcional."""

from __future__ import annotations

import numpy as np
from sklearn.isotonic import IsotonicRegression

from src.odds_utils import log_loss_multiclass as log_loss
from src.odds_utils import market_probs


def devig_1x2(odds: np.ndarray) -> np.ndarray:
    return market_probs(odds)


def shrink_probs(p_model: np.ndarray, p_market: np.ndarray, w: float) -> np.ndarray:
    """w=1 → modelo puro; w=0 → mercado puro."""
    out = w * p_model + (1.0 - w) * p_market
    out = np.clip(out, 1e-12, 1.0)
    out /= out.sum(axis=1, keepdims=True)
    return out


def tune_shrink_weight(
    p_model: np.ndarray,
    p_market: np.ndarray,
    y: np.ndarray,
    grid: np.ndarray | None = None,
) -> float:
    """Escolhe w que minimiza log-loss no conjunto de calibração (sem vazamento se calib ⊂ treino)."""
    if grid is None:
        grid = np.round(np.arange(0.0, 1.01, 0.05), 2)
    best_w, best_ll = 1.0, float("inf")
    for w in grid:
        p = shrink_probs(p_model, p_market, float(w))
        ll = log_loss(p, y)
        if ll < best_ll:
            best_ll = ll
            best_w = float(w)
    return best_w


class Isotonic1x2:
    """Isotônica por resultado (casa/empate/fora), renormalizada."""

    def __init__(self) -> None:
        self._home = IsotonicRegression(out_of_bounds="clip")
        self._draw = IsotonicRegression(out_of_bounds="clip")
        self._away = IsotonicRegression(out_of_bounds="clip")
        self.fitted = False

    def fit(self, p: np.ndarray, y: np.ndarray) -> Isotonic1x2:
        self._home.fit(p[:, 0], (y == 0).astype(float))
        self._draw.fit(p[:, 1], (y == 1).astype(float))
        self._away.fit(p[:, 2], (y == 2).astype(float))
        self.fitted = True
        return self

    def transform(self, p: np.ndarray) -> np.ndarray:
        if not self.fitted:
            return p
        ch = self._home.predict(p[:, 0])
        cd = self._draw.predict(p[:, 1])
        ca = self._away.predict(p[:, 2])
        out = np.stack([ch, cd, ca], axis=1)
        out = np.clip(out, 1e-12, 1.0)
        out /= out.sum(axis=1, keepdims=True)
        return out


def apply_market_pipeline(
    p_model: np.ndarray,
    odds_close: np.ndarray,
    y: np.ndarray,
    calib_slice: slice,
    eval_slice: slice,
    use_isotonic: bool = True,
) -> tuple[np.ndarray, float, dict]:
    """Calibra w (e isotônica) só em calib_slice; devolve probs em eval_slice."""
    p_mkt = devig_1x2(odds_close)
    w = tune_shrink_weight(p_model[calib_slice], p_mkt[calib_slice], y[calib_slice])
    p_cal = shrink_probs(p_model, p_mkt, w)
    iso = Isotonic1x2()
    if use_isotonic and w >= 0.05 and len(y[calib_slice]) >= 50:
        iso.fit(p_cal[calib_slice], y[calib_slice])
        trial = iso.transform(p_cal[calib_slice])
        if log_loss(trial, y[calib_slice]) > log_loss(p_cal[calib_slice], y[calib_slice]):
            iso.fitted = False
    ev = p_cal[eval_slice]
    if len(ev) == 0:
        meta = {"shrink_w": w, "isotonic": False}
        return ev, w, meta
    p_eval = iso.transform(ev) if iso.fitted else ev
    meta = {"shrink_w": w, "isotonic": iso.fitted}
    return p_eval, w, meta
