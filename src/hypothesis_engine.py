"""Motor walk-forward para testar hipóteses de apostas com CLV vs fechamento Pinnacle.

Regras:
  - Entrada só com informação disponível no momento (odds de abertura, histórico).
  - CLV medido vs probabilidade devig Pinnacle no fechamento (psc_* / ou_psc_*).
  - Holdout final: últimos ``holdout_days`` dias, intocado na calibração de limiares.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Iterable

import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor
from backtest_markets import _odds_validas, _over25, _sem_vig, roda
from src.features import add_form_adjustment
from src.market_blend import apply_market_pipeline
from src.odds_utils import market_probs
from src.reporting import bootstrap_ci

HOLDOUT_DAYS = 365
CLV_REF = "Pinnacle close (devig)"
ENTRY_DEFAULT = "Bet365 open"

LOWER_TIER = {
    "Championship",
    "2. Bundesliga",
    "Scottish Championship",
    "League One",
    "League Two",
    "Brasileirao Serie A",
    "Ekstraklasa",
    "Eliteserien",
    "Allsvenskan",
    "Danish Superliga",
    "Austria Bundesliga",
    "Chinese Super League",
}


@dataclass
class BetRecord:
    date: pd.Timestamp
    competition: str
    market: str
    clv: float
    return_: float


@dataclass
class HypothesisResult:
    id: str
    description: str
    n_dev: int
    clv_mean_dev: float
    clv_ci95_dev: list[float]
    roi_dev: float
    roi_ci95_dev: list[float]
    n_holdout: int
    clv_mean_holdout: float
    clv_ci95_holdout: list[float]
    roi_holdout: float
    roi_ci95_holdout: list[float]
    viable: bool
    notes: str = ""
    extra: dict = field(default_factory=dict)


def _y1x2(df: pd.DataFrame) -> np.ndarray:
    return np.where(
        df.home_goals > df.away_goals,
        0,
        np.where(df.home_goals == df.away_goals, 1, 2),
    )


def load_prediction_panel(df: pd.DataFrame, competitions: Iterable[str] | None = None) -> pd.DataFrame:
    """Predições OOS (Poisson) + odds por liga."""
    comps = sorted(competitions or df.competition.unique())
    parts = []
    for comp in comps:
        try:
            p = roda(df, comp)
        except Exception:
            continue
        if p is None or len(p) < 100:
            continue
        p = p.copy()
        p["competition"] = comp
        parts.append(p)
    if not parts:
        raise ValueError("Nenhuma liga com predições válidas")
    panel = pd.concat(parts, ignore_index=True).sort_values("date").reset_index(drop=True)
    return panel


def split_holdout(panel: pd.DataFrame, holdout_days: int = HOLDOUT_DAYS) -> tuple[pd.DataFrame, pd.DataFrame]:
    end = panel.date.max()
    cut = end - pd.Timedelta(days=holdout_days)
    dev = panel[panel.date < cut].copy()
    hold = panel[panel.date >= cut].copy()
    return dev, hold


def _summarize_records(recs: list[BetRecord]) -> dict:
    if not recs:
        return {
            "n": 0,
            "clv_mean": None,
            "clv_ci95": None,
            "roi": None,
            "roi_ci95": None,
        }
    clv = np.array([r.clv for r in recs], dtype=float)
    ret = np.array([r.return_ for r in recs], dtype=float)
    clv_m, clv_lo, clv_hi = bootstrap_ci(clv)
    roi_m, roi_lo, roi_hi = bootstrap_ci(ret)
    return {
        "n": len(recs),
        "clv_mean": round(clv_m, 5),
        "clv_ci95": [round(clv_lo, 5), round(clv_hi, 5)],
        "roi": round(roi_m, 4),
        "roi_ci95": [round(roi_lo, 4), round(roi_hi, 4)],
    }


def is_viable(summary: dict, n_min: int = 300) -> bool:
    if summary["n"] < n_min:
        return False
    if summary["clv_mean"] is None:
        return False
    lo = summary["clv_ci95"][0]
    return summary["clv_mean"] > 0 and lo is not None and lo > 0


def _records_from_mask(
    sub: pd.DataFrame,
    sel: np.ndarray,
    k: np.ndarray,
    open_o: np.ndarray,
    p_close: np.ndarray,
    p_open: np.ndarray,
    y: np.ndarray,
    market: str,
) -> list[BetRecord]:
    out: list[BetRecord] = []
    idx = np.where(sel)[0]
    for i in idx:
        side = int(k[i])
        clv = float(p_close[i, side] - p_open[i, side])
        won = side == y[i]
        ret = float(open_o[i, side] - 1.0) if won else -1.0
        out.append(
            BetRecord(
                date=sub.date.iloc[i],
                competition=sub.competition.iloc[i],
                market=market,
                clv=clv,
                return_=ret,
            )
        )
    return out


def sim_model_edge_1x2(
    sub: pd.DataFrame,
    p_model: np.ndarray,
    edge_min: float,
    league_filter: Callable[[str], bool] | None = None,
) -> list[BetRecord]:
    open_o = sub[["odds_open_h", "odds_open_d", "odds_open_a"]].to_numpy()
    close_o = sub[["psc_h", "psc_d", "psc_a"]].to_numpy()
    ok = _odds_validas(open_o) & _odds_validas(close_o)
    p_open = market_probs(open_o)
    p_close = market_probs(close_o)
    y = _y1x2(sub)
    k = np.argmax(p_model - p_open, axis=1)
    edge = p_model[np.arange(len(y)), k] - p_open[np.arange(len(y)), k]
    sel = ok & (edge >= edge_min)
    if league_filter:
        lf = np.array([league_filter(c) for c in sub.competition])
        sel = sel & lf
    return _records_from_mask(sub, sel, k, open_o, p_close, p_open, y, "1X2")


def sim_model_edge_ou(
    sub: pd.DataFrame,
    p_over: np.ndarray,
    edge_min: float,
    league_filter: Callable[[str], bool] | None = None,
) -> list[BetRecord]:
    open_o = sub[["ou_open_over", "ou_open_under"]].to_numpy()
    close_o = sub[["ou_psc_over", "ou_psc_under"]].to_numpy()
    ok = _odds_validas(open_o) & _odds_validas(close_o)
    y = (sub.home_goals + sub.away_goals > 2.5).astype(int).to_numpy()
    p_open = _sem_vig(open_o)
    p_close = _sem_vig(close_o)
    p_mod = np.stack([p_over, 1 - p_over], axis=1)
    k = np.argmax(p_mod - p_open, axis=1)
    edge = p_mod[np.arange(len(y)), k] - p_open[np.arange(len(y)), k]
    sel = ok & (edge >= edge_min)
    if league_filter:
        lf = np.array([league_filter(c) for c in sub.competition])
        sel = sel & lf
    y_pick = np.where(k == 0, y, 1 - y)
    # remap k for returns: side 0 = over, 1 = under
    recs = []
    for i in np.where(sel)[0]:
        side = int(k[i])
        clv = float(p_close[i, side] - p_open[i, side])
        won = (side == 0 and y[i] == 1) or (side == 1 and y[i] == 0)
        ret = float(open_o[i, side] - 1.0) if won else -1.0
        recs.append(
            BetRecord(
                date=sub.date.iloc[i],
                competition=sub.competition.iloc[i],
                market="O/U",
                clv=clv,
                return_=ret,
            )
        )
    return recs


def sim_pin_open_vs_b365_1x2(sub: pd.DataFrame, edge_min: float = 0.03) -> list[BetRecord]:
    """Aposta onde Pinnacle abertura > B365 abertura (sharp mais alto que execução)."""
    open_b = sub[["odds_open_h", "odds_open_d", "odds_open_a"]].to_numpy()
    open_p = sub[["ps_h", "ps_d", "ps_a"]].to_numpy()
    close_o = sub[["psc_h", "psc_d", "psc_a"]].to_numpy()
    ok = _odds_validas(open_b) & _odds_validas(open_p) & _odds_validas(close_o)
    p_b = market_probs(open_b)
    p_p = market_probs(open_p)
    p_close = market_probs(close_o)
    y = _y1x2(sub)
    k = np.argmax(p_p - p_b, axis=1)
    edge = p_p[np.arange(len(y)), k] - p_b[np.arange(len(y)), k]
    sel = ok & (edge >= edge_min)
    return _records_from_mask(sub, sel, k, open_b, p_close, p_b, y, "1X2")


def sim_pin_open_vs_b365_ou(sub: pd.DataFrame, edge_min: float = 0.03) -> list[BetRecord]:
    """Indisponível no dataset atual (sem ou_ps_*); mantido para compatibilidade."""
    open_b = sub[["ou_open_over", "ou_open_under"]].to_numpy()
    cols_p = [c for c in ("ou_ps_over", "ou_ps_under", "ou_psc_over") if c in sub.columns]
    if "ou_ps_over" not in sub.columns:
        return []
    open_p = sub[cols_p].to_numpy()
    close_o = sub[["ou_psc_over", "ou_psc_under"]].to_numpy()
    ok = _odds_validas(open_b) & _odds_validas(open_p) & _odds_validas(close_o)
    p_b = _sem_vig(open_b)
    p_p = _sem_vig(open_p)
    p_close = _sem_vig(close_o)
    y = (sub.home_goals + sub.away_goals > 2.5).astype(int).to_numpy()
    k = np.argmax(p_p - p_b, axis=1)
    edge = p_p[np.arange(len(y)), k] - p_b[np.arange(len(y)), k]
    sel = ok & (edge >= edge_min)
    recs = []
    for i in np.where(sel)[0]:
        side = int(k[i])
        clv = float(p_close[i, side] - p_b[i, side])
        won = (side == 0 and y[i] == 1) or (side == 1 and y[i] == 0)
        ret = float(open_b[i, side] - 1.0) if won else -1.0
        recs.append(
            BetRecord(
                date=sub.date.iloc[i],
                competition=sub.competition.iloc[i],
                market="O/U",
                clv=clv,
                return_=ret,
            )
        )
    return recs


def sim_b365_vs_pin_close_discrepancy_1x2(sub: pd.DataFrame, edge_min: float = 0.04) -> list[BetRecord]:
    """Proxy Polymarket: B365 abertura vs Pinnacle fechamento (só onde B365 está 'barato').

    Seleção usa apenas ps_* (Pinnacle abertura) como estimativa de fair — NÃO usa psc no filtro.
    """
    open_b = sub[["odds_open_h", "odds_open_d", "odds_open_a"]].to_numpy()
    open_p = sub[["ps_h", "ps_d", "ps_a"]].to_numpy()
    close_o = sub[["psc_h", "psc_d", "psc_a"]].to_numpy()
    ok = _odds_validas(open_b) & _odds_validas(open_p) & _odds_validas(close_o)
    p_b = market_probs(open_b)
    p_pin_open = market_probs(open_p)
    p_close = market_probs(close_o)
    y = _y1x2(sub)
    k = np.argmax(p_pin_open - p_b, axis=1)
    edge = p_pin_open[np.arange(len(y)), k] - p_b[np.arange(len(y)), k]
    sel = ok & (edge >= edge_min)
    return _records_from_mask(sub, sel, k, open_b, p_close, p_b, y, "1X2")


def _blend_1x2_probs(sub: pd.DataFrame, calib_frac: float = 0.5) -> np.ndarray:
    sub = sub.sort_values("date").reset_index(drop=True)
    pm = sub[["home_win_probability", "draw_probability", "away_win_probability"]].to_numpy()
    close = sub[["psc_h", "psc_d", "psc_a"]].to_numpy()
    ok = _odds_validas(close)
    out = pm.copy()
    if ok.sum() < 80:
        return out
    pm_ok = pm[ok]
    close_ok = close[ok]
    y_ok = _y1x2(sub)[ok]
    n = len(pm_ok)
    n_cal = max(40, int(n * calib_frac))
    calib = slice(0, n_cal)
    eval_sl = slice(n_cal, n)
    p_eval, _, _ = apply_market_pipeline(
        pm_ok, close_ok, y_ok, calib, eval_sl, use_isotonic=True,
    )
    idx_ok = np.where(ok)[0]
    out[idx_ok[eval_sl]] = p_eval
    return out


def _movement_features_1x2(sub: pd.DataFrame) -> np.ndarray:
    open_b = sub[["odds_open_h", "odds_open_d", "odds_open_a"]].to_numpy()
    open_p = sub[["ps_h", "ps_d", "ps_a"]].to_numpy()
    p_b = market_probs(open_b)
    p_p = market_probs(open_p)
    diff = p_p - p_b
    fav = np.max(p_b, axis=1, keepdims=True)
    return np.column_stack([
        p_b,
        p_p,
        diff,
        fav,
        (sub.date.dt.month / 12.0).to_numpy(),
    ])


def sim_movement_gbm_1x2_v2(
    sub: pd.DataFrame,
    edge_min: float = 0.015,
    min_train: int = 1200,
) -> list[BetRecord]:
    """GBM: prevê CLV do lado com maior (ps_open − b365_open); treino expansivo por data."""
    sub = sub.sort_values("date").reset_index(drop=True)
    open_b = sub[["odds_open_h", "odds_open_d", "odds_open_a"]].to_numpy()
    open_p = sub[["ps_h", "ps_d", "ps_a"]].to_numpy()
    close_o = sub[["psc_h", "psc_d", "psc_a"]].to_numpy()
    ok = _odds_validas(open_b) & _odds_validas(open_p) & _odds_validas(close_o)
    p_b = market_probs(open_b)
    p_p = market_probs(open_p)
    p_c = market_probs(close_o)
    y = _y1x2(sub)
    X = _movement_features_1x2(sub)
    k_side = np.argmax(p_p - p_b, axis=1)
    y_clv = p_c[np.arange(len(sub)), k_side] - p_b[np.arange(len(sub)), k_side]

    recs: list[BetRecord] = []
    dates = sub.date.to_numpy()
    unique_months = pd.to_datetime(sub.date).dt.to_period("M").unique()

    for mi, period in enumerate(unique_months):
        if mi < 24:  # ~2 anos de warm-up
            continue
        train_mask = pd.to_datetime(sub.date).dt.to_period("M") < period
        test_mask = pd.to_datetime(sub.date).dt.to_period("M") == period
        tr = train_mask.to_numpy() & ok
        te = test_mask.to_numpy() & ok
        if tr.sum() < min_train or te.sum() == 0:
            continue
        model = GradientBoostingRegressor(
            random_state=42, max_depth=3, n_estimators=100, learning_rate=0.06,
        )
        model.fit(X[tr], y_clv[tr])
        pred = model.predict(X[te])
        for j, idx in enumerate(np.where(te)[0]):
            if pred[j] < edge_min:
                continue
            side = int(k_side[idx])
            clv = float(p_c[idx, side] - p_b[idx, side])
            won = side == y[idx]
            ret = float(open_b[idx, side] - 1.0) if won else -1.0
            recs.append(
                BetRecord(
                    date=sub.date.iloc[idx],
                    competition=sub.competition.iloc[idx],
                    market="1X2",
                    clv=clv,
                    return_=ret,
                )
            )
    return recs


def _xg_panel_features(df_hist: pd.DataFrame, sub: pd.DataFrame) -> np.ndarray:
    """Features de forma + xG rolling (sem vazamento: só jogos anteriores)."""
    hist = add_form_adjustment(df_hist.sort_values("date"))
    # merge form by date+teams — heavy; approximate with competition-level lambdas from preds
    lam_h = sub.expected_goals_home.to_numpy()
    lam_a = sub.expected_goals_away.to_numpy()
    p_over_model = _over25(lam_h, lam_a, rho=0.0)
    open_o = sub[["ou_open_over", "ou_open_under"]].to_numpy()
    p_open = _sem_vig(open_o)
    return np.column_stack([p_over_model, p_open[:, 0], lam_h, lam_a, lam_h + lam_a])


def sim_xg_boost_ou(
    sub: pd.DataFrame,
    df_all: pd.DataFrame,
    edge_min: float = 0.03,
    min_train: int = 1500,
) -> list[BetRecord]:
    sub = sub.sort_values("date").reset_index(drop=True)
    open_o = sub[["ou_open_over", "ou_open_under"]].to_numpy()
    close_o = sub[["ou_psc_over", "ou_psc_under"]].to_numpy()
    ok = _odds_validas(open_o) & _odds_validas(close_o)
    p_b = _sem_vig(open_o)
    p_c = _sem_vig(close_o)
    y = (sub.home_goals + sub.away_goals > 2.5).astype(int).to_numpy()
    lam_h = sub.expected_goals_home.to_numpy()
    lam_a = sub.expected_goals_away.to_numpy()
    p_over = _over25(lam_h, lam_a, rho=0.0)
    X = np.column_stack([p_over, p_b[:, 0], lam_h, lam_a, lam_h + lam_a])
    k_side = np.argmax(np.stack([p_over, 1 - p_over], axis=1) - p_b, axis=1)
    y_clv = p_c[np.arange(len(sub)), k_side] - p_b[np.arange(len(sub)), k_side]

    recs: list[BetRecord] = []
    periods = pd.to_datetime(sub.date).dt.to_period("M").unique()
    for mi, period in enumerate(periods):
        if mi < 30:
            continue
        train_mask = pd.to_datetime(sub.date).dt.to_period("M") < period
        test_mask = pd.to_datetime(sub.date).dt.to_period("M") == period
        tr = train_mask.to_numpy() & ok
        te = test_mask.to_numpy() & ok
        if tr.sum() < min_train or te.sum() == 0:
            continue
        model = GradientBoostingRegressor(random_state=42, max_depth=3, n_estimators=80)
        model.fit(X[tr], y_clv[tr])
        pred = model.predict(X[te])
        for j, idx in enumerate(np.where(te)[0]):
            side = int(k_side[idx])
            model_edge = (p_over[idx] if side == 0 else 1 - p_over[idx]) - p_b[idx, side]
            if pred[j] < edge_min or model_edge < edge_min:
                continue
            clv = float(p_c[idx, side] - p_b[idx, side])
            won = (side == 0 and y[idx] == 1) or (side == 1 and y[idx] == 0)
            ret = float(open_o[idx, side] - 1.0) if won else -1.0
            recs.append(
                BetRecord(
                    date=sub.date.iloc[idx],
                    competition=sub.competition.iloc[idx],
                    market="O/U",
                    clv=clv,
                    return_=ret,
                )
            )
    return recs


def sim_odds_band(
    sub: pd.DataFrame,
    p_model: np.ndarray,
    edge_min: float,
    odds_lo: float,
    odds_hi: float,
) -> list[BetRecord]:
    open_o = sub[["odds_open_h", "odds_open_d", "odds_open_a"]].to_numpy()
    close_o = sub[["psc_h", "psc_d", "psc_a"]].to_numpy()
    ok = _odds_validas(open_o) & _odds_validas(close_o)
    p_open = market_probs(open_o)
    p_close = market_probs(close_o)
    y = _y1x2(sub)
    k = np.argmax(p_model - p_open, axis=1)
    edge = p_model[np.arange(len(y)), k] - p_open[np.arange(len(y)), k]
    odds_pick = open_o[np.arange(len(y)), k]
    sel = ok & (edge >= edge_min) & (odds_pick >= odds_lo) & (odds_pick <= odds_hi)
    return _records_from_mask(sub, sel, k, open_o, p_close, p_open, y, "1X2")


def run_hypothesis_on_splits(
    hid: str,
    description: str,
    dev: pd.DataFrame,
    hold: pd.DataFrame,
    fn_dev: Callable[[pd.DataFrame], list[BetRecord]],
    fn_hold: Callable[[pd.DataFrame], list[BetRecord]] | None = None,
    n_min: int = 300,
) -> HypothesisResult:
    if fn_hold is None:
        fn_hold = fn_dev
    dev_s = _summarize_records(fn_dev(dev))
    hold_s = _summarize_records(fn_hold(hold))
    return HypothesisResult(
        id=hid,
        description=description,
        n_dev=dev_s["n"],
        clv_mean_dev=dev_s["clv_mean"] or float("nan"),
        clv_ci95_dev=dev_s["clv_ci95"] or [float("nan"), float("nan")],
        roi_dev=dev_s["roi"] or float("nan"),
        roi_ci95_dev=dev_s["roi_ci95"] or [float("nan"), float("nan")],
        n_holdout=hold_s["n"],
        clv_mean_holdout=hold_s["clv_mean"] or float("nan"),
        clv_ci95_holdout=hold_s["clv_ci95"] or [float("nan"), float("nan")],
        roi_holdout=hold_s["roi"] or float("nan"),
        roi_ci95_holdout=hold_s["roi_ci95"] or [float("nan"), float("nan")],
        viable=is_viable(hold_s, n_min=n_min),
    )


def bonferroni_adjusted_alpha(n_tests: int, alpha: float = 0.05) -> float:
    return alpha / max(n_tests, 1)
