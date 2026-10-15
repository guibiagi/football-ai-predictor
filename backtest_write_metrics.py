#!/usr/bin/env python3
"""Grava reports/metrics_YYYYMMDD.json a partir de backtests walk-forward."""

from __future__ import annotations

import sys
from datetime import datetime, timezone

import numpy as np
import pandas as pd

sys.path.insert(0, ".")

from backtest_markets import _odds_validas, roda
from backtest_skill import MIN_MATCHES, analisa, market_probs, log_loss
from src.market_blend import apply_market_pipeline
from src.reporting import write_metrics_report


def _y1x2(df: pd.DataFrame) -> np.ndarray:
    return np.where(
        df.home_goals > df.away_goals, 0,
        np.where(df.home_goals == df.away_goals, 1, 2),
    )


def blend_gains(df: pd.DataFrame, comp: str) -> dict | None:
    p = roda(df, comp)
    if p is None or len(p) < MIN_MATCHES:
        return None
    p = p.sort_values("date").reset_index(drop=True)
    pm = p[["home_win_probability", "draw_probability", "away_win_probability"]].to_numpy()
    close = p[["psc_h", "psc_d", "psc_a"]].to_numpy()
    ok = _odds_validas(close)
    if ok.sum() < 200:
        return None
    y = _y1x2(p)[ok]
    pm_ok = pm[ok]
    close_ok = close[ok]
    n_ok = len(pm_ok)
    n_cal = max(100, n_ok // 2)
    calib = slice(0, n_cal)
    eval_sl = slice(n_cal, n_ok)
    p_eval, w, meta = apply_market_pipeline(
        pm_ok, close_ok, y, calib, eval_sl, use_isotonic=True,
    )
    y_ev = y[eval_sl]
    ll_base = log_loss(pm_ok[eval_sl], y_ev)
    ll_blend = log_loss(p_eval, y_ev)
    ll_close = log_loss(market_probs(close_ok[eval_sl]), y_ev)
    return {
        "liga": comp,
        "shrink_w": w,
        "isotonic": meta["isotonic"],
        "log_loss_baseline": round(ll_base, 4),
        "log_loss_blend": round(ll_blend, 4),
        "log_loss_close": round(ll_close, 4),
        "gain_blend_vs_close": round(ll_close - ll_blend, 4),
        "gain_baseline_vs_close": round(ll_close - ll_base, 4),
    }


def main() -> int:
    df = pd.read_csv("data/matches_xg.csv", parse_dates=["date"], low_memory=False)
    skill = []
    for comp in sorted(df.competition.unique()):
        try:
            r = analisa(df, comp, com_xg=False)
        except Exception:
            continue
        if r is None:
            continue
        skill.append({
            "liga": comp,
            "n": r["n"],
            "log_loss_model": round(r["ll_modelo"], 4),
            "log_loss_close": round(r["ll_fechamento"], 4),
            "gain_vs_close": round(r["ganho_ll"], 4),
            "brier_model": round(r["brier_modelo"], 4),
            "brier_close": round(r["brier_fechamento"], 4),
        })

    blend_rows = []
    for comp in sorted(df[df.psc_h.notna()].competition.unique()):
        g = blend_gains(df, comp)
        if g:
            blend_rows.append(g)
            print(f"  blend {comp}: gain {g['gain_blend_vs_close']:+.4f} (base {g['gain_baseline_vs_close']:+.4f})")

    payload = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "skill_by_league": skill,
        "blend_isotonic_by_league": blend_rows,
        "aggregate": {
            "mean_gain_vs_close": round(float(np.mean([s["gain_vs_close"] for s in skill])), 4),
            "leagues_model_beats_close": int(sum(1 for s in skill if s["gain_vs_close"] > 0)),
            "n_leagues_skill": len(skill),
            "mean_blend_gain_vs_close": round(
                float(np.mean([b["gain_blend_vs_close"] for b in blend_rows])), 4
            ) if blend_rows else None,
        },
    }
    path = write_metrics_report(payload)
    print(f"Métricas gravadas em {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
