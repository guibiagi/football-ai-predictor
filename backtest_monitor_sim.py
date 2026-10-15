#!/usr/bin/env python3
"""Simulação walk-forward do monitor (últimas 2 temporadas, stake fixa 1u).

Entrada: odds de abertura B365; referência CLV: fechamento Pinnacle.
Compara baseline (Poisson) vs blend com mercado + isotônica.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, ".")

from backtest_ah import _margens, retorno
from backtest_markets import MODEL_PARAMS, PoissonModel, _log_loss_bin, _odds_validas, _over25, _sem_vig, roda
from backtest_skill import MIN_MATCHES
from src.market_blend import apply_market_pipeline, shrink_probs, tune_shrink_weight, Isotonic1x2
from src.monitor_gates import DEFAULT_GAIN_MIN, load_gates, market_allowed
from src.odds_utils import market_probs
from src.reporting import summarize_bets, summarize_bets_binary, write_metrics_report

EDGE_MIN = 0.08
SEASONS = 2
CALIB_FRAC = 0.5  # metade das 2 temporadas para tunar blend/isotônica


def _y1x2(df: pd.DataFrame) -> np.ndarray:
    return np.where(
        df.home_goals > df.away_goals, 0,
        np.where(df.home_goals == df.away_goals, 1, 2),
    )


def _filter_last_seasons(p: pd.DataFrame, seasons: float = SEASONS) -> pd.DataFrame:
    end = p.date.max()
    start = end - pd.Timedelta(days=365.25 * seasons)
    return p[p.date >= start].sort_values("date").reset_index(drop=True)


def _sim_1x2(
    p: pd.DataFrame,
    p_model: np.ndarray,
    gates: dict,
    liga: str,
    edge_min: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    open_o = p[["odds_open_h", "odds_open_d", "odds_open_a"]].to_numpy()
    close_o = p[["psc_h", "psc_d", "psc_a"]].to_numpy()
    ok = _odds_validas(open_o) & _odds_validas(close_o)
    y = _y1x2(p)
    p_open = market_probs(open_o)
    p_close = market_probs(close_o)
    k = np.argmax(p_model - p_open, axis=1)
    edge = p_model[np.arange(len(y)), k] - p_open[np.arange(len(y)), k]
    gate_ok = market_allowed(gates, liga, "1X2")
    sel = ok & (edge >= edge_min) & gate_ok
    ret = np.where(k[sel] == y[sel], open_o[sel, k[sel]] - 1.0, -1.0)
    clv = p_close[sel, k[sel]] - p_open[sel, k[sel]]
    return sel, ret, clv, p_model[sel], p_close[sel], y[sel]


def _sim_ou(
    p: pd.DataFrame,
    p_over: np.ndarray,
    gates: dict,
    liga: str,
    edge_min: float,
) -> tuple:
    open_o = p[["ou_open_over", "ou_open_under"]].to_numpy()
    close_o = p[["ou_psc_over", "ou_psc_under"]].to_numpy()
    ok = _odds_validas(open_o) & _odds_validas(close_o)
    y = (p.home_goals + p.away_goals > 2.5).astype(float).to_numpy()
    p_open = _sem_vig(open_o)
    p_close = _sem_vig(close_o)
    p_mod = np.stack([p_over, 1 - p_over], axis=1)
    k = np.argmax(p_mod - p_open, axis=1)
    edge = p_mod[np.arange(len(y)), k] - p_open[np.arange(len(y)), k]
    gate_ok = market_allowed(gates, liga, "O/U")
    sel = ok & (edge >= edge_min) & gate_ok
    ret = np.where((k[sel] == 0) == y[sel], open_o[sel, k[sel]] - 1.0, -1.0)
    clv = p_close[sel, k[sel]] - p_open[sel, k[sel]]
    y_bin = y[sel].astype(int)
    p_m = p_mod[sel]
    p_c = p_close[sel]
    return sel, ret, clv, p_m, p_c, y_bin


def _sim_ah(p: pd.DataFrame, rho: float, p_model_1x2: np.ndarray, edge_min: float) -> tuple:
    """AH: só fechamento Bet365 disponível — entrada = fechamento (CLV ~0 vs mesma casa)."""
    if "ah_line" not in p.columns:
        return None
    oa = p[["ah_close_h", "ah_close_a"]].to_numpy()
    ok = _odds_validas(oa) & p.ah_line.notna().to_numpy()
    if ok.sum() == 0:
        return None
    y1 = _y1x2(p)
    # proxy de edge: melhor lado 1X2 do modelo vs mercado implícito AH (devig)
    close_1x2 = p[["psc_h", "psc_d", "psc_a"]].to_numpy()
    if not _odds_validas(close_1x2).any():
        return None
    p_close = market_probs(close_1x2)
    k = np.argmax(p_model_1x2 - p_close, axis=1)
    edge = p_model_1x2[np.arange(len(y1)), k] - p_close[np.arange(len(y1)), k]
    sel = ok & (edge >= edge_min)
    rets, clvs = [], []
    for i in np.where(sel)[0]:
        r = p.iloc[i]
        d, pr = _margens(r.expected_goals_home, r.expected_goals_away, rho)
        real = r.home_goals - r.away_goals
        qa = _sem_vig(oa[i:i + 1])[0]
        best_s, best_ev = 0, -1.0
        for s in (0, 1):
            ev = qa[s] * oa[i, s] - 1
            if ev > best_ev:
                best_ev, best_s = ev, s
        if best_ev < edge_min:
            continue
        sinal = 1 if best_s == 0 else -1
        rets.append(float(retorno(sinal * real, r.ah_line, oa[i, best_s])))
        clvs.append(0.0)
    if not rets:
        return None
    ret = np.array(rets)
    clv = np.array(clvs)
    return sel, ret, clv, None, None, None


def _blend_probs_league(p: pd.DataFrame, calib_frac: float) -> np.ndarray:
    p = p.sort_values("date").reset_index(drop=True)
    pm = p[["home_win_probability", "draw_probability", "away_win_probability"]].to_numpy()
    close = p[["psc_h", "psc_d", "psc_a"]].to_numpy()
    ok = _odds_validas(close)
    out = pm.copy()
    if ok.sum() < 100:
        return out
    pm_ok = pm[ok]
    close_ok = close[ok]
    y_ok = _y1x2(p)[ok]
    n = len(pm_ok)
    n_cal = max(50, int(n * calib_frac))
    calib = slice(0, n_cal)
    eval_sl = slice(n_cal, n)
    p_eval, _, _ = apply_market_pipeline(
        pm_ok, close_ok, y_ok, calib, eval_sl, use_isotonic=True,
    )
    idx_ok = np.where(ok)[0]
    out[idx_ok[eval_sl]] = p_eval
    return out


def run_league(comp: str, df: pd.DataFrame, gates: dict) -> dict | None:
    p = roda(df, comp)
    if p is None or len(p) < MIN_MATCHES:
        return None
    p = _filter_last_seasons(p)
    if len(p) < 80:
        return None
    corte = p.date.min()
    try:
        rho = PoissonModel(**MODEL_PARAMS).fit(
            df[(df.competition == comp) & (df.date < corte)]
        )._rho
    except ValueError:
        rho = 0.0

    pm_base = p[["home_win_probability", "draw_probability", "away_win_probability"]].to_numpy()
    pm_blend = _blend_probs_league(p, CALIB_FRAC)
    p_over = _over25(p.expected_goals_home.to_numpy(), p.expected_goals_away.to_numpy(), rho)

    res = {"liga": comp, "n_matches": len(p), "markets": {}}
    for label, pm in (("baseline", pm_base), ("blend_isotonic", pm_blend)):
        mkt = {}
        s1 = _sim_1x2(p, pm, gates, comp, EDGE_MIN)
        if s1[0].sum() > 0:
            mkt["1X2"] = summarize_bets(s1[1], s1[2], s1[3], s1[4], s1[5])
        s2 = _sim_ou(p, p_over, gates, comp, EDGE_MIN)
        if s2[0].sum() > 0:
            mkt["O/U"] = summarize_bets_binary(s2[1], s2[2], s2[3], s2[4], s2[5])
        sah = _sim_ah(p, rho, pm, EDGE_MIN)
        if sah is not None and len(sah[1]) > 0:
            mkt["AH"] = summarize_bets(sah[1], sah[2], np.zeros(len(sah[1])), np.zeros((len(sah[1]), 3)), np.zeros(len(sah[1]), dtype=int))
        res["markets"][label] = mkt
    return res


def aggregate(results: list[dict]) -> dict:
    agg = {"baseline": {}, "blend_isotonic": {}}
    for model in agg:
        for mkt in ("1X2", "O/U", "AH"):
            parts = []
            for r in results:
                block = r.get("markets", {}).get(model, {}).get(mkt)
                if block and block.get("n_bets", 0) > 0:
                    parts.append(block)
            if not parts:
                continue
            n = sum(x["n_bets"] for x in parts)
            roi = sum(x["roi"] * x["n_bets"] for x in parts) / n
            clv = np.nanmean([x["clv_mean"] for x in parts if x["clv_mean"] is not None])
            agg[model][mkt] = {
                "n_bets": n,
                "roi_weighted": round(roi, 4),
                "clv_mean": round(float(clv), 4) if np.isfinite(clv) else None,
            }
    return agg


def main() -> int:
    gates = load_gates()
    if not gates:
        print("Aviso: monitor_gates.json ausente — rodando scripts/build_monitor_gates.py")
        import subprocess
        subprocess.run([sys.executable, "scripts/build_monitor_gates.py"], check=False)
        gates = load_gates()

    df = pd.read_csv("data/matches_xg.csv", parse_dates=["date"], low_memory=False)
    ligas = sorted(df[df.psc_h.notna() & df.odds_open_h.notna()].competition.unique())
    results = []
    for comp in ligas:
        try:
            r = run_league(comp, df, gates)
        except Exception as e:
            print(f"  {comp}: {type(e).__name__}: {e}")
            continue
        if r:
            results.append(r)
            print(f"  ok {comp} n={r['n_matches']}")

    payload = {
        "meta": {
            "edge_min": EDGE_MIN,
            "seasons": SEASONS,
            "gate_min_log_loss_gain": DEFAULT_GAIN_MIN,
            "entry": "B365 open (1X2/O/U); AH Bet365 close (sem abertura no dataset)",
            "clv_reference": "Pinnacle close devig",
        },
        "by_league": results,
        "aggregate": aggregate(results),
    }
    reports = Path("reports")
    reports.mkdir(exist_ok=True)
    out = reports / "monitor_sim_2seasons.json"
    out.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\nRelatório: {out}")
    write_metrics_report({"monitor_sim": payload}, reports_dir=reports)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
