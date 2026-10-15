#!/usr/bin/env python3
"""Gera data/monitor_gates.json a partir de backtests walk-forward."""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from backtest_markets import MODEL_PARAMS, PoissonModel, _log_loss_bin, _odds_validas, _over25, _sem_vig, roda
from backtest_skill import MIN_MATCHES, analisa
from src.monitor_gates import DEFAULT_GAIN_MIN, save_gates


def main() -> int:
    df = pd.read_csv("data/matches_xg.csv", parse_dates=["date"], low_memory=False)
    gates: dict[str, dict] = {"1X2": {}, "O/U": {}}

    detalhes = []
    for comp in sorted(df.competition.unique()):
        try:
            r = analisa(df, comp, com_xg=False)
        except Exception:
            continue
        if r is None:
            continue
        detalhes.append(r)
        gain = r["ganho_ll"]
        gates["1X2"][comp] = {
            "log_loss_gain_vs_close": round(gain, 4),
            "alert": gain >= DEFAULT_GAIN_MIN,
            "n": r["n"],
        }

    ligas_ou = sorted(df[df.psc_h.notna() & df.ou_psc_over.notna()].competition.unique())
    for comp in ligas_ou:
        p = roda(df, comp)
        if p is None or len(p) < MIN_MATCHES:
            continue
        y2 = (p.home_goals + p.away_goals > 2.5).astype(float).to_numpy()
        odds = p[["ou_psc_over", "ou_psc_under"]].to_numpy()
        ok = _odds_validas(odds)
        if ok.sum() < 100:
            continue
        corte = p.date.min()
        try:
            rho = PoissonModel(**MODEL_PARAMS).fit(
                df[(df.competition == comp) & (df.date < corte)]
            )._rho
        except ValueError:
            rho = 0.0
        p_over = _over25(p.expected_goals_home.to_numpy(), p.expected_goals_away.to_numpy(), rho)
        mk = _sem_vig(odds[ok])[:, 0]
        gain = _log_loss_bin(mk, y2[ok]) - _log_loss_bin(p_over[ok], y2[ok])
        gates["O/U"][comp] = {
            "log_loss_gain_vs_close": round(gain, 4),
            "alert": gain >= DEFAULT_GAIN_MIN,
            "n": int(ok.sum()),
        }

    path = save_gates(gates)
    n_alert = sum(1 for m in gates.values() for v in m.values() if v.get("alert"))
    print(f"Gates gravados em {path} ({n_alert} liga/mercado com alerta >= {DEFAULT_GAIN_MIN})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
