#!/usr/bin/env python3
"""backtest_ah.py — o modelo contra o Asian handicap (linha e odds Bet365 de fechamento).

A linha varia por jogo, então a matriz de placares vira distribuição da margem
(gols casa − gols fora) e cada aposta é liquidada com push / meia-linha.
Linhas de quarto (−0,25, −0,75…) = metade da aposta em cada linha vizinha.

Duas medidas, fora da amostra (walk-forward do backtest_markets):
  1. log loss nas linhas de meio gol (±0,5, ±1,5…) — sem push, binário limpo
  2. ROI apostando o lado com EV do modelo > limiar, em todas as linhas

Uso:
    python backtest_ah.py
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from backtest_markets import (MODEL_PARAMS, PoissonModel, _log_loss_bin,
                              _odds_validas, _sem_vig, roda)
from src.poisson_model import build_score_matrix

LIMIARES = (0.03, 0.05, 0.10)


def _componentes(linha: float) -> tuple[float, ...]:
    """Linha de quarto vira duas meias apostas; as outras ficam inteiras."""
    if abs(linha * 4) % 2 == 1:  # x,25 ou x,75
        return (linha - 0.25, linha + 0.25)
    return (linha,)


def retorno(margem, linha: float, odd: float):
    """Retorno por unidade apostada no lado cuja margem é `margem` com handicap `linha`.

    Aceita margem escalar (liquidação real) ou array (para calcular a esperança).
    """
    r = 0.0
    for h in _componentes(linha):
        x = np.asarray(margem) + h
        r = r + np.where(x > 0, odd - 1, np.where(x == 0, 0.0, -1.0))
    return r / len(_componentes(linha))


def _margens(lh: float, la: float, rho: float) -> tuple[np.ndarray, np.ndarray]:
    """Distribuição da margem casa − fora: (valores, probabilidades)."""
    m = build_score_matrix(lh, la, rho=rho)
    i = np.arange(m.shape[0])
    d = np.subtract.outer(i, i).ravel()
    return d, m.ravel()


def _check() -> None:
    # −0,25 com empate: metade perde, metade devolve
    assert retorno(0, -0.25, 2.0) == -0.5
    assert retorno(1, -0.25, 2.0) == 1.0
    assert retorno(0, -0.5, 2.0) == -1.0
    assert retorno(0, 0.0, 2.0) == 0.0
    # +0,75 perdendo por 1: metade (+0,5) perde, metade (+1) devolve
    assert retorno(-1, 0.75, 2.0) == -0.5
    assert retorno(-1, 1.25, 2.0) == 0.5   # +1 push, +1,5 ganha meia


def main() -> int:
    _check()
    df = pd.read_csv("data/matches_xg.csv", parse_dates=["date"], low_memory=False)
    ligas = sorted(df[df.psc_h.notna() & df.ah_line.notna()].competition.unique())
    tot = {t: [] for t in LIMIARES}
    ll = []
    for comp in ligas:
        p = roda(df, comp)
        if p is None or "ah_line" not in p.columns:
            continue
        p = p[_odds_validas(p[["ah_close_h", "ah_close_a"]].to_numpy()) & p.ah_line.notna()]
        if len(p) < 100:
            continue
        try:  # rho só na janela inicial de treino
            rho = PoissonModel(**MODEL_PARAMS).fit(
                df[(df.competition == comp) & (df.date < p.date.min())])._rho
        except ValueError:
            rho = 0.0

        apostas = {t: [] for t in LIMIARES}
        pm_meio, pk_meio, y_meio = [], [], []
        for r in p.itertuples():
            d, pr = _margens(r.expected_goals_home, r.expected_goals_away, rho)
            real = r.home_goals - r.away_goals
            ev_h = float((retorno(d, r.ah_line, r.ah_close_h) * pr).sum())
            ev_a = float((retorno(-d, -r.ah_line, r.ah_close_a) * pr).sum())
            for t in LIMIARES:
                if ev_h > t:
                    apostas[t].append(float(retorno(real, r.ah_line, r.ah_close_h)))
                if ev_a > t:
                    apostas[t].append(float(retorno(-real, -r.ah_line, r.ah_close_a)))
            if r.ah_line % 1 == 0.5:  # meio gol: binário sem push
                pm_meio.append(pr[d + r.ah_line > 0].sum())
                pk_meio.append(_sem_vig(np.array([[r.ah_close_h, r.ah_close_a]]))[0, 0])
                y_meio.append(float(real + r.ah_line > 0))

        g = np.nan
        if len(y_meio) >= 50:
            y = np.array(y_meio)
            g = _log_loss_bin(np.array(pk_meio), y) - _log_loss_bin(np.array(pm_meio), y)
            ll.append((comp, g, len(y)))
        for t in LIMIARES:
            tot[t] += apostas[t]
        print(f"  {comp:22s} n={len(p):4d}  ll meio-gol {g:+.4f}  " + "  ".join(
            f"EV>{t:.0%}: {len(a):4d} ROI {np.mean(a) if a else 0:+.1%}"
            for t, a in apostas.items()), flush=True)

    g = np.array([x[1] for x in ll])
    print(f"\nlog loss nas linhas de meio gol vs Bet365 (positivo = modelo melhor): "
          f"média {g.mean():+.4f}, modelo melhor em {int((g > 0).sum())}/{len(g)} ligas")
    rng = np.random.default_rng(0)
    for t, a in tot.items():
        a = np.array(a)
        bs = [rng.choice(a, len(a)).mean() for _ in range(1000)]
        print(f"TOTAL EV>{t:.0%}: {len(a)} apostas, ROI {a.mean():+.2%}  "
              f"IC95 [{np.percentile(bs, 2.5):+.1%}, {np.percentile(bs, 97.5):+.1%}]")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
