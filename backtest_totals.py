#!/usr/bin/env python3
"""backtest_totals.py — recalibrar o modelo para o total de gols (Over/Under 2.5).

Hipótese (backtest_markets.py): os λ são ajustados para o resultado, então
P(over 2.5) sai como subproduto. Recalibrar mirando o total deveria fechar a
distância de ~0,015 até a Pinnacle.

Sem vazamento: as predições já são walk-forward. A calibração é ajustada no
começo do período de teste e medida SÓ no resto. Dois cortes (50% e 33%) —
um efeito que troca de sinal entre cortes é ruído (regra do skill).

Métodos:
  raw    — P(over) da matriz de placares, como hoje
  escala — λ_home e λ_away multiplicados por um k da liga (k ajustado no corte)
  platt  — regressão logística sobre logit(P_raw) (corrige viés e confiança)

Uso:
    python backtest_totals.py
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression

from backtest_markets import (MODEL_PARAMS, PoissonModel, _log_loss_bin,
                              _odds_validas, _over25, _sem_vig, roda)

GRADE_K = np.round(np.arange(0.85, 1.151, 0.01), 2)
CORTES = (0.5, 0.33)


def _logit(p: np.ndarray) -> np.ndarray:
    p = np.clip(p, 1e-6, 1 - 1e-6)
    return np.log(p / (1 - p))


def avalia(p: pd.DataFrame, rho: float, frac: float) -> dict:
    """Ganho vs Pinnacle O/U no trecho de avaliação, por método."""
    odds = p[["ou_psc_over", "ou_psc_under"]].to_numpy()
    p = p[_odds_validas(odds)].sort_values("date").reset_index(drop=True)
    y = (p.home_goals + p.away_goals > 2.5).astype(float).to_numpy()
    mk = _sem_vig(p[["ou_psc_over", "ou_psc_under"]].to_numpy())[:, 0]
    lh, la = p.expected_goals_home.to_numpy(), p.expected_goals_away.to_numpy()
    n_cal = int(len(p) * frac)
    cal, ev = slice(0, n_cal), slice(n_cal, None)

    raw = _over25(lh, la, rho)
    # k: o multiplicador de λ que minimiza log loss no trecho de calibração
    perdas = [_log_loss_bin(_over25(lh[cal] * k, la[cal] * k, rho), y[cal]) for k in GRADE_K]
    k = float(GRADE_K[int(np.argmin(perdas))])
    esc = _over25(lh * k, la * k, rho)
    lr = LogisticRegression(C=1e6).fit(_logit(raw[cal]).reshape(-1, 1), y[cal])
    platt = lr.predict_proba(_logit(raw).reshape(-1, 1))[:, 1]

    ll_mk = _log_loss_bin(mk[ev], y[ev])
    return {"n_ev": len(y[ev]), "k": k,
            "raw": ll_mk - _log_loss_bin(raw[ev], y[ev]),
            "escala": ll_mk - _log_loss_bin(esc[ev], y[ev]),
            "platt": ll_mk - _log_loss_bin(platt[ev], y[ev]),
            # viés: P(over) média prevista - frequência real, no trecho de avaliação
            "vies_raw": float(raw[ev].mean() - y[ev].mean())}


def main() -> int:
    df = pd.read_csv("data/matches_xg.csv", parse_dates=["date"], low_memory=False)
    ligas = sorted(df[df.psc_h.notna() & df.ou_psc_over.notna()].competition.unique())
    res = {f: [] for f in CORTES}
    for comp in ligas:
        p = roda(df, comp)
        if p is None or "ou_psc_over" not in p.columns:
            continue
        try:  # rho só na janela inicial de treino (mesma regra do backtest_markets)
            rho = PoissonModel(**MODEL_PARAMS).fit(
                df[(df.competition == comp) & (df.date < p.date.min())])._rho
        except ValueError:
            rho = 0.0
        for f in CORTES:
            r = avalia(p, rho, f)
            res[f].append({"liga": comp, **r})
        r = res[0.5][-1]
        print(f"  {comp:22s} n_ev={r['n_ev']:4d} k={r['k']:.2f} viés={r['vies_raw']:+.3f}  "
              f"raw {r['raw']:+.4f}  escala {r['escala']:+.4f}  platt {r['platt']:+.4f}",
              flush=True)

    print("\n=== ganho vs Pinnacle O/U 2.5 no trecho de avaliação "
          "(positivo = modelo melhor) ===")
    for f in CORTES:
        t = pd.DataFrame(res[f])
        print(f"\ncalibra nos primeiros {f:.0%} do teste, avalia no resto:")
        for m in ("raw", "escala", "platt"):
            print(f"  {m:7s} média {t[m].mean():+.4f}   modelo melhor em "
                  f"{int((t[m] > 0).sum())}/{len(t)} ligas")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
