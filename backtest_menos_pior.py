#!/usr/bin/env python3
"""backtest_menos_pior.py — qual combinação de mercado + método perde MENOS.

Nenhum mercado deu vantagem (backtest_markets/totals/ah). Pergunta agora:
qual estratégia tem o ROI menos negativo, escolhida sem olhar o resultado.

Método: EV encolhido para o mercado — ev = w·ev_modelo + (1−w)·ev_mercado.
w=1 é o modelo puro (o que já perdeu); w<1 só aposta quando o modelo discorda
E o preço sustenta. Para mercado binário isso é exatamente misturar as
probabilidades. Escolha de (mercado, w, limiar) na 1ª metade de cada liga;
o número que vale é o da 2ª metade, intocada.

Uso:
    python backtest_menos_pior.py
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from backtest_ah import _margens, retorno
from backtest_markets import (MODEL_PARAMS, PoissonModel, _odds_validas,
                              _over25, _sem_vig, roda)

PESOS = (1.0, 0.5, 0.25)
LIMIARES = (0.02, 0.05, 0.10)


def candidatos(p: pd.DataFrame, rho: float) -> list[dict]:
    """Uma linha por (jogo, lado) apostável: ev do modelo, ev do mercado, retorno real."""
    out = []
    y = np.where(p.home_goals > p.away_goals, 0, np.where(p.home_goals == p.away_goals, 1, 2))
    m1 = p[["home_win_probability", "draw_probability", "away_win_probability"]].to_numpy()
    o1 = p[["psc_h", "psc_d", "psc_a"]].to_numpy()
    ok = _odds_validas(o1)
    q1 = np.full_like(o1, np.nan)
    q1[ok] = _sem_vig(o1[ok])
    over = (p.home_goals + p.away_goals > 2.5).to_numpy()
    po = _over25(p.expected_goals_home.to_numpy(), p.expected_goals_away.to_numpy(), rho)
    oo = p[["ou_psc_over", "ou_psc_under"]].to_numpy()
    oko = _odds_validas(oo)
    qo = np.full_like(oo, np.nan)
    qo[oko] = _sem_vig(oo[oko])
    oa = p[["ah_close_h", "ah_close_a"]].to_numpy() if "ah_line" in p else None
    oka = _odds_validas(oa) & p.ah_line.notna().to_numpy() if oa is not None else None

    for i, r in enumerate(p.itertuples()):
        base = {"date": r.date}
        if ok[i]:
            for k in range(3):
                out.append({**base, "mercado": "1X2 Pinnacle", "ev_m": m1[i, k] * o1[i, k] - 1,
                            "ev_k": q1[i, k] * o1[i, k] - 1,
                            "ret": o1[i, k] - 1 if y[i] == k else -1.0})
        if oko[i]:
            for k, pm, win in ((0, po[i], over[i]), (1, 1 - po[i], not over[i])):
                out.append({**base, "mercado": "O/U Pinnacle", "ev_m": pm * oo[i, k] - 1,
                            "ev_k": qo[i, k] * oo[i, k] - 1,
                            "ret": oo[i, k] - 1 if win else -1.0})
        if oka is not None and oka[i]:
            d, pr = _margens(r.expected_goals_home, r.expected_goals_away, rho)
            real = r.home_goals - r.away_goals
            qa = _sem_vig(oa[i:i + 1])[0]
            for s, sinal, linha in ((0, 1, r.ah_line), (1, -1, -r.ah_line)):
                # ponytail: ev_mercado ignora push nas linhas inteiras/quarto; usar a
                # distribuição de margem implícita se o AH virar a estratégia final
                out.append({**base, "mercado": "AH Bet365",
                            "ev_m": float((retorno(sinal * d, linha, oa[i, s]) * pr).sum()),
                            "ev_k": qa[s] * oa[i, s] - 1,
                            "ret": float(retorno(sinal * real, linha, oa[i, s]))})
    return out


def roi(c: pd.DataFrame, w: float, t: float) -> tuple[int, float]:
    sel = c[w * c.ev_m + (1 - w) * c.ev_k > t]
    return len(sel), (sel.ret.mean() if len(sel) else np.nan)


def main() -> int:
    df = pd.read_csv("data/matches_xg.csv", parse_dates=["date"], low_memory=False)
    ligas = sorted(df[df.psc_h.notna() & df.ou_psc_over.notna()].competition.unique())
    partes = []
    for comp in ligas:
        p = roda(df, comp)
        if p is None:
            continue
        try:
            rho = PoissonModel(**MODEL_PARAMS).fit(
                df[(df.competition == comp) & (df.date < p.date.min())])._rho
        except ValueError:
            rho = 0.0
        c = pd.DataFrame(candidatos(p.sort_values("date").reset_index(drop=True), rho))
        corte = p.date.sort_values().iloc[len(p) // 2]
        c["metade"] = np.where(c.date < corte, 1, 2)
        partes.append(c)
        print(f"  {comp}: {len(c)} candidatos", flush=True)
    c = pd.concat(partes)

    linhas = []
    for mk, g in c.groupby("mercado"):
        cego = g.ret.mean()  # apostar tudo no escuro = só pagar a margem
        for w in PESOS:
            for t in LIMIARES:
                n1, r1 = roi(g[g.metade == 1], w, t)
                n2, r2 = roi(g[g.metade == 2], w, t)
                linhas.append({"mercado": mk, "w": w, "limiar": t, "n1": n1, "roi1": r1,
                               "n2": n2, "roi2": r2, "cego": cego})
    t = pd.DataFrame(linhas)
    t = t[(t.n1 >= 200) & (t.n2 >= 200)]
    print("\n" + t.to_string(index=False, float_format="%+.4f"))

    best = t.sort_values("roi1", ascending=False).iloc[0]  # escolhido SÓ na 1ª metade
    g = c[(c.mercado == best.mercado) & (c.metade == 2)]
    sel = g[best.w * g.ev_m + (1 - best.w) * g.ev_k > best.limiar].ret.to_numpy()
    rng = np.random.default_rng(0)
    bs = [rng.choice(sel, len(sel)).mean() for _ in range(2000)]
    print(f"\nescolhido na 1ª metade: {best.mercado}, w={best.w}, EV>{best.limiar:.0%} "
          f"(1ª metade ROI {best.roi1:+.2%})")
    print(f"2ª metade, intocada: {len(sel)} apostas, ROI {sel.mean():+.2%}  "
          f"IC95 [{np.percentile(bs, 2.5):+.1%}, {np.percentile(bs, 97.5):+.1%}]  "
          f"| apostar no escuro nesse mercado: {best.cego:+.2%}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
