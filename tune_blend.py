#!/usr/bin/env python3
"""tune_blend.py — qual peso de xG na mistura, medido fora da amostra?

Testa w = 0 (só gols) até 1 (só xG) em todas as ligas com xG, e reporta log loss
agregado contra a linha de fechamento.

Aviso metodológico embutido: escolher o melhor w NESTA amostra já é um ajuste.
Por isso o relatório mostra a curva inteira, e o script avisa se o ótimo é um
pico estreito (provável sorte) ou um platô largo (provável efeito real). Um w
escolhido num pico não deve ser usado.

Uso:
    python tune_blend.py
"""

from __future__ import annotations

import sys

import numpy as np
import pandas as pd

sys.path.insert(0, ".")

from backtest_skill import MIN_MATCHES, log_loss, market_probs  # noqa: E402
from src.validation import expanding_window_backtest  # noqa: E402
from src.xg_model import XGModel  # noqa: E402

PESOS = [0.0, 0.25, 0.5, 0.75, 1.0]
INITIAL_TRAIN_YEARS = 4
STEP_MONTHS = 12


def avalia(df: pd.DataFrame, comp: str, w: float) -> tuple[int, float] | None:
    sub = df[df["competition"] == comp].dropna(
        subset=["odds_h", "odds_d", "odds_a", "home_goals", "away_goals", "xg_h", "xg_a"]
    )
    if len(sub) < MIN_MATCHES:
        return None
    preds = expanding_window_backtest(
        sub, lambda: XGModel(blend_xg=w),
        initial_train_years=INITIAL_TRAIN_YEARS, step_months=STEP_MONTHS,
    )
    preds = preds.merge(
        sub[["date", "home_team", "away_team", "odds_h", "odds_d", "odds_a"]],
        on=["date", "home_team", "away_team"], how="inner",
    )
    if len(preds) < 100:
        return None
    p = preds[["home_win_probability", "draw_probability", "away_win_probability"]].to_numpy()
    o = preds[["odds_h", "odds_d", "odds_a"]].to_numpy()
    y = np.where(preds["home_goals"] > preds["away_goals"], 0,
                 np.where(preds["home_goals"] == preds["away_goals"], 1, 2))
    return len(preds), log_loss(market_probs(o), y) - log_loss(p, y)


def main() -> int:
    df = pd.read_csv("data/matches_xg.csv", parse_dates=["date"])
    ligas = sorted(df[df.xg_h.notna()].competition.unique())
    print(f"ligas com xG: {len(ligas)}\n")

    tabela: dict[float, dict[str, float]] = {}
    for w in PESOS:
        tabela[w] = {}
        for comp in ligas:
            try:
                r = avalia(df, comp, w)
            except Exception as e:
                print(f"  w={w} {comp}: falhou ({type(e).__name__}: {e})")
                continue
            if r:
                tabela[w][comp] = r[1]
        if tabela[w]:
            print(f"  w={w:.2f}: ganho_ll médio "
                  f"{np.mean(list(tabela[w].values())):+.4f}")

    res = pd.DataFrame(tabela)
    res.index.name = "liga"
    print("\n=== ganho de log loss vs fechamento, por peso de xG ===")
    print("(positivo = melhor que o mercado; negativo = pior)\n")
    print(res.to_string(float_format="%.4f"))
    print("\n--- média ---")
    med = res.mean()
    for w in PESOS:
        if w in med:
            print(f"  w={w:.2f}: {med[w]:+.4f}")
    melhor = med.idxmax()
    print(f"\nmelhor w = {melhor:.2f} ({med[melhor]:+.4f})")
    if med[melhor] > 0:
        print("ATENÇÃO: positivo aqui precisa sobreviver ao forward test antes de virar aposta.")

    # Pico estreito ou platô? Comparar o melhor com os vizinhos.
    vizinhos = [w for w in PESOS if abs(w - melhor) <= 0.26]
    if len(vizinhos) >= 3:
        spread = max(med[w] for w in vizinhos) - min(med[w] for w in vizinhos)
        print(f"variação entre vizinhos de w={melhor:.2f}: {spread:.4f} "
              f"({'platô — efeito provável' if spread < 0.005 else 'pico estreito — provável sorte'})")

    # Calibração do melhor modelo, para ver COMO ele erra.
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
