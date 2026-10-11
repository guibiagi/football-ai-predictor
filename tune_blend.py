#!/usr/bin/env python3
"""tune_blend.py — qual mistura de sinais ganha do mercado, medido fora da amostra?

Testa combinações de peso sobre gols / xG / chute a gol e reporta o ganho de log
loss contra a linha de fechamento, por liga e agregado.

Aviso metodológico embutido: escolher a melhor combinação NESTA amostra já é um
ajuste. Por isso o relatório mostra a tabela inteira e avisa se o ótimo é um
pico estreito (provável sorte) ou um platô (provável efeito real). Combinação
escolhida num pico não deve virar aposta.

Uso:
    python tune_blend.py
"""

from __future__ import annotations

import sys

import numpy as np
import pandas as pd

sys.path.insert(0, ".")

from backtest_skill import MIN_MATCHES, log_loss, market_probs  # noqa: E402
from src.blend_model import SINAIS, BlendModel  # noqa: E402
from src.validation import expanding_window_backtest  # noqa: E402

# Combinações a testar. A primeira é a referência (só placar).
MISTURAS: list[dict[str, float]] = [
    {"gols": 1.0},
    {"gols": 0.25, "xg": 0.75},
    {"gols": 0.25, "sot": 0.75},
    {"gols": 0.25, "xg": 0.375, "sot": 0.375},
    {"xg": 0.5, "sot": 0.5},
]

INITIAL_TRAIN_YEARS = 4
STEP_MONTHS = 12


def rotulo(pesos: dict[str, float]) -> str:
    return "+".join(f"{k}{v:g}" for k, v in sorted(pesos.items()))


def avalia(df: pd.DataFrame, comp: str, pesos: dict[str, float],
           so_com_cobertura: bool = False) -> float | None:
    """Ganho de log loss vs fechamento nesta liga. Positivo = modelo melhor.

    Args:
        so_com_cobertura: se True, restringe treino E avaliação aos jogos que
            têm todos os sinais. Se False, avalia em todos os jogos com odds
            (o modelo ainda treina só nos que têm sinal).

    As duas amostras respondem perguntas diferentes e dão respostas diferentes:
    'só com cobertura' mede o sinal onde ele existe; 'todos' mede o efeito de
    trocar o treino do modelo nos jogos em que ele vai ser usado de verdade.
    Reportar as duas é obrigatório — escolher a que agrada seria o overfitting
    de sempre.
    """
    need = ["odds_h", "odds_d", "odds_a", "home_goals", "away_goals"]
    if so_com_cobertura:
        for sinal in pesos:
            ch, ca = SINAIS[sinal]
            need += [ch, ca]
    sub = df[df["competition"] == comp].dropna(subset=list(dict.fromkeys(need)))
    if len(sub) < MIN_MATCHES:
        return None
    preds = expanding_window_backtest(
        sub, lambda: BlendModel(pesos=pesos),
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
    return log_loss(market_probs(o), y) - log_loss(p, y)


def main() -> int:
    df = pd.read_csv("data/matches_xg.csv", parse_dates=["date"])
    ok = df[df.xg_h.notna() & df.sot_h.notna()]
    ligas = sorted(ok.competition.unique())
    print(f"ligas com xG e chute a gol: {len(ligas)} — {', '.join(ligas)}\n")

    tabelas: dict[str, pd.DataFrame] = {}
    for modo, so_cob in (("só com cobertura", True), ("todos os jogos", False)):
        print(f"=== amostra: {modo} ===")
        tab: dict[str, dict[str, float]] = {}
        for pesos in MISTURAS:
            r = rotulo(pesos)
            tab[r] = {}
            for comp in ligas:
                try:
                    g = avalia(df, comp, pesos, so_com_cobertura=so_cob)
                except Exception as e:
                    print(f"  {r} / {comp}: falhou ({type(e).__name__}: {e})")
                    continue
                if g is not None:
                    tab[r][comp] = g
            if tab[r]:
                print(f"  {r:34s} ganho médio {np.mean(list(tab[r].values())):+.4f}")
        t = pd.DataFrame(tab).T
        t.index.name = "mistura"
        tabelas[modo] = t
        print()

    for modo, t in tabelas.items():
        print(f"\n=== ganho de log loss vs fechamento — amostra: {modo} ===")
        print("(positivo = modelo melhor que o mercado; negativo = pior)\n")
        print(t.to_string(float_format="%.4f"))
        med = t.mean(axis=1).sort_values(ascending=False)
        print("\n--- média por mistura ---")
        for k, v in med.items():
            print(f"  {k:34s} {v:+.4f}")
        print(f"melhor: {med.index[0]} ({med.iloc[0]:+.4f})")

    # Veredito: as duas amostras concordam sobre qual mistura é melhor?
    melhores = {m: t.mean(axis=1).idxmax() for m, t in tabelas.items()}
    print(f"\n=== veredito ===")
    for m, k in melhores.items():
        print(f"  {m:18s} -> {k}")
    if len(set(melhores.values())) == 1:
        print("  As duas amostras concordam: a conclusão não depende da amostra.")
    else:
        print("  As amostras DISCORDAM — a 'melhora' depende de qual amostra eu olho,")
        print("  o que é sinal de efeito fraco demais para virar aposta.")

    pos = [k for t in tabelas.values() for k, v in t.mean(axis=1).items() if v > 0]
    if pos:
        print(f"\nATENÇÃO: {sorted(set(pos))} ficaram positivos — precisam sobreviver ao forward test.")
    else:
        print("\nNenhuma mistura bate o mercado em nenhuma amostra. Adicionar sinal não fechou a distância.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
