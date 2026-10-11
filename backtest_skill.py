#!/usr/bin/env python3
"""backtest_skill.py — o modelo sabe mais que o mercado? (teste de habilidade)

Mede log loss e Brier do modelo contra as probabilidades implícitas das odds,
fora da amostra, por liga e por resultado. Testa DOIS preços:

  fechamento (AvgC*) — a linha mais eficiente que existe. Perder dela é o normal.
  abertura   (B365)  — a linha antiga, antes do dinheiro afiado mover o preço.

Por que log loss e não ROI: ROI em 200 apostas é sorte; log loss mede calibração
e converge rápido. Se o modelo perde da linha de fechamento, apostar contra ela
tem expectativa negativa, por mais que um mês de ROI pareça bom. A pergunta útil
é a segunda: o modelo ganha da linha de ABERTURA? Aí existe dinheiro, porque é o
preço que ainda está disponível quando se aposta cedo.

Uso:
    python backtest_skill.py                  # todas as ligas com dado
    python backtest_skill.py --liga "La Liga"
    python backtest_skill.py --cache preds.pkl
"""

from __future__ import annotations

import argparse
import logging
import pickle
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, ".")

from src.poisson_model import PoissonModel
from src.validation import expanding_window_backtest
from src.xg_model import XGModel

logging.basicConfig(level=logging.WARNING)

MODEL_PARAMS = dict(decay_lambda=0.5)
INITIAL_TRAIN_YEARS = 4
STEP_MONTHS = 12
MIN_MATCHES = 600
MIN_PREDICTIONS = 300
EDGES = (0.05, 0.10)


def market_probs(odds: np.ndarray) -> np.ndarray:
    """Odds → probabilidades, removendo a margem da casa.

    A margem faz as probabilidades implícitas somarem mais de 1. Normalizar é o
    mínimo para comparar com o modelo. Não é perfeito (a margem não é
    distribuída igualmente entre os três resultados), e isso deixa o mercado
    levemente otimista em TODOS os resultados — ou seja, o teste do modelo fica
    conservador, não generoso.
    """
    raw = 1.0 / odds
    return raw / raw.sum(axis=1, keepdims=True)


def log_loss(probs: np.ndarray, y: np.ndarray) -> float:
    return float(-np.mean(np.log(np.clip(probs[np.arange(len(y)), y], 1e-15, 1))))


def brier(probs: np.ndarray, y: np.ndarray) -> float:
    onehot = np.zeros_like(probs)
    onehot[np.arange(len(y)), y] = 1
    return float(np.mean(np.sum((probs - onehot) ** 2, axis=1)))


def roi(p_mod: np.ndarray, y: np.ndarray, odds: np.ndarray, edge_min: float) -> dict:
    """ROI apostando nas odds dadas onde o modelo vê vantagem contra elas.

    Uma aposta por jogo, no resultado de maior vantagem, stake fixo. Sem
    Kelly: o objetivo aqui é medir se há vantagem, não dimensionar banca.
    """
    p_imp = market_probs(odds)
    k = np.argmax(p_mod - p_imp, axis=1)
    edge = p_mod[np.arange(len(y)), k] - p_imp[np.arange(len(y)), k]
    sel = edge >= edge_min
    if sel.sum() == 0:
        return {"n": 0, "roi": 0.0, "acerto": 0.0}
    kk, oo, yy = k[sel], odds[np.arange(len(y)), k][sel], y[sel]
    retorno = np.where(kk == yy, oo, 0.0)
    return {
        "n": int(sel.sum()),
        "roi": float(retorno.mean() - 1.0),
        "acerto": float((kk == yy).mean()),
    }


def analisa(df: pd.DataFrame, comp: str, com_xg: bool = True) -> dict | None:
    sub = df[df["competition"] == comp].dropna(
        subset=["odds_h", "odds_d", "odds_a", "home_goals", "away_goals"]
    )
    if len(sub) < MIN_MATCHES:
        return None

    preds = expanding_window_backtest(
        sub, lambda: PoissonModel(**MODEL_PARAMS),
        initial_train_years=INITIAL_TRAIN_YEARS, step_months=STEP_MONTHS,
    )
    cols = ["date", "home_team", "away_team", "odds_h", "odds_d", "odds_a",
            "odds_open_h", "odds_open_d", "odds_open_a"]
    preds = preds.merge(sub[cols], on=["date", "home_team", "away_team"], how="inner")
    if len(preds) < MIN_PREDICTIONS:
        return None

    p_mod = preds[["home_win_probability", "draw_probability",
                   "away_win_probability"]].to_numpy()
    close = preds[["odds_h", "odds_d", "odds_a"]].to_numpy()
    y = np.where(preds["home_goals"] > preds["away_goals"], 0,
                 np.where(preds["home_goals"] == preds["away_goals"], 1, 2))

    r = {
        "liga": comp, "n": len(preds),
        "ll_modelo": log_loss(p_mod, y),
        "ll_fechamento": log_loss(market_probs(close), y),
        "brier_modelo": brier(p_mod, y),
        "brier_fechamento": brier(market_probs(close), y),
        "roi_fech_5": roi(p_mod, y, close, EDGES[0])["roi"],
        "roi_fech_10": roi(p_mod, y, close, EDGES[1])["roi"],
        "preds": preds, "y": y, "p_mod": p_mod,
    }
    r["ganho_ll"] = r["ll_fechamento"] - r["ll_modelo"]

    # Mesma coisa, mas com o modelo treinado em xG. Só roda nas ligas com xG.
    if com_xg and "xg_h" in df.columns and df[df.competition == comp].xg_h.notna().sum() >= MIN_MATCHES:
        try:
            px = expanding_window_backtest(
                sub, lambda: XGModel(**MODEL_PARAMS),
                initial_train_years=INITIAL_TRAIN_YEARS, step_months=STEP_MONTHS,
            )
            px = px.merge(sub[cols], on=["date", "home_team", "away_team"], how="inner")
            if len(px) >= MIN_PREDICTIONS:
                px = px.merge(
                    sub[["date", "home_team", "away_team", "xg_h", "xg_a"]],
                    on=["date", "home_team", "away_team"], how="inner",
                ).dropna(subset=["xg_h", "xg_a"])
                if len(px) >= MIN_PREDICTIONS:
                    pmx = px[["home_win_probability", "draw_probability",
                              "away_win_probability"]].to_numpy()
                    cx = px[["odds_h", "odds_d", "odds_a"]].to_numpy()
                    yx = np.where(px["home_goals"] > px["away_goals"], 0,
                                  np.where(px["home_goals"] == px["away_goals"], 1, 2))
                    r.update({
                        "n_xg": len(px),
                        "ll_xg": log_loss(pmx, yx),
                        "ganho_ll_xg": log_loss(market_probs(cx), yx) - log_loss(pmx, yx),
                        "roi_xg_5": roi(pmx, yx, cx, EDGES[0])["roi"],
                        "roi_xg_10": roi(pmx, yx, cx, EDGES[1])["roi"],
                        "preds_xg": px, "y_xg": yx, "p_mod_xg": pmx,
                    })
        except Exception as e:
            print(f"    xG falhou em {comp}: {type(e).__name__}: {e}")

    # Teste da linha de abertura — só nas ligas que têm B365.
    op = preds[["odds_open_h", "odds_open_d", "odds_open_a"]].to_numpy()
    ok = ~np.isnan(op).any(axis=1)
    if ok.sum() >= MIN_PREDICTIONS:
        op, yo, pm = op[ok], y[ok], p_mod[ok]
        r.update({
            "n_abertura": int(ok.sum()),
            "ll_abertura": log_loss(market_probs(op), yo),
            "roi_abert_5": roi(pm, yo, op, EDGES[0])["roi"],
            "roi_abert_10": roi(pm, yo, op, EDGES[1])["roi"],
            "n_apost_abert_5": roi(pm, yo, op, EDGES[0])["n"],
            "mov_favorito": float(np.mean(1.0 / op.max(axis=1) - 1.0 / close[ok].max(axis=1))),
        })
    return r


def relatorio_calibracao(detalhes: list[dict]) -> None:
    """Quando o modelo diz X%, acontece X%? Viés mostra onde ele mente."""
    p_all = np.vstack([r["p_mod"] for r in detalhes]).ravel()
    y_all = np.concatenate([r["y"] for r in detalhes])
    ok = np.tile(np.array([0, 1, 2]), len(y_all)) == np.repeat(y_all, 3)
    bins = np.array([0, .1, .2, .3, .4, .5, .6, .7, .8, .9, 1.01])
    idx = np.digitize(p_all, bins) - 1
    print("\n--- calibração do modelo (todos os jogos, todas as ligas) ---")
    print("  faixa      n       diz     acontece    viés")
    for b in range(len(bins) - 1):
        m = idx == b
        if m.sum() < 100:
            continue
        diz, acon = p_all[m].mean(), ok[m].mean()
        print(f"  {bins[b]:.0%}-{bins[b+1]:.0%}  {m.sum():7d}  {diz:7.1%}  {acon:8.1%}  {acon-diz:+7.1%}")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--liga", default=None)
    ap.add_argument("--cache", default=None, help="pickle com os detalhes, para não retreinar")
    ap.add_argument("--dados", default="data/matches_xg.csv",
                    help="dataset (matches_xg.csv tem xG; matches_football.csv não)")
    ap.add_argument("--sem-xg", action="store_true", help="não rodar o modelo de xG")
    args = ap.parse_args()

    if args.cache:
        with open(args.cache, "rb") as f:
            detalhes = pickle.load(f)
    else:
        df = pd.read_csv(args.dados, parse_dates=["date"])
        tem_xg = "xg_h" in df.columns and df.xg_h.notna().sum() > 0
        print(f"{args.dados}: {len(df)} jogos, {df.competition.nunique()} ligas"
              + (f", {df.xg_h.notna().sum()} com xG" if tem_xg else ", sem xG"))
        ligas = sorted(df["competition"].unique())
        if args.liga:
            ligas = [l for l in ligas if args.liga.lower() in l.lower()]
        detalhes = []
        for comp in ligas:
            try:
                r = analisa(df, comp, com_xg=tem_xg and not args.sem_xg)
            except Exception as e:
                print(f"  {comp}: falhou ({type(e).__name__}: {e})")
                continue
            if r is not None:
                detalhes.append(r)
                extra = f", ganho_ll_xg {r['ganho_ll_xg']:+.4f}" if "ganho_ll_xg" in r else ""
                print(f"  ok {comp}: {r['n']} previsões, ganho_ll {r['ganho_ll']:+.4f}{extra}")

    if not detalhes:
        print("Nenhuma liga com dado suficiente.")
        return 1

    esc = lambda d: {k: v for k, v in d.items() if k not in ("preds", "y", "p_mod")}
    res = pd.DataFrame([esc(d) for d in detalhes]).sort_values("ganho_ll", ascending=False)
    fmt = "%.4f"

    print(f"\n=== Modelo vs mercado — {len(res)} ligas, fora da amostra ===\n")
    print("ganho_ll > 0 = modelo melhor que o FECHAMENTO (log loss menor)\n")
    print(res[["liga", "n", "ll_modelo", "ll_fechamento", "ganho_ll",
               "roi_fech_5", "roi_fech_10"]].to_string(index=False, float_format=fmt))

    print("\n--- agregado ---")
    print(f"log loss modelo     : {res.ll_modelo.mean():.4f}")
    print(f"log loss fechamento : {res.ll_fechamento.mean():.4f}")
    print(f"ligas em que o modelo bate o fechamento: {(res.ganho_ll > 0).sum()} de {len(res)}")

    # A comparação que importa: gols vs xG, contra o mesmo mercado.
    if "ganho_ll_xg" in res.columns and res["ganho_ll_xg"].notna().any():
        x = res[res["ganho_ll_xg"].notna()]
        print(f"\n=== Gols vs xG (mesmo mercado, mesma janela) — {len(x)} ligas ===\n")
        print("melhora_xg > 0 = treinar em xG ficou MELHOR que treinar em gols\n")
        print(x[["liga", "n_xg", "ll_modelo", "ll_xg", "ll_fechamento",
                 "ganho_ll", "ganho_ll_xg", "roi_xg_5", "roi_xg_10"]].to_string(
            index=False, float_format=fmt))
        print("\n--- agregado ---")
        print(f"log loss treinando em gols : {x.ll_modelo.mean():.4f}")
        print(f"log loss treinando em xG   : {x.ll_xg.mean():.4f}")
        print(f"log loss do fechamento     : {x.ll_fechamento.mean():.4f}")
        print(f"melhora média do xG        : {x.ll_modelo.mean() - x.ll_xg.mean():+.4f}")
        print(f"ligas em que o xG bate o fechamento: {(x.ganho_ll_xg > 0).sum()} de {len(x)}")
        print(f"ROI médio xG edge>=5%      : {x.roi_xg_5.mean():+.2%}")

    abert = res[res["n_abertura"].notna()] if "n_abertura" in res.columns else pd.DataFrame()
    if len(abert):
        print(f"\n=== Linha de ABERTURA (B365) — {len(abert)} ligas com esse dado ===\n")
        print(abert[["liga", "n_abertura", "ll_modelo", "ll_abertura", "ll_fechamento",
                     "roi_abert_5", "roi_abert_10", "n_apost_abert_5",
                     "mov_favorito"]].to_string(index=False, float_format=fmt))
        print("\n--- agregado ---")
        print(f"log loss modelo na abertura : {abert.ll_modelo.mean():.4f}")
        print(f"log loss abertura           : {abert.ll_abertura.mean():.4f}")
        print(f"ROI médio apostando a abertura, edge>=5%: {abert.roi_abert_5.mean():+.2%}")
        print(f"ROI médio apostando a abertura, edge>=10%: {abert.roi_abert_10.mean():+.2%}")
        print(f"ROI médio apostando o fechamento, edge>=5%: {abert.roi_fech_5.mean():+.2%}")
        print(f"movimento do mercado (prob. do favorito, abertura→fechamento): "
              f"{abert.mov_favorito.mean():+.2%}")

    print("\n--- erro por resultado (modelo - fechamento, negativo = modelo melhor) ---")
    for nome, k in (("casa", 0), ("empate", 1), ("fora", 2)):
        d_ll = []
        for r in detalhes:
            m = r["y"] == k
            if m.sum() < 30:
                continue
            p_mkt = market_probs(r["preds"][["odds_h", "odds_d", "odds_a"]].to_numpy())
            d_ll.append(-np.mean(np.log(np.clip(r["p_mod"][m, k], 1e-15, 1)))
                        - -np.mean(np.log(np.clip(p_mkt[m, k], 1e-15, 1))))
        if d_ll:
            print(f"  {nome:6s}: {np.mean(d_ll):+.4f}  "
                  f"({'modelo melhor' if np.mean(d_ll) < 0 else 'mercado melhor'})")

    relatorio_calibracao(detalhes)

    if args.cache is None:
        with open("data/skill_preds.pkl", "wb") as f:
            pickle.dump(detalhes, f)
        print("\ndetalhes salvos em data/skill_preds.pkl (use --cache para reanalisar)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
