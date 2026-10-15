#!/usr/bin/env python3
"""backtest_markets.py — o modelo contra VÁRIOS mercados e VÁRIAS casas.

Por que existe: o teste anterior (1X2 contra a linha de fechamento) é o mais
duro que existe e o modelo perde em 32/32 ligas. Mas o mercado de 1X2 é também
o MAIS eficiente de todos — todo o dinheiro afiado está nele. Duas perguntas
nunca foram feitas:

  1. O Poisson produz a matriz de placares inteira, então precifica total de
     gols de graça. O mercado de Over/Under 2.5 é mais fino que o de 1X2.
     O modelo perde dele também?

  2. O dataset tem Pinnacle (a casa mais afiada) E Bet365 (mais mole) lado a
     lado. Se o modelo concorda com a Pinnacle mas a Bet365 está fora, a
     vantagem existe e é executável — não porque o modelo bate o mercado, mas
     porque a Bet365 está mais longe do mercado que a Pinnacle.

Uso:
    python backtest_markets.py
"""

from __future__ import annotations

import sys

import numpy as np
import pandas as pd

sys.path.insert(0, ".")

from backtest_skill import MIN_MATCHES, log_loss  # noqa: E402
from src.poisson_model import PoissonModel, build_score_matrix  # noqa: E402
from src.validation import expanding_window_backtest  # noqa: E402

INITIAL_TRAIN_YEARS = 4
STEP_MONTHS = 12

MODEL_PARAMS = {"decay_lambda": 0.5, "regularization_k": 10.0, "min_games": 10}


def _sem_vig(odds: np.ndarray) -> np.ndarray:
    """Probabilidade implícita com a margem da casa removida (método proporcional).

    Odds <= 1 são impossíveis (probabilidade >= 100%) e aparecem como zero em
    alguns arquivos antigos. Sem filtrar, 1/0 vira inf e a liga inteira sai NaN.
    """
    inv = 1.0 / odds
    return inv / inv.sum(axis=1, keepdims=True)


def _odds_validas(odds: np.ndarray) -> np.ndarray:
    """Máscara de linhas com todas as odds utilizáveis."""
    return (~np.isnan(odds).any(axis=1)) & (odds > 1.0).all(axis=1)


def _log_loss_bin(p: np.ndarray, y: np.ndarray) -> float:
    p = np.clip(p, 1e-12, 1 - 1e-12)
    return float(-np.mean(y * np.log(p) + (1 - y) * np.log(1 - p)))


def _over25(lh: np.ndarray, la: np.ndarray, rho: float) -> np.ndarray:
    """P(total de gols > 2.5) direto da matriz de placares."""
    out = np.empty(len(lh))
    for k, (h, a) in enumerate(zip(lh, la)):
        m = build_score_matrix(float(h), float(a), rho=rho)
        idx = np.arange(m.shape[0])
        out[k] = m[np.add.outer(idx, idx) >= 3].sum()
    return out


def roda(df: pd.DataFrame, comp: str) -> pd.DataFrame | None:
    """Predições fora da amostra desta liga, com os preços de cada casa."""
    sub = df[df["competition"] == comp].dropna(
        subset=["home_goals", "away_goals", "psc_h", "b365c_h"]
    )
    if len(sub) < MIN_MATCHES:
        return None
    preds = expanding_window_backtest(
        sub, lambda: PoissonModel(**MODEL_PARAMS),
        initial_train_years=INITIAL_TRAIN_YEARS, step_months=STEP_MONTHS,
    )
    # home_goals/away_goals já vêm do backtest (resultado real); trazer de novo
    # no merge criaria home_goals_x/_y e quebraria tudo abaixo.
    cols = ["date", "home_team", "away_team",
            "psc_h", "psc_d", "psc_a", "b365c_h", "b365c_d", "b365c_a",
            "ou_close_over", "ou_close_under", "ou_psc_over", "ou_psc_under",
            "ah_line", "ah_close_h", "ah_close_a"]
    preds = preds.merge(sub[[c for c in cols if c in sub.columns]],
                        on=["date", "home_team", "away_team"], how="inner")
    return preds if len(preds) >= 100 else None


def main() -> int:
    df = pd.read_csv("data/matches_xg.csv", parse_dates=["date"])
    ligas = sorted(df[df.psc_h.notna() & df.ou_psc_over.notna()].competition.unique())
    print(f"ligas com Pinnacle e Over/Under: {len(ligas)}\n")

    linhas = []
    cache: dict[str, pd.DataFrame] = {}
    for comp in ligas:
        try:
            p = roda(df, comp)
        except Exception as e:
            print(f"  {comp}: falhou ({type(e).__name__}: {e})")
            continue
        if p is None:
            continue
        cache[comp] = p

        y1 = np.where(p.home_goals > p.away_goals, 0,
                      np.where(p.home_goals == p.away_goals, 1, 2))
        m1 = p[["home_win_probability", "draw_probability", "away_win_probability"]].to_numpy()
        y2 = (p.home_goals + p.away_goals > 2.5).astype(float).to_numpy()

        # rho (correção Dixon-Coles) estimado SÓ na janela inicial de treino.
        # p.date.min() é a primeira data prevista = fim do treino inicial.
        # Estimar na liga inteira vazaria o período de teste para dentro do
        # modelo — o mesmo erro que fez o grid search parecer bom e não ser.
        corte = p.date.min()
        try:
            rho = PoissonModel(**MODEL_PARAMS).fit(
                df[(df.competition == comp) & (df.date < corte)])._rho
        except ValueError:
            rho = 0.0
        p_over = _over25(p.expected_goals_home.to_numpy(),
                         p.expected_goals_away.to_numpy(), rho)

        linha = {"liga": comp, "n": len(p)}
        for nome, col in (("psc", ("psc_h", "psc_d", "psc_a")),
                          ("b365c", ("b365c_h", "b365c_d", "b365c_a"))):
            odds = p[list(col)].to_numpy()
            ok = _odds_validas(odds)
            if ok.sum() < 100:
                continue
            linha[f"1X2 {nome}"] = log_loss(_sem_vig(odds[ok]), y1[ok]) - log_loss(m1[ok], y1[ok])
        for nome, c2 in (("psc", ("ou_psc_over", "ou_psc_under")),
                         ("b365c", ("ou_close_over", "ou_close_under"))):
            if c2[0] not in p.columns:
                continue
            odds = p[list(c2)].to_numpy()
            ok = _odds_validas(odds)
            if ok.sum() < 100:
                continue
            linha[f"O/U {nome}"] = _log_loss_bin(_sem_vig(odds[ok])[:, 0], y2[ok]) - \
                _log_loss_bin(p_over[ok], y2[ok])
        linhas.append(linha)
        print(f"  {comp:24s} n={len(p):5d}  " +
              "  ".join(f"{k.split(' ',1)[1]}:{v:+.4f}" for k, v in linha.items()
                        if isinstance(v, float)))

    res = pd.DataFrame(linhas).set_index("liga").drop(columns=["n"])
    print("\n=== ganho de log loss vs cada casa/mercado ===")
    print("(positivo = MODELO melhor; negativo = casa melhor)\n")
    print(res.to_string(float_format="%.4f"))
    print("\n--- média ---")
    for c in res.columns:
        v = res[c].dropna()
        print(f"  {c:12s} {v.mean():+.4f}   (modelo melhor em {int((v>0).sum())}/{len(v)} ligas)")

    # Soft vs sharp: a Bet365 está mais longe do consenso que a Pinnacle?
    print("\n=== Bet365 vs Pinnacle (mesmo jogo, mesmo instante) ===")
    for comp, p in list(cache.items())[:10]:
        o = p[["b365c_h", "psc_h"]].dropna()
        if len(o) < 100:
            continue
        diff = (o.b365c_h / o.psc_h - 1) * 100
        print(f"  {comp:24s} B365 paga {diff.mean():+.2f}% em média vs Pinnacle "
              f"(desvio {diff.std():.1f}%, n={len(o)})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
