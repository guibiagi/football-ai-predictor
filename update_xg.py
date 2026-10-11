#!/usr/bin/env python3
"""update_xg.py — baixa xG por jogo do understat e junta com o dataset principal.

Por que isto existe: gols são uma amostra ruidosa da força de um time. Um 1x0
com 3 gols esperados não é um time melhor que o 1x0 com 0,2 gols esperados. O
xG mede a qualidade da chance; é a informação que o mercado usa e que o modelo
de gols puro não tinha.

O endpoint (descoberto em js/league.min.js, não documentado):
    GET https://understat.com/getLeagueData/{liga}/{temporada}
devolve {teams, players, dates} — e 'dates' traz xG de cada partida.

Só cobre 6 ligas (as grandes + Rússia). O resto do dataset fica sem xG, e o
modelo cai para gols nessas ligas em vez de inventar um xG que não existe.

Uso:
    python update_xg.py                # baixa e escreve data/matches_xg.csv
    python update_xg.py --so-juntar    # rejunta sem baixar de novo
"""

from __future__ import annotations

import argparse
import gzip
import json
import sys
import time
import urllib.request

import pandas as pd

sys.path.insert(0, ".")
from monitor import _clean, build_map, lookup  # noqa: E402

UA = {
    "User-Agent": ("Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 "
                   "(KHTML, like Gecko) Chrome/120 Safari/537.36"),
    "X-Requested-With": "XMLHttpRequest",
    "Referer": "https://understat.com/",
}

# liga no understat -> competição no nosso dataset
LIGAS = {
    "EPL": "Premier League",
    "La_liga": "La Liga",
    "Bundesliga": "Bundesliga",
    "Serie_A": "Serie A",
    "Ligue_1": "Ligue 1",
    "RFPL": "Russian Premier League",
}

PRIMEIRA_TEMPORADA = 2014  # o understat começa em 2014/15
ULTIMA_TEMPORADA = 2026    # 2026/27


def busca(liga: str, temporada: int, tentativas: int = 3) -> dict | None:
    url = f"https://understat.com/getLeagueData/{liga}/{temporada}"
    for t in range(tentativas):
        try:
            r = urllib.request.urlopen(urllib.request.Request(url, headers=UA), timeout=45)
            b = r.read()
            if b[:2] == b"\x1f\x8b":
                b = gzip.decompress(b)
            return json.loads(b.decode("utf-8", "ignore"))
        except Exception as e:
            if t == tentativas - 1:
                print(f"    {liga} {temporada}: falhou ({type(e).__name__}: {e})")
                return None
            time.sleep(2 * (t + 1))
    return None


def baixa_tudo() -> pd.DataFrame:
    linhas = []
    for liga, comp in LIGAS.items():
        for temp in range(PRIMEIRA_TEMPORADA, ULTIMA_TEMPORADA + 1):
            d = busca(liga, temp)
            if not d or not d.get("dates"):
                continue
            n = 0
            for j in d["dates"]:
                # isResult=False = jogo ainda não aconteceu: não tem xG nem placar
                if not j.get("isResult"):
                    continue
                try:
                    linhas.append({
                        "date": pd.to_datetime(j["datetime"]),
                        "competition": comp,
                        "understat_h": j["h"]["title"],
                        "understat_a": j["a"]["title"],
                        "xg_h": float(j["xG"]["h"]),
                        "xg_a": float(j["xG"]["a"]),
                        "ug_home_goals": int(j["goals"]["h"]),
                        "ug_away_goals": int(j["goals"]["a"]),
                    })
                    n += 1
                except (KeyError, TypeError, ValueError):
                    continue
            print(f"  {comp} {temp}: {n} jogos com xG")
            time.sleep(0.4)  # não martelar o servidor
    return pd.DataFrame(linhas)


def junta(xg: pd.DataFrame, base: pd.DataFrame) -> pd.DataFrame:
    """Casa o xG com o dataset principal por (data, times), dentro da liga.

    Casa por dia (o horário diverge entre as fontes) e por nome de time
    resolvido com o mesmo lookup do monitor, que já lida com prefixo, apelido e
    variante de grafia. Se não casar, fica NaN — o modelo cai para gols naquele
    jogo, em vez de usar xG de outro time.
    """
    base = base.copy()
    base["dia"] = base["date"].dt.date
    xg = xg.copy()
    xg["dia"] = xg["date"].dt.date

    saida = []
    for comp, sub in base.groupby("competition"):
        xsub = xg[xg["competition"] == comp]
        if xsub.empty:
            sub = sub.copy()
            sub["xg_h"] = float("nan")
            sub["xg_a"] = float("nan")
            saida.append(sub)
            continue
        idx = build_map(sorted(set(sub.home_team) | set(sub.away_team)))
        # índice de xG por (dia, casa, fora) já resolvido para o nome do dataset
        por_dia: dict[tuple, tuple] = {}
        for r in xsub.itertuples():
            h = lookup(idx, r.understat_h)
            a = lookup(idx, r.understat_a)
            if h and a:
                por_dia[(r.dia, h, a)] = (r.xg_h, r.xg_a)
        sub = sub.copy()
        pares = list(zip(sub["dia"], sub["home_team"], sub["away_team"]))
        sub["xg_h"] = [por_dia.get(p, (float("nan"),))[0] for p in pares]
        sub["xg_a"] = [por_dia.get(p, (float("nan"), float("nan")))[1] for p in pares]
        casados = sub["xg_h"].notna().sum()
        print(f"  {comp}: {casados}/{len(sub)} jogos com xG casado "
              f"({casados/len(sub):.0%})")
        saida.append(sub)

    out = pd.concat(saida, ignore_index=True).drop(columns=["dia"])
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--so-juntar", action="store_true")
    args = ap.parse_args()

    if args.so_juntar and __import__("pathlib").Path("data/understat_xg.csv").exists():
        xg = pd.read_csv("data/understat_xg.csv", parse_dates=["date"])
        print(f"xG carregado do cache: {len(xg)} jogos")
    else:
        print("baixando do understat...")
        xg = baixa_tudo()
        if xg.empty:
            print("nada baixado — abortando sem sobrescrever o dataset")
            return 1
        xg.to_csv("data/understat_xg.csv", index=False)
        print(f"\nsalvo data/understat_xg.csv: {len(xg)} jogos")

    base = pd.read_csv("data/matches_football.csv", parse_dates=["date"])
    print(f"\ndataset principal: {len(base)} jogos")
    junto = junta(xg, base)
    junto.to_csv("data/matches_xg.csv", index=False)
    print(f"\nescrito data/matches_xg.csv: {len(junto)} jogos, "
          f"{junto.xg_h.notna().sum()} com xG "
          f"({junto.xg_h.notna().sum()/len(junto):.1%})")

    # Verificação que falha se o join escorregar: xG e gols devem ter médias
    # próximas na MESMA liga (xG é calibrado nos gols). Diferença grande = nomes
    # casados errado, não futebol.
    print("\n--- xG médio vs gols médios, por liga com xG ---")
    g = junto.dropna(subset=["xg_h"]).groupby("competition").agg(
        jogos=("xg_h", "size"), xg=("xg_h", "mean"), gols=("home_goals", "mean"))
    g["diff"] = g["xg"] - g["gols"]
    print(g.to_string(float_format="%.3f"))
    assert len(g) > 0, "nenhum xG casado — o join quebrou"
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
