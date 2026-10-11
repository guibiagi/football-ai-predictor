#!/usr/bin/env python3
"""update_data.py — baixa o histórico de várias ligas para data/matches_football.csv.

Duas fontes do football-data.co.uk, com esquemas de coluna diferentes:
  * Ligas grandes  → CSV por temporada: /mmz4281/<temporada>/<cod>.csv
                     colunas Div,Date,Time,HomeTeam,AwayTeam,FTHG,FTAG,AvgCH...
  * Ligas extras   → CSV único: /new/<cod>.csv
                     colunas Country,League,Season,Date,Time,Home,Away,HG,AG,AvgCH...

A saída tem o mesmo esquema do resto do projeto, mais uma coluna `competition`,
e guarda as odds de fechamento (AvgC*) para poder comparar com o mercado.

Uso:
    python update_data.py            # baixa tudo e escreve o CSV
    python update_data.py --list     # só mostra o que seria baixado
"""
from __future__ import annotations

import argparse
import io
import sys
import urllib.request
from pathlib import Path

import pandas as pd

BASE = "https://www.football-data.co.uk"
OUT = Path(__file__).resolve().parent / "data" / "matches_football.csv"
UA = {"User-Agent": "Mozilla/5.0"}

# Ligas por temporada (código do football-data → nome no dataset)
SEASONAL = {
    "E0": "Premier League", "E1": "Championship",
    "SP1": "La Liga", "SP2": "LaLiga2",
    "I1": "Serie A", "I2": "Serie B",
    "D1": "Bundesliga", "D2": "2. Bundesliga",
    "F1": "Ligue 1", "F2": "Ligue 2",
    "N1": "Eredivisie", "P1": "Primeira Liga",
    "T1": "Super Lig", "G1": "Super League Greece",
    "B1": "Belgium Pro League", "SC0": "Scottish Premiership",
}
# Ligas em CSV único (/new/), com a temporada dentro do arquivo
EXTRA = {
    "BRA": "Brasileirao Serie A", "ARG": "Liga Profesional de Futbol",
    "MEX": "Liga MX", "USA": "MLS", "JPN": "J1 League",
    "SWZ": "Swiss Super League", "AUT": "Austria Bundesliga",
    "DNK": "Danish Superliga", "NOR": "Eliteserien",
    "POL": "Ekstraklasa", "ROU": "Superliga Romania",
    "RUS": "Russian Premier League", "SWE": "Allsvenskan",
    "IRL": "League of Ireland Premier Division", "FIN": "Veikkausliiga",
    "CHN": "Chinese Super League",
}
FIRST_SEASON = 2012  # 2012/13 em diante


def _fetch(url: str) -> pd.DataFrame | None:
    try:
        raw = urllib.request.urlopen(urllib.request.Request(url, headers=UA), timeout=45).read()
    except Exception:
        return None
    if not raw or len(raw) < 200:
        return None
    # Nem todo CSV da fonte é UTF-8 (temporadas antigas vêm em latin-1); sem o
    # fallback o download inteiro morre num byte 0xa0.
    for enc in ("utf-8-sig", "latin-1"):
        try:
            return pd.read_csv(io.BytesIO(raw), encoding=enc, on_bad_lines="skip")
        except UnicodeDecodeError:
            continue
    return None


def _season_code(year: int) -> str:
    return f"{year % 100:02d}{(year + 1) % 100:02d}"


def _from_seasonal(df: pd.DataFrame, comp: str, season: str) -> pd.DataFrame | None:
    need = {"Date", "HomeTeam", "AwayTeam", "FTHG", "FTAG"}
    if not need.issubset(df.columns):
        return None
    out = pd.DataFrame({
        "date": pd.to_datetime(df["Date"], dayfirst=True, errors="coerce"),
        "competition": comp,
        "season": season,
        "stage": "league",
        "home_team": df["HomeTeam"],
        "away_team": df["AwayTeam"],
        "home_goals": pd.to_numeric(df["FTHG"], errors="coerce"),
        "away_goals": pd.to_numeric(df["FTAG"], errors="coerce"),
        "neutral": False,
        "odds_h": pd.to_numeric(df.get("AvgCH"), errors="coerce"),
        "odds_d": pd.to_numeric(df.get("AvgCD"), errors="coerce"),
        "odds_a": pd.to_numeric(df.get("AvgCA"), errors="coerce"),
        # Bet365 (sem sufixo C) é a linha mais ANTIGA do arquivo; AvgC* é o
        # fechamento. A diferença entre as duas é o movimento do mercado — e é
        # onde pode estar a vantagem de quem aposta antes do dinheiro afiado.
        "odds_open_h": pd.to_numeric(df.get("B365H"), errors="coerce"),
        "odds_open_d": pd.to_numeric(df.get("B365D"), errors="coerce"),
        "odds_open_a": pd.to_numeric(df.get("B365A"), errors="coerce"),
    })
    return out.dropna(subset=["date", "home_goals", "away_goals"])


def _from_extra(df: pd.DataFrame, comp: str) -> pd.DataFrame | None:
    need = {"Date", "Home", "Away", "HG", "AG"}
    if not need.issubset(df.columns):
        return None
    season = df["Season"].astype(str) if "Season" in df.columns else pd.Series([""] * len(df))
    out = pd.DataFrame({
        "date": pd.to_datetime(df["Date"], dayfirst=True, errors="coerce"),
        "competition": comp,
        "season": season,
        "stage": "league",
        "home_team": df["Home"],
        "away_team": df["Away"],
        "home_goals": pd.to_numeric(df["HG"], errors="coerce"),
        "away_goals": pd.to_numeric(df["AG"], errors="coerce"),
        "neutral": False,
        "odds_h": pd.to_numeric(df.get("AvgCH"), errors="coerce"),
        "odds_d": pd.to_numeric(df.get("AvgCD"), errors="coerce"),
        "odds_a": pd.to_numeric(df.get("AvgCA"), errors="coerce"),
        # Os arquivos extras só trazem odds de fechamento (PSCH/AvgCH) — não há
        # linha de abertura neles. Fica NaN e essas ligas saem do teste de
        # abertura, em vez de eu inventar uma abertura que não existe.
        "odds_open_h": float("nan"),
        "odds_open_d": float("nan"),
        "odds_open_a": float("nan"),
    })
    return out.dropna(subset=["date", "home_goals", "away_goals"])


def collect(verbose: bool = True) -> pd.DataFrame:
    parts = []
    for code, comp in SEASONAL.items():
        for year in range(FIRST_SEASON, 2027):
            season = _season_code(year)
            df = _fetch(f"{BASE}/mmz4281/{season}/{code}.csv")
            if df is None:
                continue
            got = _from_seasonal(df, comp, season)
            if got is not None and len(got):
                parts.append(got)
                if verbose:
                    print(f"   {comp:24s} {season}: {len(got)} jogos")
    for code, comp in EXTRA.items():
        df = _fetch(f"{BASE}/new/{code}.csv")
        if df is None:
            continue
        got = _from_extra(df, comp)
        if got is not None and len(got):
            parts.append(got)
            if verbose:
                print(f"   {comp:24s} (novo): {len(got)} jogos")
    if not parts:
        raise RuntimeError("nenhuma liga baixada — a fonte mudou ou está fora do ar")
    df = pd.concat(parts, ignore_index=True)
    df["home_goals"] = df["home_goals"].astype(int)
    df["away_goals"] = df["away_goals"].astype(int)
    return df.sort_values("date").reset_index(drop=True)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--list", action="store_true", help="só listar as ligas")
    a = ap.parse_args()
    if a.list:
        print("por temporada:", ", ".join(f"{k}={v}" for k, v in SEASONAL.items()))
        print("extras:", ", ".join(f"{k}={v}" for k, v in EXTRA.items()))
        return 0

    df = collect()
    # checagem: o download só vale se cobriu ligas de verdade
    assert df["competition"].nunique() >= 15, "poucas ligas baixadas"
    assert df["home_goals"].ge(0).all() and df["away_goals"].ge(0).all(), "gol negativo"
    assert df["date"].max() >= pd.Timestamp("2026-01-01"), "dados velhos demais"
    df.to_csv(OUT, index=False)
    print(f"\n{len(df)} jogos, {df['competition'].nunique()} ligas, "
          f"{df.date.min().date()} → {df.date.max().date()}")
    print(f"escrito em {OUT}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
