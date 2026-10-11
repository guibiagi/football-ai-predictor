#!/usr/bin/env python3
"""Refresh data/matches_brasileirao.csv from football-data.co.uk (Série A 2012→today, with closing odds)."""
import io
import urllib.request

import pandas as pd

URL = "https://www.football-data.co.uk/new/BRA.csv"


def fetch() -> pd.DataFrame:
    req = urllib.request.Request(URL, headers={"User-Agent": "Mozilla/5.0"})
    raw = pd.read_csv(io.BytesIO(urllib.request.urlopen(req, timeout=60).read()), encoding="utf-8-sig")
    raw = raw.dropna(subset=["HG", "AG"])  # future fixtures have no score
    return pd.DataFrame({
        "date": pd.to_datetime(raw["Date"], dayfirst=True),
        "competition": "Brasileirao Serie A",
        "season": raw["Season"].astype(int),
        "stage": "league",
        "home_team": raw["Home"],
        "away_team": raw["Away"],
        "home_goals": raw["HG"].astype(int),
        "away_goals": raw["AG"].astype(int),
        "neutral": False,
        # closing avg odds, kept for backtesting vs market
        "odds_h": raw["AvgCH"], "odds_d": raw["AvgCD"], "odds_a": raw["AvgCA"],
    }).sort_values("date")


if __name__ == "__main__":
    df = fetch()
    assert len(df) > 5000 and df["home_goals"].ge(0).all(), "suspicious download"
    df.to_csv("data/matches_brasileirao.csv", index=False)
    print(f"{len(df)} matches {df.date.min().date()} → {df.date.max().date()}")
