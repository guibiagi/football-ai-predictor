#!/usr/bin/env python3
"""Atualiza cache de fixtures/odds Pinnacle (football-data.co.uk)."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.football_data_fixtures import load_upcoming_pinnacle_cached


def main() -> None:
    df = load_upcoming_pinnacle_cached(force_refresh=True)
    print(f"Fixtures futuros com odds: {len(df)}")
    if len(df):
        print(df[["date", "competition", "home_team", "away_team", "ps_h"]].head(10).to_string())


if __name__ == "__main__":
    main()
