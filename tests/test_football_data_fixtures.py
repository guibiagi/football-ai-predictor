"""Fixtures Pinnacle via football-data fixtures.csv."""

from __future__ import annotations

import pandas as pd

from src.football_data_fixtures import _odds_triplet, fetch_fixtures_csv_pinnacle


def test_odds_triplet_prefers_pph_when_no_psh():
    row = pd.Series({"PSH": float("nan"), "PPH": 2.1, "PPD": 3.4, "PPA": 3.5})
    assert _odds_triplet(row) == (2.1, 3.4, 3.5)


def test_fetch_fixtures_csv_pinnacle_returns_rows(monkeypatch):
    csv = (
        "Div,Date,Time,HomeTeam,AwayTeam,PPH,PPD,PPA\n"
        "E0,17/10/2026,15:00,Arsenal,Chelsea,2.50,3.30,2.90\n"
        "ZZZ,17/10/2026,15:00,A,B,2.0,3.0,4.0\n"
    )

    def fake_fetch(_url: str):
        import io

        return pd.read_csv(io.StringIO(csv))

    monkeypatch.setattr("src.football_data_fixtures._fetch_csv", lambda url: fake_fetch(url))
    df = fetch_fixtures_csv_pinnacle(["Premier League"])
    assert len(df) == 1
    assert df.iloc[0]["home_team"] == "Arsenal"
    assert df.iloc[0]["ps_h"] == 2.5
