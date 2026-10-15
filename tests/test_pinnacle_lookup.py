"""Testes pinnacle_lookup (timezone)."""

from __future__ import annotations

import pandas as pd

from src.pinnacle_lookup import as_utc, lookup_1x2_close


def test_as_utc_from_z_string():
    t = as_utc("2026-10-15T19:00:00Z")
    assert t.tzinfo is not None
    assert str(t.tz) in ("UTC", "UTC+00:00")


def test_lookup_1x2_close_naive_hist_aware_kickoff():
    df = pd.DataFrame({
        "date": [pd.Timestamp("2026-10-15")],
        "competition": ["Premier League"],
        "home_team": ["Arsenal"],
        "away_team": ["Chelsea"],
        "psc_h": [2.1],
        "psc_d": [3.4],
        "psc_a": [3.5],
    })
    out = lookup_1x2_close(
        df, "Premier League", "Arsenal", "Chelsea",
        "2026-10-15T20:00:00Z",
    )
    assert out == (2.1, 3.4, 3.5)
