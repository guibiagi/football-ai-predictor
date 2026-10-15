"""Fechamento Pinnacle no dataset histórico (para forward test)."""

from __future__ import annotations

from datetime import datetime

import pandas as pd


def lookup_1x2_close(
    df: pd.DataFrame,
    liga: str,
    casa: str,
    fora: str,
    kickoff: datetime | str,
    tol_days: int = 1,
) -> tuple[float, float, float] | None:
    if isinstance(kickoff, str):
        kickoff = pd.to_datetime(kickoff, utc=True)
    sub = df[
        (df.competition == liga)
        & (df.home_team == casa)
        & (df.away_team == fora)
    ]
    if sub.empty:
        return None
    sub = sub.copy()
    sub["delta"] = (sub["date"] - kickoff).abs()
    row = sub.loc[sub["delta"] <= pd.Timedelta(days=tol_days)].sort_values("delta").head(1)
    if row.empty or row[["psc_h", "psc_d", "psc_a"]].isna().any(axis=None):
        return None
    r = row.iloc[0]
    return float(r.psc_h), float(r.psc_d), float(r.psc_a)
