"""Fechamento Pinnacle no dataset histórico (para forward test)."""

from __future__ import annotations

from datetime import datetime

import pandas as pd


def as_utc(ts: datetime | pd.Timestamp | str) -> pd.Timestamp:
    """Normaliza para Timestamp UTC (evita naive vs aware)."""
    if isinstance(ts, str):
        t = pd.to_datetime(ts, utc=True)
    else:
        t = pd.Timestamp(ts)
    if t.tzinfo is None:
        return t.tz_localize("UTC")
    return t.tz_convert("UTC")


def lookup_1x2_close(
    df: pd.DataFrame,
    liga: str,
    casa: str,
    fora: str,
    kickoff: datetime | str,
    tol_days: int = 1,
) -> tuple[float, float, float] | None:
    kickoff = as_utc(kickoff)
    sub = df[
        (df.competition == liga)
        & (df.home_team == casa)
        & (df.away_team == fora)
    ]
    if sub.empty:
        return None
    sub = sub.copy()
    dates_utc = sub["date"].map(as_utc)
    sub["delta"] = (dates_utc - kickoff).abs()
    row = sub.loc[sub["delta"] <= pd.Timedelta(days=tol_days)].sort_values("delta").head(1)
    if row.empty or row[["psc_h", "psc_d", "psc_a"]].isna().any(axis=None):
        return None
    r = row.iloc[0]
    return float(r.psc_h), float(r.psc_d), float(r.psc_a)
