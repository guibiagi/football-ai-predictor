"""
features.py — Feature engineering for football prediction.

For Brasileirão (club football), form and context matter much more
than for international tournaments. Teams change every transfer window,
travel matters, and the calendar is brutal (Wed-Sun rhythm).

Key features:
  - Rolling form: goals scored/conceded in last N matches
  - Rest days: how many days since the team's last match
  - Home/away streak: consecutive home/away games
"""

from __future__ import annotations

import logging
from typing import Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


def add_team_form(
    df: pd.DataFrame,
    window: int = 5,
    min_matches: int = 3,
) -> pd.DataFrame:
    """Add rolling form features for both teams.

    For each match, computes the team's average goals scored and conceded
    in their previous `window` matches (any venue).

    These features help the model answer:
      "Is this team in good form RIGHT NOW?"

    Args:
        df: Match DataFrame sorted by date.
        window: Number of recent matches to average.
        min_matches: Minimum matches needed for a reliable form estimate.

    Returns:
        DataFrame with added columns:
          home_form_gf, home_form_ga, away_form_gf, away_form_ga
    """
    df = df.sort_values("date").reset_index(drop=True)

    home_form_gf = []
    home_form_ga = []
    away_form_gf = []
    away_form_ga = []

    # Build a lookup: for each team, list of (date, goals_for, goals_against)
    team_history: dict[str, list] = {}

    for idx, row in df.iterrows():
        home = row["home_team"]
        away = row["away_team"]
        match_date = row["date"]

        # Compute form for home team
        history = team_history.get(home, [])
        recent = _get_recent(history, window)
        home_form_gf.append(_safe_mean(recent, "gf", min_matches))
        home_form_ga.append(_safe_mean(recent, "ga", min_matches))

        # Compute form for away team
        history = team_history.get(away, [])
        recent = _get_recent(history, window)
        away_form_gf.append(_safe_mean(recent, "gf", min_matches))
        away_form_ga.append(_safe_mean(recent, "ga", min_matches))

        # Update history AFTER computing form (no look-ahead)
        team_history.setdefault(home, []).append({
            "date": match_date,
            "gf": row["home_goals"],
            "ga": row["away_goals"],
        })
        team_history.setdefault(away, []).append({
            "date": match_date,
            "gf": row["away_goals"],
            "ga": row["home_goals"],
        })

    df["home_form_gf"] = home_form_gf
    df["home_form_ga"] = home_form_ga
    df["away_form_gf"] = away_form_gf
    df["away_form_ga"] = away_form_ga

    return df


def _get_recent(history: list[dict], window: int) -> list[dict]:
    """Get the last `window` entries from a team's history."""
    return history[-window:] if len(history) >= window else history


def _safe_mean(history: list[dict], key: str, min_matches: int) -> float:
    """Mean of a key from history, or NaN if too few matches."""
    if len(history) < min_matches:
        return float("nan")
    return float(np.mean([h[key] for h in history]))


def add_rest_days(df: pd.DataFrame) -> pd.DataFrame:
    """Add days of rest since each team's previous match.

    Less rest = more fatigue = fewer goals expected.

    Args:
        df: Match DataFrame sorted by date.

    Returns:
        DataFrame with columns: home_rest_days, away_rest_days
    """
    df = df.sort_values("date").reset_index(drop=True)

    # Track last match date for each team
    last_match: dict[str, pd.Timestamp] = {}

    home_rest = []
    away_rest = []

    for _, row in df.iterrows():
        match_date = row["date"]
        home = row["home_team"]
        away = row["away_team"]

        # Rest for home team
        if home in last_match:
            rest = (match_date - last_match[home]).days
            home_rest.append(max(rest, 1))  # Min 1 day
        else:
            home_rest.append(7)  # Default: ~1 week (season start)

        # Rest for away team
        if away in last_match:
            rest = (match_date - last_match[away]).days
            away_rest.append(max(rest, 1))
        else:
            away_rest.append(7)

        # Update last match dates
        last_match[home] = match_date
        last_match[away] = match_date

    df["home_rest_days"] = home_rest
    df["away_rest_days"] = away_rest

    return df


def add_form_adjustment(
    df: pd.DataFrame,
    window: int = 5,
) -> pd.DataFrame:
    """Full feature pipeline: form + rest days."""
    df = add_team_form(df, window=window)
    df = add_rest_days(df)
    return df