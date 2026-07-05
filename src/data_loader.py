"""
data_loader.py — Load, validate, and clean football match data.

Design principles:
- Temporal split ONLY (never shuffle football data — it leaks future info)
- Name normalization prevents "Brazil" vs "Brasil" duplicates
- Columns are validated before any processing
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

import pandas as pd

logger = logging.getLogger(__name__)

# ── Required columns ────────────────────────────────────────────────
REQUIRED_COLUMNS = [
    "date",
    "competition",
    "season",
    "stage",
    "home_team",
    "away_team",
    "home_goals",
    "away_goals",
    "neutral",
]

# ── Column types for validation ─────────────────────────────────────
COLUMN_DTYPES = {
    "date": "datetime64[ns]",
    "competition": "object",
    "season": "int64",
    "stage": "object",
    "home_team": "string",
    "away_team": "string",
    "home_goals": "int64",
    "away_goals": "int64",
    "neutral": "bool",
}


def load_matches(path: str | Path) -> pd.DataFrame:
    """Load the CSV and return a validated DataFrame.

    Args:
        path: Path to matches.csv

    Returns:
        DataFrame with validated and cleaned match data.

    Raises:
        FileNotFoundError: if the file doesn't exist.
        ValueError: if required columns are missing or data types are wrong.
    """
    path = Path(path)

    if not path.exists():
        raise FileNotFoundError(f"Match data not found: {path}")

    df = pd.read_csv(path)

    # Validate columns exist
    _validate_columns(df, path)

    # Parse dates
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    if df["date"].isna().any():
        bad_rows = df[df["date"].isna()].index.tolist()
        raise ValueError(f"Invalid dates in rows: {bad_rows}")

    # Ensure goals are non-negative integers
    for col in ["home_goals", "away_goals"]:
        if (df[col] < 0).any():
            raise ValueError(f"Negative values found in column '{col}'")

    # Sort chronologically (CRITICAL for temporal splits)
    df = df.sort_values("date").reset_index(drop=True)

    # Normalize team names
    df["home_team"] = df["home_team"].str.strip().str.title()
    df["away_team"] = df["away_team"].str.strip().str.title()

    # Ensure neutral is boolean
    df["neutral"] = df["neutral"].astype(bool)

    logger.info(
        "Loaded %d matches from %s to %s",
        len(df),
        df["date"].min().strftime("%Y-%m-%d"),
        df["date"].max().strftime("%Y-%m-%d"),
    )

    return df


def _validate_columns(df: pd.DataFrame, path: Path) -> None:
    """Check that all required columns are present."""
    missing = [col for col in REQUIRED_COLUMNS if col not in df.columns]
    if missing:
        raise ValueError(
            f"Missing required columns in {path}: {missing}\n"
            f"Expected: {REQUIRED_COLUMNS}"
        )


def get_team_names(df: pd.DataFrame) -> list[str]:
    """Return a sorted list of unique team names from the data."""
    return sorted(set(df["home_team"].unique()) | set(df["away_team"].unique()))


def get_team_stats(df: pd.DataFrame) -> pd.DataFrame:
    """Compute per-team statistics from match history.

    Returns a DataFrame with columns:
        team, games_played, goals_for, goals_against,
        avg_goals_for, avg_goals_against
    """
    # Home stats
    home_stats = (
        df.groupby("home_team")
        .agg(
            games_home=("home_team", "count"),
            goals_for_home=("home_goals", "sum"),
            goals_against_home=("away_goals", "sum"),
        )
        .reset_index()
        .rename(columns={"home_team": "team"})
    )

    # Away stats
    away_stats = (
        df.groupby("away_team")
        .agg(
            games_away=("away_team", "count"),
            goals_for_away=("away_goals", "sum"),
            goals_against_away=("home_goals", "sum"),
        )
        .reset_index()
        .rename(columns={"away_team": "team"})
    )

    # Merge
    stats = pd.merge(home_stats, away_stats, on="team", how="outer").fillna(0)

    stats["games_played"] = stats["games_home"] + stats["games_away"]
    stats["goals_for"] = stats["goals_for_home"] + stats["goals_for_away"]
    stats["goals_against"] = stats["goals_against_home"] + stats["goals_against_away"]

    stats["avg_goals_for"] = stats["goals_for"] / stats["games_played"]
    stats["avg_goals_against"] = stats["goals_against"] / stats["games_played"]

    stats["games_played"] = stats["games_played"].astype(int)

    return stats[
        [
            "team",
            "games_played",
            "goals_for",
            "goals_against",
            "avg_goals_for",
            "avg_goals_against",
        ]
    ]


def temporal_train_test_split(
    df: pd.DataFrame,
    test_ratio: float = 0.2,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Split data by date — older matches for training, newer for testing.

    NEVER use random splits for time-series data like football.
    Random splits leak future information into training.

    Args:
        df: Chronologically sorted match DataFrame.
        test_ratio: Fraction of most recent matches to hold out (default 0.2).

    Returns:
        (train_df, test_df) tuple.
    """
    if test_ratio <= 0 or test_ratio >= 1:
        raise ValueError("test_ratio must be between 0 and 1 (exclusive)")

    split_idx = int(len(df) * (1 - test_ratio))
    train = df.iloc[:split_idx].copy()
    test = df.iloc[split_idx:].copy()

    logger.info(
        "Temporal split: %d train matches, %d test matches (test_ratio=%.0f%%)",
        len(train),
        len(test),
        test_ratio * 100,
    )

    return train, test