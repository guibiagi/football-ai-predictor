"""Tests for data_loader.py"""

import tempfile
from pathlib import Path

import pytest

from src.data_loader import (
    load_matches,
    get_team_names,
    get_team_stats,
    temporal_train_test_split,
    REQUIRED_COLUMNS,
)


def test_load_matches():
    """Load the actual dataset and verify structure."""
    df = load_matches("data/matches.csv")
    assert len(df) > 1000  # Should have plenty of matches
    assert list(df.columns) == REQUIRED_COLUMNS
    assert df["home_goals"].min() >= 0
    assert df["away_goals"].min() >= 0
    assert df["neutral"].dtype == bool
    # Dates should be sorted
    assert df["date"].is_monotonic_increasing


def test_get_team_names():
    df = load_matches("data/matches.csv")
    teams = get_team_names(df)
    assert "Brazil" in teams
    assert "France" in teams
    assert len(teams) > 50  # Many international teams


def test_get_team_stats():
    df = load_matches("data/matches.csv")
    stats = get_team_stats(df)
    assert len(stats) > 50
    assert "avg_goals_for" in stats.columns
    assert "avg_goals_against" in stats.columns
    # Average should be reasonable (between 0 and ~4)
    assert 0 < stats["avg_goals_for"].mean() < 4


def test_temporal_train_test_split():
    df = load_matches("data/matches.csv")
    train, test = temporal_train_test_split(df, test_ratio=0.2)
    assert len(train) + len(test) == len(df)
    assert len(test) > 0
    # Training data should be strictly before test data
    assert train["date"].max() <= test["date"].min()


def test_load_missing_file():
    with pytest.raises(FileNotFoundError):
        load_matches("data/nonexistent.csv")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])