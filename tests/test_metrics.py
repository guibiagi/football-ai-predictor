"""Tests for metrics.py"""

import pytest
import numpy as np
import pandas as pd

from src.metrics import (
    log_loss_score,
    brier_score,
    accuracy,
    evaluate_predictions,
    calibration_summary,
)


def test_log_loss_perfect_prediction():
    """Perfect prediction should give log loss ≈ 0."""
    # Home win predicted at 99.99%
    ll = log_loss_score(2, 0, 0.9999, 0.00005, 0.00005)
    assert ll < 0.01


def test_log_loss_wrong_and_confident():
    """Wrong + overconfident = heavily penalized."""
    # Home win predicted at 99%, but away won
    ll_wrong = log_loss_score(0, 2, 0.99, 0.005, 0.005)
    # Home win predicted at 51%, but away won
    ll_uncertain = log_loss_score(0, 2, 0.51, 0.25, 0.24)
    assert ll_wrong > ll_uncertain, "Overconfident wrong prediction should be worse"


def test_log_loss_draw():
    """Draw prediction."""
    ll = log_loss_score(1, 1, 0.30, 0.40, 0.30)
    assert ll > 0


def test_brier_score_perfect():
    """Perfect prediction = Brier 0."""
    bs = brier_score(3, 0, 1.0, 0.0, 0.0)
    assert bs < 0.001


def test_brier_score_worst():
    """Completely wrong + overconfident."""
    bs = brier_score(0, 3, 1.0, 0.0, 0.0)
    assert bs > 1.0


def test_brier_score_uniform():
    """Uniform prediction (1/3 each) for any outcome."""
    bs = brier_score(2, 1, 1 / 3, 1 / 3, 1 / 3)
    expected = (2 / 3) ** 2 + (1 / 3) ** 2 + (1 / 3) ** 2
    assert abs(bs - expected) < 0.001


def test_accuracy_home_win():
    assert accuracy(2, 0, 0.50, 0.30, 0.20) is True
    assert accuracy(2, 0, 0.30, 0.35, 0.35) is False  # Model said draw


def test_accuracy_draw():
    assert accuracy(1, 1, 0.30, 0.40, 0.30) is True
    assert accuracy(1, 1, 0.45, 0.25, 0.30) is False  # Model said home


def test_evaluate_predictions():
    df = pd.DataFrame({
        "home_goals": [2, 1, 0],
        "away_goals": [0, 1, 2],
        "home_win_probability": [0.60, 0.25, 0.15],
        "draw_probability": [0.25, 0.50, 0.25],
        "away_win_probability": [0.15, 0.25, 0.60],
    })
    result = evaluate_predictions(df)
    assert result["n_matches"] == 3
    assert 0 < result["log_loss"] < 5
    assert 0 < result["brier_score"] < 2
    assert "naive_log_loss" in result
    assert "verdict" in result


def test_calibration_summary():
    df = pd.DataFrame({
        "home_goals": [2, 1, 0, 3, 1],
        "away_goals": [0, 0, 2, 1, 0],
        "home_win_probability": [0.80, 0.60, 0.20, 0.70, 0.55],
        "draw_probability": [0.10, 0.25, 0.30, 0.15, 0.20],
        "away_win_probability": [0.10, 0.15, 0.50, 0.15, 0.25],
    })
    cal = calibration_summary(df, n_bins=5)
    assert len(cal) > 0
    assert "predicted_pct" in cal.columns
    assert "actual_pct" in cal.columns
    assert "error" in cal.columns