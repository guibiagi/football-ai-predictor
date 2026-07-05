"""
validation.py — Temporal backtesting for football models.

Football is a time series. You CANNOT shuffle matches randomly —
that leaks future information into training. This module provides
proper temporal cross-validation.

Example: with 25,000 matches over 26 years (2000-2026), we might:
  - Train on 2000-2014
  - Validate on 2015-2018
  - Test on 2019-2026

Or use expanding windows for more granular evaluation.
"""

from __future__ import annotations

import logging
from typing import Any, Callable

import numpy as np
import pandas as pd
from datetime import timedelta

from src.poisson_model import PoissonModel

logger = logging.getLogger(__name__)


def expanding_window_backtest(
    df: pd.DataFrame,
    model_factory: Callable[[], PoissonModel],
    initial_train_years: float = 4,
    step_months: int = 12,
) -> pd.DataFrame:
    """Backtest using expanding windows — realistic out-of-sample evaluation.

    For each window:
      1. Train on all data before the test period
      2. Predict the test period matches
      3. Record predictions with actual outcomes
      4. Expand the training window and repeat

    This simulates what would have happened if you used this model
    historically — you only know the past, never the future.

    Args:
        df: Chronologically sorted match DataFrame with date column.
        model_factory: Function that returns a fresh model instance.
                       This allows configuring hyperparameters per window.
        initial_train_years: Minimum years of training data before first test.
        step_months: How often to re-evaluate (e.g., 12 = annual).

    Returns:
        DataFrame with all out-of-sample predictions + actual results.
    """
    df = df.sort_values("date").reset_index(drop=True)

    # Determine test windows
    min_date = df["date"].min()
    first_test_date = min_date + timedelta(days=int(initial_train_years * 365.25))
    last_date = df["date"].max()

    # Generate test window boundaries
    test_starts = pd.date_range(
        first_test_date, last_date, freq=f"{step_months}MS"
    )

    all_predictions = []
    all_metrics = []

    for i, test_start in enumerate(test_starts):
        # Train on everything before test_start
        train = df[df["date"] < test_start]
        # Test on the step_months window starting at test_start
        test_end = test_start + timedelta(days=step_months * 30)
        test = df[(df["date"] >= test_start) & (df["date"] < test_end)]

        if len(train) < 100 or len(test) < 5:
            logger.debug("Skipping window %s: too few matches (train=%d, test=%d)",
                         test_start.date(), len(train), len(test))
            continue

        model = model_factory()
        model.fit(train)

        for _, row in test.iterrows():
            pred = model.predict_match(
                row["home_team"], row["away_team"], neutral=bool(row["neutral"])
            )
            all_predictions.append({
                "date": row["date"],
                "train_until": test_start,
                "home_team": row["home_team"],
                "away_team": row["away_team"],
                "home_goals": int(row["home_goals"]),
                "away_goals": int(row["away_goals"]),
                "neutral": bool(row["neutral"]),
                "expected_goals_home": pred["expected_goals_home"],
                "expected_goals_away": pred["expected_goals_away"],
                "home_win_probability": pred["home_win_probability"],
                "draw_probability": pred["draw_probability"],
                "away_win_probability": pred["away_win_probability"],
            })

        logger.info(
            "Window %s: train=%d, test=%d matches",
            test_start.date(), len(train), len(test),
        )

    if not all_predictions:
        raise ValueError(
            "No predictions generated. Check initial_train_years and "
            "step_months — the training windows may be too large for the data."
        )

    result = pd.DataFrame(all_predictions)
    logger.info(
        "Backtest complete: %d predictions over %d windows",
        len(result), len(test_starts),
    )
    return result


def compare_models(
    df: pd.DataFrame,
    model_configs: list[dict[str, Any]],
    initial_train_years: float = 4,
    step_months: int = 12,
) -> pd.DataFrame:
    """Compare multiple model configurations using the same backtest windows.

    Args:
        df: Match DataFrame.
        model_configs: List of configs, each with 'name' and kwargs for PoissonModel.
                       Example: [
                           {"name": "no_decay", "decay_lambda": 0.0},
                           {"name": "decay_0.5", "decay_lambda": 0.5},
                           {"name": "decay_1.0", "decay_lambda": 1.0},
                       ]

    Returns:
        DataFrame with one row per model config, columns = metrics.
    """
    from src.metrics import evaluate_predictions

    results = []

    for config in model_configs:
        name = config.pop("name", "unnamed")

        def factory(cfg=config.copy()):
            return PoissonModel(**cfg)

        preds = expanding_window_backtest(
            df, factory,
            initial_train_years=initial_train_years,
            step_months=step_months,
        )

        metrics = evaluate_predictions(preds)
        metrics["model"] = name
        metrics["n_predictions"] = len(preds)
        results.append(metrics)

        # Restore name for next iteration
        config["name"] = name

    return pd.DataFrame(results)