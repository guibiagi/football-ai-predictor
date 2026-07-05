#!/usr/bin/env python3
"""
evaluate.py — Full model evaluation with temporal backtesting.

Compares:
  - Model without time decay (baseline)
  - Model with time decay (our improvement)
  - Naive baseline (always predict historical averages)

Usage:
  python evaluate.py           # Quick evaluation
  python evaluate.py --full    # Full backtest (takes ~30s)
"""

from __future__ import annotations

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np
import pandas as pd

from src.data_loader import load_matches, temporal_train_test_split
from src.poisson_model import PoissonModel
from src.metrics import (
    evaluate_predictions,
    calibration_summary,
    log_loss_score,
    brier_score,
    accuracy,
)
from src.validation import expanding_window_backtest


def quick_eval():
    """Fast evaluation: single temporal split, compare 3 models."""
    print("=" * 65)
    print("  ⚽ MVP 2 — Model Evaluation Report")
    print("=" * 65)

    df = load_matches("data/matches.csv")
    train, test = temporal_train_test_split(df, test_ratio=0.2)
    print(f"\n📊 Split: {len(train)} train / {len(test)} test matches")
    print(f"   Train range: {train['date'].min().date()} → {train['date'].max().date()}")
    print(f"   Test range:  {test['date'].min().date()} → {test['date'].max().date()}")

    # ── Models to compare ──
    models = {
        "Naive baseline": None,  # Special case
        "Poisson (no decay)": PoissonModel(decay_lambda=0.0, min_games=3),
        "Poisson (λ=0.5)": PoissonModel(decay_lambda=0.5, min_games=3),
    }

    results = []

    for name, model in models.items():
        if model is not None:
            model.fit(train)

        predictions = []
        for _, row in test.iterrows():
            if model is not None:
                pred = model.predict_match(
                    row["home_team"], row["away_team"],
                    neutral=bool(row["neutral"]),
                )
                p_home = pred["home_win_probability"]
                p_draw = pred["draw_probability"]
                p_away = pred["away_win_probability"]
            else:
                # Naive baseline: use training set outcome distribution
                home_pct = (train["home_goals"] > train["away_goals"]).mean()
                draw_pct = (train["home_goals"] == train["away_goals"]).mean()
                away_pct = (train["home_goals"] < train["away_goals"]).mean()
                p_home, p_draw, p_away = home_pct, draw_pct, away_pct

            predictions.append({
                "home_goals": int(row["home_goals"]),
                "away_goals": int(row["away_goals"]),
                "home_win_probability": p_home,
                "draw_probability": p_draw,
                "away_win_probability": p_away,
            })

        pred_df = pd.DataFrame(predictions)
        eval_result = evaluate_predictions(pred_df)
        eval_result["model"] = name
        results.append(eval_result)

    # ── Print results table ──
    print(f"\n{'Model':<25s} {'LogLoss':>8s} {'Brier':>8s} {'Acc':>8s} {'vsNaiveLL':>10s}")
    print("-" * 62)
    for r in results:
        vs_ll = r.get("log_loss_vs_naive", 0)
        sign = "+" if vs_ll > 0 else ""
        print(
            f"{r['model']:<25s} "
            f"{r['log_loss']:>8.4f} "
            f"{r['brier_score']:>8.4f} "
            f"{r['accuracy']:>7.1%} "
            f"{sign}{vs_ll:>9.4f}"
        )

    # ── Verdict ──
    best = min(
        [r for r in results if r["model"] != "Naive baseline"],
        key=lambda x: x["log_loss"],
    )
    naive = [r for r in results if r["model"] == "Naive baseline"][0]

    print(f"\n{'─' * 62}")
    print(f"🏆 Best model: {best['model']}")
    print(f"   Log loss: {best['log_loss']:.4f} (naive = {naive['log_loss']:.4f})")
    improvement = (naive["log_loss"] - best["log_loss"]) / naive["log_loss"] * 100
    if improvement > 0:
        print(f"   Improvement over naive: {improvement:.1f}%")
    else:
        print(f"   ⚠️ Model is WORSE than naive baseline!")

    # ── Calibration ──
    print(f"\n📐 Calibration analysis for: {best['model']}")
    best_model = models[best["model"]]
    best_model.fit(train)

    cal_predictions = []
    for _, row in test.iterrows():
        pred = best_model.predict_match(
            row["home_team"], row["away_team"], neutral=bool(row["neutral"])
        )
        cal_predictions.append({
            "home_goals": int(row["home_goals"]),
            "away_goals": int(row["away_goals"]),
            "home_win_probability": pred["home_win_probability"],
            "draw_probability": pred["draw_probability"],
            "away_win_probability": pred["away_win_probability"],
        })

    cal = calibration_summary(pd.DataFrame(cal_predictions), n_bins=10)
    print(f"\n{'Bin':>7s} {'N':>5s} {'Predicted':>10s} {'Actual':>10s} {'Error':>8s}")
    print("-" * 45)
    for _, row in cal.iterrows():
        direction = "⚠️ " if abs(row["error"]) > 0.1 else "  "
        print(
            f"{direction}{row['bin_center']:>5.2f} "
            f"{int(row['n_matches']):>5d} "
            f"{row['predicted_pct']:>9.1%} "
            f"{row['actual_pct']:>9.1%} "
            f"{row['error']:>+8.3f}"
        )

    print(f"\n💡 Calibration error = predicted - actual")
    print(f"   Positive = overconfident, negative = underconfident")
    print(f"   |error| < 0.05 is good, > 0.10 needs attention")

    return results


def full_backtest():
    """Full expanding-window backtest (takes ~30-60 seconds)."""
    print("=" * 65)
    print("  ⚽ Full Temporal Backtest")
    print("=" * 65)

    df = load_matches("data/matches.csv")

    from src.validation import compare_models

    configs = [
        {"name": "No decay (λ=0)", "decay_lambda": 0.0, "min_games": 3},
        {"name": "Decay λ=0.5", "decay_lambda": 0.5, "min_games": 3},
        {"name": "Decay λ=1.0", "decay_lambda": 1.0, "min_games": 3},
    ]

    print("Running expanding-window backtest...")
    print("(This evaluates the model as if used historically — no future leakage)")
    results = compare_models(df, configs, initial_train_years=4, step_months=12)

    print(f"\n{'Model':<20s} {'Preds':>6s} {'LogLoss':>8s} {'Brier':>8s} {'Acc':>7s}")
    print("-" * 54)
    best_ll = float("inf")
    best_name = ""
    for _, row in results.iterrows():
        print(
            f"{row['model']:<20s} "
            f"{int(row['n_predictions']):>6d} "
            f"{row['log_loss']:>8.4f} "
            f"{row['brier_score']:>8.4f} "
            f"{row['accuracy']:>7.1%}"
        )
        if row["log_loss"] < best_ll:
            best_ll = row["log_loss"]
            best_name = row["model"]

    print(f"\n🏆 Best backtest model: {best_name} (log loss = {best_ll:.4f})")


if __name__ == "__main__":
    if "--full" in sys.argv:
        full_backtest()
    else:
        quick_eval()