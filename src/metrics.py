"""
metrics.py — Proper probabilistic evaluation for football predictions.

Key principle: accuracy alone is misleading. A model that says
"Brazil 99%, Draw 0.5%, Norway 0.5%" and gets it right looks
perfect on accuracy — but it's dangerously overconfident.

Log loss and Brier score punish overconfidence. That's what we use.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def log_loss_score(y_true_home: int, y_true_away: int,
                   p_home: float, p_draw: float, p_away: float) -> float:
    """Compute log loss for a single match.

    Log loss = -ln(p_correct_outcome)

    Lower is better. Perfect = 0. Random guessing (33/33/33) = -ln(0.33) ≈ 1.10.
    A model that's overconfident and wrong gets massively penalized:
      -ln(0.01) = 4.6 (ouch!)

    Args:
        y_true_home: Actual home goals.
        y_true_away: Actual away goals.
        p_home: Model's home win probability.
        p_draw: Model's draw probability.
        p_away: Model's away win probability.

    Returns:
        Log loss value (non-negative, lower = better).
    """
    if y_true_home > y_true_away:
        p_correct = p_home
    elif y_true_home == y_true_away:
        p_correct = p_draw
    else:
        p_correct = p_away

    # Clip to avoid log(0)
    p_correct = np.clip(p_correct, 1e-15, 1.0)
    return float(-np.log(p_correct))


def brier_score(y_true_home: int, y_true_away: int,
                p_home: float, p_draw: float, p_away: float) -> float:
    """Compute multi-class Brier score for a single match.

    Brier = (1/N) * Σ (p_i - o_i)²

    Where o_i = 1 for the correct outcome, 0 otherwise.
    Lower is better. 0 = perfect, 1 = worst.

    This is more interpretable than log loss:
      - 0.00 = perfect
      - 0.20 = good (well-calibrated)
      - 0.30+ = poor

    Args:
        y_true_home, y_true_away: Actual goals.
        p_home, p_draw, p_away: Model probabilities.

    Returns:
        Brier score (0 to ~2, lower = better).
    """
    # One-hot encode the true outcome
    if y_true_home > y_true_away:
        o_home, o_draw, o_away = 1.0, 0.0, 0.0
    elif y_true_home == y_true_away:
        o_home, o_draw, o_away = 0.0, 1.0, 0.0
    else:
        o_home, o_draw, o_away = 0.0, 0.0, 1.0

    squared_errors = (
        (p_home - o_home) ** 2
        + (p_draw - o_draw) ** 2
        + (p_away - o_away) ** 2
    )
    return float(squared_errors)


def accuracy(y_true_home: int, y_true_away: int,
             p_home: float, p_draw: float, p_away: float) -> bool:
    """Did the model pick the correct winner? (1 = correct, 0 = wrong).

    This is a SECONDARY metric. A coin that always says "home win"
    gets ~45% accuracy in football. That doesn't make it a good model.

    Use this for intuition, not for model selection.
    """
    if y_true_home > y_true_away:
        predicted_correct = p_home > p_draw and p_home > p_away
    elif y_true_home == y_true_away:
        predicted_correct = p_draw > p_home and p_draw > p_away
    else:
        predicted_correct = p_away > p_home and p_away > p_draw

    return predicted_correct


def evaluate_predictions(df_predictions: pd.DataFrame) -> dict:
    """Evaluate a DataFrame of predictions against actual results.

    Args:
        df_predictions: DataFrame with columns:
            home_goals, away_goals (actual)
            home_win_probability, draw_probability, away_win_probability

    Returns:
        Dict with aggregate metrics.
    """
    n = len(df_predictions)
    if n == 0:
        return {"error": "No predictions to evaluate"}

    log_losses = []
    brier_scores = []
    accuracies = []

    for _, row in df_predictions.iterrows():
        log_losses.append(log_loss_score(
            row["home_goals"], row["away_goals"],
            row["home_win_probability"], row["draw_probability"],
            row["away_win_probability"],
        ))
        brier_scores.append(brier_score(
            row["home_goals"], row["away_goals"],
            row["home_win_probability"], row["draw_probability"],
            row["away_win_probability"],
        ))
        accuracies.append(accuracy(
            row["home_goals"], row["away_goals"],
            row["home_win_probability"], row["draw_probability"],
            row["away_win_probability"],
        ))

    avg_log_loss = np.mean(log_losses)
    avg_brier = np.mean(brier_scores)
    acc_rate = np.mean(accuracies)

    # Naive baseline: always predict global outcome distribution
    home_win_pct = (df_predictions["home_goals"] > df_predictions["away_goals"]).mean()
    draw_pct = (df_predictions["home_goals"] == df_predictions["away_goals"]).mean()
    away_win_pct = (df_predictions["home_goals"] < df_predictions["away_goals"]).mean()

    naive_log_loss = -(
        home_win_pct * np.log(max(home_win_pct, 1e-15))
        + draw_pct * np.log(max(draw_pct, 1e-15))
        + away_win_pct * np.log(max(away_win_pct, 1e-15))
    )
    naive_accuracy = max(home_win_pct, draw_pct, away_win_pct)
    naive_brier = (
        home_win_pct * ((1 - home_win_pct) ** 2 + draw_pct ** 2 + away_win_pct ** 2)
        + draw_pct * (home_win_pct ** 2 + (1 - draw_pct) ** 2 + away_win_pct ** 2)
        + away_win_pct * (home_win_pct ** 2 + draw_pct ** 2 + (1 - away_win_pct) ** 2)
    )

    return {
        "n_matches": n,
        "log_loss": round(avg_log_loss, 4),
        "brier_score": round(avg_brier, 4),
        "accuracy": round(acc_rate, 4),
        "naive_log_loss": round(naive_log_loss, 4),
        "naive_brier_score": round(naive_brier, 4),
        "naive_accuracy": round(naive_accuracy, 4),
        "log_loss_vs_naive": round(naive_log_loss - avg_log_loss, 4),
        "brier_vs_naive": round(naive_brier - avg_brier, 4),
        "verdict": _verdict(avg_log_loss, naive_log_loss, avg_brier, naive_brier),
    }


def _verdict(ll: float, nll: float, br: float, nbr: float) -> str:
    """Generate a human-readable verdict."""
    parts = []
    if ll < nll:
        parts.append(f"✅ Log loss {ll:.3f} beats naive {nll:.3f}")
    else:
        parts.append(f"❌ Log loss {ll:.3f} worse than naive {nll:.3f}")

    if br < nbr:
        parts.append(f"✅ Brier {br:.3f} beats naive {nbr:.3f}")
    else:
        parts.append(f"❌ Brier {br:.3f} worse than naive {nbr:.3f}")

    return " | ".join(parts)


def calibration_summary(df_predictions: pd.DataFrame,
                        n_bins: int = 10) -> pd.DataFrame:
    """Group predictions into probability bins and check calibration.

    A well-calibrated model: when it says "30% win probability",
    that team should win ~30% of the time.

    Args:
        df_predictions: Predictions with actual outcomes.
        n_bins: Number of probability bins.

    Returns:
        DataFrame with columns: bin, n_predictions, predicted_pct,
        actual_pct, calibration_error.
    """
    df = df_predictions.copy()
    df["actual_win"] = df["home_goals"] > df["away_goals"]
    df["pred_win"] = df["home_win_probability"]

    bins = np.linspace(0, 1, n_bins + 1)
    df["bin"] = pd.cut(df["pred_win"], bins=bins, labels=False, include_lowest=True)

    summary = []
    for i in range(n_bins):
        mask = df["bin"] == i
        n = mask.sum()
        if n == 0:
            continue
        pred_pct = df.loc[mask, "pred_win"].mean()
        actual_pct = df.loc[mask, "actual_win"].mean()
        summary.append({
            "bin_center": round((bins[i] + bins[i + 1]) / 2, 2),
            "n_matches": int(n),
            "predicted_pct": round(pred_pct, 3),
            "actual_pct": round(actual_pct, 3),
            "error": round(pred_pct - actual_pct, 3),
        })

    return pd.DataFrame(summary)