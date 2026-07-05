"""
poisson_model.py — Probabilistic football match prediction using Poisson.

Core idea:
  1. Estimate expected goals (lambda) for each team
  2. Use Poisson distribution to get goal probabilities
  3. Multiply to build a score matrix
  4. Derive all match probabilities from that matrix

Mathematical foundation:
  P(k goals | lambda) = (lambda^k * e^(-lambda)) / k!
"""

from __future__ import annotations

import logging
from typing import Any

import numpy as np
import pandas as pd
from scipy.stats import poisson

logger = logging.getLogger(__name__)

# Maximum goals to model in the score matrix.
# Matches rarely exceed 10 goals; this covers 99.9%+ of cases.
MAX_GOALS = 10


# ── Poisson probability helpers ────────────────────────────────────

def poisson_prob(lambda_: float, max_goals: int = MAX_GOALS) -> np.ndarray:
    """Return P(k goals) for k = 0, 1, ..., max_goals given lambda.

    Args:
        lambda_: Expected goals (must be > 0).
        max_goals: Upper bound for goal count.

    Returns:
        Array of probabilities, shape (max_goals + 1,), summing to ~1.
    """
    ks = np.arange(max_goals + 1)
    probs = poisson.pmf(ks, lambda_)
    return probs / probs.sum()  # Normalize so it sums to exactly 1


def build_score_matrix(lambda_home: float, lambda_away: float) -> np.ndarray:
    """Build a (MAX_GOALS+1) × (MAX_GOALS+1) matrix of score probabilities.

    Entry [i, j] is P(home_score=i, away_score=j), assuming independence.

    Args:
        lambda_home: Expected goals for the home team.
        lambda_away: Expected goals for the away team.

    Returns:
        2D numpy array of shape (MAX_GOALS+1, MAX_GOALS+1).
    """
    home_probs = poisson_prob(lambda_home)
    away_probs = poisson_prob(lambda_away)
    # Outer product: P(home=i) * P(away=j) for all i, j
    return np.outer(home_probs, away_probs)


# ── Match outcome extraction ───────────────────────────────────────

def extract_probabilities(score_matrix: np.ndarray) -> dict[str, float]:
    """Derive all probabilities from a score matrix.

    Args:
        score_matrix: 2D array where [i, j] = P(home=i, away=j).

    Returns:
        Dictionary with keys:
          - home_win_probability
          - draw_probability
          - away_win_probability
          - both_teams_score_probability
          - over_2_5_probability
          - under_2_5_probability
    """
    max_g = score_matrix.shape[0] - 1

    home_win = 0.0
    draw = 0.0
    away_win = 0.0
    both_score = 0.0
    over_25 = 0.0

    for i in range(max_g + 1):
        for j in range(max_g + 1):
            p = score_matrix[i, j]
            if i > j:
                home_win += p
            elif i == j:
                draw += p
            else:
                away_win += p

            if i > 0 and j > 0:
                both_score += p

            if i + j > 2:
                over_25 += p

    return {
        "home_win_probability": float(home_win),
        "draw_probability": float(draw),
        "away_win_probability": float(away_win),
        "both_teams_score_probability": float(both_score),
        "over_2_5_probability": float(over_25),
        "under_2_5_probability": float(1.0 - over_25),
    }


def most_likely_scores(
    score_matrix: np.ndarray, top_n: int = 5
) -> list[dict[str, Any]]:
    """Return the top-N most likely scorelines.

    Args:
        score_matrix: 2D probability matrix.
        top_n: Number of top scores to return.

    Returns:
        List of dicts with keys: score (str like "2-1"), probability (float).
    """
    max_g = score_matrix.shape[0] - 1
    scores = []

    for i in range(max_g + 1):
        for j in range(max_g + 1):
            scores.append({"score": f"{i}-{j}", "probability": float(score_matrix[i, j])})

    scores.sort(key=lambda x: x["probability"], reverse=True)
    return scores[:top_n]


# ── Expected goals estimation ──────────────────────────────────────

class PoissonModel:
    """Simple Poisson-based football match predictor.

    Estimates expected goals using:
      - Global average goals (baseline)
      - Team offensive strength (goals scored relative to average)
      - Team defensive strength (goals conceded relative to average)
      - Home advantage multiplier (ignored when neutral=True)

    For teams with very few games, falls back to global averages.
    """

    def __init__(self, min_games: int = 3, decay_lambda: float = 0.5):
        """
        Args:
            min_games: Minimum games a team needs before using its own stats.
                       Teams below this threshold use global averages.
            decay_lambda: Time decay rate for match weighting.
                          0.0 = all matches equal (no decay)
                          0.5 = half-life ~1.4 years (balanced)
                          1.0 = half-life ~0.7 years (strong recency)
                          2.0 = half-life ~4 months (very short memory)
        """
        self.min_games = min_games
        self.decay_lambda = decay_lambda
        self._global_home_avg: float = 0.0
        self._global_away_avg: float = 0.0
        self._global_avg: float = 0.0
        self._home_advantage: float = 1.0
        self._attack: dict[str, float] = {}
        self._defense: dict[str, float] = {}
        self._team_games: dict[str, int] = {}
        self._team_weighted_games: dict[str, float] = {}
        self._fitted = False

    def fit(self, df: pd.DataFrame) -> PoissonModel:
        """Learn team strengths from historical match data.

        Uses exponential time decay so recent matches carry more weight.
        Formula: weight(match) = exp(-λ × years_since_match)

        Args:
            df: DataFrame with columns home_team, away_team, home_goals,
                away_goals, neutral.

        Returns:
            self (for method chaining).
        """
        df = df.copy()
        reference_date = df["date"].max()

        # ── Time decay weights ──
        years_ago = (reference_date - df["date"]).dt.days / 365.25
        weight = np.exp(-self.decay_lambda * years_ago)
        df["_weight"] = weight

        # ── Global averages (time-weighted) ──
        total_weight = weight.sum()
        self._global_home_avg = (df["home_goals"] * weight).sum() / total_weight
        self._global_away_avg = (df["away_goals"] * weight).sum() / total_weight
        self._global_avg = (self._global_home_avg + self._global_away_avg) / 2

        # ── Home advantage (time-weighted, non-neutral only) ──
        non_neutral = df[~df["neutral"]]
        if len(non_neutral) > 0:
            nn_weight = non_neutral["_weight"]
            nn_total = nn_weight.sum()
            home_avg_nn = (non_neutral["home_goals"] * nn_weight).sum() / nn_total
            away_avg_nn = (non_neutral["away_goals"] * nn_weight).sum() / nn_total
            self._home_advantage = home_avg_nn / away_avg_nn if away_avg_nn > 0 else 1.0
        else:
            self._home_advantage = 1.0

        # ── Per-team stats (time-weighted) ──
        def weighted_mean(group, col):
            w = group["_weight"]
            return (group[col] * w).sum() / w.sum()

        home_for = df.groupby("home_team").apply(weighted_mean, "home_goals")
        away_for = df.groupby("away_team").apply(weighted_mean, "away_goals")
        home_against = df.groupby("home_team").apply(weighted_mean, "away_goals")
        away_against = df.groupby("away_team").apply(weighted_mean, "home_goals")
        home_games = df.groupby("home_team").size()
        away_games = df.groupby("away_team").size()
        home_weighted = df.groupby("home_team")["_weight"].sum()
        away_weighted = df.groupby("away_team")["_weight"].sum()

        all_teams = set(home_for.index) | set(away_for.index)

        for team in all_teams:
            games = int(home_games.get(team, 0) + away_games.get(team, 0))
            weighted_g = float(home_weighted.get(team, 0.0) + away_weighted.get(team, 0.0))
            self._team_games[team] = games
            self._team_weighted_games[team] = weighted_g

            if games >= self.min_games:
                gf_home = float(home_for.get(team, 0.0))
                gf_away = float(away_for.get(team, 0.0))
                gf_avg = (gf_home + gf_away) / 2

                ga_home = float(home_against.get(team, 0.0))
                ga_away = float(away_against.get(team, 0.0))
                ga_avg = (ga_home + ga_away) / 2

                self._attack[team] = gf_avg / self._global_avg if self._global_avg > 0 else 1.0
                self._defense[team] = ga_avg / self._global_avg if self._global_avg > 0 else 1.0
            else:
                self._attack[team] = 1.0
                self._defense[team] = 1.0

        self._fitted = True
        half_life = np.log(2) / self.decay_lambda if self.decay_lambda > 0 else float("inf")
        logger.info(
            "Fitted PoissonModel on %d matches (λ_decay=%.1f, half-life=%.1fy). "
            "%d teams. Global avg: %.2f goals. Home advantage: %.2fx",
            len(df),
            self.decay_lambda,
            half_life,
            len(all_teams),
            self._global_avg,
            self._home_advantage,
        )
        logger.info(
            "Teams with fallback (<%d games): %s",
            self.min_games,
            [t for t, g in self._team_games.items() if g < self.min_games],
        )
        return self

    def expected_goals(
        self, home_team: str, away_team: str, neutral: bool = True
    ) -> tuple[float, float]:
        """Estimate expected goals for both teams.

        Formula:
          lambda_home = global_home_avg × attack[home] × defense[away] × home_factor
          lambda_away = global_away_avg × attack[away] × defense[home]

        The home_factor is applied only when neutral=False.

        Args:
            home_team: Name of the home team.
            away_team: Name of the away team.
            neutral: Whether the match is on neutral ground.

        Returns:
            (lambda_home, lambda_away) tuple.

        Raises:
            RuntimeError: if the model hasn't been fitted yet.
        """
        if not self._fitted:
            raise RuntimeError("Model not fitted. Call .fit(df) first.")

        home_team = home_team.strip().title()
        away_team = away_team.strip().title()

        # Get strengths (default to 1.0 for unknown teams)
        home_att = self._attack.get(home_team, 1.0)
        home_def = self._defense.get(home_team, 1.0)
        away_att = self._attack.get(away_team, 1.0)
        away_def = self._defense.get(away_team, 1.0)

        # Home advantage factor
        home_factor = 1.0 if neutral else self._home_advantage

        lambda_home = self._global_home_avg * home_att * away_def * home_factor
        lambda_away = self._global_away_avg * away_att * home_def

        # Safety floor: expected goals should never be negative or zero
        lambda_home = max(lambda_home, 0.01)
        lambda_away = max(lambda_away, 0.01)

        return lambda_home, lambda_away

    # ── Model persistence ──────────────────────────────────────────

    def save(self, path: str | Path) -> None:
        """Save the trained model to disk so you don't need to retrain.

        Uses pickle — the model is just floats, dicts, and bools.
        Typical file size: ~30 KB for 300 teams.

        Args:
            path: Where to save (e.g., 'models/poisson_wc2026.pkl').
        """
        import pickle
        from pathlib import Path

        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        if not self._fitted:
            raise RuntimeError("Cannot save an unfitted model.")

        data = {
            "min_games": self.min_games,
            "decay_lambda": self.decay_lambda,
            "global_home_avg": self._global_home_avg,
            "global_away_avg": self._global_away_avg,
            "global_avg": self._global_avg,
            "home_advantage": self._home_advantage,
            "attack": self._attack,
            "defense": self._defense,
            "team_games": self._team_games,
            "team_weighted_games": self._team_weighted_games,
            "fitted": True,
        }

        with open(path, "wb") as f:
            pickle.dump(data, f)

        logger.info("Model saved to %s (%d teams)", path, len(self._attack))

    @classmethod
    def load(cls, path: str | Path) -> "PoissonModel":
        """Load a previously saved model.

        Args:
            path: Path to the .pkl file saved by PoissonModel.save().

        Returns:
            A fully fitted PoissonModel ready for predictions.
        """
        import pickle
        from pathlib import Path

        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"Model file not found: {path}")

        with open(path, "rb") as f:
            data = pickle.load(f)

        model = cls(
            min_games=data["min_games"],
            decay_lambda=data["decay_lambda"],
        )
        model._global_home_avg = data["global_home_avg"]
        model._global_away_avg = data["global_away_avg"]
        model._global_avg = data["global_avg"]
        model._home_advantage = data["home_advantage"]
        model._attack = data["attack"]
        model._defense = data["defense"]
        model._team_games = data["team_games"]
        model._team_weighted_games = data["team_weighted_games"]
        model._fitted = data["fitted"]

        logger.info(
            "Model loaded from %s (%d teams, λ=%.1f)",
            path,
            len(model._attack),
            model.decay_lambda,
        )
        return model

    # ── Prediction ─────────────────────────────────────────────────

    def predict_match(
        self, home_team: str, away_team: str, neutral: bool = True
    ) -> dict[str, Any]:
        """Full match prediction — the main function you'll call.

        Args:
            home_team: Home team name.
            away_team: Away team name.
            neutral: Whether the match is on neutral ground.

        Returns:
            Dictionary with all predictions and probabilities.
        """
        lambda_home, lambda_away = self.expected_goals(home_team, away_team, neutral)

        score_matrix = build_score_matrix(lambda_home, lambda_away)
        probs = extract_probabilities(score_matrix)
        scores = most_likely_scores(score_matrix, top_n=5)

        return {
            "home_team": home_team,
            "away_team": away_team,
            "neutral": neutral,
            "expected_goals_home": round(lambda_home, 2),
            "expected_goals_away": round(lambda_away, 2),
            **probs,
            "most_likely_scores": scores,
        }