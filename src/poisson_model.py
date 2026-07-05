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
from typing import Any, Optional

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


def build_score_matrix(
    lambda_home: float,
    lambda_away: float,
    rho: float = 0.0,
) -> np.ndarray:
    """Build a (MAX_GOALS+1) × (MAX_GOALS+1) matrix of score probabilities.

    Entry [i, j] is P(home_score=i, away_score=j).

    With rho=0.0: standard Poisson (independence assumption).
    With rho>0.0: Dixon-Coles adjustment — low scores (0-0, 1-0, 0-1, 1-1)
                   are correlated, happening more/less often than independence predicts.

    Dixon & Coles (1997) showed that 0-0 and 1-1 happen MORE often,
    while 1-0 and 0-1 happen LESS often than Poisson predicts.

    Args:
        lambda_home: Expected goals for the home team.
        lambda_away: Expected goals for the away team.
        rho: Dixon-Coles dependence parameter. 0 = independence.
             Typical values: 0.0 to 0.1. Negative means anti-correlation.

    Returns:
        2D numpy array of shape (MAX_GOALS+1, MAX_GOALS+1).
    """
    home_probs = poisson_prob(lambda_home)
    away_probs = poisson_prob(lambda_away)
    matrix = np.outer(home_probs, away_probs)

    if rho == 0.0 or lambda_home == 0 or lambda_away == 0:
        return matrix

    # ── Dixon-Coles adjustment for low scores ──
    # τ(i,j) modifies P(i,j) for i,j ≤ 1
    # τ(0,0) = 1 + λ·μ·ρ      — more 0-0 draws
    # τ(1,0) = 1 − λ·ρ        — fewer 1-0 wins
    # τ(0,1) = 1 − μ·ρ        — fewer 0-1 wins
    # τ(1,1) = 1 + ρ          — more 1-1 draws
    # τ(i,j) = 1              — unchanged for i>1 or j>1

    lm = lambda_home
    mu = lambda_away

    # Apply tau corrections
    tau_00 = 1.0 + lm * mu * rho
    tau_10 = 1.0 - lm * rho
    tau_01 = 1.0 - mu * rho
    tau_11 = 1.0 + rho

    matrix[0, 0] *= tau_00
    matrix[1, 0] *= tau_10
    matrix[0, 1] *= tau_01
    matrix[1, 1] *= tau_11

    # Re-normalize so the whole matrix sums to 1
    total = matrix.sum()
    if total > 0:
        matrix /= total

    return matrix


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

    def __init__(
        self,
        min_games: int = 3,
        decay_lambda: float = 0.5,
        regularization_k: float = 10.0,
    ):
        """
        Args:
            min_games: Minimum games a team needs before using its own stats.
                       Teams below this threshold use global averages.
            decay_lambda: Time decay rate for match weighting.
                          0.0 = all matches equal (no decay)
                          0.5 = half-life ~1.4 years (balanced)
                          1.0 = half-life ~0.7 years (strong recency)
                          2.0 = half-life ~4 months (very short memory)
            regularization_k: Bayesian shrinkage strength.
                              Higher = more pull toward global average.
                              0 = no regularization (raw averages)
                              10 = moderate (recommended)
                              50 = strong (conservative estimates)
        """
        self.min_games = min_games
        self.decay_lambda = decay_lambda
        self.regularization_k = regularization_k
        self._global_home_avg: float = 0.0
        self._global_away_avg: float = 0.0
        self._global_avg: float = 0.0
        self._home_advantage: float = 1.0
        self._attack: dict[str, float] = {}
        self._defense: dict[str, float] = {}
        self._team_games: dict[str, int] = {}
        self._team_weighted_games: dict[str, float] = {}
        self._rho: float = 0.0
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

                raw_attack = gf_avg / self._global_avg if self._global_avg > 0 else 1.0
                raw_defense = ga_avg / self._global_avg if self._global_avg > 0 else 1.0

                # Bayesian shrinkage toward 1.0 (global average)
                # Formula: (raw × effective_games + prior × k) / (effective_games + k)
                # k = regularization_k pseudo-games of "average" performance
                k = self.regularization_k
                eff = max(weighted_g, 1.0)  # Use weighted games as effective sample size

                self._attack[team] = (raw_attack * eff + 1.0 * k) / (eff + k)
                self._defense[team] = (raw_defense * eff + 1.0 * k) / (eff + k)
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

        # ── Estimate Dixon-Coles rho ──
        self._rho = self._estimate_rho(df)
        logger.info("Dixon-Coles ρ = %.4f", self._rho)

        return self

    def _estimate_rho(self, df: pd.DataFrame) -> float:
        """Estimate the Dixon-Coles dependence parameter from training data.

        Compares observed low-score frequencies to what independence predicts.
        Uses method-of-moments: average the discrepancy across (0,0), (1,1).

        Returns:
            Rho value (typically 0.0 to 0.15). Higher = stronger low-score dependence.
        """
        n = len(df)
        if n == 0:
            return 0.0

        # Count observed low scores
        obs_00 = ((df["home_goals"] == 0) & (df["away_goals"] == 0)).sum() / n
        obs_11 = ((df["home_goals"] == 1) & (df["away_goals"] == 1)).sum() / n

        # Compute expected under independence for each match
        exp_00_total = 0.0
        exp_11_total = 0.0

        for _, row in df.iterrows():
            lm, mu = self.expected_goals(
                row["home_team"], row["away_team"], neutral=bool(row["neutral"])
            )
            p_home = poisson_prob(lm)
            p_away = poisson_prob(mu)
            exp_00_total += p_home[0] * p_away[0]
            exp_11_total += p_home[1] * p_away[1]

        exp_00 = exp_00_total / n
        exp_11 = exp_11_total / n

        # Rho from the (1,1) discrepancy: obs_11 ≈ exp_11 × (1 + ρ)
        if exp_11 > 0 and obs_11 > 0:
            rho_11 = (obs_11 / exp_11) - 1.0
        else:
            rho_11 = 0.0

        # Rho from the (0,0) discrepancy: obs_00 ≈ exp_00 × (1 + λμρ)
        # Average λμ across all matches
        avg_lm_mu = 0.0
        for _, row in df.iterrows():
            lm, mu = self.expected_goals(
                row["home_team"], row["away_team"], neutral=bool(row["neutral"])
            )
            avg_lm_mu += lm * mu
        avg_lm_mu /= n

        if exp_00 > 0 and obs_00 > 0 and avg_lm_mu > 0:
            rho_00 = (obs_00 / exp_00 - 1.0) / avg_lm_mu
        else:
            rho_00 = 0.0

        # Average the two estimates, clamp to reasonable range
        rho = (rho_00 + rho_11) / 2.0
        rho = max(-0.05, min(rho, 0.30))

        return float(rho)

    def expected_goals(
        self,
        home_team: str,
        away_team: str,
        neutral: bool = True,
        home_form: Optional[dict[str, float]] = None,
        away_form: Optional[dict[str, float]] = None,
    ) -> tuple[float, float]:
        """Estimate expected goals for both teams.

        Formula:
          lambda_home = global_home_avg × attack[home] × defense[away] × home_factor × form_factor
          lambda_away = global_away_avg × attack[away] × defense[home] × form_factor

        The home_factor is applied only when neutral=False.
        form_factor adjusts based on recent goals scored/conceded (Brasileirão).

        Args:
            home_team: Name of the home team.
            away_team: Name of the away team.
            neutral: Whether the match is on neutral ground.
            home_form: Optional dict with 'gf' (recent goals scored) and
                       'ga' (recent goals conceded) for the home team.
            away_form: Same for away team.

        Returns:
            (lambda_home, lambda_away) tuple.
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

        # ── Form adjustment (Brasileirão) ──
        # If a team is scoring 2.0 recently vs 1.5 historically → 1.33x boost
        if home_form and home_form.get("gf", 0) > 0:
            hist_gf_home = home_att * self._global_avg
            form_boost_att = home_form["gf"] / hist_gf_home if hist_gf_home > 0 else 1.0
            home_att *= max(0.5, min(form_boost_att, 2.0))  # Clamp 0.5x to 2.0x

        if home_form and home_form.get("ga", 0) > 0:
            hist_ga_home = home_def * self._global_avg
            form_boost_def = home_form["ga"] / hist_ga_home if hist_ga_home > 0 else 1.0
            home_def *= max(0.5, min(form_boost_def, 2.0))

        if away_form and away_form.get("gf", 0) > 0:
            hist_gf_away = away_att * self._global_avg
            form_boost_att = away_form["gf"] / hist_gf_away if hist_gf_away > 0 else 1.0
            away_att *= max(0.5, min(form_boost_att, 2.0))

        if away_form and away_form.get("ga", 0) > 0:
            hist_ga_away = away_def * self._global_avg
            form_boost_def = away_form["ga"] / hist_ga_away if hist_ga_away > 0 else 1.0
            away_def *= max(0.5, min(form_boost_def, 2.0))

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
            "regularization_k": self.regularization_k,
            "global_home_avg": self._global_home_avg,
            "global_away_avg": self._global_away_avg,
            "global_avg": self._global_avg,
            "home_advantage": self._home_advantage,
            "attack": self._attack,
            "defense": self._defense,
            "team_games": self._team_games,
            "team_weighted_games": self._team_weighted_games,
            "rho": self._rho,
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
            regularization_k=data.get("regularization_k", 0.0),
        )
        model._global_home_avg = data["global_home_avg"]
        model._global_away_avg = data["global_away_avg"]
        model._global_avg = data["global_avg"]
        model._home_advantage = data["home_advantage"]
        model._attack = data["attack"]
        model._defense = data["defense"]
        model._team_games = data["team_games"]
        model._team_weighted_games = data["team_weighted_games"]
        model._rho = data.get("rho", 0.0)
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
        self,
        home_team: str,
        away_team: str,
        neutral: bool = True,
        home_form: Optional[dict[str, float]] = None,
        away_form: Optional[dict[str, float]] = None,
    ) -> dict[str, Any]:
        """Full match prediction — the main function you'll call.

        Args:
            home_team: Home team name.
            away_team: Away team name.
            neutral: Whether the match is on neutral ground.
            home_form: Optional dict with 'gf', 'ga' for recent form (Brasileirão).
            away_form: Optional dict with 'gf', 'ga' for recent form (Brasileirão).

        Returns:
            Dictionary with all predictions and probabilities.
        """
        lambda_home, lambda_away = self.expected_goals(
            home_team, away_team, neutral,
            home_form=home_form, away_form=away_form,
        )

        score_matrix = build_score_matrix(lambda_home, lambda_away, rho=self._rho)
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