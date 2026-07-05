#!/usr/bin/env python3
"""
run_prediction.py — Quick match prediction from the command line.

Usage:
    python run_prediction.py
    python run_prediction.py "Brazil" "Argentina"

The script loads match data, fits the Poisson model, and prints
a formatted prediction for the requested matchup.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

# Allow running from project root
sys.path.insert(0, str(Path(__file__).resolve().parent))

from src.data_loader import load_matches
from src.poisson_model import PoissonModel


def format_prediction(pred: dict) -> str:
    """Format a prediction dict into a readable string."""
    lines = [
        "",
        "=" * 50,
        f"  {pred['home_team']} vs {pred['away_team']}",
        f"  {'(Neutral ground)' if pred['neutral'] else '(Home advantage)'}",
        "=" * 50,
        "",
        "📊 Expected Goals:",
        f"  {pred['home_team']}: {pred['expected_goals_home']:.2f}",
        f"  {pred['away_team']}: {pred['expected_goals_away']:.2f}",
        "",
        "🎲 Probabilities:",
        f"  {pred['home_team']} win:  {pred['home_win_probability']:.1%}",
        f"  Draw:               {pred['draw_probability']:.1%}",
        f"  {pred['away_team']} win:  {pred['away_win_probability']:.1%}",
        "",
        "⚽ Both teams score:",
        f"  Yes: {pred['both_teams_score_probability']:.1%}",
        f"  No:  {1 - pred['both_teams_score_probability']:.1%}",
        "",
        "📈 Over/Under 2.5 goals:",
        f"  Over 2.5:  {pred['over_2_5_probability']:.1%}",
        f"  Under 2.5: {pred['under_2_5_probability']:.1%}",
        "",
        "🏆 Most Likely Scores:",
    ]

    for entry in pred["most_likely_scores"]:
        lines.append(f"  {entry['score']}: {entry['probability']:.1%}")

    lines.append("")
    lines.append("─" * 50)
    lines.append(
        "⚡ Remember: this is a probabilistic model, not a crystal ball.\n"
        "   It tells you what's LIKELY, not what WILL happen."
    )
    lines.append("─" * 50)
    lines.append("")

    return "\n".join(lines)


def main():
    # Paths
    data_path = Path(__file__).resolve().parent / "data" / "matches.csv"
    model_path = Path(__file__).resolve().parent / "models" / "poisson_model.pkl"

    # ── Load or train model ──
    if model_path.exists():
        print(f"\n📦 Loading cached model from: {model_path}")
        model = PoissonModel.load(model_path)
        print(f"   (Delete {model_path} to force retrain)")
    else:
        print(f"\n📂 Loading match data from: {data_path}")
        df = load_matches(data_path)
        print(f"   {len(df)} matches loaded.")

        model = PoissonModel(min_games=3, decay_lambda=0.5)
        model.fit(df)

        model.save(model_path)
        print(f"   Model cached to: {model_path}")

    # Choose teams
    if len(sys.argv) >= 3:
        home_team = sys.argv[1]
        away_team = sys.argv[2]
    else:
        home_team = "Brazil"
        away_team = "France"

    print(f"\n🔮 Predicting: {home_team} vs {away_team}")

    pred = model.predict_match(home_team, away_team, neutral=True)
    print(format_prediction(pred))

    # Bonus: show team stats
    print("\n📋 Team profiles (from training data):")
    half_life = np.log(2) / model.decay_lambda
    print(f"   Time decay: λ={model.decay_lambda} (half-life: {half_life:.1f} years)")
    for team in [home_team, away_team]:
        att = model._attack.get(team, 1.0)
        def_ = model._defense.get(team, 1.0)
        games = model._team_games.get(team, 0)
        w_games = model._team_weighted_games.get(team, 0.0)
        tag = "⚠️ FALLBACK" if games < model.min_games else "✓"
        print(
            f"  {team}: {games} games ({w_games:.1f} weighted) | "
            f"Attack: {att:.2f}x | Defense: {def_:.2f}x {tag}"
        )


if __name__ == "__main__":
    main()