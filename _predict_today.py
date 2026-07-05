"""Predict today's matches (July 5, 2026) — true out-of-sample."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))

import pandas as pd
from src.data_loader import load_matches
from src.poisson_model import PoissonModel

df = load_matches("data/matches.csv")

# Today's matches are NOT in the training data (NaN goals = filtered out by load_matches)
# Train on everything with actual results
model = PoissonModel(min_games=3)
model.fit(df)

print("TODAY'S MATCHES — TRUE OUT-OF-SAMPLE PREDICTION")
print("Model has NO knowledge of these results.\n")

# Predict manually
games = [
    ("Brazil", "Norway", True),
    ("Mexico", "England", True),
]

for home, away, neutral in games:
    pred = model.predict_match(home, away, neutral=neutral)
    
    print("=" * 56)
    print(f"  ⚽ {home} vs {away}")
    print("=" * 56)
    
    games_h = model._team_games.get(home, 0)
    games_a = model._team_games.get(away, 0)
    att_h = model._attack.get(home, 1.0)
    def_h = model._defense.get(home, 1.0)
    att_a = model._attack.get(away, 1.0)
    def_a = model._defense.get(away, 1.0)
    
    print(f"  {home}: {games_h}g | Atk {att_h:.2f}x | Def {def_h:.2f}x")
    print(f"  {away}: {games_a}g | Atk {att_a:.2f}x | Def {def_a:.2f}x")
    print()
    print(f"  📊 xG:  {home} {pred['expected_goals_home']:.2f}  —  {away} {pred['expected_goals_away']:.2f}")
    print()
    print(f"  🎲 {home} win:  {pred['home_win_probability']:.1%}")
    print(f"     Draw:       {pred['draw_probability']:.1%}")
    print(f"     {away} win:  {pred['away_win_probability']:.1%}")
    print()
    print(f"  ⚽ Both score: {pred['both_teams_score_probability']:.1%}")
    print(f"  📈 Over 2.5:   {pred['over_2_5_probability']:.1%}")
    print()
    print(f"  🏆 Most likely:")
    for s in pred['most_likely_scores'][:5]:
        bar = '█' * int(s['probability'] * 100)
        print(f"     {s['score']}  {s['probability']:.1%}  {bar}")
    print()