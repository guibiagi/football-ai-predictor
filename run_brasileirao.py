#!/usr/bin/env python3
"""
run_brasileirao.py — Predict Brasileirão matches with home advantage + form.

Usage:
  python run_brasileirao.py "Flamengo" "Palmeiras"
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np
import pandas as pd

from src.data_loader import load_matches
from src.poisson_model import PoissonModel
from src.features import add_form_adjustment


def main():
    data_path = Path("data/matches_brasileirao.csv")
    model_path = Path("models/poisson_brasileirao.pkl")

    # ── Load or train ──
    if model_path.exists():
        print(f"📦 Loading cached model from: {model_path}")
        model = PoissonModel.load(model_path)
        df = pd.read_csv(data_path, parse_dates=["date"])
    else:
        print(f"📂 Loading Brasileirão data: {data_path}")
        df = load_matches(data_path)
        print(f"   {len(df)} matches, {df['home_team'].nunique()} teams")
        print(f"   {df['date'].min().date()} → {df['date'].max().date()}")

        # Brasileirão-specific: high decay (short memory), home advantage ON
        model = PoissonModel(
            min_games=5,
            decay_lambda=2.0,   # Half-life ~4 months (transfer windows!)
            regularization_k=15.0,
        )
        model.fit(df)
        model.save(model_path)
        print(f"   Model cached. Home advantage: {model._home_advantage:.2f}x")
        print(f"   Dixon-Coles ρ: {model._rho:.4f}")

    # ── Add form features ──
    df = add_form_adjustment(df, window=5)

    # ── Teams ──
    if len(sys.argv) >= 3:
        home_team = sys.argv[1]
        away_team = sys.argv[2]
    else:
        home_team = "Flamengo"
        away_team = "Palmeiras"

    # ── Recent form for these teams ──
    home_matches = df[
        (df["home_team"] == home_team) | (df["away_team"] == home_team)
    ].tail(5)
    away_matches = df[
        (df["home_team"] == away_team) | (df["away_team"] == away_team)
    ].tail(5)

    home_form = None
    away_form = None
    if len(home_matches) >= 3:
        gf = home_matches.apply(
            lambda r: r["home_goals"] if r["home_team"] == home_team else r["away_goals"],
            axis=1,
        ).mean()
        ga = home_matches.apply(
            lambda r: r["away_goals"] if r["home_team"] == home_team else r["home_goals"],
            axis=1,
        ).mean()
        home_form = {"gf": gf, "ga": ga}

    if len(away_matches) >= 3:
        gf = away_matches.apply(
            lambda r: r["home_goals"] if r["home_team"] == away_team else r["away_goals"],
            axis=1,
        ).mean()
        ga = away_matches.apply(
            lambda r: r["away_goals"] if r["home_team"] == away_team else r["home_goals"],
            axis=1,
        ).mean()
        away_form = {"gf": gf, "ga": ga}

    # ── Predict: with and without form ──
    print(f"\n{'='*56}")
    print(f"  ⚽ {home_team} vs {away_team} — Brasileirão")
    print(f"{'='*56}")

    # No form (historical only)
    pred_base = model.predict_match(home_team, away_team, neutral=False)

    # With form
    pred_form = model.predict_match(
        home_team, away_team, neutral=False,
        home_form=home_form, away_form=away_form,
    )

    print(f"\n  {'':30s} {'Histórico':>12s} {'Com forma':>12s}")
    print(f"  {'─'*54}")
    print(f"  {'xG Casa':30s} {pred_base['expected_goals_home']:>12.2f} {pred_form['expected_goals_home']:>12.2f}")
    print(f"  {'xG Fora':30s} {pred_base['expected_goals_away']:>12.2f} {pred_form['expected_goals_away']:>12.2f}")
    print(f"  {'Vitória Casa':30s} {pred_base['home_win_probability']:>11.1%} {pred_form['home_win_probability']:>11.1%}")
    print(f"  {'Empate':30s} {pred_base['draw_probability']:>11.1%} {pred_form['draw_probability']:>11.1%}")
    print(f"  {'Vitória Fora':30s} {pred_base['away_win_probability']:>11.1%} {pred_form['away_win_probability']:>11.1%}")

    if home_form:
        print(f"\n  🔥 {home_team} form (last 5): GF {home_form['gf']:.1f}/game, GA {home_form['ga']:.1f}/game")
        hist_gf = model._attack.get(home_team, 1.0) * model._global_avg
        print(f"     Historical: GF {hist_gf:.1f}/game → form boost: {home_form['gf']/hist_gf:.1%}")

    if away_form:
        print(f"  🔥 {away_team} form (last 5): GF {away_form['gf']:.1f}/game, GA {away_form['ga']:.1f}/game")
        hist_gf = model._attack.get(away_team, 1.0) * model._global_avg
        print(f"     Historical: GF {hist_gf:.1f}/game → form boost: {away_form['gf']/hist_gf:.1%}")

    # ── Compare with Copa mode ──
    pred_neutral = model.predict_match(home_team, away_team, neutral=True)
    print(f"\n  📊 Impacto do mando de campo:")
    print(f"     Casa (xG): {pred_base['expected_goals_home']:.2f} → Neutro: {pred_neutral['expected_goals_home']:.2f}")
    print(f"     Fora (xG): {pred_base['expected_goals_away']:.2f} → Neutro: {pred_neutral['expected_goals_away']:.2f}")
    print(f"     Vitória casa: {pred_base['home_win_probability']:.1%} → Neutro: {pred_neutral['home_win_probability']:.1%}")


if __name__ == "__main__":
    main()