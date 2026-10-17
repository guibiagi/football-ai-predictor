#!/usr/bin/env python3
"""Junta colunas Pinnacle/Max abertura O/U ao matches_xg.csv a partir do football-data."""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from update_data import collect  # noqa: E402

KEY = ["date", "competition", "home_team", "away_team"]
COLS = [
    "ou_ps_over", "ou_ps_under",
    "ou_max_over", "ou_max_under",
    "ou_avg_over", "ou_avg_under",
]


def main() -> int:
    target = ROOT / "data" / "matches_xg.csv"
    if not target.exists():
        print(f"ausente: {target}")
        return 1
    print("Baixando football-data.co.uk (pode levar alguns minutos)...")
    fresh = collect(verbose=True)
    sub = fresh[KEY + [c for c in COLS if c in fresh.columns]].drop_duplicates(KEY)
    base = pd.read_csv(target, parse_dates=["date"], low_memory=False)
    merged = base.drop(columns=[c for c in COLS if c in base.columns], errors="ignore")
    merged = merged.merge(sub, on=KEY, how="left")
    for c in COLS:
        if c not in merged.columns:
            merged[c] = float("nan")
    merged.to_csv(target, index=False)
    n = merged["ou_ps_over"].notna().sum()
    print(f"Atualizado {target}: {n}/{len(merged)} linhas com ou_ps_over")
    fb = ROOT / "data" / "matches_football.csv"
    if fb.exists():
        fresh.to_csv(fb, index=False)
        print(f"Atualizado {fb}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
