#!/usr/bin/env python3
"""Execução rápida H-RN1 + H-COH (usa cache de eventos)."""

import json
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.poly_coherence_backtest import run_coherence_backtest
from src.poly_rn1_backtest import run_rn1_offset_grid

def main():
    df = pd.read_csv("data/matches_xg.csv", parse_dates=["date"], low_memory=False)
    cache = Path("data/polymarket_closed_matches.json")
    events = json.loads(cache.read_text()) if cache.exists() else None
    rn1 = run_rn1_offset_grid(df, events=events)
    coh = run_coherence_backtest(df)
    out = {"rn1": rn1, "coh": coh}
    p = Path("reports/rn1_coh_quick.json")
    p.write_text(json.dumps(out, indent=2, default=str), encoding="utf-8")
    for r in rn1:
        h = r["aggregate"]["holdout"]
        print("RN1 off", r["offset"], "fills", h.get("n_fills"), "fill_rate", h.get("fill_rate"),
              "roi", h.get("roi_mean"), "ci", h.get("roi_ci95"), "adverse", h.get("adverse_gap"))
        band = r.get("by_price_band", {}).get("0.60-0.80", {}).get("holdout", {})
        print("  band 0.60-0.80 hold", band)
    print("COH trades", coh.get("n_trades"), "hold", coh.get("holdout"))
    print("wrote", p)

if __name__ == "__main__":
    main()
