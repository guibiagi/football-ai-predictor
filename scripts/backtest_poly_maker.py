#!/usr/bin/env python3
"""Backtest maker Polymarket (CLOB prices-history) vs fair Pinnacle.

Uso:
  python scripts/backtest_poly_maker.py
  python scripts/backtest_poly_maker.py --refresh-events
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import date, datetime, timezone
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.hypothesis_engine import HOLDOUT_DAYS
from src.poly_maker_backtest import (
    LEAGUE_TAG_TO_COMP,
    fetch_closed_soccer_events,
    run_maker_backtest,
    summarize_maker_fills,
)

EVENTS_CACHE = Path("data/polymarket_closed_matches.json")
REPORTS = Path("reports")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--refresh-events", action="store_true")
    ap.add_argument("--max-per-tag", type=int, default=150)
    ap.add_argument("--max-events", type=int, default=None, help="limite de fills (None = todos)")
    args = ap.parse_args()

    tags = list(LEAGUE_TAG_TO_COMP.keys()) + ["soccer"]
    if args.refresh_events or not EVENTS_CACHE.exists():
        print(f"Buscando eventos fechados ({len(tags)} tags)...")
        events = fetch_closed_soccer_events(tags, max_per_tag=args.max_per_tag)
        EVENTS_CACHE.parent.mkdir(parents=True, exist_ok=True)
        EVENTS_CACHE.write_text(json.dumps(events, ensure_ascii=False), encoding="utf-8")
        print(f"  {len(events)} eventos salvos em {EVENTS_CACHE}")
    else:
        events = json.loads(EVENTS_CACHE.read_text(encoding="utf-8"))
        print(f"Cache: {len(events)} eventos")

    df = pd.read_csv("data/matches_xg.csv", parse_dates=["date"], low_memory=False)
    holdout_cut = df.date.max() - pd.Timedelta(days=HOLDOUT_DAYS)

    scenarios = [
        ("H13_maker_pin_open_spread2c", False, 0.02),
        ("H14_maker_pin_open_spread1c", False, 0.01),
        ("H15_maker_pin_open_spread3c", False, 0.03),
        ("H16_maker_pin_close_oracle_spread2c", True, 0.02),
    ]

    results = []
    for sid, fair_close, spread in scenarios:
        print(f"\n=== {sid} (fair={'psc' if fair_close else 'ps'}, spread={spread}) ===")
        fills = run_maker_backtest(
            events,
            df,
            half_spread=spread,
            fair_use_close=fair_close,
            min_volume=1000.0,
            max_events=args.max_events,
        )
        summary = summarize_maker_fills(fills, holdout_cut)
        summary["id"] = sid
        summary["fair"] = "Pinnacle close (oracle — só sensibilidade)" if fair_close else "Pinnacle open"
        summary["half_spread"] = spread
        summary["n_fills_total"] = len(fills)
        results.append(summary)
        h = summary["holdout"]
        print(
            f"  fills total={len(fills)} | holdout n={h['n']} CLV={h.get('clv_mean')} "
            f"ROI={h.get('roi')} viable={summary['viable']}"
        )

    REPORTS.mkdir(exist_ok=True)
    out = REPORTS / f"poly_maker_{date.today().strftime('%Y%m%d')}.json"
    payload = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "meta": {
            "clob": "prices-history fidelity=60",
            "holdout_days": HOLDOUT_DAYS,
            "holdout_cut": str(holdout_cut.date()),
            "fill_rule": "primeiro toque: p<=bid compra, p>=ask venda",
            "rewards": "estimativa conservadora (5% do pool diário se spread elegível)",
            "events_cache": str(EVENTS_CACHE),
        },
        "scenarios": results,
    }
    out.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\nRelatório: {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
