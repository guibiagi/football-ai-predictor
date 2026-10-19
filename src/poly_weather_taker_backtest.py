"""Clima via trades taker (data-api), estilo paper arXiv 2610.13593."""

from __future__ import annotations

import json

from src.polymarket_fees import infer_category, net_return_taker
from src.polymarket_gamma_client import fetch_events_closed
from src.poly_trades_client import fetch_trades
from src.poly_weather_backtest import event_peak_ts, is_daily_max_temp_event
from datetime import datetime, timezone

import pandas as pd

from src.hypothesis_engine import HOLDOUT_DAYS
from src.poly_market_utils import pnl_viable, summarize_returns


def run_weather_taker_trades(
    holdout_frac: float | None = None,
    holdout_cut: pd.Timestamp | None = None,
    fav_lo: float = 0.75,
    fav_hi: float = 0.95,
    hours_lo: float = 1.0,
    hours_hi: float = 12.0,
    spread_cents: float = 0.06,
    min_depth_usd: float = 60.0,
    max_events: int = 200,
    cache_only: bool = False,
) -> dict:
    events = fetch_events_closed(
        tag_slug="temperature", max_pages=150, title_filter="Highest temperature",
    )
    events = [e for e in events if is_daily_max_temp_event(e)]
    records: list[dict] = []

    for event in events[:max_events]:
        peak_ts = event_peak_ts(event)
        if peak_ts is None:
            continue
        cat = infer_category(event)
        t_lo = peak_ts - int(hours_hi * 3600)
        t_hi = peak_ts - int(hours_lo * 3600)

        for market in event.get("markets") or []:
            if float(market.get("volume") or 0) < min_depth_usd:
                continue
            cid = market.get("conditionId")
            if not cid:
                continue
            try:
                yw = float(json.loads(market["outcomePrices"])[0]) > 0.9
            except Exception:
                continue
            trades = fetch_trades(cid, limit=500, max_pages=10, cache_only=cache_only)
            if not trades:
                continue
            # Taker BUY YES em favorito na janela
            for tr in trades:
                ts = int(tr.get("timestamp") or 0)
                if ts < t_lo or ts > t_hi:
                    continue
                if tr.get("side") != "BUY" or tr.get("outcome") != "Yes":
                    continue
                p_yes = float(tr.get("price") or 0)
                if not (fav_lo <= p_yes <= fav_hi):
                    continue
                size = float(tr.get("size") or 0)
                if size * p_yes < 1.0:
                    continue
                p_no = min(0.99, (1.0 - p_yes) + spread_cents)
                gross = (1.0 - p_no) / p_no if not yw else -1.0
                net = net_return_taker(gross, p_no, cat)
                records.append({
                    "date": (event.get("endDate") or "")[:10],
                    "p_yes": p_yes,
                    "net": net,
                    "gross": gross,
                })
                break  # um sinal por bucket/evento na janela

    if not records:
        return {"hypothesis": "W2_weather_taker_trades", "n": 0, "error": "sem trades"}

    records.sort(key=lambda r: r["date"])
    if holdout_cut is None:
        if holdout_frac is None:
            holdout_frac = 0.15
        cut = int(len(records) * (1 - holdout_frac))
        dev, hold = records[:cut], records[cut:]
    else:
        dev = [r for r in records if r["date"] < str(holdout_cut.date())]
        hold = [r for r in records if r["date"] >= str(holdout_cut.date())]
    hold_s = summarize_returns([r["net"] for r in hold])
    return {
        "hypothesis": "W2_weather_taker_trades",
        "spread_cents": spread_cents,
        "min_depth_usd": min_depth_usd,
        "n_total": len(records),
        "dev": summarize_returns([r["net"] for r in dev]),
        "holdout": hold_s,
        "holdout_gross": summarize_returns([r["gross"] for r in hold]),
        "pnl_viable_holdout": pnl_viable(hold_s),
    }
