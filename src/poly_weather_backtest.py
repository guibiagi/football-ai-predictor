"""Clima: Buy NO em favoritos YES 0.75–0.95, 1–12h antes do pico (arXiv:2610.13593)."""

from __future__ import annotations

import re
from datetime import datetime, timezone

from src.polymarket_fees import infer_category, net_return_taker, taker_fee_fraction
from src.polymarket_gamma_client import fetch_events_closed
from src.poly_market_utils import (
    max_price_in_range,
    mean_price_in_window,
    parse_kickoff,
    parse_token_yes,
    pnl_viable,
    price_history,
    summarize_returns,
    yes_won,
)

TEMP_TITLE = "highest temperature"
# Pico diário: proxy = endDate 16:00 local ≈ usar endDate noon UTC + offset omitted → endDate 18:00 UTC
PEAK_HOUR_UTC = 18


def is_daily_max_temp_event(event: dict) -> bool:
    title = (event.get("title") or "").lower()
    return TEMP_TITLE in title and len(event.get("markets") or []) >= 5


def event_peak_ts(event: dict) -> int | None:
    ko = parse_kickoff(event)
    if ko is None:
        return None
    if ko.tzinfo is None:
        ko = ko.replace(tzinfo=timezone.utc)
    # pico da máxima diária ≈ tarde no dia do evento
    peak = ko.replace(hour=PEAK_HOUR_UTC, minute=0, second=0, microsecond=0)
    return int(peak.timestamp())


def open_meteo_fair_yes(city: str, date_str: str, bucket_question: str) -> float | None:
    """Opcional: prevê P(bucket) via forecast archive — simplificado: skip se sem coords."""
    return None  # placeholder: integração futura


def run_weather_buy_no(
    holdout_frac: float = 0.15,
    hours_before: float = 6.0,
    fav_lo: float = 0.75,
    fav_hi: float = 0.95,
    max_events: int | None = None,
    spread_cents: float = 0.06,
    min_depth_usd: float = 60.0,
) -> dict:
    events = fetch_events_closed(tag_slug="temperature", max_pages=100, title_filter="Highest temperature")
    events = [e for e in events if is_daily_max_temp_event(e)]
    events.sort(key=lambda e: e.get("endDate") or "")
    if max_events:
        events = events[-max_events:]

    records: list[dict] = []
    for event in events:
        peak_ts = event_peak_ts(event)
        if peak_ts is None:
            continue
        cat = infer_category(event)
        hist_start = peak_ts - 72 * 3600
        for market in event.get("markets") or []:
            token = parse_token_yes(market)
            if not token:
                continue
            hist = price_history(token, hist_start, peak_ts)
            if len(hist) < 3:
                continue
            p_yes = max_price_in_range(hist, peak_ts, hours_before_lo=1.0, hours_before_hi=12.0)
            if p_yes is None or not (fav_lo <= p_yes <= fav_hi):
                continue
            vol = float(market.get("volume") or 0)
            if vol < min_depth_usd:
                continue
            # spread 4–10¢: piora preço efetivo do NO (taker)
            p_no = min(0.99, (1.0 - p_yes) + spread_cents)
            yw = yes_won(market)
            if yw is None:
                continue
            gross = (1.0 - p_no) / p_no if not yw else -1.0
            net = net_return_taker(gross, p_no, cat)
            records.append({
                "date": (event.get("endDate") or "")[:10],
                "p_yes": p_yes,
                "gross": gross,
                "net": net,
                "fee_frac": taker_fee_fraction(p_no, cat),
            })
        if len(records) > 5000:
            break

    if not records:
        return {"error": "sem trades", "n": 0}

    records.sort(key=lambda r: r["date"])
    cut = int(len(records) * (1 - holdout_frac))
    dev, hold = records[:cut], records[cut:]
    return {
        "hypothesis": "W1_weather_buy_no_favorite",
        "hours_before_peak": hours_before,
        "fav_band": [fav_lo, fav_hi],
        "spread_cents": spread_cents,
        "min_depth_usd": min_depth_usd,
        "n_total": len(records),
        "dev": summarize_returns([r["net"] for r in dev]),
        "holdout": summarize_returns([r["net"] for r in hold]),
        "dev_gross": summarize_returns([r["gross"] for r in dev]),
        "holdout_gross": summarize_returns([r["gross"] for r in hold]),
        "pnl_viable_holdout": pnl_viable(summarize_returns([r["net"] for r in hold])),
    }
