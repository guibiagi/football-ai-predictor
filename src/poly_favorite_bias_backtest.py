"""Crypto/Politics: favoritos YES ≥0.90 — taker buy vs maker simulado."""

from __future__ import annotations

from src.polymarket_fees import infer_category, net_return_taker, taker_fee_fraction
from src.polymarket_gamma_client import fetch_events_closed
from src.poly_market_utils import (
    mean_price_in_window,
    parse_kickoff,
    parse_token_yes,
    pnl_viable,
    price_history,
    summarize_returns,
    yes_won,
)


def _binary_markets(event: dict) -> list[dict]:
    markets = event.get("markets") or []
    if len(markets) == 1:
        return markets
    if len(markets) == 2:
        return [m for m in markets if "yes" in (m.get("outcome") or "").lower() or True][:1] or markets[:1]
    return []


def run_favorite_taker(
    tag: str,
    price_min: float = 0.90,
    holdout_frac: float = 0.15,
    max_events: int = 800,
) -> dict:
    events = fetch_events_closed(tag_slug=tag, max_pages=15)
    trades: list[dict] = []
    for event in events[:max_events]:
        for market in event.get("markets") or []:
            if len(event.get("markets") or []) > 3:
                continue  # skip multi-outcome except weather-style
            token = parse_token_yes(market)
            if not token:
                continue
            ko = parse_kickoff(event, market)
            if ko is None:
                continue
            if ko.tzinfo is None:
                from datetime import timezone
                ko = ko.replace(tzinfo=timezone.utc)
            end_ts = int(ko.timestamp())
            hist = price_history(token, end_ts - 14 * 86400, end_ts)
            if len(hist) < 5:
                continue
            p = mean_price_in_window(hist, end_ts, hours_before=24.0)
            if p is None or p < price_min:
                continue
            yw = yes_won(market)
            if yw is None:
                continue
            cat = infer_category(event, market)
            gross = (1.0 - p) / p if yw else -1.0
            net = net_return_taker(gross, p, cat)
            trades.append({"date": str(ko.date()), "net": net, "gross": gross, "p": p})
    trades.sort(key=lambda t: t["date"])
    cut = int(len(trades) * (1 - holdout_frac))
    dev, hold = trades[:cut], trades[cut:]
    return {
        "tag": tag,
        "mode": "taker_buy_yes",
        "price_min": price_min,
        "n_total": len(trades),
        "dev": summarize_returns([t["net"] for t in dev]),
        "holdout": summarize_returns([t["net"] for t in hold]),
        "holdout_gross": summarize_returns([t["gross"] for t in hold]),
        "pnl_viable_holdout": pnl_viable(summarize_returns([t["net"] for t in hold])),
    }


def run_favorite_maker_sim(
    tag: str,
    price_min: float = 0.90,
    half_spread: float = 0.01,
    holdout_frac: float = 0.15,
    max_events: int = 600,
) -> dict:
    """Maker: posta bid em price_min - spread; fill se histórico cruza."""
    events = fetch_events_closed(tag_slug=tag, max_pages=35)
    trades: list[dict] = []
    for event in events[:max_events]:
        for market in event.get("markets") or []:
            if len(event.get("markets") or []) > 3:
                continue
            token = parse_token_yes(market)
            if not token:
                continue
            ko = parse_kickoff(event, market)
            if ko is None:
                continue
            from datetime import timezone
            if ko.tzinfo is None:
                ko = ko.replace(tzinfo=timezone.utc)
            end_ts = int(ko.timestamp())
            hist = sorted(
                price_history(token, end_ts - 14 * 86400, end_ts),
                key=lambda h: h.get("t", 0),
            )
            if len(hist) < 5:
                continue
            bid = price_min - half_spread
            fill_p = None
            for pt in hist[-48:]:
                if float(pt.get("p", 0)) <= bid:
                    fill_p = bid
                    break
            if fill_p is None:
                continue
            yw = yes_won(market)
            if yw is None:
                continue
            gross = (1.0 - fill_p) / fill_p if yw else -1.0
            # maker: sem taxa taker na entrada
            trades.append({"date": str(ko.date()), "net": gross, "gross": gross})
    trades.sort(key=lambda t: t["date"])
    cut = int(len(trades) * (1 - holdout_frac))
    dev, hold = trades[:cut], trades[cut:]
    return {
        "tag": tag,
        "mode": "maker_bid",
        "price_min": price_min,
        "half_spread": half_spread,
        "n_total": len(trades),
        "dev": summarize_returns([t["net"] for t in dev]),
        "holdout": summarize_returns([t["net"] for t in hold]),
        "pnl_viable_holdout": pnl_viable(summarize_returns([t["net"] for t in hold])),
    }
