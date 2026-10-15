"""BTC Polymarket vs vol implícita Deribit (público)."""

from __future__ import annotations

import json
import urllib.request
from datetime import datetime, timezone

from src.polymarket_gamma_client import fetch_events_closed
from src.polymarket_fees import infer_category, net_return_taker
from src.poly_market_utils import (
    mean_price_in_window,
    parse_kickoff,
    parse_token_yes,
    pnl_viable,
    price_history,
    summarize_returns,
    yes_won,
)

DERIBIT = "https://www.deribit.com/api/v2"


def deribit_historical_iv(currency: str = "BTC", start_ts_ms: int | None = None) -> list[dict]:
    """Vol index histórico (DVOL) — amostra recente."""
    url = f"{DERIBIT}/public/get_volatility_index_data?currency={currency}&resolution=3600"
    if start_ts_ms:
        url += f"&start_timestamp={start_ts_ms}"
    try:
        req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
        with urllib.request.urlopen(req, timeout=30) as resp:
            data = json.load(resp)
        return data.get("result", {}).get("data") or []
    except Exception:
        return []


def run_btc_above_markets(holdout_frac: float = 0.2) -> dict:
    events = fetch_events_closed(tag_slug="bitcoin", max_pages=30)
    events += fetch_events_closed(tag_slug="crypto", max_pages=15)
    seen = set()
    trades: list[dict] = []
    iv_cache: list[dict] = []

    for event in events:
        slug = event.get("slug")
        if slug in seen:
            continue
        seen.add(slug)
        title = (event.get("title") or "").lower()
        if "above" not in title and "below" not in title and "bitcoin" not in title:
            continue
        for market in event.get("markets") or []:
            token = parse_token_yes(market)
            if not token:
                continue
            ko = parse_kickoff(event, market)
            if ko is None:
                continue
            if ko.tzinfo is None:
                ko = ko.replace(tzinfo=timezone.utc)
            end_ts = int(ko.timestamp())
            hist = price_history(token, end_ts - 7 * 86400, end_ts)
            p = mean_price_in_window(hist, end_ts, hours_before=12.0)
            if p is None:
                continue
            yw = yes_won(market)
            if yw is None:
                continue
            # Estratégia simples: comprar YES quando p<0.45 e IV alta (proxy: não temos IV histórica alinhada)
            if not iv_cache:
                iv_cache = deribit_historical_iv(start_ts_ms=(end_ts - 30 * 86400) * 1000)
            iv_row = None
            for row in iv_cache:
                if row[0] <= end_ts * 1000:
                    iv_row = row
            iv = float(iv_row[4]) if iv_row and len(iv_row) > 4 else None
            if iv is None or p >= 0.45:
                continue
            cat = infer_category(event, market)
            gross = (1.0 - p) / p if yw else -1.0
            net = net_return_taker(gross, p, cat)
            trades.append({"date": str(ko.date()), "net": net, "iv": iv})
    trades.sort(key=lambda t: t["date"])
    cut = int(len(trades) * (1 - holdout_frac))
    dev, hold = trades[:cut], trades[cut:]
    return {
        "hypothesis": "B1_btc_low_prob_vs_iv",
        "note": "Amostra limitada; IV Deribit alinhamento aproximado",
        "n_total": len(trades),
        "dev": summarize_returns([t["net"] for t in dev]),
        "holdout": summarize_returns([t["net"] for t in hold]),
        "pnl_viable_holdout": pnl_viable(summarize_returns([t["net"] for t in hold])),
    }
