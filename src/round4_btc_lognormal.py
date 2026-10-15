"""H4-BTC: faixas BTC/ETH vs lognormal (Deribit IV ou vol realizada)."""

from __future__ import annotations

import math
import urllib.request
import json
from datetime import timezone

import numpy as np
import pandas as pd

from src.hypothesis_engine import HOLDOUT_DAYS
from src.poly_hypothesis_registry import HypothesisRegistry
from src.poly_market_utils import mean_price_in_window, parse_kickoff, parse_token_yes, price_history, yes_won
from src.polymarket_fees import infer_category, net_return_taker
from src.polymarket_gamma_client import load_merged_cached_events

DERIBIT = "https://www.deribit.com/api/v2/public/get_volatility_index_data"


def _norm_cdf(x: float) -> float:
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))


def lognormal_above_prob(spot: float, strike: float, sigma_ann: float, days: float) -> float:
    if spot <= 0 or strike <= 0 or sigma_ann <= 0 or days <= 0:
        return 0.5
    t = days / 365.0
    vol = sigma_ann * math.sqrt(t)
    if vol <= 0:
        return 0.5
    d2 = (math.log(spot / strike) - 0.5 * vol * vol) / vol
    return _norm_cdf(d2)


def _deribit_iv_at(ts_sec: int, currency: str = "BTC") -> float | None:
    url = f"{DERIBIT}?currency={currency}&resolution=3600&start_timestamp={(ts_sec - 30 * 86400) * 1000}"
    try:
        req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
        with urllib.request.urlopen(req, timeout=25) as resp:
            data = json.load(resp)
        rows = data.get("result", {}).get("data") or []
        iv = None
        for row in rows:
            if row[0] <= ts_sec * 1000:
                iv = float(row[4]) / 100.0
        return iv
    except Exception:
        return None


def run_btc_eth_grid(
    registry: HypothesisRegistry,
    holdout_cut: pd.Timestamp,
    events: list[dict] | None = None,
    cache_only: bool = True,
    edge_min: float = 0.05,
) -> None:
    events = events or load_merged_cached_events()
    holds = {"buy_yes_under": [], "buy_yes_over": []}
    devs = {k: [] for k in holds}

    for event in events:
        title = (event.get("title") or "").lower()
        if "bitcoin" not in title and "ethereum" not in title and "btc" not in title:
            continue
        cur = "ETH" if "ethereum" in title or "eth" in title else "BTC"
        for market in event.get("markets") or []:
            q = (market.get("question") or market.get("groupItemTitle") or "").lower()
            if "above" not in q and "below" not in q and "$" not in q:
                continue
            token = parse_token_yes(market)
            if not token:
                continue
            ko = parse_kickoff(event, market)
            if ko is None:
                continue
            if ko.tzinfo is None:
                ko = ko.replace(tzinfo=timezone.utc)
            end_ts = int(ko.timestamp())
            hist = price_history(token, end_ts - 7 * 86400, end_ts, cache_only=cache_only)
            p_poly = mean_price_in_window(hist, end_ts, 12.0) if hist else None
            if p_poly is None:
                continue
            yw = yes_won(market)
            if yw is None:
                continue
            # Strike parsing rudimentar
            strike = None
            for part in q.replace(",", "").split():
                if part.startswith("$"):
                    try:
                        strike = float(part.replace("$", ""))
                    except ValueError:
                        pass
            if strike is None:
                continue
            iv = _deribit_iv_at(end_ts, cur) or 0.55
            spot = strike  # sem spot histórico: neutro no strike
            p_model = lognormal_above_prob(spot, strike, iv, 7.0)
            if "below" in q:
                p_model = 1.0 - p_model
            edge = p_model - p_poly
            exec_p = p_poly + 0.01
            gross = (1.0 - exec_p) / exec_p if yw else -1.0
            net = net_return_taker(gross, exec_p, infer_category(event, market))
            kt = pd.Timestamp(ko)
            if kt.tzinfo is not None:
                kt = kt.tz_convert(None)
            bucket = "buy_yes_under" if edge > edge_min else ("buy_yes_over" if edge < -edge_min else None)
            if bucket is None:
                continue
            if kt >= holdout_cut:
                holds[bucket].append(net)
            else:
                devs[bucket].append(net)

    registry.add(
        "H4-BTC-ln-buy-yes-cheap",
        "btc_lognormal",
        f"Compra YES quando p_poly < p_lognormal(Deribit IV)−{edge_min:.0%}",
        holds["buy_yes_under"],
        devs["buy_yes_under"],
    )
    registry.add(
        "H4-BTC-ln-buy-yes-rich",
        "btc_lognormal",
        f"Compra YES quando p_poly > p_lognormal+{edge_min:.0%} (fade modelo)",
        holds["buy_yes_over"],
        devs["buy_yes_over"],
    )
