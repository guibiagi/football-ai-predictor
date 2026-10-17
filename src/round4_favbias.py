"""H4-FAV: viés favorito/azarão por categoria e faixa de preço."""

from __future__ import annotations

import json
from datetime import timezone

import pandas as pd

from src.hypothesis_engine import HOLDOUT_DAYS
from src.poly_hypothesis_registry import HypothesisRegistry
from src.poly_market_utils import mean_price_in_window, parse_kickoff, parse_token_yes, price_history, yes_won
from src.polymarket_fees import infer_category, net_return_taker
from src.polymarket_gamma_client import load_merged_cached_events

CATEGORY_TAGS = {
    "sports": ("soccer", "nba", "nfl", "ufc", "mma", "atp", "wta", "mlb", "nhl", "epl"),
    "crypto": ("crypto", "bitcoin", "ethereum"),
    "politics": ("politics", "elections", "us-politics"),
    "culture": ("pop-culture", "entertainment", "music", "movies"),
    "climate": ("temperature", "weather"),
}

PRICE_BUCKETS = [(0.55, 0.65), (0.65, 0.75), (0.75, 0.85), (0.85, 0.95)]
LONGSHOT_HI = 0.12


def _event_category(event: dict) -> str:
    slugs = {t.get("slug", "").lower() for t in (event.get("tags") or []) if isinstance(t, dict)}
    for cat, tags in CATEGORY_TAGS.items():
        if slugs & set(tags):
            return cat
    title = (event.get("title") or "").lower()
    if "temperature" in title:
        return "climate"
    if "bitcoin" in title or "ethereum" in title:
        return "crypto"
    if " vs " in title:
        return "sports"
    if any(x in title for x in ("election", "president", "trump", "biden")):
        return "politics"
    return "other"


def run_favbias_grid(
    registry: HypothesisRegistry,
    events: list[dict] | None = None,
    cache_only: bool = True,
    holdout_cut: pd.Timestamp | None = None,
    max_events_per_cat: int = 2500,
) -> None:
    events = events or load_merged_cached_events()
    if holdout_cut is None:
        holdout_cut = pd.Timestamp("2024-10-01")  # placeholder; overridden by caller

    holds: dict[tuple[str, str, tuple], list[float]] = {}
    devs: dict[tuple[str, str, tuple], list[float]] = {}

    per_cat: dict[str, int] = {c: 0 for c in CATEGORY_TAGS}
    per_cat["other"] = 0

    for event in events:
        cat = _event_category(event)
        if per_cat.get(cat, 0) >= max_events_per_cat:
            continue
        per_cat[cat] = per_cat.get(cat, 0) + 1
        markets = event.get("markets") or []
        if len(markets) > 4 or len(markets) == 0:
            continue
        for market in markets[:1] if len(markets) > 1 else markets:
            token = parse_token_yes(market)
            if not token:
                continue
            ko = parse_kickoff(event, market)
            if ko is None:
                continue
            if ko.tzinfo is None:
                ko = ko.replace(tzinfo=timezone.utc)
            end_ts = int(ko.timestamp())
            hist = price_history(token, end_ts - 14 * 86400, end_ts, cache_only=cache_only)
            if len(hist) < 3:
                continue
            p = mean_price_in_window(hist, end_ts, 24.0)
            if p is None:
                continue
            yw = yes_won(market)
            if yw is None:
                continue
            fee_cat = infer_category(event, market)

            def store(side: str, bucket: tuple, gross: float, price: float) -> None:
                net = net_return_taker(gross, price, fee_cat)
                key = (cat, side, bucket)
                kt = pd.Timestamp(ko)
                if kt.tzinfo is not None:
                    kt = kt.tz_convert(None)
                if kt >= holdout_cut:
                    holds.setdefault(key, []).append(net)
                else:
                    devs.setdefault(key, []).append(net)

            for lo, hi in PRICE_BUCKETS:
                if lo <= p < hi:
                    gross_yes = (1.0 - p) / p if yw else -1.0
                    store("buy_yes", (lo, hi), gross_yes, p)
                    break
            if cat == "culture" and p <= LONGSHOT_HI:
                p_no = min(0.99, 1.0 - p + 0.02)
                gross_no = (1.0 - p_no) / p_no if not yw else -1.0
                store("buy_no_long", (0.0, LONGSHOT_HI), gross_no, p_no)

    planned: set[tuple[str, str, tuple]] = set()
    for cat in list(CATEGORY_TAGS) + ["other"]:
        for lo, hi in PRICE_BUCKETS:
            planned.add((cat, "buy_yes", (lo, hi)))
        if cat == "culture":
            planned.add((cat, "buy_no_long", (0.0, LONGSHOT_HI)))

    for key in planned:
        rets = holds.get(key, [])
        cat, side, bucket = key
        registry.add(
            hid=f"H4-FAV-{cat}-{side}-{bucket[0]:.2f}-{bucket[1]:.2f}",
            family="favorite_bias",
            description=f"Taker {side} cat={cat} faixa YES~{bucket}",
            holdout_returns=rets,
            dev_returns=devs.get(key, []),
            category=cat,
            side=side,
            bucket=bucket,
        )
