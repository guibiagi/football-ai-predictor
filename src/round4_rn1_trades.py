"""H4-RN1-TRADES: fill maker via trades data-api (não prices-history)."""

from __future__ import annotations

from datetime import timezone

import numpy as np
import pandas as pd

from src.hypothesis_engine import HOLDOUT_DAYS
from src.poly_hypothesis_registry import HypothesisRegistry
from src.poly_maker_backtest import (
    _parse_kickoff,
    _pin_probs,
    _winner_from_markets,
    is_match_event,
    match_event_to_row,
    outcome_roles_from_markets,
    parse_token_ids,
)
from src.poly_rewards import estimate_maker_reward_usd, reward_roi_addition
from src.poly_rn1_backtest import _settlement_buy_yes
from src.poly_trades_client import fetch_trades
from src.polymarket_gamma_client import load_merged_cached_events

FAIR_LO, FAIR_HI = 0.60, 0.80
OFFSETS = [0.02, 0.025, 0.03]


def run_rn1_trades_fills(
    df: pd.DataFrame,
    registry: HypothesisRegistry,
    events: list[dict] | None = None,
    cache_only: bool = True,
) -> None:
    events = events or [e for e in load_merged_cached_events() if is_match_event(e)]
    holdout_cut = df.date.max() - pd.Timedelta(days=HOLDOUT_DAYS)
    teams_idx: dict[str, dict[str, str]] = {}
    buckets: dict[float, list[float]] = {o: [] for o in OFFSETS}
    buckets_dev: dict[float, list[float]] = {o: [] for o in OFFSETS}

    for event in events:
        markets = event.get("markets") or []
        kickoff = _parse_kickoff(event)
        if kickoff is None:
            continue
        row = match_event_to_row(event, df, teams_idx)
        if row is None:
            continue
        roles = outcome_roles_from_markets(markets, row.home_team, row.away_team)
        winner = _winner_from_markets(markets, roles) if roles else None
        if roles is None or winner is None:
            continue
        pin_open = _pin_probs(row, use_close=False)
        if pin_open is None:
            continue
        fav_i = int(np.argmax(pin_open))
        fair = float(pin_open[fav_i])
        if not (FAIR_LO <= fair <= FAIR_HI):
            continue
        market = markets[fav_i]
        cid = market.get("conditionId")
        if not cid:
            continue
        trades = fetch_trades(cid, limit=300, max_pages=5, cache_only=cache_only)
        if not trades:
            continue
        if kickoff.tzinfo is None:
            kickoff = kickoff.replace(tzinfo=timezone.utc)
        end_ts = int(kickoff.timestamp())
        start_ts = end_ts - 72 * 3600
        fav_won = winner == roles[fav_i]

        for off in OFFSETS:
            bid = round(fair - off, 3)
            filled = False
            fill_p = bid
            for tr in trades:
                ts = int(tr.get("timestamp") or 0)
                if ts < start_ts or ts > end_ts:
                    continue
                if tr.get("side") != "SELL" or tr.get("outcome") != "Yes":
                    continue
                px = float(tr.get("price") or 1)
                if px <= bid:
                    filled = True
                    fill_p = px
                    break
            if not filled:
                continue
            gross = _settlement_buy_yes(fill_p, fav_won)
            rew = estimate_maker_reward_usd(market, 24.0, size_usd=50.0, half_spread_cents=off * 100)
            roi = gross + reward_roi_addition(rew, fill_p)
            kt = pd.Timestamp(kickoff)
            if kt.tzinfo is not None:
                kt = kt.tz_convert(None)
            if kt >= holdout_cut:
                buckets[off].append(roi)
            else:
                buckets_dev[off].append(roi)

    for off in OFFSETS:
        registry.add(
            hid=f"H4-RN1-trades-off{int(off*1000):03d}",
            family="rn1_trades",
            description=f"RN1 maker fill via trades SELL≤bid (offset {off:.3f})",
            holdout_returns=buckets[off],
            dev_returns=buckets_dev[off],
            offset=off,
        )
