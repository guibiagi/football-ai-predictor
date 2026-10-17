"""H4-LAG: atraso Poly vs movimento Pinnacle; reversão após salto."""

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
    fetch_price_history,
    is_match_event,
    match_event_to_row,
    outcome_roles_from_markets,
    parse_token_ids,
)
from src.poly_market_utils import mean_price_in_window
from src.polymarket_fees import net_return_taker
from src.polymarket_gamma_client import load_merged_cached_events


def run_pin_lag_tests(
    df: pd.DataFrame,
    registry: HypothesisRegistry,
    events: list[dict] | None = None,
    cache_only: bool = True,
) -> None:
    events = events or [e for e in load_merged_cached_events() if is_match_event(e)]
    holdout_cut = df.date.max() - pd.Timedelta(days=HOLDOUT_DAYS)
    teams_idx: dict[str, dict[str, str]] = {}

    lag_hold: dict[str, list[float]] = {"mom_2c": [], "mom_3c": [], "rev_8c": []}
    lag_dev: dict[str, list[float]] = {k: [] for k in lag_hold}

    for event in events:
        kickoff = _parse_kickoff(event)
        if kickoff is None:
            continue
        row = match_event_to_row(event, df, teams_idx)
        if row is None:
            continue
        markets = event.get("markets") or []
        roles = outcome_roles_from_markets(markets, row.home_team, row.away_team)
        winner = _winner_from_markets(markets, roles) if roles else None
        if roles is None or winner is None:
            continue
        pin_open = _pin_probs(row, use_close=False)
        pin_close = _pin_probs(row, use_close=True)
        if pin_open is None or pin_close is None:
            continue
        fav_i = int(np.argmax(pin_open))
        delta_pin = float(pin_close[fav_i] - pin_open[fav_i])
        market = markets[fav_i]
        token = parse_token_ids(market)
        if not token:
            continue
        hist = fetch_price_history(token, kickoff, cache_only=cache_only)
        if len(hist) < 5:
            continue
        if kickoff.tzinfo is None:
            kickoff = kickoff.replace(tzinfo=timezone.utc)
        end_ts = int(kickoff.timestamp())
        p1 = mean_price_in_window(hist, end_ts, 6.0)
        p2 = mean_price_in_window(hist, end_ts, 1.0)
        if p1 is None or p2 is None:
            continue
        won = winner == roles[fav_i]
        exec_p = p2 + 0.01

        def _push(key: str, active: bool) -> None:
            if not active:
                return
            gross = (1.0 - exec_p) / exec_p if won else -1.0
            net = net_return_taker(gross, exec_p, "sports")
            kt = pd.Timestamp(kickoff)
            if kt.tzinfo is not None:
                kt = kt.tz_convert(None)
            if kt >= holdout_cut:
                lag_hold[key].append(net)
            else:
                lag_dev[key].append(net)

        # Momentum: Pin subiu, Poly em T-1h ainda barato vs close Pin
        _push("mom_2c", delta_pin >= 0.02 and p2 < pin_close[fav_i] - 0.02)
        _push("mom_3c", delta_pin >= 0.03 and p2 < pin_close[fav_i] - 0.03)
        # Reversão: salto Poly 6h→1h sem Pin acompanhar
        jump = p2 - p1
        _push("rev_8c", abs(jump) >= 0.08 and delta_pin * jump < 0)

    registry.add(
        "H4-LAG-mom-pin2c",
        "pin_lag",
        "Pin +2pp no fav; compra Poly T-1h se ainda < Pin_close−2¢",
        lag_hold["mom_2c"],
        lag_dev["mom_2c"],
    )
    registry.add(
        "H4-LAG-mom-pin3c",
        "pin_lag",
        "Pin +3pp; Poly defasado vs close",
        lag_hold["mom_3c"],
        lag_dev["mom_3c"],
    )
    registry.add(
        "H4-LAG-rev-poly8c",
        "pin_lag",
        "Reversão: salto Poly ≥8¢ entre T-6h e T-1h contra movimento Pin",
        lag_hold["rev_8c"],
        lag_dev["rev_8c"],
    )
