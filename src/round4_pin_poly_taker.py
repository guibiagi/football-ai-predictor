"""H4-PIN: Pinnacle close (devig) vs Poly no mesmo instante — taker."""

from __future__ import annotations

from datetime import timezone

import numpy as np
import pandas as pd

from src.hypothesis_engine import HOLDOUT_DAYS
from src.odds_utils import market_probs
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

TIMINGS = {
    "T24h": 24.0,
    "T6h": 6.0,
    "T1h": 1.0,
    "T10m": 10.0 / 60.0,
}
GAPS = [0.01, 0.02, 0.03, 0.04, 0.05]
SPREAD_HALF = 0.01  # metade do spread taker simulado (2¢ total)


def _split_holdout(kickoff, holdout_cut: pd.Timestamp) -> bool:
    kt = pd.Timestamp(kickoff)
    if kt.tzinfo is not None:
        kt = kt.tz_convert(None)
    return kt >= holdout_cut


def run_pin_poly_grid(
    df: pd.DataFrame,
    registry: HypothesisRegistry,
    events: list[dict] | None = None,
    cache_only: bool = True,
    leg_mode: str = "favorite",
) -> None:
    events = events or load_merged_cached_events()
    events = [e for e in events if is_match_event(e)]
    holdout_cut = df.date.max() - pd.Timedelta(days=HOLDOUT_DAYS)
    teams_idx: dict[str, dict[str, str]] = {}

    buckets: dict[tuple[str, float], list[float]] = {
        (t, g): [] for t in TIMINGS for g in GAPS
    }
    buckets_dev: dict[tuple[str, float], list[float]] = {
        (t, g): [] for t in TIMINGS for g in GAPS
    }

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
        pin = _pin_probs(row, use_close=True)
        if pin is None:
            continue
        fav_i = int(np.argmax(pin))
        indices = [fav_i] if leg_mode == "favorite" else [0, 1, 2]

        for i in indices:
            market = markets[i]
            token = parse_token_ids(market)
            if not token:
                continue
            hist = fetch_price_history(token, kickoff, cache_only=cache_only)
            if len(hist) < 3:
                continue
            if kickoff.tzinfo is None:
                kickoff = kickoff.replace(tzinfo=timezone.utc)
            end_ts = int(kickoff.timestamp())
            role = roles[i]
            won = winner == role
            pin_p = float(pin[i])

            for tname, hours in TIMINGS.items():
                p_poly = mean_price_in_window(hist, end_ts, hours)
                if p_poly is None:
                    continue
                exec_p = p_poly + SPREAD_HALF
                for gap in GAPS:
                    if exec_p >= pin_p - gap:
                        continue
                    gross = (1.0 - exec_p) / exec_p if won else -1.0
                    net = net_return_taker(gross, exec_p, "sports")
                    key = (tname, gap)
                    if _split_holdout(kickoff, holdout_cut):
                        buckets[key].append(net)
                    else:
                        buckets_dev[key].append(net)

    for (tname, gap), hold_rets in buckets.items():
        registry.add(
            hid=f"H4-PIN-{tname}-gap{int(gap*100):02d}-{leg_mode[:3]}",
            family="pin_poly_taker",
            description=f"Taker YES se Poly+spread < Pin−{gap:.0%} em {tname} ({leg_mode})",
            holdout_returns=hold_rets,
            dev_returns=buckets_dev[(tname, gap)],
            timing=tname,
            gap=gap,
            leg_mode=leg_mode,
        )
