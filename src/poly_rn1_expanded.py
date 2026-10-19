"""RN1 ampliado: futebol (Pin) + NBA/NFL/UFC/tênis + clobRewards no PnL."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone

import numpy as np
import pandas as pd

from src.hypothesis_engine import HOLDOUT_DAYS
from src.poly_maker_backtest import (
    LEAGUE_TAG_TO_COMP,
    _parse_kickoff,
    _pin_probs,
    _winner_from_markets,
    fetch_price_history,
    is_match_event,
    match_event_to_row,
    outcome_roles_from_markets,
    parse_token_ids,
)
from src.poly_market_utils import price_history as price_history_util
from src.poly_rewards import estimate_maker_reward_usd, reward_roi_addition
from src.poly_rn1_backtest import Rn1Opp, _settlement_buy_yes

SPORT_TAGS = list(LEAGUE_TAG_TO_COMP.keys()) + [
    "soccer", "epl", "nba", "nfl", "ufc", "mma", "atp", "wta", "mlb", "nhl",
]


def fetch_multi_sport_closed(max_per_tag: int = 120) -> list[dict]:
    seen: set[str] = set()
    out: list[dict] = []
    for tag in SPORT_TAGS:
        try:
            from src.polymarket_gamma_client import fetch_events_closed

            batch = fetch_events_closed(tag_slug=tag, max_pages=max(1, max_per_tag // 100))
        except Exception:
            continue
        for e in batch[:max_per_tag]:
            slug = e.get("slug")
            if slug in seen:
                continue
            seen.add(slug)
            out.append(e)
    return out


def _fair_from_history(
    token: str, kickoff: datetime, hours_before: float = 24.0, cache_only: bool = False,
) -> float | None:
    if kickoff.tzinfo is None:
        kickoff = kickoff.replace(tzinfo=timezone.utc)
    end_ts = int(kickoff.timestamp())
    hist = price_history_util(
        token, end_ts - 96 * 3600, end_ts, cache_only=cache_only,
    )
    if not hist:
        return None
    target = end_ts - int(hours_before * 3600)
    pts = sorted(hist, key=lambda h: abs(h.get("t", 0) - target))
    if not pts:
        return None
    return float(pts[0].get("p", 0))


@dataclass
class _Rn1Raw:
    kickoff: datetime
    fair_fav: float
    pin_close_fav: float
    fav_won: bool
    market: dict
    hist: list
    sport: str


def build_rn1_raws(
    df: pd.DataFrame,
    events: list[dict],
    fair_lo: float = 0.60,
    fair_hi: float = 0.80,
    min_volume: float = 200.0,
    cache_only: bool = True,
) -> list[_Rn1Raw]:
    soccer = [e for e in events if is_match_event(e)]
    other = [e for e in events if e not in soccer and " vs " in (e.get("title") or "").lower()]
    teams_idx: dict[str, dict[str, str]] = {}
    raws: list[_Rn1Raw] = []

    for event in soccer:
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
        pin_close = _pin_probs(row, use_close=True)
        if pin_open is None or pin_close is None:
            continue
        fav_i = int(np.argmax(pin_open))
        fair_fav = float(pin_open[fav_i])
        if not (fair_lo <= fair_fav <= fair_hi):
            continue
        market = markets[fav_i]
        if float(market.get("volume") or 0) < min_volume:
            continue
        token = parse_token_ids(market)
        if not token:
            continue
        hist = fetch_price_history(token, kickoff, cache_only=cache_only)
        if not hist:
            continue
        role = roles[fav_i]
        raws.append(
            _Rn1Raw(
                kickoff=kickoff,
                fair_fav=fair_fav,
                pin_close_fav=float(pin_close[fav_i]),
                fav_won=winner == role,
                market=market,
                hist=hist,
                sport="soccer",
            )
        )

    for event in other:
        markets = event.get("markets") or []
        if len(markets) != 1:
            continue
        m = markets[0]
        token = parse_token_ids(m)
        if not token:
            continue
        kickoff = _parse_kickoff(event)
        if kickoff is None:
            continue
        fair_yes = _fair_from_history(token, kickoff, cache_only=cache_only)
        if fair_yes is None:
            continue
        try:
            import json

            yes_won_flag = float(json.loads(m["outcomePrices"])[0]) > 0.9
        except Exception:
            continue
        fair_fav = fair_yes if fair_yes >= 0.5 else 1.0 - fair_yes
        if not (fair_lo <= fair_fav <= fair_hi):
            continue
        if float(m.get("volume") or 0) < min_volume:
            continue
        hist = fetch_price_history(token, kickoff, cache_only=cache_only)
        if not hist:
            continue
        tag_slug = "other"
        for t in event.get("tags") or []:
            if t.get("slug"):
                tag_slug = t["slug"]
                break
        fav_won = yes_won_flag if fair_yes >= 0.5 else not yes_won_flag
        raws.append(
            _Rn1Raw(
                kickoff=kickoff,
                fair_fav=fair_fav,
                pin_close_fav=fair_fav,
                fav_won=fav_won,
                market=m,
                hist=hist,
                sport=tag_slug,
            )
        )
    return raws


def run_rn1_expanded(
    df: pd.DataFrame,
    offset: float = 0.025,
    fair_lo: float = 0.60,
    fair_hi: float = 0.80,
    min_volume: float = 200.0,
    include_rewards: bool = True,
    events: list[dict] | None = None,
    cache_only: bool = False,
    raws: list[_Rn1Raw] | None = None,
) -> dict:
    if raws is None:
        events = events or fetch_multi_sport_closed(max_per_tag=200)
        raws = build_rn1_raws(
            df, events, fair_lo, fair_hi, min_volume, cache_only=cache_only,
        )
    holdout_cut = df.date.max() - pd.Timedelta(days=HOLDOUT_DAYS)
    sports = [r.sport for r in raws]
    opps = _opps_from_raws(raws, offset, include_rewards)
    return _summarize_expanded(opps, holdout_cut, offset, fair_lo, fair_hi, sports, include_rewards)


def run_rn1_expanded_offsets(
    df: pd.DataFrame,
    offsets: tuple[float, ...] = (0.02, 0.025, 0.03),
    events: list[dict] | None = None,
    cache_only: bool = True,
    include_rewards: bool = True,
    fair_lo: float = 0.60,
    fair_hi: float = 0.80,
    min_volume: float = 200.0,
) -> tuple[list[_Rn1Raw], list[dict]]:
    events = events or []
    raws = build_rn1_raws(
        df, events, fair_lo=fair_lo, fair_hi=fair_hi, min_volume=min_volume, cache_only=cache_only,
    )
    return raws, [
        run_rn1_expanded(
            df, offset=o, raws=raws, include_rewards=include_rewards,
            fair_lo=fair_lo, fair_hi=fair_hi,
        )
        for o in offsets
    ]


def _opps_from_raws(raws: list[_Rn1Raw], offset: float, include_rewards: bool) -> list[Rn1Opp]:
    opps: list[Rn1Opp] = []
    for raw in raws:
        bid = round(raw.fair_fav - offset, 3)
        filled, fill_p = _try_fill(raw.hist, bid)
        roi, _ = _roi_with_rewards(
            raw.market, filled, fill_p, raw.fav_won, raw.kickoff, raw.hist,
            include_rewards, offset,
        )
        opps.append(_mk_opp(
            raw.kickoff, raw.fair_fav, bid, filled, fill_p, raw.pin_close_fav, raw.fav_won, roi,
        ))
    return opps


def _try_fill(hist: list, bid: float) -> tuple[bool, float | None]:
    for pt in sorted(hist, key=lambda h: h.get("t", 0)):
        if float(pt.get("p", 1)) <= bid:
            return True, bid
    return False, None


def _roi_with_rewards(
    market, filled, fill_p, fav_won, kickoff, hist, include_rewards, offset,
) -> tuple[float | None, float]:
    if not filled or fill_p is None:
        return None, 0.0
    gross = _settlement_buy_yes(fill_p, fav_won)
    if not include_rewards:
        return gross, 0.0
    hours = 36.0
    if hist:
        hours = max(1.0, (hist[-1].get("t", 0) - hist[0].get("t", 0)) / 3600.0)
    rew = estimate_maker_reward_usd(market, hours, size_usd=50.0, half_spread_cents=offset * 100)
    stake = fill_p
    return gross + reward_roi_addition(rew, stake), reward_roi_addition(rew, stake)


def _mk_opp(kickoff, fair_fav, bid, filled, fill_p, pin_close_fav, fav_won, roi):
    clv = (pin_close_fav - fill_p) if filled and fill_p else None
    return Rn1Opp(
        kickoff=kickoff, fair_fav=fair_fav, bid=bid, band="0.60-0.80",
        filled=filled, fill_price=fill_p, pin_close_fav=pin_close_fav,
        fav_won=fav_won, roi=roi, clv=clv,
    )


def _summarize_expanded(opps, holdout_cut, offset, fair_lo, fair_hi, sports, include_rewards):
    from src.poly_rn1_backtest import _summarize_rn1

    base = _summarize_rn1(opps, holdout_cut, offset, fair_lo, fair_hi)
    base["hypothesis"] = "H-RN1_expanded_rewards"
    base["include_rewards"] = include_rewards
    base["fair_band"] = [fair_lo, fair_hi]
    fills = [o for o in opps if o.filled and o.roi is not None]
    hold_fills = [
        o for o in fills
        if pd.Timestamp(o.kickoff).tz_localize(None) >= holdout_cut
    ]
    base["n_sports_breakdown"] = {
        "soccer_opps": sum(1 for s in sports if s == "soccer"),
        "other_opps": sum(1 for s in sports if s != "soccer"),
    }
    base["holdout_fill_count"] = len(hold_fills)
    base["offset"] = offset
    return base
