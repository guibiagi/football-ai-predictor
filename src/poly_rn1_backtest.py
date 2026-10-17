"""H-RN1: maker passivo só no favorito (bid abaixo do fair Pinnacle)."""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime

import numpy as np
import pandas as pd

from src.hypothesis_engine import HOLDOUT_DAYS
from src.poly_maker_backtest import (
    fetch_closed_soccer_events,
    fetch_price_history,
    match_event_to_row,
    outcome_roles_from_markets,
    _parse_kickoff,
    _pin_probs,
    _winner_from_markets,
    parse_token_ids,
)
from src.reporting import bootstrap_ci


@dataclass
class Rn1Opp:
    kickoff: datetime
    fair_fav: float
    bid: float
    band: str
    filled: bool
    fill_price: float | None
    pin_close_fav: float
    fav_won: bool
    roi: float | None
    clv: float | None


BANDS = [
    ("0.50-0.85", 0.50, 0.85),
    ("0.60-0.80", 0.60, 0.80),
    ("0.60-0.70", 0.60, 0.70),
    ("0.70-0.80", 0.70, 0.80),
]


def _band_label(fair: float, lo: float = 0.50, hi: float = 0.85) -> str | None:
    if lo <= fair <= hi:
        return f"{lo:.2f}-{hi:.2f}"
    return None


def _settlement_buy_yes(fill_p: float, won: bool) -> float:
    if fill_p <= 0 or fill_p >= 1:
        return float("nan")
    return float((1.0 - fill_p) / fill_p) if won else -1.0


def run_rn1_favorite_bid_maker(
    df: pd.DataFrame,
    events: list[dict] | None = None,
    offset_below_fair: float = 0.03,
    fair_lo: float = 0.50,
    fair_hi: float = 0.85,
    min_volume: float = 500.0,
    holdout_days: int = HOLDOUT_DAYS,
) -> dict:
    """Posta bid no favorito a fair_pin - offset; fill se histórico ≤ bid."""
    if events is None:
        events = fetch_closed_soccer_events(list({"epl", "bundesliga", "soccer"}), max_per_tag=80)

    holdout_cut = df.date.max() - pd.Timedelta(days=holdout_days)
    teams_idx: dict[str, dict[str, str]] = {}
    opps: list[Rn1Opp] = []

    role_to_idx = {"casa": 0, "empate": 1, "fora": 2}

    for event in events:
        markets = event.get("markets") or []
        if len(markets) != 3:
            continue
        kickoff = _parse_kickoff(event)
        if kickoff is None:
            continue
        row = match_event_to_row(event, df, teams_idx)
        if row is None:
            continue
        home, away = row.home_team, row.away_team
        roles = outcome_roles_from_markets(markets, home, away)
        if roles is None:
            continue
        winner = _winner_from_markets(markets, roles)
        if winner is None:
            continue
        pin_open = _pin_probs(row, use_close=False)
        pin_close = _pin_probs(row, use_close=True)
        if pin_open is None or pin_close is None:
            continue
        fair_probs = pin_open
        fav_i = int(np.argmax(fair_probs))
        fair_fav = float(fair_probs[fav_i])
        if not (fair_lo <= fair_fav <= fair_hi):
            continue
        role = roles[fav_i]
        market = markets[fav_i]
        vol = float(market.get("volume") or 0)
        if vol < min_volume:
            continue
        token = parse_token_ids(market)
        if not token:
            continue
        hist = fetch_price_history(token, kickoff)
        bid = round(fair_fav - offset_below_fair, 3)
        if bid < 0.01 or bid > 0.99:
            continue
        filled = False
        fill_p = None
        for pt in sorted(hist, key=lambda h: h.get("t", 0)):
            if float(pt.get("p", 1)) <= bid:
                filled = True
                fill_p = bid
                break
        fav_won = winner == role
        roi = _settlement_buy_yes(fill_p, fav_won) if filled and fill_p else None
        p_close_fav = float(pin_close[fav_i])
        clv = (p_close_fav - fill_p) if filled and fill_p is not None else None
        opps.append(
            Rn1Opp(
                kickoff=kickoff,
                fair_fav=fair_fav,
                bid=bid,
                band=_band_label(fair_fav, fair_lo, fair_hi) or "out",
                filled=filled,
                fill_price=fill_p,
                pin_close_fav=p_close_fav,
                fav_won=fav_won,
                roi=roi,
                clv=clv,
            )
        )

    return _summarize_rn1(opps, holdout_cut, offset_below_fair, fair_lo, fair_hi)


def _summarize_rn1(
    opps: list[Rn1Opp],
    holdout_cut: pd.Timestamp,
    offset: float,
    fair_lo: float,
    fair_hi: float,
) -> dict:
    def split_hold(lst: list[Rn1Opp]) -> tuple[list[Rn1Opp], list[Rn1Opp]]:
        dev, hold = [], []
        for o in lst:
            kt = pd.Timestamp(o.kickoff)
            if kt.tzinfo is not None:
                kt = kt.tz_convert(None)
            if kt < holdout_cut:
                dev.append(o)
            else:
                hold.append(o)
        return dev, hold

    def pack_fills(sub: list[Rn1Opp]) -> dict:
        fills = [o for o in sub if o.filled and o.roi is not None]
        if not fills:
            return {"n_opps": len(sub), "n_fills": 0, "fill_rate": 0.0}
        rois = np.array([f.roi for f in fills], dtype=float)
        clvs = np.array([f.clv for f in fills if f.clv is not None], dtype=float)
        rm, rlo, rhi = bootstrap_ci(rois)
        cm, clo, chi = bootstrap_ci(clvs) if len(clvs) else (float("nan"), float("nan"), float("nan"))
        # seleção adversa: ROI vs fair implícito (ganhou menos que fair_fav)
        hit = np.mean([f.fav_won for f in fills])
        fair_mean = np.mean([f.fair_fav for f in fills])
        return {
            "n_opps": len(sub),
            "n_fills": len(fills),
            "fill_rate": round(len(fills) / len(sub), 4) if sub else 0.0,
            "roi_mean": round(rm, 4),
            "roi_ci95": [round(rlo, 4), round(rhi, 4)],
            "clv_mean": round(cm, 5) if np.isfinite(cm) else None,
            "clv_ci95": [round(clo, 5), round(chi, 5)] if np.isfinite(cm) else None,
            "hit_rate": round(float(hit), 4),
            "fair_mean": round(float(fair_mean), 4),
            "adverse_gap": round(float(hit - fair_mean), 4),
        }

    by_band: dict[str, dict] = {}
    for name, lo, hi in BANDS:
        sub = [o for o in opps if lo <= o.fair_fav <= hi]
        if not sub:
            continue
        _, hold_sub = split_hold(sub)
        by_band[name] = {
            "all": pack_fills(sub),
            "holdout": pack_fills(hold_sub),
        }

    dev, hold = split_hold(opps)
    hold_pack = pack_fills(hold)
    pnl_viable = (
        hold_pack.get("n_fills", 0) >= 300
        and hold_pack.get("roi_ci95")
        and hold_pack["roi_mean"] > 0
        and hold_pack["roi_ci95"][0] > 0
    )
    return {
        "hypothesis": "H-RN1_favorite_bid_maker",
        "offset_below_pin_fair": offset,
        "fair_band": [fair_lo, fair_hi],
        "note": "Maker: sem taxa taker; inspirado em perfil RN1 (não verificado)",
        "aggregate": {"dev": pack_fills(dev), "holdout": hold_pack},
        "by_price_band": by_band,
        "pnl_viable_holdout": pnl_viable,
    }


def run_rn1_offset_grid(df: pd.DataFrame, events: list[dict] | None = None) -> list[dict]:
    events = events or fetch_closed_soccer_events(["epl", "bundesliga", "soccer"], max_per_tag=60)
    out = []
    for off in (0.02, 0.03, 0.04, 0.05):
        r = run_rn1_favorite_bid_maker(df, events=events, offset_below_fair=off)
        r["offset"] = off
        out.append(r)
    return out
