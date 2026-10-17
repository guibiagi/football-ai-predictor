"""Rodada 5: validação realista H4-NEG (asks, profundidade, taxas, neg-risk)."""

from __future__ import annotations

import json
import time
import urllib.error
import urllib.request
from dataclasses import dataclass, field
from datetime import datetime, timezone

import numpy as np
import pandas as pd

from src.hypothesis_engine import HOLDOUT_DAYS
from src.poly_clob_client import best_ask, best_bid, fetch_order_book
from src.poly_market_utils import parse_token_yes
from src.poly_trades_client import fetch_trades
from src.polymarket_fees import infer_category, taker_fee_fraction
from src.polymarket_gamma_client import fetch_events_closed, load_merged_cached_events
from src.reporting import bootstrap_ci

GAMMA = "https://gamma-api.polymarket.com"
SYNC_WINDOW_SEC = 120
MIN_PROFIT_USDC = 0.50
MIN_SIZE_USDC = 5.0
SUM_LO = 0.98
SUM_HI = 1.02


@dataclass
class NegRiskSnapshot:
    event_slug: str
    ts: int
    direction: str  # buy_all_yes | sell_all_yes
    n_legs: int
    sum_prices: float
    size_shares: float
    size_usdc: float
    gross_profit_usdc: float
    fee_usdc: float
    net_profit_usdc: float
    days_to_resolution: float
    annualized_roi: float
    neg_risk: bool
    uma_flag: str | None
    source: str  # trades | live_book


@dataclass
class EventNegRiskReport:
    slug: str
    title: str
    n_markets: int
    neg_risk: bool
    uma_flag: str | None
    snapshots_executable: list[NegRiskSnapshot] = field(default_factory=list)
    max_net_profit_usdc: float = 0.0
    max_size_usdc: float = 0.0


def _parse_kickoff(event: dict) -> datetime | None:
    for k in ("endDate", "startTime", "closedTime"):
        v = event.get(k)
        if v:
            try:
                return datetime.fromisoformat(str(v).replace("Z", "+00:00"))
            except ValueError:
                continue
    return None


def _uma_risk(event: dict) -> str | None:
    for k in ("umaResolutionStatus", "resolutionStatus"):
        v = event.get(k)
        if v:
            return str(v)
    for m in event.get("markets") or []:
        if m.get("umaResolutionStatus"):
            return str(m["umaResolutionStatus"])
    return None


def _fee_usdc_buy_yes_legs(asks: list[float], size: float, category: str) -> float:
    total = 0.0
    for p in asks:
        total += taker_fee_fraction(p, category) * (p * size)
    return total


def _arb_buy_all_yes(
    asks: list[float], size: float, category: str,
) -> tuple[float, float, float]:
    """Retorna gross, fee, net em USDC para size shares."""
    cost = sum(asks) * size
    payout = size
    gross = payout - cost
    fee = _fee_usdc_buy_yes_legs(asks, size, category)
    return gross, fee, gross - fee


def _arb_sell_all_yes(
    bids: list[float], size: float, category: str,
) -> tuple[float, float, float]:
    revenue = sum(bids) * size
    payout_obligation = size
    gross = revenue - payout_obligation
    fee = 0.0
    for p in bids:
        fee += taker_fee_fraction(p, category) * (p * size)
    return gross, fee, gross - fee


def _asks_from_trades(trades: list[dict], t_lo: int, t_hi: int) -> tuple[float | None, float]:
    """Melhor ask (menor preço BUY Yes) e profundidade na janela."""
    candidates = []
    for tr in trades:
        ts = int(tr.get("timestamp") or 0)
        if ts < t_lo or ts > t_hi:
            continue
        if str(tr.get("side", "")).upper() != "BUY":
            continue
        out = tr.get("outcome")
        if out is not None and str(out).lower() not in ("yes", "y"):
            continue
        px = float(tr.get("price") or 0)
        sz = float(tr.get("size") or 0)
        if px <= 0 or sz <= 0:
            continue
        candidates.append((px, sz))
    if not candidates:
        return None, 0.0
    best = min(c[0] for c in candidates)
    depth = sum(sz for px, sz in candidates if px <= best + 0.002)
    return best, depth


def _bids_from_trades(trades: list[dict], t_lo: int, t_hi: int) -> tuple[float | None, float]:
    candidates = []
    for tr in trades:
        ts = int(tr.get("timestamp") or 0)
        if ts < t_lo or ts > t_hi:
            continue
        if str(tr.get("side", "")).upper() != "SELL":
            continue
        px = float(tr.get("price") or 0)
        sz = float(tr.get("size") or 0)
        if px <= 0 or sz <= 0:
            continue
        candidates.append((px, sz))
    if not candidates:
        return None, 0.0
    best = max(c[0] for c in candidates)
    depth = sum(sz for px, sz in candidates if px >= best - 0.002)
    return best, depth


def validate_event_trades(
    event: dict,
    lookback_days: int = 14,
    bucket_sec: int = 60,
    cache_only: bool = False,
) -> EventNegRiskReport:
    markets = event.get("markets") or []
    legs: list[tuple[str, str, list[dict]]] = []
    for m in markets:
        cid = m.get("conditionId")
        tok = parse_token_yes(m)
        if not cid or not tok:
            continue
        trades = fetch_trades(cid, limit=500, max_pages=15, cache_only=cache_only)
        legs.append((cid, tok, trades))

    ko = _parse_kickoff(event)
    rep = EventNegRiskReport(
        slug=event.get("slug") or "",
        title=event.get("title") or "",
        n_markets=len(legs),
        neg_risk=bool(event.get("negRisk")),
        uma_flag=_uma_risk(event),
    )
    if len(legs) < 2 or ko is None:
        return rep

    if ko.tzinfo is None:
        ko = ko.replace(tzinfo=timezone.utc)
    end_ts = int(ko.timestamp())
    start_ts = end_ts - lookback_days * 86400
    cat = infer_category(event)

    # Buckets de 1 min com trades em todas as pernas
    all_ts: set[int] = set()
    for _, _, tr in legs:
        for t in tr:
            ts = int(t.get("timestamp") or 0)
            if start_ts <= ts <= end_ts:
                all_ts.add(ts // bucket_sec * bucket_sec)

    for bucket in sorted(all_ts):
        t_lo, t_hi = bucket, bucket + SYNC_WINDOW_SEC
        asks: list[float] = []
        bids: list[float] = []
        depths_ask: list[float] = []
        depths_bid: list[float] = []
        for _, _, tr in legs:
            a, da = _asks_from_trades(tr, t_lo, t_hi)
            asks.append(a if a is not None else -1.0)
            depths_ask.append(da)
            b, db = _bids_from_trades(tr, t_lo, t_hi)
            bids.append(b if b is not None else -1.0)
            depths_bid.append(db)

        days_left = max(1 / 24, (end_ts - bucket) / 86400)

        if all(a > 0 for a in asks):
            size = min(depths_ask)
            s_ask = sum(asks)
        else:
            size = 0.0
            s_ask = 99.0

        if s_ask < SUM_LO and size >= 0.01:
            size_usdc = size * s_ask
            gross, fee, net = _arb_buy_all_yes(asks, size, cat)
            if net >= MIN_PROFIT_USDC and size_usdc >= MIN_SIZE_USDC:
                ann = (net / max(size_usdc, 0.01)) * (365 / days_left)
                snap = NegRiskSnapshot(
                    event_slug=rep.slug,
                    ts=bucket,
                    direction="buy_all_yes",
                    n_legs=len(legs),
                    sum_prices=s_ask,
                    size_shares=size,
                    size_usdc=round(size_usdc, 2),
                    gross_profit_usdc=round(gross, 4),
                    fee_usdc=round(fee, 4),
                    net_profit_usdc=round(net, 4),
                    days_to_resolution=round(days_left, 2),
                    annualized_roi=round(ann, 4),
                    neg_risk=rep.neg_risk,
                    uma_flag=rep.uma_flag,
                    source="trades",
                )
                rep.snapshots_executable.append(snap)

        if all(b > 0 for b in bids):
            size_s = min(depths_bid)
            s_bid = sum(bids)
        else:
            size_s = 0.0
            s_bid = 0.0

        if s_bid > SUM_HI and size_s >= 0.01:
            size_usdc = size_s * s_bid
            gross, fee, net = _arb_sell_all_yes(bids, size_s, cat)
            if net >= MIN_PROFIT_USDC and size_usdc >= MIN_SIZE_USDC:
                ann = (net / max(size_usdc, 0.01)) * (365 / days_left)
                snap = NegRiskSnapshot(
                    event_slug=rep.slug,
                    ts=bucket,
                    direction="sell_all_yes",
                    n_legs=len(legs),
                    sum_prices=s_bid,
                    size_shares=size_s,
                    size_usdc=round(size_usdc, 2),
                    gross_profit_usdc=round(gross, 4),
                    fee_usdc=round(fee, 4),
                    net_profit_usdc=round(net, 4),
                    days_to_resolution=round(days_left, 2),
                    annualized_roi=round(ann, 4),
                    neg_risk=rep.neg_risk,
                    uma_flag=rep.uma_flag,
                    source="trades",
                )
                rep.snapshots_executable.append(snap)

    if rep.snapshots_executable:
        rep.max_net_profit_usdc = max(s.net_profit_usdc for s in rep.snapshots_executable)
        rep.max_size_usdc = max(s.size_usdc for s in rep.snapshots_executable)
    return rep


def scan_live_negrisk(max_events: int = 40) -> list[dict]:
    """Varredura ao vivo (somente leitura) em eventos negRisk abertos."""
    url = f"{GAMMA}/events?closed=false&limit=50&offset=0"
    try:
        req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
        with urllib.request.urlopen(req, timeout=60) as resp:
            events = json.load(resp)
    except Exception:
        events = []

    opps: list[dict] = []
    scanned = 0
    for event in events:
        if not event.get("negRisk"):
            continue
        markets = event.get("markets") or []
        if len(markets) < 2 or len(markets) > 12:
            continue
        scanned += 1
        if scanned > max_events:
            break
        cat = infer_category(event)
        asks: list[float] = []
        bids: list[float] = []
        depth_a: list[float] = []
        depth_b: list[float] = []
        for m in markets:
            tok = parse_token_yes(m)
            if not tok:
                continue
            book = fetch_order_book(tok, use_cache=False)
            a, da = best_ask(book)
            b, db = best_bid(book)
            if a is None or b is None:
                asks = []
                break
            asks.append(a)
            bids.append(b)
            depth_a.append(da)
            depth_b.append(db)
            time.sleep(0.05)
        if len(asks) < 2:
            continue
        size_buy = min(depth_a)
        size_sell = min(depth_b)
        s_ask = sum(asks)
        s_bid = sum(bids)
        ko = _parse_kickoff(event)
        days = 30.0
        if ko:
            if ko.tzinfo is None:
                ko = ko.replace(tzinfo=timezone.utc)
            days = max(1.0, (ko - datetime.now(timezone.utc)).total_seconds() / 86400)

        if s_ask < SUM_LO and size_buy >= 1:
            g, f, n = _arb_buy_all_yes(asks, size_buy, cat)
            if n > 0:
                opps.append({
                    "slug": event.get("slug"),
                    "direction": "buy_all_yes",
                    "sum_asks": round(s_ask, 4),
                    "size_usdc": round(size_buy * s_ask, 2),
                    "net_profit_usdc": round(n, 2),
                    "annualized_roi": round((n / (size_buy * s_ask)) * (365 / days), 4),
                    "neg_risk": True,
                    "n_legs": len(asks),
                })
        if s_bid > SUM_HI and size_sell >= 1:
            g, f, n = _arb_sell_all_yes(bids, size_sell, cat)
            if n > 0:
                opps.append({
                    "slug": event.get("slug"),
                    "direction": "sell_all_yes",
                    "sum_bids": round(s_bid, 4),
                    "size_usdc": round(size_sell * s_bid, 2),
                    "net_profit_usdc": round(n, 2),
                    "annualized_roi": round((n / (size_sell * s_bid)) * (365 / days), 4),
                    "neg_risk": True,
                    "n_legs": len(bids),
                })
    return opps


def run_historical_validation(
    max_events: int = 120,
    holdout_only: bool = True,
    df_matches: pd.DataFrame | None = None,
    cache_only: bool = False,
) -> dict:
    events = load_merged_cached_events()
    neg = [
        e for e in events
        if e.get("negRisk") and 2 <= len(e.get("markets") or []) <= 10
    ]
    if holdout_only and df_matches is not None:
        cut = df_matches.date.max() - pd.Timedelta(days=HOLDOUT_DAYS)
        filtered = []
        for e in neg:
            ko = _parse_kickoff(e)
            if ko is None:
                continue
            kt = pd.Timestamp(ko)
            if kt.tzinfo is not None:
                kt = kt.tz_convert(None)
            if kt >= cut:
                filtered.append(e)
        neg = filtered

    neg = neg[:max_events]
    reports: list[EventNegRiskReport] = []
    proxy_count = 0
    for i, e in enumerate(neg):
        if i % 20 == 0:
            print(f"  validate {i}/{len(neg)}", flush=True)
        rep = validate_event_trades(e, cache_only=cache_only)
        reports.append(rep)
        # proxy: sum resolution prices
        try:
            prices = [float(json.loads(m["outcomePrices"])[0]) for m in e.get("markets") or []]
            if sum(prices) > SUM_HI or sum(prices) < SUM_LO:
                proxy_count += 1
        except Exception:
            pass

    all_snaps = [s for r in reports for s in r.snapshots_executable]
    holdout_returns = []
    for s in all_snaps:
        if s.size_usdc > 0:
            holdout_returns.append(s.net_profit_usdc / s.size_usdc)

    summary = summarize_returns_proxy(holdout_returns, all_snaps, reports, proxy_count, len(neg))
    return {
        "n_events_scanned": len(neg),
        "n_events_with_executable_snap": sum(1 for r in reports if r.snapshots_executable),
        "n_executable_snapshots": len(all_snaps),
        "proxy_mispricing_events": proxy_count,
        "event_reports": [
            {
                "slug": r.slug,
                "n_markets": r.n_markets,
                "neg_risk": r.neg_risk,
                "uma_flag": r.uma_flag,
                "n_executable": len(r.snapshots_executable),
                "max_net_profit_usdc": r.max_net_profit_usdc,
                "max_size_usdc": r.max_size_usdc,
                "best": [
                    {
                        "ts": s.ts,
                        "direction": s.direction,
                        "sum": s.sum_prices,
                        "size_usdc": s.size_usdc,
                        "net_profit_usdc": s.net_profit_usdc,
                        "annualized_roi": s.annualized_roi,
                    }
                    for s in sorted(r.snapshots_executable, key=lambda x: -x.net_profit_usdc)[:3]
                ],
            }
            for r in sorted(reports, key=lambda x: -x.max_net_profit_usdc)[:40]
        ],
        "aggregate": summary,
        "executable_for_paper": summary.get("executable_for_paper", False),
    }


def summarize_returns_proxy(
    rois: list[float],
    snaps: list[NegRiskSnapshot],
    reports: list[EventNegRiskReport],
    proxy_count: int,
    n_events: int,
) -> dict:
    if not rois:
        return {
            "n_trades_realistic": 0,
            "roi_mean": None,
            "roi_ci95": None,
            "total_net_profit_usdc": 0,
            "proxy_events": proxy_count,
            "false_positive_rate_proxy": 1.0 if proxy_count else 0,
            "executable_for_paper": False,
            "notes": "Nenhum snapshot com asks/bids sincronizados em todas as pernas.",
        }
    arr = np.asarray(rois, dtype=float)
    m, lo, hi = bootstrap_ci(arr)
    total_net = sum(s.net_profit_usdc for s in snaps)
    executable = (
        len(snaps) >= 5
        and m > 0
        and lo > 0
        and total_net > 10
    )
    return {
        "n_trades_realistic": len(rois),
        "roi_mean": round(m, 4),
        "roi_ci95": [round(lo, 4), round(hi, 4)],
        "total_net_profit_usdc": round(total_net, 2),
        "median_size_usdc": round(float(np.median([s.size_usdc for s in snaps])), 2),
        "median_annualized_roi": round(float(np.median([s.annualized_roi for s in snaps])), 4),
        "proxy_events": proxy_count,
        "false_positive_rate_proxy": round(1 - len(snaps) / max(proxy_count, 1), 4),
        "executable_for_paper": executable,
        "notes": (
            "ROI sobre capital USDC por snapshot; não equivale ao proxy +52% da Rodada 4 "
            "(que usava preço médio + slippage fixo sem sincronização)."
        ),
    }


def summarize_returns(returns: list[float]) -> dict:
    from src.poly_market_utils import summarize_returns as sr
    return sr(returns)
