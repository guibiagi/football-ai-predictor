"""Preços Polymarket: execução (ask/bid) e fechamento antes do kickoff."""

from __future__ import annotations

import json
import urllib.request
from datetime import datetime, timezone

CLOB = "https://clob.polymarket.com"


def _get(url: str) -> dict | list:
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    with urllib.request.urlopen(req, timeout=30) as resp:
        return json.load(resp)


def parse_token_ids(market: dict) -> tuple[str | None, str | None]:
    raw = market.get("clobTokenIds")
    if not raw:
        return None, None
    try:
        ids = json.loads(raw) if isinstance(raw, str) else raw
        return ids[0], ids[1] if len(ids) > 1 else None
    except (json.JSONDecodeError, TypeError, IndexError):
        return None, None


def execution_price_yes(market: dict) -> float | None:
    """Preço de compra do YES: bestAsk quando disponível, senão outcomePrices[0]."""
    ask = market.get("bestAsk")
    if ask is not None:
        try:
            a = float(ask)
            if 0 < a < 1:
                return a
        except (TypeError, ValueError):
            pass
    bid = market.get("bestBid")
    if bid is not None:
        try:
            b = float(bid)
            if 0 < b < 1:
                return b
        except (TypeError, ValueError):
            pass
    ltp = market.get("lastTradePrice")
    if ltp is not None:
        try:
            p = float(ltp)
            if 0 < p < 1:
                return p
        except (TypeError, ValueError):
            pass
    try:
        p = float(json.loads(market["outcomePrices"])[0])
        return p if 0 < p < 1 else None
    except Exception:
        return None


def closing_price_before_kickoff(token_id: str, kickoff: datetime) -> float | None:
    """Último preço registrado no CLOB até o kickoff (UTC)."""
    if not token_id or kickoff is None:
        return None
    if kickoff.tzinfo is None:
        kickoff = kickoff.replace(tzinfo=timezone.utc)
    end_ts = int(kickoff.timestamp())
    start_ts = end_ts - 48 * 3600
    url = (
        f"{CLOB}/prices-history?market={token_id}"
        f"&startTs={start_ts}&endTs={end_ts}&fidelity=60"
    )
    try:
        data = _get(url)
        hist = data.get("history") or []
        if not hist:
            return None
        last = max(hist, key=lambda h: h.get("t", 0))
        return float(last.get("p"))
    except Exception:
        return None
