from __future__ import annotations

import json
import re
from dataclasses import dataclass, asdict
from typing import Any

from src.macro_poly.http_util import get_json

GAMMA = "https://gamma-api.polymarket.com"
TAGS = ("fed", "inflation", "cpi", "fomc", "interest-rates", "economics")
MACRO_RE = re.compile(r"fed|fomc|cpi|inflation|pce|interest rate|bps|rate cut|rate hike", re.I)
CPI_RELEASE_RE = re.compile(
    r"released on ([A-Za-z]+ \d{1,2}, \d{4}), at (\d{1,2}):(\d{2}) (AM|PM) ET",
    re.I,
)
CPI_THRESHOLD_RE = re.compile(
    r"inflation\s*([><]=?)\s*([\d.]+)\s*%|>\s*([\d.]+)\s*percent",
    re.I,
)


@dataclass
class MacroMarket:
    event_id: str
    event_title: str
    event_slug: str
    market_id: str
    question: str
    market_slug: str
    token_yes: str
    end_date: str
    volume: float
    category: str  # cpi | fomc | other
    release_ts: int | None
    threshold_op: str | None  # gt | lt
    threshold_pct: float | None
    outcome_yes: int | None  # 1 or 0


def _parse_release_ts(description: str) -> int | None:
    if not description:
        return None
    m = CPI_RELEASE_RE.search(description)
    if not m:
        return None
    from datetime import datetime
    from zoneinfo import ZoneInfo

    month_day_year = m.group(1)
    hour = int(m.group(2))
    minute = int(m.group(3))
    ampm = m.group(4).upper()
    if ampm == "PM" and hour != 12:
        hour += 12
    if ampm == "AM" and hour == 12:
        hour = 0
    dt_str = f"{month_day_year} {hour:02d}:{minute:02d}"
    try:
        local = datetime.strptime(dt_str, "%B %d, %Y %H:%M")
        local = local.replace(tzinfo=ZoneInfo("America/New_York"))
        return int(local.timestamp())
    except ValueError:
        return None


def _parse_cpi_threshold(question: str, description: str) -> tuple[str | None, float | None]:
    text = f"{question} {description}"
    m = CPI_THRESHOLD_RE.search(text)
    if not m:
        return None, None
    if m.group(1) and m.group(2):
        op = m.group(1).replace(" ", "")
        val = float(m.group(2))
        if ">" in op:
            return "gt", val
        return "lt", val
    if m.group(3):
        return "gt", float(m.group(3))
    return None, None


def _classify(title: str, question: str) -> str:
    t = f"{title} {question}".lower()
    if "inflation" in t or "cpi" in t:
        return "cpi"
    if "fed" in t and ("bps" in t or "interest rate" in t or "decision" in t or "meeting" in t):
        return "fomc"
    if "fomc" in t:
        return "fomc"
    if "pce" in t:
        return "pce"
    return "other"


def _fomc_release_ts(end_date: str) -> int | None:
    """FOMC statement ~14:00 ET no dia útil do endDate (aprox.)."""
    if not end_date:
        return None
    from datetime import datetime
    from zoneinfo import ZoneInfo

    try:
        day = datetime.fromisoformat(end_date.replace("Z", "+00:00")).date()
        local = datetime(day.year, day.month, day.day, 14, 0, tzinfo=ZoneInfo("America/New_York"))
        return int(local.timestamp())
    except ValueError:
        return None


def _outcome_yes(market: dict[str, Any]) -> int | None:
    try:
        prices = market.get("outcomePrices")
        if isinstance(prices, str):
            prices = json.loads(prices)
        if prices[0] == "1":
            return 1
        if prices[0] == "0":
            return 0
    except (TypeError, IndexError, json.JSONDecodeError):
        pass
    return None


def discover_macro_markets(max_per_tag: int = 400) -> list[MacroMarket]:
    seen_events: set[str] = set()
    rows: list[MacroMarket] = []

    for tag in TAGS:
        offset = 0
        while offset < max_per_tag:
            batch = get_json(
                f"{GAMMA}/events?tag_slug={tag}&closed=true&limit=100&offset={offset}"
            )
            if not batch:
                break
            for ev in batch:
                eid = str(ev.get("id", ""))
                if eid in seen_events:
                    continue
                seen_events.add(eid)
                title = ev.get("title") or ""
                desc = ev.get("description") or ""
                if not MACRO_RE.search(title) and not MACRO_RE.search(desc[:500]):
                    continue
                for m in ev.get("markets") or []:
                    oy = _outcome_yes(m)
                    if oy is None:
                        continue
                    raw_ids = m.get("clobTokenIds")
                    if not raw_ids:
                        continue
                    try:
                        ids = json.loads(raw_ids) if isinstance(raw_ids, str) else raw_ids
                        token_yes = ids[0]
                    except (json.JSONDecodeError, TypeError, IndexError):
                        continue
                    cat = _classify(title, m.get("question") or "")
                    rel = _parse_release_ts(m.get("description") or desc)
                    th_op, th_pct = _parse_cpi_threshold(
                        m.get("question") or "", m.get("description") or desc
                    )
                    if cat == "fomc" and rel is None:
                        rel = _fomc_release_ts(m.get("endDate") or "")
                    vol = float(m.get("volume") or ev.get("volume") or 0)
                    rows.append(
                        MacroMarket(
                            event_id=eid,
                            event_title=title,
                            event_slug=ev.get("slug") or "",
                            market_id=str(m.get("id", "")),
                            question=m.get("question") or "",
                            market_slug=m.get("slug") or "",
                            token_yes=token_yes,
                            end_date=m.get("endDate") or "",
                            volume=vol,
                            category=cat,
                            release_ts=rel,
                            threshold_op=th_op,
                            threshold_pct=th_pct,
                            outcome_yes=oy,
                        )
                    )
            offset += 100
    return rows


def markets_to_manifest(markets: list[MacroMarket]) -> list[dict]:
    return [asdict(m) for m in markets]
