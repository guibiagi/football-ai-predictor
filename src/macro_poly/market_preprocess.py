from __future__ import annotations

from collections import defaultdict

from src.macro_poly.discovery import MacroMarket
from src.macro_poly.fomc_calendar import parse_meeting_year_month


def dedupe_fomc_events(markets: list[MacroMarket]) -> list[MacroMarket]:
    """Um evento FOMC por reunião (mês/ano): mantém o de maior volume agregado."""
    vol_by_event: dict[str, float] = defaultdict(float)
    meeting_key_by_event: dict[str, tuple[int, int] | int] = {}
    for m in markets:
        if m.category != "fomc":
            continue
        vol_by_event[m.event_id] += m.volume
        ym = parse_meeting_year_month(m.event_title, m.end_date)
        meeting_key_by_event[m.event_id] = ym if ym else (m.release_ts or 0)

    best_per_meeting: dict[tuple | int, str] = {}
    vol_best: dict[tuple | int, float] = {}
    for eid, key in meeting_key_by_event.items():
        v = vol_by_event[eid]
        if key not in vol_best or v > vol_best[key]:
            vol_best[key] = v
            best_per_meeting[key] = eid

    keep = set(best_per_meeting.values())
    return [m for m in markets if m.category != "fomc" or m.event_id in keep]
