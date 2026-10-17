"""Datas oficiais de comunicado FOMC (statement ~14:00 ET no dia da decisão)."""

from __future__ import annotations

import re
from datetime import datetime
from zoneinfo import ZoneInfo

# Dia do mês da decisão (calendário publicado pelo Fed; Mar/2024 = 20, não 18).
FOMC_STATEMENT_DAY: dict[tuple[int, int], int] = {
    (2023, 1): 31,
    (2023, 3): 22,
    (2023, 5): 3,
    (2023, 6): 14,
    (2023, 7): 26,
    (2023, 9): 20,
    (2023, 11): 1,
    (2023, 12): 13,
    (2024, 1): 31,
    (2024, 3): 20,
    (2024, 5): 1,
    (2024, 6): 12,
    (2024, 7): 31,
    (2024, 9): 18,
    (2024, 11): 7,
    (2024, 12): 18,
    (2025, 1): 29,
    (2025, 3): 19,
    (2025, 5): 7,
    (2025, 6): 18,
    (2025, 7): 30,
    (2025, 9): 17,
    (2025, 10): 29,
    (2025, 12): 10,
    (2026, 1): 28,
    (2026, 3): 18,
    (2026, 4): 29,
    (2026, 6): 17,
    (2026, 7): 29,
    (2026, 9): 16,
    (2026, 10): 28,
    (2026, 12): 9,
}

MONTHS = {
    "january": 1,
    "february": 2,
    "march": 3,
    "april": 4,
    "may": 5,
    "june": 6,
    "july": 7,
    "august": 8,
    "september": 9,
    "october": 10,
    "november": 11,
    "december": 12,
}

MONTH_IN_TITLE = re.compile(
    r"\b(january|february|march|april|may|june|july|august|september|october|november|december)\b",
    re.I,
)


def parse_meeting_year_month(title: str, end_date_iso: str) -> tuple[int, int] | None:
    m = MONTH_IN_TITLE.search(title or "")
    if not m:
        return None
    month = MONTHS[m.group(1).lower()]
    try:
        year = datetime.fromisoformat(end_date_iso.replace("Z", "+00:00")).year
    except ValueError:
        return None
    return year, month


def fomc_statement_ts(year: int, month: int) -> int | None:
    day = FOMC_STATEMENT_DAY.get((year, month))
    if day is None:
        return None
    local = datetime(year, month, day, 14, 0, tzinfo=ZoneInfo("America/New_York"))
    return int(local.timestamp())


def fomc_statement_ts_from_event(title: str, end_date_iso: str) -> int | None:
    ym = parse_meeting_year_month(title, end_date_iso)
    if ym is None:
        return None
    return fomc_statement_ts(ym[0], ym[1])
