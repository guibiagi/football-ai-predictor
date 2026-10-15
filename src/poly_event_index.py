"""Índice de eventos Poly por data/slug para parear 1X2, O/U, margens."""

from __future__ import annotations

import re
from datetime import datetime


def _norm(s: str) -> str:
    return re.sub(r"[^a-z0-9]+", "", s.lower())


def event_date_key(event: dict) -> str:
    for k in ("endDate", "startTime", "gameStartTime"):
        v = event.get(k)
        if v:
            return str(v)[:10]
    return ""


def slug_core(slug: str) -> str:
    """Remove sufixos de mercado derivado."""
    s = slug.lower()
    for suf in ("-total-goals", "-over-under", "-o-u", "-spread", "-moneyline", "-to-advance"):
        if s.endswith(suf):
            s = s[: -len(suf)]
    return s


def teams_from_title(title: str) -> tuple[str, str] | None:
    m = re.search(r"([\w\s\.\']+?)\s+vs\.?\s+([\w\s\.\']+)", title, re.I)
    if not m:
        return None
    return _norm(m.group(1)), _norm(m.group(2))


def build_event_index(events: list[dict]) -> dict[str, list[dict]]:
    """Chave: date|teamA|teamB (ordenado) → lista de eventos."""
    idx: dict[str, list[dict]] = {}
    for e in events:
        pair = teams_from_title(e.get("title") or "")
        d = event_date_key(e)
        if not pair or not d:
            core = slug_core(e.get("slug") or "")
            if d and core:
                key = f"{d}|{core}"
            else:
                continue
        else:
            a, b = sorted(pair)
            key = f"{d}|{a}|{b}"
        idx.setdefault(key, []).append(e)
    return idx


def classify_event(event: dict) -> str:
    t = (event.get("title") or "").lower()
    slug = (event.get("slug") or "").lower()
    if "over" in t or "under" in t or "total goal" in t or "o/u" in t:
        return "ou"
    if "2 goal" in t or "by 2" in t or "margin" in t:
        return "margin"
    if " vs " in t and len(event.get("markets") or []) in (1, 2, 3):
        if len(event.get("markets") or []) == 3:
            return "1x2"
        return "moneyline"
    return "other"
