from __future__ import annotations

from src.macro_poly.discovery import MacroMarket
from src.macro_poly.prices import iso_to_ts


def volume_at_decision(m: MacroMarket, decision_ts: int) -> float:
    """
    Proxy de volume acumulado até decision_ts (sem usar volume final do Gamma).
    Interpola linearmente entre startDate e endDate com volume total reportado.
    """
    total = max(float(m.volume or 0), 0.0)
    start = m.start_ts
    if start is None and m.release_ts:
        start = m.release_ts - 30 * 86400
    end = iso_to_ts(m.end_date) or m.release_ts
    if start is None or end is None or end <= start:
        return total
    if decision_ts <= start:
        return 0.0
    if decision_ts >= end:
        return total
    frac = (decision_ts - start) / (end - start)
    return total * frac
