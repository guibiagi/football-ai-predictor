from __future__ import annotations

from src.macro_poly.discovery import MacroMarket
from src.macro_poly.trade_volume import TradeVolumeStore


def volume_at_decision(
    m: MacroMarket,
    decision_ts: int,
    store: TradeVolumeStore | None,
) -> float | None:
    if store is None:
        return None
    return store.volume_at(m.token_yes, decision_ts)
