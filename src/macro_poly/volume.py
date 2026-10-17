from __future__ import annotations

from src.macro_poly.discovery import MacroMarket
from src.macro_poly.trade_volume import TradeVolumeStore


def gamma_liquidity_usd(m: MacroMarket) -> float:
    return max(float(m.volume or 0.0), 1.0)


def volume_at_decision(
    m: MacroMarket,
    decision_ts: int,
    store: TradeVolumeStore | None,
) -> float | None:
    if store is not None and store.trade_volume_usable(m.token_yes):
        v = store.volume_at(m.token_yes, decision_ts)
        if v is not None:
            return v
    return gamma_liquidity_usd(m)
