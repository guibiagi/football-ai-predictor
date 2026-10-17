from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

# Aposta típica do usuário (US$); sensibilidade 10–50 modelada via impacto linear.
DEFAULT_STAKE_USD = 25.0
BASE_HALF_SPREAD = 0.005  # meio-spread fixo ~1% round-trip


@dataclass(frozen=True)
class LiquidityTerciles:
    """Limites em volume Gamma (USD) — tercis no log(volume)."""

    q33: float
    q66: float

    def label(self, volume: float) -> str:
        v = max(volume, 0.0)
        if v <= self.q33:
            return "T1_low"
        if v <= self.q66:
            return "T2_mid"
        return "T3_high"


def compute_terciles(volumes: list[float]) -> LiquidityTerciles:
    if not volumes:
        return LiquidityTerciles(0.0, 0.0)
    logv = np.log1p(np.array(volumes, dtype=float))
    q33, q66 = np.percentile(logv, [33.33, 66.67])
    return LiquidityTerciles(
        q33=float(np.expm1(q33)),
        q66=float(np.expm1(q66)),
    )


def taker_ask_with_slippage(
    mid: float,
    volume_gamma_usd: float,
    stake_usd: float = DEFAULT_STAKE_USD,
) -> float:
    """
    Preço efetivo de compra (probabilidade) para ordem taker pequena.

    - volume_gamma_usd: proxy de liquidez (volume acumulado Gamma no mercado).
    - Profundidade no touch ≈ max(250 USD, 2% do volume).
    - Impacto linear em pontos de prob: 0.5 * stake / depth (mais agressivo em T1).
    """
    if mid <= 0 or mid >= 1:
        return float("nan")
    touch_depth = max(250.0, volume_gamma_usd * 0.02)
    impact = 0.5 * (stake_usd / touch_depth)
    return min(0.99, mid + BASE_HALF_SPREAD + impact)


def stake_sensitivity_rois(
    mid: float,
    volume: float,
    won: bool,
    fee_rate: float,
    roi_fn,
) -> dict[str, float]:
    """ROI em stakes 10/25/50 USD (diagnóstico, não testes separados)."""
    out = {}
    for stake in (10.0, 25.0, 50.0):
        p = taker_ask_with_slippage(mid, volume, stake_usd=stake)
        out[f"stake_{int(stake)}"] = roi_fn(p, won, fee_rate)
    return out
