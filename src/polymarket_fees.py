"""Taxas taker Polymarket (docs: polymarket-learn/trading/fees)."""

from __future__ import annotations

# fee = C * feeRate * p * (1-p)  por share; aqui normalizamos C=1 share, retorno em fração do custo.

CATEGORY_FEE_RATE: dict[str, float] = {
    "crypto": 0.07,
    "sports": 0.05,
    "finance": 0.04,
    "politics": 0.04,
    "economics": 0.05,
    "culture": 0.05,
    "weather": 0.05,
    "other": 0.05,
    "mentions": 0.04,
    "tech": 0.04,
    "geopolitics": 0.0,
}


def infer_category(event: dict, market: dict | None = None) -> str:
    tags = [t.get("slug", "").lower() for t in (event.get("tags") or [])]
    for key in CATEGORY_FEE_RATE:
        if key in tags:
            return key
    title = (event.get("title") or "").lower()
    if "bitcoin" in title or "btc" in title or "ethereum" in title:
        return "crypto"
    if "temperature" in title or "highest temperature" in title:
        return "weather"
    if "trump" in title or "election" in title or "president" in title:
        return "politics"
    return "other"


def taker_fee_fraction(price: float, category: str, shares: float = 1.0) -> float:
    """Taxa em USDC dividida pelo custo (price*shares) → fração do stake."""
    if price <= 0 or price >= 1:
        return 0.0
    rate = CATEGORY_FEE_RATE.get(category, 0.05)
    if rate <= 0:
        return 0.0
    fee_usdc = shares * rate * price * (1.0 - price)
    cost = shares * price
    return float(fee_usdc / cost) if cost > 0 else 0.0


def net_return_taker(gross_return: float, entry_price: float, category: str) -> float:
    """Retorno líquido aproximado: perde taxa na entrada (taker)."""
    fee = taker_fee_fraction(entry_price, category)
    # gross_return é ROI sobre stake (ex: -1 ou (1-p)/p)
    return gross_return - fee
