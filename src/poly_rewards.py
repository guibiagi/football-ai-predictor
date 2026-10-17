"""Estimativa de clobRewards para PnL maker (metadata Gamma)."""

from __future__ import annotations


def estimate_maker_reward_usd(
    market: dict,
    quote_hours: float,
    size_usd: float = 50.0,
    half_spread_cents: float = 2.5,
    pool_share: float = 0.08,
) -> float:
    rewards = market.get("clobRewards") or []
    if not rewards:
        return 0.0
    rate = float(rewards[0].get("rewardsDailyRate") or 0)
    if rate <= 0:
        return 0.0
    min_size = float(market.get("rewardsMinSize") or 50)
    max_spread = float(market.get("rewardsMaxSpread") or 3.5)
    if size_usd < min_size or half_spread_cents > max_spread:
        return 0.0
    days = max(quote_hours / 24.0, 0.05)
    return rate * days * pool_share


def reward_roi_addition(reward_usd: float, stake_usd: float) -> float:
    if stake_usd <= 0:
        return 0.0
    return reward_usd / stake_usd
