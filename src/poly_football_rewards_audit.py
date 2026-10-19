"""Auditoria de clobRewards em mercados de futebol abertos/fechados."""

from __future__ import annotations

import json
import urllib.request

from src.poly_maker_backtest import fetch_closed_soccer_events, LEAGUE_TAG_TO_COMP


GAMMA = "https://gamma-api.polymarket.com"


def _get(url: str) -> list:
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    with urllib.request.urlopen(req, timeout=60) as resp:
        return json.load(resp)


def scan_open_soccer_rewards() -> dict:
    tags = list(LEAGUE_TAG_TO_COMP.keys()) + ["soccer", "epl"]
    with_rewards = []
    total_mkt = 0
    for tag in tags:
        try:
            events = _get(f"{GAMMA}/events?tag_slug={tag}&closed=false&limit=50")
        except Exception:
            continue
        for e in events:
            for m in e.get("markets") or []:
                total_mkt += 1
                rewards = m.get("clobRewards") or []
                if rewards and float(rewards[0].get("rewardsDailyRate") or 0) > 0:
                    with_rewards.append({
                        "tag": tag,
                        "event": e.get("title"),
                        "question": m.get("question"),
                        "rewardsDailyRate": rewards[0].get("rewardsDailyRate"),
                        "rewardsMinSize": m.get("rewardsMinSize"),
                        "rewardsMaxSpread": m.get("rewardsMaxSpread"),
                        "volume": m.get("volume"),
                    })
    return {
        "open_markets_scanned": total_mkt,
        "markets_with_rewards": len(with_rewards),
        "sample": with_rewards[:15],
    }


def rewards_on_closed_football_sample(max_events: int = 200) -> dict:
    events = fetch_closed_soccer_events(list(LEAGUE_TAG_TO_COMP.keys())[:8], max_per_tag=30)
    n = 0
    with_r = 0
    rates = []
    for e in events[:max_events]:
        for m in e.get("markets") or []:
            n += 1
            rewards = m.get("clobRewards") or []
            if rewards and float(rewards[0].get("rewardsDailyRate") or 0) > 0:
                with_r += 1
                rates.append(float(rewards[0]["rewardsDailyRate"]))
    return {
        "markets": n,
        "with_rewards": with_r,
        "pct": round(100 * with_r / n, 2) if n else 0,
        "median_daily_rate": sorted(rates)[len(rates) // 2] if rates else None,
    }
