"""Backtest de market maker passivo na Polymarket vs fair Pinnacle (CLOB prices-history)."""

from __future__ import annotations

import json
import time
import urllib.error
import urllib.request
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from src.odds_utils import market_probs
from src.reporting import bootstrap_ci
from src.team_matching import build_map, load_aliases, lookup

GAMMA = "https://gamma-api.polymarket.com"
CLOB = "https://clob.polymarket.com"

LEAGUE_TAG_TO_COMP = {
    "premier-league": "Premier League",
    "epl": "Premier League",
    "la-liga": "La Liga",
    "bundesliga": "Bundesliga",
    "ligue-1": "Ligue 1",
    "primeira-liga": "Primeira Liga",
    "brazil-serie-a": "Brasileirao Serie A",
    "mls": "MLS",
    "efl-championship": "Championship",
    "scottish-premiership": "Scottish Premiership",
    "austria-bundesliga": "Austria Bundesliga",
    "denmark-superliga": "Danish Superliga",
    "norway-eliteserien": "Eliteserien",
    "sweden-allsvenskan": "Allsvenskan",
    "poland-ekstraklasa": "Ekstraklasa",
    "chinese-super-league": "Chinese Super League",
}


@dataclass
class MakerFill:
    kickoff: datetime
    competition: str
    slug: str
    outcome_label: str  # casa | empate | fora
    side: str  # buy | sell
    fill_price: float
    fair_pin: float
    pin_close_prob: float
    clv: float
    settlement_return: float
    reward_estimate: float
    volume: float


def _get(url: str, retries: int = 4) -> dict | list:
    for attempt in range(retries):
        try:
            req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
            with urllib.request.urlopen(req, timeout=60) as resp:
                return json.load(resp)
        except (urllib.error.URLError, TimeoutError):
            time.sleep(2 ** attempt)
    raise RuntimeError(f"Falha ao buscar {url}")


def is_match_event(event: dict) -> bool:
    markets = event.get("markets") or []
    if len(markets) != 3:
        return False
    mid = (markets[1].get("groupItemTitle") or "").lower()
    return "draw" in mid


def parse_token_ids(market: dict) -> str | None:
    raw = market.get("clobTokenIds")
    if not raw:
        return None
    try:
        ids = json.loads(raw) if isinstance(raw, str) else raw
        return ids[0] if ids else None
    except (json.JSONDecodeError, TypeError, IndexError):
        return None


def outcome_roles_from_markets(markets: list[dict], home: str, away: str) -> list[str] | None:
    """Retorna ['casa','empate','fora'] na ordem dos mercados."""
    idx = build_map([home, away], load_aliases())
    roles = []
    for m in markets:
        title = m.get("groupItemTitle") or m.get("question") or ""
        if "draw" in title.lower():
            roles.append("empate")
            continue
        who = lookup(idx, title)
        if who == home:
            roles.append("casa")
        elif who == away:
            roles.append("fora")
        else:
            roles.append(None)
    if None in roles or sorted(roles) != ["casa", "empate", "fora"]:
        return None
    return roles


def _parse_kickoff(event: dict) -> datetime | None:
    raw = event.get("startTime") or event.get("gameStartTime") or event.get("endDate")
    if not raw:
        return None
    try:
        return datetime.fromisoformat(str(raw).replace("Z", "+00:00"))
    except ValueError:
        return None


def _event_competition(event: dict, comp_index: dict[str, str]) -> str | None:
    tags = [t.get("slug") for t in (event.get("tags") or []) if t.get("slug")]
    for slug in tags:
        if slug in LEAGUE_TAG_TO_COMP:
            return LEAGUE_TAG_TO_COMP[slug]
    title = (event.get("title") or "").lower()
    for key, comp in LEAGUE_TAG_TO_COMP.items():
        if key.replace("-", " ") in title:
            return comp
    return None


def _teams_from_markets(markets: list[dict]) -> tuple[str | None, str | None]:
    """Extrai nomes Polymarket dos mercados não-empate."""
    names = []
    for m in markets:
        t = m.get("groupItemTitle") or ""
        if "draw" in t.lower():
            continue
        names.append(t)
    if len(names) != 2:
        return None, None
    return names[0], names[1]


HISTORY_CACHE_DIR = Path(__file__).resolve().parent.parent / "data" / "poly_price_history"


def fetch_price_history(
    token_id: str,
    kickoff: datetime,
    hours_before: int = 72,
    use_cache: bool = True,
    cache_only: bool = False,
) -> list[dict]:
    from pathlib import Path

    if kickoff.tzinfo is None:
        kickoff = kickoff.replace(tzinfo=timezone.utc)
    end_ts = int(kickoff.timestamp())
    start_ts = end_ts - hours_before * 3600
    cache_path = HISTORY_CACHE_DIR / f"{token_id}_{start_ts}_{end_ts}.json"
    if use_cache and cache_path.exists():
        try:
            return json.loads(cache_path.read_text(encoding="utf-8"))
        except Exception:
            pass
    if cache_only:
        return []
    url = (
        f"{CLOB}/prices-history?market={token_id}"
        f"&startTs={start_ts}&endTs={end_ts}&fidelity=60"
    )
    try:
        data = _get(url)
        hist = data.get("history") or []
        if use_cache and hist:
            HISTORY_CACHE_DIR.mkdir(parents=True, exist_ok=True)
            cache_path.write_text(json.dumps(hist), encoding="utf-8")
        return hist
    except Exception:
        return []


def _winner_from_markets(markets: list[dict], roles: list[str]) -> str | None:
    prices = []
    for m in markets:
        try:
            prices.append(float(json.loads(m["outcomePrices"])[0]))
        except Exception:
            prices.append(0.0)
    if max(prices) < 0.9:
        return None
    return roles[prices.index(max(prices))]


def _pin_probs(row: pd.Series, use_close: bool) -> tuple[float, float, float] | None:
    if use_close:
        cols = ("psc_h", "psc_d", "psc_a")
    else:
        cols = ("ps_h", "ps_d", "ps_a")
    if any(pd.isna(row.get(c)) or row.get(c, 0) <= 1 for c in cols):
        return None
    o = row[list(cols)].to_numpy(dtype=float)
    p = market_probs(o.reshape(1, -3))[0]
    return float(p[0]), float(p[1]), float(p[2])


def _settlement_return_buy(fill_p: float, won: bool) -> float:
    if fill_p <= 0 or fill_p >= 1:
        return float("nan")
    return float((1.0 - fill_p) / fill_p) if won else -1.0


def _settlement_return_sell(fill_p: float, won_yes: bool) -> float:
    """Vendeu YES a fill_p: ganha se NO vence."""
    if fill_p <= 0 or fill_p >= 1:
        return float("nan")
    return float(fill_p) if not won_yes else float(-(1.0 - fill_p))


def estimate_liquidity_reward(
    market: dict,
    half_spread: float,
    quote_hours: float,
    size_usd: float = 50.0,
    competition_share: float = 0.05,
) -> float:
    """Estimativa conservadora de rewards (compartilhados entre makers)."""
    rewards = market.get("clobRewards") or []
    if not rewards:
        return 0.0
    rate = float(rewards[0].get("rewardsDailyRate") or 0)
    min_size = float(market.get("rewardsMinSize") or 50)
    max_spread_cents = float(market.get("rewardsMaxSpread") or 3.5)
    if rate <= 0 or size_usd < min_size:
        return 0.0
    if half_spread * 100 > max_spread_cents:
        return 0.0
    days = quote_hours / 24.0
    return rate * days * competition_share


def simulate_one_outcome(
    history: list[dict],
    fair: float,
    half_spread: float,
    pin_close: float,
    won_yes: bool,
    market: dict,
    kickoff: datetime,
) -> MakerFill | None:
    if not history or fair is None or np.isnan(fair):
        return None
    bid = max(0.01, fair - half_spread)
    ask = min(0.99, fair + half_spread)
    hist = sorted(history, key=lambda h: h.get("t", 0))
    fill_side = None
    fill_p = None
    fill_t = None
    for pt in hist:
        p = float(pt.get("p", 0))
        if p <= bid:
            fill_side, fill_p, fill_t = "buy", bid, pt.get("t")
            break
        if p >= ask:
            fill_side, fill_p, fill_t = "sell", ask, pt.get("t")
            break
    if fill_side is None:
        return None
    if fill_side == "buy":
        clv = pin_close - fill_p
        ret = _settlement_return_buy(fill_p, won_yes)
    else:
        clv = fill_p - pin_close
        ret = _settlement_return_sell(fill_p, won_yes)
    quote_hours = 48.0
    if fill_t and hist:
        quote_hours = max(1.0, (hist[-1].get("t", 0) - hist[0].get("t", 0)) / 3600.0)
    reward = estimate_liquidity_reward(market, half_spread, quote_hours)
    return MakerFill(
        kickoff=kickoff,
        competition="",
        slug="",
        outcome_label="",
        side=fill_side,
        fill_price=fill_p,
        fair_pin=fair,
        pin_close_prob=pin_close,
        clv=clv,
        settlement_return=ret,
        reward_estimate=reward,
        volume=float(market.get("volume") or 0),
    )


def fetch_closed_soccer_events(
    tags: list[str],
    max_per_tag: int = 200,
    page_size: int = 100,
) -> list[dict]:
    seen: set[str] = set()
    out: list[dict] = []
    for tag in tags:
        offset = 0
        fetched = 0
        while fetched < max_per_tag:
            url = f"{GAMMA}/events?tag_slug={tag}&closed=true&limit={page_size}&offset={offset}"
            batch = _get(url)
            if not batch:
                break
            for e in batch:
                if not is_match_event(e):
                    continue
                slug = e.get("slug")
                if slug in seen:
                    continue
                seen.add(slug)
                out.append(e)
            fetched += page_size
            offset += page_size
            if len(batch) < page_size:
                break
            time.sleep(0.15)
    return out


def match_event_to_row(
    event: dict,
    df: pd.DataFrame,
    teams_idx: dict[str, dict[str, str]],
) -> pd.Series | None:
    comp = _event_competition(event, {})
    markets = event.get("markets") or []
    t1, t2 = _teams_from_markets(markets)
    kickoff = _parse_kickoff(event)
    if not t1 or not t2 or kickoff is None:
        return None
    if comp is None:
        comp = _event_competition(event, {c: c for c in df.competition.unique()})
    if comp is None:
        return None
    sub = df[df.competition == comp]
    if sub.empty:
        return None
    idx = teams_idx.get(comp) or build_map(sub.home_team.unique().tolist() + sub.away_team.unique().tolist())
    teams_idx[comp] = idx
    h = lookup(idx, t1)
    a = lookup(idx, t2)
    if not h or not a:
        h2, a2 = lookup(idx, t2), lookup(idx, t1)
        if h2 and a2:
            h, a = h2, a2
    if not h or not a:
        return None
    day = pd.Timestamp(kickoff.date())
    cand = sub[
        (sub.home_team == h) & (sub.away_team == a)
        & (sub.date >= day - pd.Timedelta(days=2))
        & (sub.date <= day + pd.Timedelta(days=2))
    ]
    if cand.empty:
        cand = sub[
            ((sub.home_team == h) & (sub.away_team == a))
            | ((sub.home_team == a) & (sub.away_team == h))
        ].copy()
        cand["delta"] = (cand.date - day).abs()
        cand = cand.sort_values("delta").head(1)
    else:
        cand = cand.sort_values("date").head(1)
    if cand.empty:
        return None
    return cand.iloc[0]


def run_maker_backtest(
    events: list[dict],
    df: pd.DataFrame,
    half_spread: float = 0.02,
    fair_use_close: bool = False,
    min_volume: float = 1000.0,
    max_events: int | None = None,
) -> list[MakerFill]:
    """Simula maker em cada mercado 1X2; fair = Pinnacle devig (abertura ou fechamento)."""
    fills: list[MakerFill] = []
    teams_idx: dict[str, dict[str, str]] = {}
    count = 0
    for event in events:
        if max_events and count >= max_events:
            break
        markets = event.get("markets") or []
        kickoff = _parse_kickoff(event)
        if kickoff is None:
            continue
        row = match_event_to_row(event, df, teams_idx)
        if row is None:
            continue
        home, away = row.home_team, row.away_team
        roles = outcome_roles_from_markets(markets, home, away)
        if roles is None:
            continue
        winner = _winner_from_markets(markets, roles)
        if winner is None:
            continue
        pin_open = _pin_probs(row, use_close=False)
        pin_close = _pin_probs(row, use_close=True)
        if pin_close is None:
            continue
        fair_probs = pin_close if fair_use_close else (pin_open or pin_close)
        if fair_probs is None:
            continue
        role_to_idx = {"casa": 0, "empate": 1, "fora": 2}
        for mi, (market, role) in enumerate(zip(markets, roles)):
            vol = float(market.get("volume") or 0)
            if vol < min_volume:
                continue
            token = parse_token_ids(market)
            if not token:
                continue
            hist = fetch_price_history(token, kickoff)
            if len(hist) < 5:
                continue
            ri = role_to_idx[role]
            fair = fair_probs[ri]
            p_close = pin_close[ri]
            try:
                yes_won = float(json.loads(market["outcomePrices"])[0]) > 0.9
            except Exception:
                continue
            mf = simulate_one_outcome(
                hist, fair, half_spread, p_close, yes_won, market, kickoff,
            )
            if mf is None:
                continue
            mf.competition = row.competition
            mf.slug = event.get("slug") or ""
            mf.outcome_label = role
            fills.append(mf)
            count += 1
        time.sleep(0.05)
    return fills


def summarize_maker_fills(
    fills: list[MakerFill],
    holdout_cut: pd.Timestamp,
) -> dict:
    if not fills:
        return {"n_dev": 0, "n_holdout": 0}

    def _pack(sub: list[MakerFill]) -> dict:
        if not sub:
            return {"n": 0, "clv_mean": None, "clv_ci95": None, "roi": None, "roi_ci95": None,
                    "reward_mean": None}
        clv = np.array([f.clv for f in sub], dtype=float)
        ret = np.array([f.settlement_return for f in sub], dtype=float)
        rew = np.array([f.reward_estimate for f in sub], dtype=float)
        cm, clo, chi = bootstrap_ci(clv)
        rm, rlo, rhi = bootstrap_ci(ret)
        rwm, _, _ = bootstrap_ci(rew)
        ret_plus = ret + rew
        rpm, rplo, rphi = bootstrap_ci(ret_plus)
        return {
            "n": len(sub),
            "clv_mean": round(cm, 5),
            "clv_ci95": [round(clo, 5), round(chi, 5)],
            "roi": round(rm, 4),
            "roi_ci95": [round(rlo, 4), round(rhi, 4)],
            "reward_mean": round(rwm, 4),
            "roi_with_rewards_est": round(rpm, 4),
            "roi_with_rewards_ci95": [round(rplo, 4), round(rphi, 4)],
        }

    cut = pd.Timestamp(holdout_cut)
    if cut.tzinfo is not None:
        cut = cut.tz_convert(None)
    dev = []
    hold = []
    for f in fills:
        kt = pd.Timestamp(f.kickoff)
        if kt.tzinfo is not None:
            kt = kt.tz_convert(None)
        if kt < cut:
            dev.append(f)
        else:
            hold.append(f)
    dev_s = _pack(dev)
    hold_s = _pack(hold)
    clv_viable = (
        hold_s["n"] >= 300
        and hold_s["clv_mean"] is not None
        and hold_s["clv_mean"] > 0
        and hold_s["clv_ci95"][0] > 0
    )
    pnl_viable = (
        hold_s["n"] >= 300
        and hold_s["roi"] is not None
        and hold_s["roi_ci95"] is not None
        and hold_s["roi_ci95"][0] > 0
    )
    return {
        "dev": dev_s,
        "holdout": hold_s,
        "viable": clv_viable and pnl_viable,
        "clv_viable": clv_viable,
        "pnl_viable": pnl_viable,
    }
