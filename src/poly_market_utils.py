"""Utilitários comuns para backtests Polymarket."""

from __future__ import annotations

import json
import time
import urllib.request
from datetime import datetime, timezone

from src.reporting import bootstrap_ci

CLOB = "https://clob.polymarket.com"
HIST_CACHE = __import__("pathlib").Path(__file__).resolve().parent.parent / "data" / "poly_price_history"


def parse_token_yes(market: dict) -> str | None:
    raw = market.get("clobTokenIds")
    if not raw:
        return None
    try:
        ids = json.loads(raw) if isinstance(raw, str) else raw
        return ids[0] if ids else None
    except (json.JSONDecodeError, TypeError, IndexError):
        return None


def parse_kickoff(event: dict, market: dict | None = None) -> datetime | None:
    for key in ("gameStartTime", "startTime", "endDate"):
        raw = (market or {}).get(key) or event.get(key)
        if not raw:
            continue
        try:
            s = str(raw).replace(" ", "T")
            if "+" in s and "T" not in s.split("+")[0]:
                s = s.replace(" ", "T", 1)
            return datetime.fromisoformat(s.replace("Z", "+00:00"))
        except ValueError:
            continue
    return None


def yes_won(market: dict) -> bool | None:
    try:
        p = float(json.loads(market["outcomePrices"])[0])
        if p > 0.9:
            return True
        if p < 0.1:
            return False
    except Exception:
        return None
    return None


def price_history(token_id: str, start_ts: int, end_ts: int, fidelity: int = 60) -> list[dict]:
    cache = HIST_CACHE / f"{token_id}_{start_ts}_{end_ts}_{fidelity}.json"
    if cache.exists():
        return json.loads(cache.read_text(encoding="utf-8"))
    url = (
        f"{CLOB}/prices-history?market={token_id}"
        f"&startTs={start_ts}&endTs={end_ts}&fidelity={fidelity}"
    )
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    try:
        with urllib.request.urlopen(req, timeout=60) as resp:
            data = json.load(resp)
        hist = data.get("history") or []
        if hist:
            HIST_CACHE.mkdir(parents=True, exist_ok=True)
            cache.write_text(json.dumps(hist), encoding="utf-8")
        return hist
    except Exception:
        return []


def max_price_in_range(
    hist: list[dict],
    window_end_ts: int,
    hours_before_lo: float,
    hours_before_hi: float,
) -> float | None:
    lo = window_end_ts - int(hours_before_hi * 3600)
    hi = window_end_ts - int(hours_before_lo * 3600)
    pts = [float(h["p"]) for h in hist if lo <= h.get("t", 0) <= hi]
    return max(pts) if pts else None


def mean_price_in_window(
    hist: list[dict],
    window_end_ts: int,
    hours_before: float,
    half_width_h: float = 0.5,
) -> float | None:
    """Média de preço YES na janela [end - hours_before ± half_width]."""
    lo = window_end_ts - int((hours_before + half_width_h) * 3600)
    hi = window_end_ts - int((hours_before - half_width_h) * 3600)
    pts = [float(h["p"]) for h in hist if lo <= h.get("t", 0) <= hi]
    if not pts:
        return None
    return float(sum(pts) / len(pts))


def summarize_returns(returns: list[float], clv: list[float] | None = None) -> dict:
    if not returns:
        return {"n": 0}
    r = __import__("numpy").asarray(returns, dtype=float)
    m, lo, hi = bootstrap_ci(r)
    out = {"n": len(r), "roi_mean": round(m, 4), "roi_ci95": [round(lo, 4), round(hi, 4)]}
    if clv:
        c = __import__("numpy").asarray(clv, dtype=float)
        cm, clo, chi = bootstrap_ci(c)
        out["clv_mean"] = round(cm, 5)
        out["clv_ci95"] = [round(clo, 5), round(chi, 5)]
    return out


def pnl_viable(summary: dict, n_min: int = 300) -> bool:
    if summary.get("n", 0) < n_min:
        return False
    ci = summary.get("roi_ci95")
    return ci is not None and summary.get("roi_mean", 0) > 0 and ci[0] > 0
