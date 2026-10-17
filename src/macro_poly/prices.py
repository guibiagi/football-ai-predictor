from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

from src.macro_poly.http_util import get_json

CLOB = "https://clob.polymarket.com"
CHUNK_SECONDS = 14 * 24 * 3600
# Fidelidade CLOB usada no backtest (minutos); tolerância de entrada alinhada.
DEFAULT_FIDELITY_MIN = 60
ENTRY_MAX_LAG_SEC = DEFAULT_FIDELITY_MIN * 60


def fetch_price_history(
    token_id: str,
    start_ts: int,
    end_ts: int,
    fidelity_min: int = DEFAULT_FIDELITY_MIN,
) -> tuple[list[dict], bool]:
    """Busca histórico em blocos. Retorna (history, had_network_error)."""
    if end_ts <= start_ts:
        return [], False
    all_pts: list[dict] = []
    had_error = False
    cur = start_ts
    while cur < end_ts:
        chunk_end = min(cur + CHUNK_SECONDS, end_ts)
        url = (
            f"{CLOB}/prices-history?market={token_id}"
            f"&startTs={cur}&endTs={chunk_end}&fidelity={fidelity_min}"
        )
        try:
            data = get_json(url)
            hist = data.get("history") or []
            all_pts.extend(hist)
        except Exception:
            had_error = True
        cur = chunk_end
    by_t = {int(p["t"]): float(p["p"]) for p in all_pts if "t" in p and "p" in p}
    return [{"t": t, "p": by_t[t]} for t in sorted(by_t)], had_error


def load_or_fetch_history(
    token_id: str,
    start_ts: int,
    end_ts: int,
    cache_dir: Path,
    fidelity_min: int = DEFAULT_FIDELITY_MIN,
) -> list[dict]:
    cache_dir.mkdir(parents=True, exist_ok=True)
    path = cache_dir / f"{token_id}_{start_ts}_{end_ts}_{fidelity_min}.json"
    if path.exists():
        with open(path, encoding="utf-8") as f:
            cached = json.load(f)
        if not cached.get("fetch_failed"):
            return cached.get("history") or []
    hist, had_error = fetch_price_history(token_id, start_ts, end_ts, fidelity_min=fidelity_min)
    if had_error and not hist:
        payload = {"token_id": token_id, "fetch_failed": True, "history": []}
        with open(path, "w", encoding="utf-8") as f:
            json.dump(payload, f)
        return []
    payload = {
        "token_id": token_id,
        "start_ts": start_ts,
        "end_ts": end_ts,
        "fidelity_min": fidelity_min,
        "fetch_failed": False,
        "history": hist,
    }
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f)
    return hist


def price_at_or_after(
    history: list[dict],
    ts: int,
    max_lag_sec: int = ENTRY_MAX_LAG_SEC,
) -> float | None:
    """Primeiro preço em t >= ts; None se não houver ponto dentro da tolerância."""
    if not history:
        return None
    for pt in history:
        t = int(pt["t"])
        if t >= ts:
            if t - ts <= max_lag_sec:
                return float(pt["p"])
            return None
    return None


def price_at_or_before(history: list[dict], ts: int) -> float | None:
    if not history:
        return None
    best = None
    for pt in history:
        t = int(pt["t"])
        if t <= ts:
            best = float(pt["p"])
        else:
            break
    return best


def spread_adjusted_ask(mid: float, spread: float = 0.01) -> float:
    return mid + spread / 2


def iso_to_ts(iso: str) -> int | None:
    if not iso:
        return None
    try:
        dt = datetime.fromisoformat(iso.replace("Z", "+00:00"))
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        return int(dt.timestamp())
    except ValueError:
        return None
