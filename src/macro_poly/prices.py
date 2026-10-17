from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

from src.macro_poly.http_util import get_json

CLOB = "https://clob.polymarket.com"
CHUNK_SECONDS = 14 * 24 * 3600


def fetch_price_history(
    token_id: str,
    start_ts: int,
    end_ts: int,
    fidelity_min: int = 60,
) -> list[dict]:
    """Busca histórico em blocos (API limita janela fina)."""
    if end_ts <= start_ts:
        return []
    all_pts: list[dict] = []
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
            pass
        cur = chunk_end
    # dedupe by timestamp
    by_t = {int(p["t"]): float(p["p"]) for p in all_pts if "t" in p and "p" in p}
    return [{"t": t, "p": by_t[t]} for t in sorted(by_t)]


def load_or_fetch_history(
    token_id: str,
    start_ts: int,
    end_ts: int,
    cache_dir: Path,
    fidelity_min: int = 60,
) -> list[dict]:
    cache_dir.mkdir(parents=True, exist_ok=True)
    path = cache_dir / f"{token_id}.json"
    if path.exists():
        with open(path, encoding="utf-8") as f:
            cached = json.load(f)
        if cached.get("end_ts", 0) >= end_ts - 3600:
            return cached.get("history") or []
    hist = fetch_price_history(token_id, start_ts, end_ts, fidelity_min=fidelity_min)
    payload = {
        "token_id": token_id,
        "start_ts": start_ts,
        "end_ts": end_ts,
        "fidelity_min": fidelity_min,
        "history": hist,
    }
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f)
    return hist


def price_at_or_after(history: list[dict], ts: int) -> float | None:
    if not history:
        return None
    best = None
    for pt in history:
        t = int(pt["t"])
        if t >= ts:
            best = float(pt["p"])
            break
    if best is not None:
        return best
    return float(history[-1]["p"])


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
    return min(0.99, mid + spread / 2)


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
