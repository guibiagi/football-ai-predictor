from __future__ import annotations

import json
from pathlib import Path

from src.macro_poly.http_util import get_json

DATA_API = "https://data-api.polymarket.com/trades"


def fetch_trades(condition_id: str, token_yes: str, max_pages: int = 80) -> list[dict]:
    rows: list[dict] = []
    offset = 0
    for _ in range(max_pages):
        url = f"{DATA_API}?market={condition_id}&limit=500&offset={offset}"
        try:
            batch = get_json(url)
        except Exception:
            break
        if not batch:
            break
        for t in batch:
            if str(t.get("asset")) == str(token_yes):
                rows.append(t)
        if len(batch) < 500:
            break
        offset += 500
    return rows


def cumulative_volume_usd(trades: list[dict], until_ts: int) -> float:
    total = 0.0
    for t in trades:
        ts = int(t.get("timestamp") or 0)
        if ts > until_ts:
            continue
        total += float(t.get("size") or 0) * float(t.get("price") or 0)
    return total


def load_trades_cached(condition_id: str, token_yes: str, cache_dir: Path) -> list[dict] | None:
    cache_dir.mkdir(parents=True, exist_ok=True)
    path = cache_dir / f"{condition_id}.json"
    if path.exists():
        data = json.loads(path.read_text(encoding="utf-8"))
        if data.get("failed"):
            return None
        return data.get("trades") or []
    try:
        trades = fetch_trades(condition_id, token_yes)
        payload = {
            "condition_id": condition_id,
            "token_yes": token_yes,
            "failed": False,
            "trades": trades,
        }
        path.write_text(json.dumps(payload), encoding="utf-8")
        return trades
    except Exception:
        path.write_text(
            json.dumps({"condition_id": condition_id, "failed": True, "trades": []}),
            encoding="utf-8",
        )
        return None


class TradeVolumeStore:
    def __init__(self, cache_dir: Path):
        self.cache_dir = cache_dir
        self._trades: dict[str, list[dict] | None] = {}
        self._keys: dict[str, tuple[str, str]] = {}

    def register(self, token_yes: str, condition_id: str) -> None:
        self._keys[token_yes] = (condition_id, token_yes)

    def volume_at(self, token_yes: str, decision_ts: int) -> float | None:
        key = self._keys.get(token_yes)
        if not key:
            return None
        cid, tid = key
        if cid not in self._trades:
            self._trades[cid] = load_trades_cached(cid, tid, self.cache_dir)
        trades = self._trades[cid]
        if trades is None:
            return None
        return cumulative_volume_usd(trades, decision_ts)

    def coverage_fraction(self, markets, decision_ts_fn) -> float:
        ok = 0
        for m in markets:
            ts = decision_ts_fn(m)
            if ts is None:
                continue
            v = self.volume_at(m.token_yes, ts)
            if v is not None and v > 0:
                ok += 1
        return ok / max(len(markets), 1)
