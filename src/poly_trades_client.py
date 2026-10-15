"""Trades taker via data-api.polymarket.com (público)."""

from __future__ import annotations

import json
import time
import urllib.error
import urllib.request
from pathlib import Path

DATA_API = "https://data-api.polymarket.com"
CACHE = Path(__file__).resolve().parent.parent / "data" / "poly_trades_cache"


def fetch_trades(
    condition_id: str,
    limit: int = 500,
    max_pages: int = 20,
    use_cache: bool = True,
    cache_only: bool = False,
) -> list[dict]:
    cid = condition_id.lower()
    cache_path = CACHE / f"{cid.replace('0x','')[:16]}_{limit}.json"
    if use_cache and cache_path.exists():
        return json.loads(cache_path.read_text(encoding="utf-8"))
    if cache_only:
        return []

    out: list[dict] = []
    offset = 0
    for _ in range(max_pages):
        url = f"{DATA_API}/trades?market={condition_id}&limit={limit}&offset={offset}"
        try:
            req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
            with urllib.request.urlopen(req, timeout=60) as resp:
                batch = json.load(resp)
        except (urllib.error.URLError, json.JSONDecodeError, TimeoutError):
            break
        if not batch:
            break
        out.extend(batch)
        if len(batch) < limit:
            break
        offset += limit
        time.sleep(0.08)

    if use_cache and out:
        CACHE.mkdir(parents=True, exist_ok=True)
        cache_path.write_text(json.dumps(out), encoding="utf-8")
    return out
