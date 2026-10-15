"""CLOB Polymarket: order book (somente leitura)."""

from __future__ import annotations

import json
import time
import urllib.error
import urllib.request
from pathlib import Path

CLOB = "https://clob.polymarket.com"
BOOK_CACHE = Path(__file__).resolve().parent.parent / "data" / "poly_book_cache"


def fetch_order_book(token_id: str, use_cache: bool = True) -> dict:
    cache_path = BOOK_CACHE / f"{token_id}.json"
    if use_cache and cache_path.exists():
        try:
            return json.loads(cache_path.read_text(encoding="utf-8"))
        except Exception:
            pass
    url = f"{CLOB}/book?token_id={token_id}"
    for attempt in range(4):
        try:
            req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
            with urllib.request.urlopen(req, timeout=30) as resp:
                data = json.load(resp)
            if use_cache and "asks" in data:
                BOOK_CACHE.mkdir(parents=True, exist_ok=True)
                cache_path.write_text(json.dumps(data), encoding="utf-8")
            return data
        except (urllib.error.URLError, TimeoutError, json.JSONDecodeError):
            time.sleep(0.5 * (attempt + 1))
    return {}


def best_ask(book: dict) -> tuple[float | None, float]:
    asks = book.get("asks") or []
    if not asks:
        return None, 0.0
    a = asks[0]
    return float(a.get("price", 0)), float(a.get("size", 0))


def best_bid(book: dict) -> tuple[float | None, float]:
    bids = book.get("bids") or []
    if not bids:
        return None, 0.0
    b = bids[0]
    return float(b.get("price", 0)), float(b.get("size", 0))
