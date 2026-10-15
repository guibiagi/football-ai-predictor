"""Cliente Gamma API com cache e paginação."""

from __future__ import annotations

import json
import time
import urllib.error
import urllib.request
from pathlib import Path

GAMMA = "https://gamma-api.polymarket.com"
CACHE_DIR = Path(__file__).resolve().parent.parent / "data" / "gamma_cache"


def _get(url: str, retries: int = 4) -> list | dict:
    for attempt in range(retries):
        try:
            req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
            with urllib.request.urlopen(req, timeout=90) as resp:
                return json.load(resp)
        except (urllib.error.URLError, TimeoutError, json.JSONDecodeError):
            time.sleep(2 ** attempt)
    raise RuntimeError(url)


def fetch_events_closed(
    tag_slug: str | None = None,
    limit: int = 100,
    max_pages: int = 50,
    title_filter: str | None = None,
) -> list[dict]:
    cache_key = f"events_{tag_slug or 'all'}_{title_filter or 'all'}_{max_pages}.json"
    cache_path = CACHE_DIR / cache_key
    if cache_path.exists():
        return json.loads(cache_path.read_text(encoding="utf-8"))

    out: list[dict] = []
    offset = 0
    for _ in range(max_pages):
        parts = [f"closed=true", f"limit={limit}", f"offset={offset}"]
        if tag_slug:
            parts.append(f"tag_slug={tag_slug}")
        url = f"{GAMMA}/events?{'&'.join(parts)}"
        try:
            batch = _get(url)
        except RuntimeError:
            break
        if not batch:
            break
        for e in batch:
            if title_filter:
                if title_filter.lower() not in (e.get("title") or "").lower():
                    continue
            out.append(e)
        offset += limit
        if len(batch) < limit:
            break
        time.sleep(0.12)

    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    cache_path.write_text(json.dumps(out, ensure_ascii=False), encoding="utf-8")
    return out
