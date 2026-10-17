from __future__ import annotations

import json
import time
import urllib.error
import urllib.request

DEFAULT_HEADERS = {"User-Agent": "football-ai-predictor-macro-backtest/1.0"}


def get_json(url: str, timeout: int = 60, retries: int = 4) -> dict | list:
    last_err: Exception | None = None
    for attempt in range(retries):
        try:
            req = urllib.request.Request(url, headers=DEFAULT_HEADERS)
            with urllib.request.urlopen(req, timeout=timeout) as resp:
                return json.load(resp)
        except (urllib.error.HTTPError, urllib.error.URLError, TimeoutError) as e:
            last_err = e
            time.sleep(0.5 * (2 ** attempt))
    raise last_err  # type: ignore[misc]
