from __future__ import annotations

import json
from pathlib import Path

import pytest

from src.macro_poly.trade_volume import (
    fetch_trades_paged,
    trade_dedupe_key,
)


@pytest.mark.integration
def test_data_api_offset_beats_single_page_without_start_window():
    """Live probe: bare offset>0 is empty; start/end window allows >100 rows."""
    import urllib.request

    manifest = json.loads(
        Path("data/macro/markets_manifest.json").read_text(encoding="utf-8")
    )
    m = next(x for x in manifest["markets"] if x.get("start_ts") and x.get("condition_id"))
    cid = m["condition_id"]
    hdr = {"User-Agent": "football-ai-predictor-macro-backtest/1.0"}
    req = urllib.request.Request(
        f"https://data-api.polymarket.com/trades?market={cid}&limit=100&offset=100",
        headers=hdr,
    )
    with urllib.request.urlopen(req, timeout=60) as resp:
        offset_only = json.load(resp)
    trades, meta = fetch_trades_paged(cid, {m["token_yes"]}, start_ts=m["start_ts"])
    assert len(offset_only) == 0, "offset without start/end window does not page"
    assert meta.trades_kept > 100, "windowed offset pagination must exceed 100 trades"
    assert meta.pagination_mode == "start_end_offset"


def test_trade_dedupe_key_uses_fill_tuple():
    t1 = {
        "transactionHash": "0xabc",
        "asset": "1",
        "side": "BUY",
        "size": "10",
    }
    t2 = {**t1, "price": "0.5"}
    assert trade_dedupe_key(t1) == trade_dedupe_key(t2)
    t3 = {**t1, "size": "11"}
    assert trade_dedupe_key(t1) != trade_dedupe_key(t3)
