"""DOGE trend signal and shared-account sizing. No network."""

import json
import socket
from datetime import datetime, timedelta
from zoneinfo import ZoneInfo

import pytest

from trading.hyperliquid import signal_check as sig

SP = ZoneInfo("America/Sao_Paulo")
HOUR = sig.HOUR_MS
START = datetime(2026, 10, 1, 0, 0, tzinfo=SP)
START_MS = int(START.timestamp() * 1000)


@pytest.fixture(autouse=True)
def _block_network(monkeypatch):
    def refuse(*_args, **_kwargs):
        raise AssertionError("tests must not open a socket")

    monkeypatch.setattr(socket, "socket", refuse)


def _bar(i, price, high=None, low=None, close=None):
    t = START_MS + i * HOUR
    c = price if close is None else close
    h = price * 1.01 if high is None else high
    l = price * 0.99 if low is None else low
    return {"t": t, "T": t + HOUR - 1, "o": float(price), "h": float(h), "l": float(l), "c": float(c)}


def _breakout(price=0.075):
    candles = [_bar(i, price, high=price, low=price * 0.99, close=price * 0.995) for i in range(20)]
    candles.append(_bar(20, price * 1.05, high=price * 1.06, low=price * 1.04, close=price * 1.05))
    return candles


def _drop(price=0.075):
    candles = [_bar(i, price, high=price * 1.001, low=price * 0.999, close=price) for i in range(30)]
    candles.append(_bar(30, price * 0.8, high=price * 0.81, low=price * 0.79, close=price * 0.8))
    return candles


def _now_after(candles):
    return datetime.fromtimestamp((candles[-1]["T"] + 60_000) / 1000, tz=SP)


def _account(value=78.0, withdrawable=40.0, btc=None, doge=None):
    return {"account_value": value, "withdrawable": withdrawable, "btc": btc, "doge": doge}


def _market(sz_decimals=0, max_leverage=10, mark=0.07875, tradeable=True):
    return {
        "coin": "DOGE",
        "exists": True,
        "asset_index": 12,
        "sz_decimals": sz_decimals,
        "max_leverage": max_leverage,
        "only_isolated": False,
        "is_delisted": False,
        "mark_px": mark,
        "tradeable": tradeable,
    }


def _meta_rows(doge_extra=None):
    doge = {"szDecimals": 0, "name": "DOGE", "maxLeverage": 10, "marginTableId": 52}
    doge.update(doge_extra or {})
    meta = {
        "universe": [
            {"szDecimals": 5, "name": "BTC", "maxLeverage": 40},
            {"szDecimals": 4, "name": "ETH", "maxLeverage": 25},
            doge,
        ]
    }
    ctxs = [{"markPx": "77883.0"}, {"markPx": "3000.0"}, {"markPx": "0.075612"}]
    return [meta, ctxs]


def _clearinghouse(btc_size=None, doge_size=None, value="78.390005", withdrawable="8.052391"):
    positions = []
    if btc_size is not None:
        positions.append(
            {"type": "oneWay", "position": {"coin": "BTC", "szi": btc_size, "entryPx": "77976.0", "unrealizedPnl": "0"}}
        )
    if doge_size is not None:
        positions.append(
            {"type": "oneWay", "position": {"coin": "DOGE", "szi": doge_size, "entryPx": "0.075", "unrealizedPnl": "0"}}
        )
    return {"marginSummary": {"accountValue": value}, "withdrawable": withdrawable, "assetPositions": positions}


def _rows(candles):
    return [
        {"t": c["t"], "T": c["T"], "o": str(c["o"]), "h": str(c["h"]), "l": str(c["l"]), "c": str(c["c"])}
        for c in candles
    ]


# --- sizing -----------------------------------------------------------------


def test_recommended_notional_is_capped_by_half_equity_and_free_balance():
    # Free balance is the cap.
    assert sig.recommended_notional(78.39, 8.052391) == pytest.approx(8.052391 * 1.8)
    # Half the equity is the cap.
    assert sig.recommended_notional(78.0, 60.0) == pytest.approx(39.0 * 1.8)
    assert sig.recommended_notional(20.0, 20.0) == pytest.approx(18.0)
    # Negative free balance never sizes a negative order.
    assert sig.recommended_notional(30.0, -2.0) == 0.0
    assert sig.recommended_notional(None, 5.0) is None
    assert sig.recommended_notional("nan", 5.0) is None


def test_round_size_down_respects_sz_decimals():
    assert sig.round_size_down(191.7, 0) == 191.0
    assert sig.round_size_down(0.000186789, 5) == pytest.approx(0.00018)
    assert sig.round_size_down(1.23456, 2) == pytest.approx(1.23)
    # Float noise just under a lot boundary does not round up.
    assert sig.round_size_down(0.29999999999, 1) == pytest.approx(0.2)


def test_size_order_uses_mark_and_floors():
    sized = sig.size_order(14.494304, 0.07, _market(mark=0.075612))
    assert sized["size_coin"] == 191.0
    assert sized["size_price"] == pytest.approx(0.075612)
    assert sized["size_notional_usd"] <= 14.494304
    no_market = sig.size_order(14.49, 0.07, None)
    assert no_market["size_coin"] is None


# --- market metadata --------------------------------------------------------


def test_parse_market_finds_doge_and_its_limits():
    market = sig.parse_market(_meta_rows(), "DOGE")
    assert market["exists"] is True
    assert market["sz_decimals"] == 0
    assert market["max_leverage"] == 10
    assert market["mark_px"] == pytest.approx(0.075612)
    assert market["tradeable"] is True
    btc = sig.parse_market(_meta_rows(), "BTC")
    assert btc["sz_decimals"] == 5 and btc["max_leverage"] == 40


def test_parse_market_flags_delisted_missing_and_low_leverage():
    assert sig.parse_market(_meta_rows({"isDelisted": True}), "DOGE")["tradeable"] is False
    assert sig.parse_market(_meta_rows({"maxLeverage": 1}), "DOGE")["tradeable"] is False
    missing = sig.parse_market(_meta_rows(), "NOPE")
    assert missing["exists"] is False and missing["tradeable"] is False
    with pytest.raises(RuntimeError):
        sig.parse_market({"error": "x"}, "DOGE")


# --- DOGE decisions ---------------------------------------------------------


def test_doge_breakout_enters_with_rounded_size():
    candles = _breakout()
    payload, state = sig.decide_doge(
        candles, _account(value=78.39, withdrawable=8.052391), None, _now_after(candles), market=_market()
    )
    assert payload["action"] == "enter"
    assert payload["coin"] == "DOGE"
    assert payload["size_usd"] == pytest.approx(8.052391 * 1.8)
    assert payload["size_coin"] == float(int(payload["size_usd"] / 0.07875))
    assert payload["size_coin"] == payload["recommended_size"]
    assert payload["sz_decimals"] == 0
    assert payload["max_leverage"] == 10
    assert payload["tradeable"] is True
    assert payload["perp_exists"] is True
    assert payload["size_notional_usd"] == pytest.approx(payload["size_coin"] * 0.07875)
    assert payload["size_notional_usd"] <= payload["size_usd"]
    assert payload["leverage"] == 2
    assert payload["margin_mode"] == "isolated"
    assert payload["expiry"] == "2026-12-01"
    assert state["position"] is None
    assert "DOGE" in state["note"]


def test_doge_skip_small_instead_of_enter_and_not_paused():
    candles = _breakout()
    now = _now_after(candles)
    small, _ = sig.decide_doge(candles, _account(value=78, withdrawable=5), None, now, market=_market())
    assert small["action"] == "skip_small"
    assert small["size_usd"] == pytest.approx(9.0)
    assert "low_balance" not in small
    # The notional clears 10 but the lot rounding does not.
    lot, _ = sig.decide_doge(
        candles, _account(value=78, withdrawable=5.8), None, now, market=_market(mark=3.0)
    )
    assert lot["size_usd"] == pytest.approx(10.44)
    assert lot["size_coin"] == 3.0
    assert lot["action"] == "skip_small"
    assert "rounded" in lot["reason"]


def test_low_equity_still_pauses():
    candles = _breakout()
    paused, _ = sig.decide_doge(
        candles, _account(value=11.99, withdrawable=11.99), None, _now_after(candles), market=_market()
    )
    assert paused["action"] == "pause_low_balance"


def test_doge_expiry_is_1_december_2026():
    candles = _breakout()
    last_day = datetime(2026, 12, 1, 23, 30, tzinfo=SP)
    still, _ = sig.decide_doge(candles, _account(), None, last_day, market=_market())
    assert still["action"] == "enter"
    after, _ = sig.decide_doge(candles, _account(), None, datetime(2026, 12, 2, 0, 5, tzinfo=SP), market=_market())
    assert after["action"] == "expired"
    assert after["reason"] == "after_2026-12-01"
    # BTC keeps its own date.
    btc, _ = sig.decide_btc(candles, _account(), None, datetime(2026, 11, 19, 0, 5, tzinfo=SP))
    assert btc["action"] == "expired"
    assert btc["reason"] == "after_2026-11-18"


def test_metadata_failure_blocks_entry_but_not_exit():
    candles = _breakout()
    blocked, _ = sig.decide_doge(
        candles, _account(), None, _now_after(candles), market=None, market_error="RuntimeError: down"
    )
    assert blocked["action"] == "error"
    assert "metadata" in blocked["reason"]
    drop = _drop()
    state = {
        "position": {"entry_price": 0.075, "size": 200.0, "entry_time": sig.ms_to_iso(drop[0]["t"])},
        "highest_close": 0.075,
        "stop": 0.07,
        "last_processed_candle": sig.ms_to_iso(drop[-2]["t"]),
    }
    account = _account(doge={"size": 200.0, "entry_px": 0.075, "upnl": 0.0})
    hit, _ = sig.decide_doge(drop, account, state, _now_after(drop), market=None, market_error="down")
    assert hit["action"] == "exit"
    assert hit["reason"] == "stop_hit"


def test_entry_without_metadata_is_refused_even_without_an_error_text():
    candles = _breakout()
    payload, _ = sig.decide_doge(candles, _account(), None, _now_after(candles))
    assert payload["action"] == "error"
    assert "not provided" in payload["reason"]
    assert payload["size_coin"] is None


def test_not_tradeable_does_not_enter():
    candles = _breakout()
    payload, _ = sig.decide_doge(candles, _account(), None, _now_after(candles), market=_market(tradeable=False))
    assert payload["action"] == "none"
    assert payload["reason"] == "not_tradeable"


def test_btc_long_does_not_leak_into_doge_and_vice_versa():
    candles = _breakout()
    now = _now_after(candles)
    account = _account(btc={"size": 0.00181, "entry_px": 77976.0, "upnl": 0.0})
    doge, doge_state = sig.decide_doge(candles, account, None, now, market=_market())
    assert doge["action"] == "enter"
    assert "state_mismatch" not in doge
    assert doge_state["position"] is None
    only_doge = _account(doge={"size": 191.0, "entry_px": 0.0756, "upnl": 0.0})
    btc_market = dict(_market(sz_decimals=5, max_leverage=40, mark=78000.0), coin="BTC")
    btc, btc_state = sig.decide_btc(candles, only_doge, None, now, market=btc_market)
    assert btc_state["position"] is None
    assert btc["action"] == "enter"


def test_parse_account_reads_doge_and_fills_filter_by_coin():
    account = sig.parse_account(_clearinghouse(btc_size="0.00181", doge_size="191"))
    assert account["btc"]["size"] == pytest.approx(0.00181)
    assert account["doge"]["size"] == pytest.approx(191)
    now_ms = START_MS + 100 * HOUR
    fills = [
        {"coin": "DOGE", "side": "B", "sz": "191", "time": START_MS + 10 * HOUR, "startPosition": "0"},
        {"coin": "BTC", "side": "B", "sz": "0.001", "time": START_MS + 20 * HOUR, "startPosition": "0"},
    ]
    assert sig.opening_buy_time_ms(fills, now_ms, "DOGE") == START_MS + 10 * HOUR
    assert sig.opening_buy_time_ms(fills, now_ms, "BTC") == START_MS + 20 * HOUR
    assert sig.opening_buy_time_ms(fills, now_ms) == START_MS + 20 * HOUR


# --- end to end -------------------------------------------------------------


def _fake_api(candles, book, seen):
    def fake(payload, timeout=20.0):
        seen.append(payload)
        kind = payload["type"]
        if kind == "candleSnapshot":
            return _rows(candles)
        if kind == "clearinghouseState":
            assert payload["user"] == sig.ACCOUNT
            return book
        if kind == "metaAndAssetCtxs":
            return _meta_rows()
        raise AssertionError(kind)

    return fake


def test_run_doge_writes_only_doge_files(tmp_path, monkeypatch):
    candles = _breakout()
    now = _now_after(candles)
    btc_state = {
        "position": {"entry_price": 77976.0, "size": 0.00181, "entry_time": "2026-11-01T03:08-03:00"},
        "stop": 77582.6,
        "highest_close": 78230.0,
    }
    btc_path = tmp_path / "btc_trend_state.json"
    btc_path.write_text(json.dumps(btc_state), encoding="utf-8")
    frozen = btc_path.read_bytes()
    seen = []
    monkeypatch.setattr(sig, "post_info", _fake_api(candles, _clearinghouse(btc_size="0.00181"), seen))
    payload = sig.run_doge(tmp_path, now=now)
    assert payload["action"] == "enter"
    assert payload["coin"] == "DOGE"
    assert payload["size_usd"] == pytest.approx(8.052391 * 1.8)
    assert payload["size_coin"] == 191.0
    assert payload["tradeable"] is True
    assert [p["req"]["coin"] for p in seen if p["type"] == "candleSnapshot"] == ["DOGE"]
    assert btc_path.read_bytes() == frozen
    saved = json.loads((tmp_path / "doge_trend_state.json").read_text(encoding="utf-8"))
    assert saved["position"] is None
    assert saved["available_usd"] == pytest.approx(8.052391)
    assert (tmp_path / ".doge_state.lock").exists()


def test_run_btc_survives_metadata_failure(tmp_path, monkeypatch):
    drop = _drop(price=78000.0)
    state = {
        "position": {"entry_price": 78000.0, "size": 0.00181, "entry_time": sig.ms_to_iso(drop[0]["t"])},
        "highest_close": 78000.0,
        "stop": 77000.0,
        "last_processed_candle": sig.ms_to_iso(drop[-2]["t"]),
    }
    (tmp_path / "btc_trend_state.json").write_text(json.dumps(state), encoding="utf-8")

    def fake(payload, timeout=20.0):
        if payload["type"] == "candleSnapshot":
            return _rows(drop)
        if payload["type"] == "clearinghouseState":
            return _clearinghouse(btc_size="0.00181")
        raise RuntimeError("meta down")

    monkeypatch.setattr(sig, "post_info", fake)
    payload = sig.run_btc(tmp_path, now=_now_after(drop))
    assert payload["action"] == "exit"
    assert "meta down" in payload["market_error"]


def test_cli_doge_record_entry_and_exit_use_doge_files(tmp_path, capsys):
    rc = sig.main(["doge", "--state-dir", str(tmp_path), "--record-entry", "0.0756", "191"])
    assert rc == 0
    entry = json.loads(capsys.readouterr().out)
    assert entry["action"] == "recorded_entry"
    assert not (tmp_path / "btc_trend_state.json").exists()
    assert not (tmp_path / "btc_trend_log.md").exists()
    state = json.loads((tmp_path / "doge_trend_state.json").read_text(encoding="utf-8"))
    assert state["position"]["size"] == 191.0
    log = (tmp_path / "doge_trend_log.md").read_text(encoding="utf-8")
    assert log.startswith("# DOGE trend log")
    rc = sig.main(["doge", "--state-dir", str(tmp_path), "--record-exit", "0.08"])
    assert rc == 0
    assert json.loads(capsys.readouterr().out)["action"] == "recorded_exit"
    state = json.loads((tmp_path / "doge_trend_state.json").read_text(encoding="utf-8"))
    assert state["position"] is None
    with pytest.raises(SystemExit):
        sig.main(["eth", "--state-dir", str(tmp_path)])
