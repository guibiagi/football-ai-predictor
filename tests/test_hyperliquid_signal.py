"""Signal checks for the read-only Hyperliquid script. No network."""

import json
import socket
from datetime import datetime, timedelta
from pathlib import Path
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd
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


def _bar(i, price, high=None, low=None, close=None, open_=None):
    t = START_MS + i * HOUR
    c = price if close is None else close
    h = price + 1 if high is None else high
    l = price - 1 if low is None else low
    o = price if open_ is None else open_
    return {"t": t, "T": t + HOUR - 1, "o": float(o), "h": float(h), "l": float(l), "c": float(c)}


def _flat(n, price=100.0, **overrides):
    return [_bar(i, price, **overrides) for i in range(n)]


def _now_after(candles):
    """A clock a few minutes after the last bar has closed, before the next one exists."""
    return datetime.fromtimestamp((candles[-1]["T"] + 60_000) / 1000, tz=SP)


def _account(value=20.0, withdrawable=20.0, btc=None):
    return {"account_value": value, "withdrawable": withdrawable, "btc": btc}


def _long(size=0.01, entry=100.0):
    return {"size": size, "entry_px": entry, "upnl": 1.0}


def _engine_atr(candles, n=14):
    """Copy of engine.atr: shift true range, then ewm(alpha=1/n, adjust=False)."""
    df = pd.DataFrame(
        {"h": [c["h"] for c in candles], "l": [c["l"] for c in candles], "c": [c["c"] for c in candles]}
    )
    pc = df.c.shift()
    tr = np.maximum(df.h - df.l, np.maximum((df.h - pc).abs(), (df.l - pc).abs()))
    return tr.ewm(alpha=1 / n, adjust=False).mean()


def _varied_candles():
    """Fixed path with a gap so true range is not just high-low."""
    rows = []
    price = 100.0
    for i in range(40):
        if i == 10:
            # Gap up: previous close stays near 100, this bar trades 110-112.
            rows.append(_bar(i, 111, high=112, low=110, close=111.5, open_=110))
            price = 111.5
            continue
        price += 0.3 if i % 2 == 0 else -0.15
        rows.append(_bar(i, price, high=price + 1.2, low=price - 0.8, close=price + 0.1, open_=price))
    return rows


def test_atr_matches_backtest_ewm():
    candles = _varied_candles()
    expected = _engine_atr(candles, 14)
    got = sig.atr_series(candles, 14)
    assert got[0] is None
    assert pd.isna(expected.iloc[0])
    for i in range(1, len(candles)):
        assert got[i] == pytest.approx(float(expected.iloc[i]), rel=1e-12, abs=1e-9)


def test_true_range_includes_gap_beyond_high_low():
    candles = [
        _bar(0, 100, high=101, low=99, close=100),
        _bar(1, 111, high=112, low=110, close=111),
    ]
    # high-low is 2; distance from the previous close is 12 and 10.
    assert sig.true_ranges(candles)[1] == pytest.approx(12.0)


def test_breakout_yes_sizes_notional_from_withdrawable():
    candles = _flat(20, price=100, high=100, low=99, close=99.5)
    candles.append(_bar(20, 101, high=102, low=100.5, close=101.2))
    payload, state = sig.decide_btc(candles, _account(value=20, withdrawable=20), None, _now_after(candles))
    assert payload["action"] == "enter"
    assert payload["reason"] == "breakout"
    assert payload["size_usd"] == sig.clean_float(20 * 2 * 0.9, 6)
    assert payload["size_usd"] >= 10
    assert payload["last_close"] == pytest.approx(101.2)
    assert payload["breakout_level"] == pytest.approx(100)
    assert payload["candle_time"].endswith("-03:00")
    assert state["position"] is None  # caller has not filled yet
    assert state["prior20_max_high"] == pytest.approx(100)
    for key in (
        "action",
        "last_close",
        "breakout_level",
        "atr",
        "stop",
        "highest_close",
        "account_value",
        "withdrawable",
        "size_usd",
        "reason",
        "candle_time",
    ):
        assert key in payload


def test_breakout_no_when_close_is_not_strictly_above_prior_high():
    equal = _flat(20, price=100, high=100, low=99, close=99.5)
    equal.append(_bar(20, 100, high=100, low=99, close=100))
    below = _flat(20, price=100, high=100, low=99, close=99.5)
    below.append(_bar(20, 99, high=99.5, low=98, close=99))
    for candles in (equal, below):
        payload, _state = sig.decide_btc(candles, _account(), None, _now_after(candles))
        assert payload["action"] == "none"
        assert payload["reason"] == "no_breakout"
        assert payload["size_usd"] is None
        assert payload["breakout_level"] == pytest.approx(100)


def test_forming_candle_is_excluded_from_breakout_and_atr():
    closed = _flat(21, price=100, high=100, low=99, close=99)
    forming = _bar(21, 500, high=99999, low=1, close=99999)
    # Still inside the forming bar: its end timestamp is not in the past.
    now = datetime.fromtimestamp(forming["t"] / 1000, tz=SP) + timedelta(minutes=15)
    assert forming["T"] >= int(now.timestamp() * 1000)
    payload, _state = sig.decide_btc(closed + [forming], _account(), None, now)
    closed_only_payload, _ = sig.decide_btc(closed, _account(), None, now)
    assert payload["action"] == "none"
    assert payload["reason"] == "no_breakout"
    assert payload["last_close"] == pytest.approx(99)
    assert payload["breakout_level"] == pytest.approx(100)
    assert payload["atr"] == pytest.approx(closed_only_payload["atr"])
    assert payload["candle_time"] == sig.ms_to_iso(closed[-1]["t"])
    # A bar whose end equals "now" is not closed yet.
    boundary = dict(forming)
    boundary["T"] = int(now.timestamp() * 1000)
    held = sig.closed_candles(closed + [boundary], int(now.timestamp() * 1000))
    assert held[-1]["t"] == closed[-1]["t"]


def test_stop_ratchet_only_moves_up_and_is_idempotent_per_candle():
    candles = [
        _bar(0, 100, high=101, low=99, close=100),
        _bar(1, 110, high=111, low=109, close=110),
        _bar(2, 108, high=109, low=107, close=108),
    ]
    atrs = [None, 2.0, 10.0]
    entry_ms = candles[0]["t"]
    highest, stop, cursor = sig.advance_trail(candles, atrs, None, None, None, entry_ms, 100.0)
    assert highest == pytest.approx(110)
    assert stop == pytest.approx(110 - 3 * 2.0)
    assert cursor == candles[2]["t"]
    # Wider ATR on a replay of the same bars must not loosen the stop or double-count.
    again_h, again_s, again_c = sig.advance_trail(candles, atrs, highest, stop, cursor, entry_ms, 100.0)
    assert again_h == pytest.approx(highest)
    assert again_s == pytest.approx(stop)
    assert again_c == cursor
    # A new lower close with a huge ATR proposes a lower stop and is ignored.
    extra = candles + [_bar(3, 90, high=91, low=89, close=90)]
    atrs2 = atrs + [50.0]
    h2, s2, c2 = sig.advance_trail(extra, atrs2, highest, stop, cursor, entry_ms, 100.0)
    assert h2 == pytest.approx(110)
    assert s2 == pytest.approx(stop)
    assert c2 == extra[-1]["t"]


def test_pre_entry_close_does_not_change_highest_or_force_exit():
    candles = _flat(25, price=100, high=101, low=99, close=100)
    candles[-1] = _bar(24, 50, high=100, low=50, close=50)
    now = _now_after(candles)
    state = {
        "position": {
            "entry_price": 100.0,
            "size": 0.01,
            "entry_time": sig.iso_sp(now),  # fill is after the last close
        },
        "highest_close": 100.0,
        "stop": 90.0,
        "last_processed_candle": sig.ms_to_iso(candles[-1]["t"]),
        "atr14_1h": 2.0,
    }
    payload, new_state = sig.decide_btc(candles, _account(btc=_long()), state, now)
    assert payload["action"] == "none"
    assert payload["reason"] == "hold"
    assert payload["highest_close"] == pytest.approx(100)
    assert payload["stop"] == pytest.approx(90)
    assert "state_mismatch" not in payload
    assert new_state["position"]["size"] == pytest.approx(0.01)


def test_exit_when_last_close_is_below_stop_and_equal_does_not_exit():
    candles = _flat(25, price=100, high=101, low=99, close=100)
    candles[-1] = _bar(24, 80, high=82, low=79, close=80)
    entry_time = sig.ms_to_iso(candles[0]["t"])
    base = {
        "position": {"entry_price": 100.0, "size": 0.01, "entry_time": entry_time},
        "highest_close": 100.0,
        "atr14_1h": 1.0,
    }
    below = dict(base)
    below["stop"] = 90.0
    below["last_processed_candle"] = sig.ms_to_iso(candles[-1]["t"])
    payload, _state = sig.decide_btc(candles, _account(btc=_long()), below, _now_after(candles))
    assert payload["action"] == "exit"
    assert payload["reason"] == "stop_hit"
    assert payload["stop"] == pytest.approx(90)
    assert payload["last_close"] == pytest.approx(80)

    candles[-1] = _bar(24, 90, high=91, low=89, close=90)
    equal = dict(base)
    equal["stop"] = 90.0
    equal["last_processed_candle"] = sig.ms_to_iso(candles[-1]["t"])
    held, _state = sig.decide_btc(candles, _account(btc=_long()), equal, _now_after(candles))
    assert held["action"] == "none"
    assert held["reason"] == "hold"
    assert held["last_close"] == pytest.approx(held["stop"])


def test_new_high_ratchets_stop_and_rerun_is_idempotent():
    candles = _flat(30, price=100, high=101, low=99, close=100)
    candles[-1] = _bar(29, 130, high=131, low=129, close=130)
    entry_time = sig.ms_to_iso(candles[0]["t"])
    state = {
        "position": {"entry_price": 100.0, "size": 0.02, "entry_time": entry_time},
        "highest_close": 100.0,
        "stop": 10.0,
        "last_processed_candle": sig.ms_to_iso(candles[-2]["t"]),
    }
    now = _now_after(candles)
    first, first_state = sig.decide_btc(candles, _account(btc=_long(size=0.02, entry=100)), state, now)
    assert first["action"] == "none"
    assert first["reason"] == "hold"
    assert first["highest_close"] == pytest.approx(130)
    assert first["stop"] > 10
    assert first["stop"] == pytest.approx(130 - 3 * first["atr"])
    second, second_state = sig.decide_btc(
        candles, _account(btc=_long(size=0.02, entry=100)), first_state, now
    )
    assert second["stop"] == pytest.approx(first["stop"])
    assert second["highest_close"] == pytest.approx(first["highest_close"])
    assert second_state["last_processed_candle"] == first_state["last_processed_candle"]
    assert second["action"] == "none"


def test_hype_thresholds_and_flat_account():
    assert sig.decide_hype(85.90, {"size": 0.25, "upnl": -0.2}, False, datetime(2026, 10, 18).date())["action"] == "close"
    assert sig.decide_hype(85.89, {"size": 0.25, "upnl": -0.2}, False, datetime(2026, 10, 18).date())["action"] == "none"
    assert sig.decide_hype(71.80, {"size": -1.5, "upnl": 3.0}, False, datetime(2026, 10, 18).date())["action"] == "close"
    mid = sig.decide_hype(71.81, {"size": 0.25, "upnl": 1.25}, False, datetime(2026, 10, 18).date())
    assert mid["action"] == "none"
    assert mid["price"] == pytest.approx(71.81)
    assert mid["size"] == pytest.approx(0.25)
    assert mid["upnl"] == pytest.approx(1.25)
    gone = sig.decide_hype(40.0, None, False, datetime(2026, 11, 16).date())
    assert gone == {"action": "gone", "price": 40.0, "size": 0.0, "upnl": 0.0}


def test_hype_deadline_flag_and_mark_asked(tmp_path):
    deadline = datetime(2026, 11, 16, 9, 0, tzinfo=SP)
    before = datetime(2026, 11, 15, 23, 0, tzinfo=SP)
    position = {"size": 0.25, "upnl": -0.1}
    assert sig.decide_hype(77.0, position, False, before.date())["action"] == "none"
    assert sig.decide_hype(77.0, position, False, deadline.date())["action"] == "ask_deadline"
    assert sig.decide_hype(77.0, position, True, deadline.date())["action"] == "none"
    # A price outside the band is a close, including on the deadline day.
    assert sig.decide_hype(90.0, position, False, deadline.date())["action"] == "close"
    # No position stays gone even after the deadline.
    assert sig.decide_hype(77.0, None, False, deadline.date())["action"] == "gone"

    calls = []

    def fetch(_payload):
        calls.append(_payload)
        raise AssertionError("mark-asked must not call the API")

    marked = sig.run_hype(tmp_path, mark_asked=True, now=deadline, fetch=fetch)
    assert calls == []
    assert marked["action"] == "marked_asked"
    saved = json.loads((tmp_path / "hype_state.json").read_text(encoding="utf-8"))
    assert saved["asked"] is True
    assert saved["asked_at"].startswith("2026-11-16T09:00")

    def info(payload):
        if payload["type"] == "allMids":
            return {"HYPE": "77.36", "BTC": "80000"}
        if payload["type"] == "clearinghouseState":
            assert payload["user"] == sig.ACCOUNT
            return _clearinghouse(hype_size="0.25", hype_upnl="-0.1785")
        raise AssertionError(payload["type"])

    after = sig.run_hype(tmp_path, mark_asked=False, now=deadline, fetch=info)
    assert after["action"] == "none"
    assert after["size"] == pytest.approx(0.25)
    fresh = sig.run_hype(tmp_path / "other", mark_asked=False, now=deadline, fetch=info)
    assert fresh["action"] == "ask_deadline"


def test_low_balance_and_exact_boundary():
    candles = _flat(20, price=100, high=100, low=99, close=99)
    candles.append(_bar(20, 110, high=111, low=109, close=110))  # would enter
    now = _now_after(candles)
    paused, _state = sig.decide_btc(candles, _account(value=11.99, withdrawable=11), None, now)
    assert paused["action"] == "pause_low_balance"
    assert paused["reason"] == "account_value_below_12"
    assert paused["account_value"] == pytest.approx(11.99)
    ok, _state = sig.decide_btc(candles, _account(value=12, withdrawable=20), None, now)
    assert ok["action"] == "enter"


def test_min_size_blocks_breakout_entry():
    candles = _flat(20, price=100, high=100, low=99, close=99)
    candles.append(_bar(20, 110, high=111, low=109, close=110))
    now = _now_after(candles)
    small, _state = sig.decide_btc(candles, _account(value=20, withdrawable=5), None, now)
    assert small["action"] == "none"
    assert small["size_usd"] == sig.clean_float(5 * 2 * 0.9, 6)
    assert small["size_usd"] < 10
    assert "10" in small["reason"]
    large, _state = sig.decide_btc(candles, _account(value=20, withdrawable=6), None, now)
    assert large["action"] == "enter"
    assert large["size_usd"] == sig.clean_float(6 * 2 * 0.9, 6)
    assert large["size_usd"] >= 10


def test_expired_after_18_nov_2026_only():
    candles = _flat(20, price=100, high=100, low=99, close=99)
    candles.append(_bar(20, 110, high=111, low=109, close=110))
    last_day = datetime(2026, 11, 18, 23, 30, tzinfo=SP)
    # Candles are in October, so they are closed on both clocks.
    still, _state = sig.decide_btc(candles, _account(), None, last_day)
    assert still["action"] == "enter"
    expired, _state = sig.decide_btc(
        candles, _account(value=5, withdrawable=1), None, datetime(2026, 11, 19, 0, 5, tzinfo=SP)
    )
    assert expired["action"] == "expired"
    assert expired["reason"] == "after_2026-11-18"


def test_state_mismatch_trusts_the_account_and_keeps_file_format():
    candles = _flat(20, price=100, high=100, low=99, close=99)
    candles.append(_bar(20, 110, high=111, low=109, close=110))
    now = _now_after(candles)
    stale = {
        "position": {"entry_price": 50.0, "size": 1.0, "entry_time": sig.ms_to_iso(candles[0]["t"])},
        "highest_close": 80.0,
        "stop": 70.0,
        "last_processed_candle": sig.ms_to_iso(candles[-1]["t"]),
        "note": "Criado na primeira execução; sem posição BTC na conta real.",
        "custom_flag": "keep-me",
    }
    payload, state = sig.decide_btc(candles, _account(), stale, now)
    assert payload["state_mismatch"] is True
    assert payload["action"] == "enter"  # account is flat, so the breakout stands
    assert state["position"] is None
    assert state["stop"] is None
    assert state["note"] == stale["note"]
    assert state["custom_flag"] == "keep-me"
    for key in (
        "last_check",
        "last_closed_candle_open",
        "last_close",
        "prior20_max_high",
        "atr14_1h",
        "account_value_usd",
        "available_usd",
    ):
        assert key in state

    discovered, discovered_state = sig.decide_btc(
        candles,
        _account(btc=_long(size=0.01, entry=10_000)),
        {"position": None, "note": stale["note"]},
        now,
    )
    assert discovered["state_mismatch"] is True
    assert discovered["action"] == "exit"
    assert discovered["highest_close"] == pytest.approx(10_000)
    assert discovered_state["position"]["size"] == pytest.approx(0.01)
    assert discovered_state["position"]["entry_price"] == pytest.approx(10_000)


def test_record_entry_and_exit_persist_log_without_network(tmp_path):
    now = datetime(2026, 10, 18, 12, 10, tzinfo=SP)
    entry = sig.run_record_entry(tmp_path, 81234.5, 0.004, now=now)
    assert entry["action"] == "recorded_entry"
    state = json.loads((tmp_path / "btc_trend_state.json").read_text(encoding="utf-8"))
    assert state["position"]["entry_price"] == pytest.approx(81234.5)
    assert state["position"]["size"] == pytest.approx(0.004)
    assert state["highest_close"] == pytest.approx(81234.5)
    assert state["position"]["entry_time"].startswith("2026-10-18T12:10")
    log = (tmp_path / "btc_trend_log.md").read_text(encoding="utf-8")
    assert "entry" in log
    assert "81234.5" in log
    assert "0.004" in log

    later = now + timedelta(hours=3)
    exited = sig.run_record_exit(tmp_path, 80500, now=later)
    assert exited["action"] == "recorded_exit"
    state = json.loads((tmp_path / "btc_trend_state.json").read_text(encoding="utf-8"))
    assert state["position"] is None
    assert state["highest_close"] is None
    assert state["stop"] is None
    log = (tmp_path / "btc_trend_log.md").read_text(encoding="utf-8")
    assert "exit" in log
    assert "80500" in log
    assert log.count("## ") == 2


def test_cli_record_and_api_error_are_one_json_line(tmp_path, monkeypatch, capsys):
    rc = sig.main(["--state-dir", str(tmp_path), "btc", "--record-entry", "100", "0.01"])
    assert rc == 0
    line = capsys.readouterr().out
    assert line.count("\n") == 1
    assert json.loads(line)["action"] == "recorded_entry"

    rc = sig.main(["hype", "--state-dir", str(tmp_path), "--mark-asked"])
    assert rc == 0
    marked = json.loads(capsys.readouterr().out)
    assert marked["action"] == "marked_asked"

    def boom(_payload, timeout=20.0):
        raise RuntimeError("connection timed out")

    monkeypatch.setattr(sig, "post_info", boom)
    rc = sig.main(["hype", "--state-dir", str(tmp_path)])
    assert rc == 0
    err_line = capsys.readouterr().out
    assert err_line.count("\n") == 1
    err = json.loads(err_line)
    assert err["action"] == "error"
    assert "timed out" in err["reason"]
    rc = sig.main(["--state-dir", str(tmp_path), "btc"])
    assert rc == 0
    assert json.loads(capsys.readouterr().out)["action"] == "error"


def test_cli_btc_uses_string_api_rows_and_only_closed_bars(tmp_path, monkeypatch):
    candles = _flat(21, price=100, high=100, low=99, close=99)
    forming = _bar(21, 500, high=99999, low=1, close=99999)
    now = datetime.fromtimestamp(forming["t"] / 1000, tz=SP) + timedelta(minutes=10)

    def fake(payload, timeout=20.0):
        kind = payload["type"]
        if kind == "candleSnapshot":
            req = payload["req"]
            assert req["coin"] == "BTC"
            assert req["interval"] == "1h"
            rows = []
            for candle in candles + [forming]:
                rows.append(
                    {
                        "t": candle["t"],
                        "T": candle["T"],
                        "s": "BTC",
                        "i": "1h",
                        "o": str(candle["o"]),
                        "h": str(candle["h"]),
                        "l": str(candle["l"]),
                        "c": str(candle["c"]),
                        "v": "1",
                        "n": 1,
                    }
                )
            return rows
        if kind == "clearinghouseState":
            assert payload["user"] == "0x48a942a27c4dd4b4dd0cecca7ff1df446c5aef77"
            return _clearinghouse()
        raise AssertionError(kind)

    monkeypatch.setattr(sig, "post_info", fake)
    monkeypatch.setattr(sig, "now_sp", lambda: now)
    rc_payload = sig.run_btc(tmp_path, now=now)
    assert rc_payload["action"] == "none"
    assert rc_payload["reason"] == "no_breakout"
    assert rc_payload["last_close"] == pytest.approx(99)
    saved = json.loads((tmp_path / "btc_trend_state.json").read_text(encoding="utf-8"))
    assert saved["position"] is None
    assert saved["last_closed_candle_open"] == sig.ms_to_iso(candles[-1]["t"])
    assert saved["available_usd"] == pytest.approx(7.541194)
    assert saved["account_value_usd"] == pytest.approx(17.110235)


def test_module_does_not_trade_or_read_a_key():
    text = sig.__file__.replace("\\", "/")
    source = Path(sig.__file__).read_text(encoding="utf-8")
    assert "api.hyperliquid.xyz/info" in source
    assert "/exchange" not in source
    assert "api_key" not in source.lower()
    assert "secret" not in source.lower()
    assert text.endswith("signal_check.py")


def _clearinghouse(hype_size="0", hype_upnl="0", btc_size=None):
    positions = []
    if hype_size not in ("0", "0.0", None):
        positions.append(
            {
                "type": "oneWay",
                "position": {
                    "coin": "HYPE",
                    "szi": hype_size,
                    "entryPx": "78.075",
                    "unrealizedPnl": hype_upnl,
                },
            }
        )
    if btc_size is not None:
        positions.append(
            {
                "type": "oneWay",
                "position": {
                    "coin": "BTC",
                    "szi": btc_size,
                    "entryPx": "80000",
                    "unrealizedPnl": "1",
                },
            }
        )
    return {
        "marginSummary": {"accountValue": "17.110235"},
        "withdrawable": "7.541194",
        "assetPositions": positions,
    }
