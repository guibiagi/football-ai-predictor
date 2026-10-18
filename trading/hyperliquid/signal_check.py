#!/usr/bin/env python3
"""Read-only Hyperliquid signals for the HYPE position and the BTC trend.

Prints one JSON line telling the caller what to do. This process never places
orders and never reads a key. It only POSTs to the public info endpoint.

State files (under --state-dir, default /workspace/trading):
  hype_state.json       deadline flag
  btc_trend_state.json  snapshot compatible with the existing BTC state
  btc_trend_log.md      append-only log of recorded entries and exits
"""

from __future__ import annotations

import argparse
import copy
import fcntl
import json
import math
import os
import sys
import tempfile
import time
from datetime import datetime, timezone
from pathlib import Path
from zoneinfo import ZoneInfo

API_URL = "https://api.hyperliquid.xyz/info"
ACCOUNT = "0x48a942a27c4dd4b4dd0cecca7ff1df446c5aef77"
DEFAULT_STATE_DIR = "/workspace/trading"

SP = ZoneInfo("America/Sao_Paulo")

# HYPE position band. Inclusive on both sides.
HYPE_HIGH = 85.90
HYPE_LOW = 71.80
HYPE_DEADLINE = datetime(2026, 11, 16, tzinfo=SP).date()

# BTC long-only Donchian breakout plus a ratcheting ATR stop.
BTC_EXPIRY = datetime(2026, 11, 18, tzinfo=SP).date()  # expired strictly after this date
BREAKOUT_BARS = 20
ATR_PERIOD = 14
ATR_MULT = 3.0
MIN_ACCOUNT_VALUE = 12.0
MIN_SIZE_USD = 10.0
CANDLE_LOOKBACK_HOURS = 400
HOUR_MS = 3_600_000
SIZE_TOLERANCE_FRAC = 0.01
PRICE_TOLERANCE_FRAC = 1e-4
# A stuck holder must not block an exit. The wait is polled so a free lock
# is taken immediately.
LOCK_TIMEOUT_S = 10.0
LOCK_POLL_S = 0.05

BTC_STATE_NAME = "btc_trend_state.json"
BTC_LOG_NAME = "btc_trend_log.md"
HYPE_STATE_NAME = "hype_state.json"


def now_sp() -> datetime:
    return datetime.now(SP)


def as_sp(dt: datetime) -> datetime:
    if dt.tzinfo is None:
        return dt.replace(tzinfo=SP)
    return dt.astimezone(SP)


def iso_sp(dt: datetime) -> str:
    """Minute precision with a numeric offset, matching btc_trend_state.json."""
    return as_sp(dt).isoformat(timespec="minutes")


def ms_to_sp(ms: int) -> datetime:
    return datetime.fromtimestamp(ms / 1000.0, tz=timezone.utc).astimezone(SP)


def ms_to_iso(ms: int) -> str:
    return ms_to_sp(ms).isoformat(timespec="minutes")


def _finite_number(value):
    """A finite float, or None. Booleans and garbage are not numbers."""
    if isinstance(value, bool) or value is None:
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(number):
        return None
    return number


def parse_time_ms(value) -> int | None:
    """Epoch milliseconds, or None when the value is missing or malformed."""
    if isinstance(value, bool) or value is None:
        return None
    if isinstance(value, (int, float)):
        if not math.isfinite(float(value)):
            return None
        return int(value)
    text = str(value).strip()
    if not text:
        return None
    try:
        dt = datetime.fromisoformat(text)
    except ValueError:
        return None
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=SP)
    try:
        return int(dt.timestamp() * 1000)
    except (OverflowError, OSError, ValueError):
        return None


def clean_float(value, nd: int = 8):
    if value is None:
        return None
    number = float(value)
    if math.isnan(number) or math.isinf(number):
        return None
    return float(f"{number:.{nd}f}")


def one_line(exc: BaseException) -> str:
    text = f"{type(exc).__name__}: {exc}"
    return " ".join(text.split())[:400]


def emit(payload: dict) -> None:
    sys.stdout.write(json.dumps(payload, ensure_ascii=False) + "\n")


# --- ATR --------------------------------------------------------------------
# Same definition as the backtest engine:
#   pc = close.shift()
#   tr = max(high-low, abs(high-pc), abs(low-pc))
#   atr = tr.ewm(alpha=1/n, adjust=False).mean()
# The first bar is NaN (no previous close). The first finite true range seeds
# the average; after that y = (1 - alpha) * y_prev + alpha * tr.


def true_ranges(candles: list[dict]) -> list[float | None]:
    out: list[float | None] = []
    prev_close = None
    for candle in candles:
        if prev_close is None:
            out.append(None)
        else:
            high = float(candle["h"])
            low = float(candle["l"])
            tr = max(high - low, abs(high - prev_close), abs(low - prev_close))
            out.append(tr)
        prev_close = float(candle["c"])
    return out


def atr_series(candles: list[dict], n: int = ATR_PERIOD) -> list[float | None]:
    """Wilder/EWM ATR. Index 0 is None. Later values match pandas ewm."""
    if n <= 0:
        raise ValueError("ATR period must be positive")
    alpha = 1.0 / n
    out: list[float | None] = []
    atr = None
    for tr in true_ranges(candles):
        if tr is None:
            out.append(None)
            continue
        if atr is None:
            atr = tr
        else:
            atr = (1.0 - alpha) * atr + alpha * tr
        out.append(atr)
    return out


def closed_candles(candles: list[dict], now_ms: int) -> list[dict]:
    """Drop the forming bar. A bar is closed only when its end time is in the past."""
    closed = [c for c in candles if int(c["T"]) < now_ms]
    closed.sort(key=lambda c: int(c["t"]))
    return closed


def parse_candles(raw) -> list[dict]:
    if not isinstance(raw, list):
        raise RuntimeError(f"candleSnapshot returned {type(raw).__name__}")
    candles = []
    for row in raw:
        if not isinstance(row, dict):
            raise RuntimeError("candle row is not an object")
        candles.append(
            {
                "t": int(row["t"]),
                "T": int(row["T"]) if row.get("T") is not None else int(row["t"]) + HOUR_MS - 1,
                "o": float(row["o"]),
                "h": float(row["h"]),
                "l": float(row["l"]),
                "c": float(row["c"]),
            }
        )
    candles.sort(key=lambda c: c["t"])
    dedup = {}
    for candle in candles:
        dedup[candle["t"]] = candle
    return [dedup[key] for key in sorted(dedup)]


def breakout_level(candles: list[dict], n: int = BREAKOUT_BARS) -> float:
    """Max high of the n candles before the last one. Mirrors rolling(n).max().shift()."""
    if len(candles) < n + 1:
        raise ValueError(f"need at least {n + 1} closed candles, got {len(candles)}")
    return max(float(c["h"]) for c in candles[-(n + 1) : -1])


# --- state files ------------------------------------------------------------


def default_btc_state() -> dict:
    return {
        "position": None,
        "last_check": None,
        "last_closed_candle_open": None,
        "last_close": None,
        "prior20_max_high": None,
        "atr14_1h": None,
        "account_value_usd": None,
        "available_usd": None,
        "highest_close": None,
        "stop": None,
        "last_processed_candle": None,
        "last_exit_candle": None,
        "note": "Criado na primeira execução; sem posição BTC na conta real.",
    }


def default_hype_state() -> dict:
    return {"asked": False, "asked_at": None}


class StateLoadError(Exception):
    """The state file exists but cannot be used. The account is still authoritative."""


class CommandError(Exception):
    """A caller command was refused. main() prints one JSON line and exits non-zero."""

    def __init__(self, reason: str, code: int = 1):
        super().__init__(reason)
        self.reason = reason
        self.code = code


def _load_json(path: Path, default: dict) -> dict:
    if not path.exists():
        return copy.deepcopy(default)
    try:
        text = path.read_text(encoding="utf-8")
    except OSError as exc:
        raise StateLoadError(f"unreadable {path.name}: {exc}") from exc
    try:
        data = json.loads(text)
    except json.JSONDecodeError as exc:
        raise StateLoadError(f"invalid JSON in {path.name}: {exc}") from exc
    if not isinstance(data, dict):
        raise StateLoadError(f"{path.name} must be a JSON object")
    merged = copy.deepcopy(default)
    merged.update(data)
    return merged


def _atomic_write(path: Path, payload: dict) -> None:
    """Unique temp file in the destination directory, fsync, then os.replace."""
    path.parent.mkdir(parents=True, exist_ok=True)
    text = json.dumps(payload, indent=2, ensure_ascii=False) + "\n"
    fd, tmp_name = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    tmp_path = Path(tmp_name)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            handle.write(text)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(tmp_path, path)
    except Exception:
        if tmp_path.exists():
            try:
                tmp_path.unlink()
            except OSError:
                pass
        raise


class _StateLock:
    """Exclusive lock so two signal runs cannot clobber the same state file.

    The wait is bounded. A stuck holder must not block an exit: the caller
    reads the state file without the lock and skips the write.
    """

    def __init__(self, state_dir: Path):
        self.state_dir = Path(state_dir)
        self.handle = None
        self.timeout = LOCK_TIMEOUT_S
        self.poll = LOCK_POLL_S

    def _close_handle(self) -> None:
        if self.handle is None:
            return
        try:
            self.handle.close()
        except OSError:
            pass
        self.handle = None

    def __enter__(self):
        self.state_dir.mkdir(parents=True, exist_ok=True)
        self.handle = open(self.state_dir / ".state.lock", "a+", encoding="utf-8")
        deadline = time.monotonic() + self.timeout
        while True:
            try:
                fcntl.flock(self.handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
                return self
            except BlockingIOError:
                if time.monotonic() >= deadline:
                    self._close_handle()
                    raise TimeoutError(
                        f"state lock held longer than {self.timeout:.0f}s"
                    ) from None
                time.sleep(self.poll)

    def __exit__(self, exc_type, exc, tb):
        if self.handle is not None:
            try:
                fcntl.flock(self.handle.fileno(), fcntl.LOCK_UN)
            except OSError:
                pass
            self._close_handle()
        return False


def _material(state: dict | None):
    """Comparable snapshot. last_check moves every run and does not count as a change."""
    if state is None:
        return None
    data = copy.deepcopy(state)
    data.pop("last_check", None)
    return json.loads(json.dumps(data, ensure_ascii=False, sort_keys=True))


def load_btc_state(state_dir: Path) -> dict:
    return _load_json(state_dir / BTC_STATE_NAME, default_btc_state())


def save_btc_state(state_dir: Path, state: dict) -> None:
    _atomic_write(state_dir / BTC_STATE_NAME, state)


def load_hype_state(state_dir: Path) -> dict:
    return _load_json(state_dir / HYPE_STATE_NAME, default_hype_state())


def save_hype_state(state_dir: Path, state: dict) -> None:
    _atomic_write(state_dir / HYPE_STATE_NAME, state)


def append_btc_log(state_dir: Path, block: str) -> None:
    path = state_dir / BTC_LOG_NAME
    path.parent.mkdir(parents=True, exist_ok=True)
    if not path.exists():
        path.write_text("# BTC trend log\n\n", encoding="utf-8")
    with path.open("a", encoding="utf-8") as handle:
        handle.write(block)
        if not block.endswith("\n"):
            handle.write("\n")


# --- decisions --------------------------------------------------------------


def decide_hype(mid: float, position: dict | None, asked: bool, today) -> dict:
    """HYPE actions. `position` is None when the account has no HYPE size.

    `close` is only for a long. A short is reported as an error so the caller
    does not send a reduce-only close in the wrong direction.
    """
    price = clean_float(mid, 8)
    if position is None:
        return {"action": "gone", "price": price, "size": 0.0, "upnl": 0.0}
    try:
        size_raw = float(position.get("size"))
    except (TypeError, ValueError):
        return {
            "action": "error",
            "reason": "unparseable hype size",
            "price": price,
            "size": None,
            "upnl": None,
        }
    if not math.isfinite(size_raw):
        return {
            "action": "error",
            "reason": "non-finite hype size",
            "price": price,
            "size": None,
            "upnl": None,
        }
    size = clean_float(size_raw, 8)
    upnl = clean_float(position.get("upnl"), 8)
    if size_raw < 0:
        return {"action": "error", "reason": "hype_short", "price": price, "size": size, "upnl": upnl}
    if size_raw == 0:
        return {"action": "gone", "price": price, "size": 0.0, "upnl": 0.0}
    if price is not None and (price >= HYPE_HIGH or price <= HYPE_LOW):
        action = "close"
    elif today >= HYPE_DEADLINE and not asked:
        action = "ask_deadline"
    else:
        action = "none"
    return {"action": action, "price": price, "size": size, "upnl": upnl}


def _position_is_open(position) -> bool:
    if position is None:
        return False
    if isinstance(position, dict):
        try:
            return float(position.get("size") or 0) > 0
        except (TypeError, ValueError):
            return False
    return True


def _same_open_position(position, price: float, size: float) -> bool:
    """True when the stored fill is the one the caller is recording again."""
    if not isinstance(position, dict):
        return False
    old_price = _finite_number(position.get("entry_price"))
    old_size = _finite_number(position.get("size"))
    if old_price is None or old_size is None or old_size <= 0:
        return False
    price_tol = max(1e-6, PRICE_TOLERANCE_FRAC * abs(price))
    size_tol = max(1e-8, SIZE_TOLERANCE_FRAC * abs(size))
    return abs(old_price - price) <= price_tol and abs(old_size - size) <= size_tol


def _sizes_disagree(state_size, account_size) -> bool:
    try:
        left = float(state_size)
        right = float(account_size)
    except (TypeError, ValueError):
        return True
    tolerance = max(1e-8, SIZE_TOLERANCE_FRAC * abs(right))
    return abs(left - right) > tolerance


def advance_trail(
    candles: list[dict],
    atrs: list[float | None],
    highest: float | None,
    stop: float | None,
    last_processed_ms: int | None,
    entry_time_ms: int | None,
    entry_price: float | None,
) -> tuple[float | None, float | None, int | None]:
    """Apply each new closed candle once. The stop only moves up.

    Candles that closed before entry are consumed (so they are not replayed)
    but they do not raise highest_close. A candle already stored as processed
    is left untouched, which keeps a re-run on the same bar idempotent.
    """
    cursor = last_processed_ms
    for index, candle in enumerate(candles):
        open_ms = int(candle["t"])
        if cursor is not None and open_ms <= cursor:
            continue
        close_ms = int(candle["T"])
        if entry_time_ms is not None and close_ms < entry_time_ms:
            cursor = open_ms
            continue
        atr = atrs[index] if index < len(atrs) else None
        if atr is None:
            cursor = open_ms
            continue
        close = float(candle["c"])
        if highest is None:
            seed = float(entry_price) if entry_price is not None else close
            highest = max(seed, close)
        else:
            highest = max(float(highest), close)
        candidate = highest - ATR_MULT * float(atr)
        stop = candidate if stop is None else max(float(stop), candidate)
        cursor = open_ms
    return highest, stop, cursor


def validate_state_fields(state) -> str | None:
    """Name the first unusable field. Never raises.

    Nulls are fine. An open position needs a numeric entry price and size and
    a parseable entry time; a file from the older format (size only) does not.
    """
    if not isinstance(state, dict):
        return "state is not an object"
    for key in (
        "stop",
        "highest_close",
        "atr14_1h",
        "last_close",
        "prior20_max_high",
        "account_value_usd",
        "available_usd",
    ):
        if key not in state or state.get(key) is None:
            continue
        if _finite_number(state.get(key)) is None:
            return f"non-numeric {key}"
    for key in ("last_processed_candle", "last_exit_candle", "last_closed_candle_open"):
        if key not in state or state.get(key) is None:
            continue
        if parse_time_ms(state.get(key)) is None:
            return f"malformed {key}"
    position = state.get("position")
    if position is None:
        return None
    if not isinstance(position, dict):
        return "position is not an object"
    if "size" not in position or position.get("size") is None:
        if position:
            return "position missing size"
        return None
    size = _finite_number(position.get("size"))
    if size is None:
        return "non-numeric position.size"
    if size <= 0:
        return None
    if _finite_number(position.get("entry_price")) is None:
        return "non-numeric entry_price"
    if parse_time_ms(position.get("entry_time")) is None:
        return "malformed entry_time"
    return None


def _without_bad_fields(state: dict) -> dict:
    """Copy state with unusable fields cleared so a rebuild can be saved cleanly."""
    cleaned = copy.deepcopy(state) if isinstance(state, dict) else {}
    for key in (
        "stop",
        "highest_close",
        "atr14_1h",
        "last_close",
        "prior20_max_high",
        "account_value_usd",
        "available_usd",
    ):
        if key in cleaned and cleaned[key] is not None and _finite_number(cleaned[key]) is None:
            cleaned[key] = None
    for key in ("last_processed_candle", "last_exit_candle", "last_closed_candle_open"):
        if key in cleaned and cleaned[key] is not None and parse_time_ms(cleaned[key]) is None:
            cleaned[key] = None
    position = cleaned.get("position")
    if position is not None and not isinstance(position, dict):
        cleaned["position"] = None
    elif isinstance(position, dict) and position:
        size = _finite_number(position.get("size")) if position.get("size") is not None else None
        broken = size is None or (
            size > 0
            and (
                _finite_number(position.get("entry_price")) is None
                or parse_time_ms(position.get("entry_time")) is None
            )
        )
        if broken:
            cleaned["position"] = None
    return cleaned


def opening_buy_time_ms(fills) -> int | None:
    """Latest BTC fill that opened the current long.

    A later buy that adds to a long already open is not the entry. `dir` of
    "Open Long", or a buy whose start position is flat or short, counts.
    """
    if not isinstance(fills, list):
        return None
    latest = None
    for fill in fills:
        if not isinstance(fill, dict) or fill.get("coin") != "BTC":
            continue
        when = _finite_number(fill.get("time"))
        if when is None:
            continue
        when = int(when)
        opened = str(fill.get("dir") or "") == "Open Long"
        if not opened and str(fill.get("side") or "") == "B":
            start = _finite_number(fill.get("startPosition"))
            opened = start is not None and start <= 0
        if not opened:
            continue
        if latest is None or when > latest:
            latest = when
    return latest


def lookup_open_long_ms(fetch, now: datetime) -> int | None:
    """Public fills for the account. Any failure returns None, never an error action."""
    now_ms = int(as_sp(now).timestamp() * 1000)
    start = now_ms - CANDLE_LOOKBACK_HOURS * HOUR_MS
    queries = (
        {
            "type": "userFillsByTime",
            "user": ACCOUNT,
            "startTime": start,
            "endTime": now_ms,
        },
        {"type": "userFills", "user": ACCOUNT},
    )
    for payload in queries:
        try:
            data = fetch(payload)
        except Exception:
            continue
        when = opening_buy_time_ms(data)
        if when is not None:
            return when
    return None


def rebuild_long_trail(
    closed: list[dict],
    atrs: list[float | None],
    entry_px,
    size,
    fill_entry_ms: int | None = None,
):
    """Rebuild a lost trail without treating the candle window as the entry.

    When the public fills name the opening buy, highest_close is the best
    close at or after that time (seeded with the fill price). When they do
    not, the stop is max(last_close, entryPx) - 3*ATR14 and the entry is
    stamped on the current candle. Bars from before the position must not
    ratchet the stop.
    """
    entry_price = None if entry_px is None else float(entry_px)
    last = closed[-1]
    last_close = float(last["c"])
    atr = atrs[-1]
    if fill_entry_ms is None:
        seeds = [last_close]
        if entry_price is not None:
            seeds.append(entry_price)
        highest = max(seeds)
        stop = float(highest) - ATR_MULT * float(atr)
        cursor = int(last["t"])
        entry_time_ms = cursor
    else:
        entry_time_ms = int(fill_entry_ms)
        highest, stop, cursor = advance_trail(
            closed, atrs, None, None, None, entry_time_ms, entry_price
        )
        if highest is None:
            highest = entry_price if entry_price is not None else last_close
        if stop is None:
            stop = float(highest) - ATR_MULT * float(atr)
        if cursor is None:
            cursor = int(last["t"])
    position = {
        "entry_price": clean_float(entry_price, 8),
        "size": clean_float(size, 8),
        "entry_time": ms_to_iso(entry_time_ms),
    }
    return position, highest, stop, cursor


def decide_btc(
    candles: list[dict],
    account: dict,
    state: dict | None,
    now: datetime,
    state_error: str | None = None,
    fill_entry_ms: int | None = None,
) -> tuple[dict, dict]:
    """BTC signal plus the state to persist. `candles` may include the forming bar."""
    now = as_sp(now)
    now_ms = int(now.timestamp() * 1000)
    today = now.date()
    closed = closed_candles(candles, now_ms)
    if len(closed) < BREAKOUT_BARS + 1:
        raise ValueError(f"not enough closed candles ({len(closed)})")
    atrs = atr_series(closed, ATR_PERIOD)
    atr = atrs[-1]
    if atr is None:
        raise ValueError("ATR is undefined")
    level = breakout_level(closed, BREAKOUT_BARS)
    last = closed[-1]
    last_close = float(last["c"])
    candle_iso = ms_to_iso(int(last["t"]))

    account_value = float(account["account_value"])
    withdrawable = float(account["withdrawable"])
    acct = account.get("btc")
    acct_size = float(acct["size"]) if acct else 0.0
    acct_long = acct is not None and acct_size > 0
    acct_short = acct is not None and acct_size < 0

    if state is not None:
        field_error = validate_state_fields(state)
        if field_error:
            state_error = state_error or field_error
            state = _without_bad_fields(state)

    new_state = default_btc_state()
    if state:
        new_state.update(copy.deepcopy(state))
    state_pos = new_state.get("position")
    state_open = _position_is_open(state_pos)

    mismatch = False
    if state_error or acct_short or state_open != acct_long:
        mismatch = True
    elif state_open and acct_long and isinstance(state_pos, dict):
        if _sizes_disagree(state_pos.get("size"), acct_size):
            mismatch = True

    highest = new_state.get("highest_close")
    stop = new_state.get("stop")
    last_processed_ms = parse_time_ms(new_state.get("last_processed_candle"))

    if acct_long and state_error:
        # The file cannot be trusted. Use the opening fill when we have it;
        # otherwise seed from this close and the account entry. Do not walk
        # back to the first candle in the lookback.
        entry_px = acct.get("entry_px")
        position, highest, stop, last_processed_ms = rebuild_long_trail(
            closed, atrs, entry_px, acct_size, fill_entry_ms=fill_entry_ms
        )
        new_state["position"] = position
    elif acct_long:
        entry_px = acct.get("entry_px")
        if not state_open:
            # Account is long and the file does not know it. Start the trail
            # from the account entry and this close; do not replay old bars.
            # Stamp entry at this candle's open so the stop is allowed to
            # fire on the close we just observed.
            seeds = [last_close]
            if entry_px is not None:
                seeds.append(float(entry_px))
            highest = max(seeds)
            stop = highest - ATR_MULT * atr
            last_processed_ms = int(last["t"])
            new_state["position"] = {
                "entry_price": clean_float(entry_px, 8),
                "size": clean_float(acct_size, 8),
                "entry_time": ms_to_iso(int(last["t"])),
            }
        else:
            if not isinstance(state_pos, dict):
                state_pos = {}
            else:
                state_pos = dict(state_pos)
            try:
                entry_time_ms = parse_time_ms(state_pos.get("entry_time"))
                if state_pos.get("entry_price") is None and entry_px is not None:
                    state_pos["entry_price"] = clean_float(entry_px, 8)
                entry_price = state_pos.get("entry_price")
                if entry_price is None:
                    entry_price = entry_px
                highest_n = _finite_number(highest)
                stop_n = _finite_number(stop)
                if highest is not None and highest_n is None:
                    raise ValueError("non-numeric highest_close")
                if stop is not None and stop_n is None:
                    raise ValueError("non-numeric stop")
                highest, stop, last_processed_ms = advance_trail(
                    closed,
                    atrs,
                    highest_n,
                    stop_n,
                    last_processed_ms,
                    entry_time_ms,
                    None if entry_price is None else float(entry_price),
                )
                if highest is None:
                    highest = float(entry_price) if entry_price is not None else last_close
                if stop is None:
                    stop = float(highest) - ATR_MULT * atr
                state_pos["size"] = clean_float(acct_size, 8)
                new_state["position"] = state_pos
            except (TypeError, ValueError) as exc:
                # A field that slipped past validation must not become `error`
                # while the account is long. Rebuild from the fill instead.
                state_error = state_error or f"state field: {exc}"
                position, highest, stop, last_processed_ms = rebuild_long_trail(
                    closed, atrs, entry_px, acct_size, fill_entry_ms=fill_entry_ms
                )
                new_state["position"] = position
    else:
        # Just observed the long disappear: do not re-enter on this same candle.
        if state_open:
            new_state["last_exit_candle"] = candle_iso
        new_state["position"] = None
        highest = None
        stop = None
        last_processed_ms = None

    new_state["last_check"] = iso_sp(now)
    new_state["last_closed_candle_open"] = candle_iso
    new_state["last_close"] = clean_float(last_close, 6)
    new_state["prior20_max_high"] = clean_float(level, 6)
    new_state["atr14_1h"] = clean_float(atr, 6)
    new_state["account_value_usd"] = clean_float(account_value, 6)
    new_state["available_usd"] = clean_float(withdrawable, 6)
    new_state["highest_close"] = clean_float(highest, 6) if highest is not None else None
    new_state["stop"] = clean_float(stop, 6) if stop is not None else None
    new_state["last_processed_candle"] = (
        ms_to_iso(last_processed_ms) if acct_long and last_processed_ms is not None else None
    )

    # Expiry and a low balance block new entries only. An open long still
    # trails, and a stop hit is still an exit.
    size_usd = None
    low_balance = account_value < MIN_ACCOUNT_VALUE
    expired_date = today > BTC_EXPIRY
    if acct_short:
        action = "none"
        reason = "unexpected_short"
    elif acct_long:
        stop_n = _finite_number(stop)
        if (
            _stop_is_active(new_state.get("position"), int(last["T"]))
            and stop_n is not None
            and last_close < stop_n
        ):
            action = "exit"
            reason = "stop_hit"
            new_state["last_exit_candle"] = candle_iso
        else:
            action = "none"
            reason = "hold"
    elif expired_date:
        action = "expired"
        reason = "after_2026-11-18"
    elif low_balance:
        action = "pause_low_balance"
        reason = "account_value_below_12"
    elif last_close > level and new_state.get("last_exit_candle") == candle_iso:
        action = "none"
        reason = "exited_this_candle"
    elif last_close > level:
        size_usd = clean_float(withdrawable * 2.0 * 0.9, 6)
        if size_usd is None or size_usd < MIN_SIZE_USD:
            action = "none"
            reason = f"size_usd {size_usd} < {MIN_SIZE_USD:.0f}"
        else:
            action = "enter"
            reason = "breakout"
    else:
        action = "none"
        reason = "no_breakout"

    payload = {
        "action": action,
        "last_close": clean_float(last_close, 6),
        "breakout_level": clean_float(level, 6),
        "atr": clean_float(atr, 6),
        "stop": new_state["stop"],
        "highest_close": new_state["highest_close"],
        "account_value": clean_float(account_value, 6),
        "withdrawable": clean_float(withdrawable, 6),
        "size_usd": size_usd,
        "reason": reason,
        "candle_time": candle_iso,
    }
    if low_balance:
        payload["low_balance"] = True
    if acct_long and expired_date:
        payload["expired"] = True
    if mismatch:
        payload["state_mismatch"] = True
    if state_error:
        payload["state_error"] = state_error
        payload["state_mismatch"] = True
    return payload, new_state


def _stop_is_active(position, last_close_ms: int) -> bool:
    """A close can hit the stop only after the entry. The signal bar is earlier."""
    if not isinstance(position, dict):
        return False
    entry_ms = parse_time_ms(position.get("entry_time"))
    if entry_ms is None:
        return True
    return last_close_ms >= entry_ms


# --- account parsing --------------------------------------------------------


def _require_float(value, field: str) -> float:
    """Parse a finite number. A bad value is an error, never a flat position."""
    try:
        number = float(value)
    except (TypeError, ValueError) as exc:
        raise RuntimeError(f"unparseable {field}") from exc
    if not math.isfinite(number):
        raise RuntimeError(f"non-finite {field}")
    return number


def parse_account(data: dict) -> dict:
    if not isinstance(data, dict):
        raise RuntimeError(f"clearinghouseState returned {type(data).__name__}")
    if "error" in data and "marginSummary" not in data:
        raise RuntimeError(f"clearinghouseState: {data.get('error')}")
    margin = data.get("marginSummary")
    if not isinstance(margin, dict) or "accountValue" not in margin:
        raise RuntimeError("clearinghouseState missing marginSummary.accountValue")
    if "withdrawable" not in data:
        raise RuntimeError("clearinghouseState missing withdrawable")
    positions = data.get("assetPositions")
    if not isinstance(positions, list):
        raise RuntimeError("clearinghouseState missing assetPositions")
    account_value = _require_float(margin["accountValue"], "accountValue")
    withdrawable = _require_float(data["withdrawable"], "withdrawable")
    found = {"BTC": None, "HYPE": None}
    for item in positions:
        if not isinstance(item, dict):
            continue
        pos = item.get("position") or {}
        if not isinstance(pos, dict):
            raise RuntimeError("asset position is not an object")
        coin = pos.get("coin")
        if coin not in found:
            continue
        if "szi" not in pos:
            raise RuntimeError(f"missing {coin}.szi")
        size = _require_float(pos.get("szi"), f"{coin}.szi")
        if abs(size) < 1e-12:
            continue
        entry_px = _require_float(pos.get("entryPx"), f"{coin}.entryPx")
        upnl_raw = pos.get("unrealizedPnl")
        upnl = 0.0 if upnl_raw in (None, "") else _require_float(upnl_raw, f"{coin}.unrealizedPnl")
        found[coin] = {"size": size, "entry_px": entry_px, "upnl": upnl}
    return {
        "account_value": account_value,
        "withdrawable": withdrawable,
        "btc": found["BTC"],
        "hype": found["HYPE"],
    }


def parse_mid(mids, coin: str) -> float:
    if not isinstance(mids, dict) or coin not in mids:
        raise RuntimeError(f"allMids missing {coin}")
    return float(mids[coin])


# --- HTTP -------------------------------------------------------------------


def post_info(payload: dict, timeout: float = 20.0):
    import requests

    kind = payload.get("type")
    try:
        response = requests.post(API_URL, json=payload, timeout=timeout)
        response.raise_for_status()
        data = response.json()
    except Exception as exc:
        raise RuntimeError(f"hyperliquid {kind} failed: {exc}") from exc
    if isinstance(data, dict) and set(data) == {"error"}:
        raise RuntimeError(f"hyperliquid {kind}: {data.get('error')}")
    return data


def fetch_account(fetch=None) -> dict:
    fetch = post_info if fetch is None else fetch
    raw = fetch({"type": "clearinghouseState", "user": ACCOUNT})
    return parse_account(raw)


def fetch_mids(fetch=None):
    fetch = post_info if fetch is None else fetch
    return fetch({"type": "allMids"})


def fetch_btc_candles(now: datetime, fetch=None) -> list[dict]:
    fetch = post_info if fetch is None else fetch
    now_ms = int(as_sp(now).timestamp() * 1000)
    start = now_ms - CANDLE_LOOKBACK_HOURS * HOUR_MS
    raw = fetch(
        {
            "type": "candleSnapshot",
            "req": {
                "coin": "BTC",
                "interval": "1h",
                "startTime": start,
                "endTime": now_ms,
            },
        }
    )
    return parse_candles(raw)


# --- commands ---------------------------------------------------------------


def _persist_hype_asked(state_dir: Path, now: datetime) -> None:
    try:
        state = load_hype_state(state_dir)
    except StateLoadError:
        state = default_hype_state()
    state["asked"] = True
    state["asked_at"] = iso_sp(now)
    save_hype_state(state_dir, state)


def run_hype(state_dir: Path, mark_asked: bool, now: datetime | None = None, fetch=None) -> dict:
    fetch = post_info if fetch is None else fetch
    now = as_sp(now or now_sp())
    if mark_asked:
        with _StateLock(state_dir):
            _persist_hype_asked(state_dir, now)
            asked_at = iso_sp(now)
        return {"action": "marked_asked", "asked_at": asked_at}
    account = fetch_account(fetch)
    mid = parse_mid(fetch_mids(fetch), "HYPE")
    position = account.get("hype")

    def _decide(asked: bool) -> dict:
        return decide_hype(mid, position, asked, now.date())

    try:
        with _StateLock(state_dir):
            try:
                asked = bool(load_hype_state(state_dir).get("asked"))
            except StateLoadError:
                asked = False
            payload = _decide(asked)
            # Printing ask_deadline is itself the ask. Persist it so the next
            # run does not repeat the prompt. A close or a short is not an ask.
            if payload.get("action") == "ask_deadline":
                try:
                    _persist_hype_asked(state_dir, now)
                except OSError:
                    pass
            return payload
    except OSError:
        try:
            asked = bool(load_hype_state(state_dir).get("asked"))
        except StateLoadError:
            asked = False
        payload = _decide(asked)
        if payload.get("action") == "ask_deadline":
            try:
                _persist_hype_asked(state_dir, now)
            except OSError:
                pass
        return payload


def _account_is_long(account: dict) -> bool:
    btc = account.get("btc") if isinstance(account, dict) else None
    if not isinstance(btc, dict):
        return False
    size = _finite_number(btc.get("size"))
    return size is not None and size > 0


def _btc_finish(state_dir: Path, candles, account, now: datetime, fetch, write: bool) -> dict:
    """Decide from the state file on disk. `write` is false when the lock was not taken."""
    state_error = None
    state = None
    try:
        state = load_btc_state(state_dir)
    except StateLoadError as exc:
        state_error = one_line(exc)
        state = None
    if state_error is None and state is not None:
        field_error = validate_state_fields(state)
        if field_error:
            state_error = field_error
    fill_entry_ms = None
    if state_error and _account_is_long(account):
        fill_entry_ms = lookup_open_long_ms(fetch, now)
    payload, new_state = decide_btc(
        candles,
        account,
        state,
        now,
        state_error=state_error,
        fill_entry_ms=fill_entry_ms,
    )
    if not write:
        payload["state_write_skipped"] = True
        return payload
    try:
        if _material(state) != _material(new_state):
            new_state["last_check"] = iso_sp(now)
            save_btc_state(state_dir, new_state)
    except Exception as exc:
        # A full disk or a permissions error must not hide an exit.
        payload["state_write_failed"] = True
        payload.setdefault("state_error", one_line(exc))
        payload["state_mismatch"] = True
    return payload


def run_btc(state_dir: Path, now: datetime | None = None, fetch=None) -> dict:
    fetch = post_info if fetch is None else fetch
    now = as_sp(now or now_sp())
    candles = fetch_btc_candles(now, fetch)
    account = fetch_account(fetch)
    try:
        with _StateLock(state_dir):
            return _btc_finish(state_dir, candles, account, now, fetch, write=True)
    except OSError:
        # TimeoutError is an OSError. The file may still be readable; do not
        # throw it away and rebuild as if the long were 400 hours old.
        return _btc_finish(state_dir, candles, account, now, fetch, write=False)


def run_record_entry(state_dir: Path, price: float, size: float, now: datetime | None = None) -> dict:
    if not math.isfinite(price) or price <= 0:
        raise ValueError("entry price must be positive")
    if not math.isfinite(size) or size <= 0:
        raise ValueError("entry size must be positive")
    now = as_sp(now or now_sp())
    stamp = iso_sp(now)
    with _StateLock(state_dir):
        try:
            state = load_btc_state(state_dir)
        except StateLoadError as exc:
            raise CommandError(f"cannot record entry: {exc}") from exc
        position = state.get("position")
        if _position_is_open(position):
            if _same_open_position(position, price, size):
                return {
                    "action": "already_recorded",
                    "price": clean_float(position.get("entry_price"), 8),
                    "size": clean_float(position.get("size"), 8),
                    "time": position.get("entry_time"),
                }
            raise CommandError("position already open")
        atr = state.get("atr14_1h")
        state["position"] = {
            "entry_price": clean_float(price, 8),
            "size": clean_float(size, 8),
            "entry_time": stamp,
        }
        state["highest_close"] = clean_float(price, 8)
        if atr is None:
            state["stop"] = None
        else:
            state["stop"] = clean_float(float(price) - ATR_MULT * float(atr), 6)
        # The signal candle already closed before the fill. Do not treat its close
        # as a post-entry high; the next closed bar is the first one that counts.
        state["last_processed_candle"] = state.get("last_closed_candle_open")
        state["last_check"] = stamp
        save_btc_state(state_dir, state)
        append_btc_log(
            state_dir,
            f"## {stamp} entry\n- price: {clean_float(price, 8)}\n- size: {clean_float(size, 8)}\n\n",
        )
    return {
        "action": "recorded_entry",
        "price": clean_float(price, 8),
        "size": clean_float(size, 8),
        "time": stamp,
    }


def run_record_exit(state_dir: Path, price: float, now: datetime | None = None) -> dict:
    if not math.isfinite(price) or price <= 0:
        raise ValueError("exit price must be positive")
    now = as_sp(now or now_sp())
    stamp = iso_sp(now)
    with _StateLock(state_dir):
        corrupt = False
        try:
            state = load_btc_state(state_dir)
        except StateLoadError:
            state = default_btc_state()
            corrupt = True
        if not corrupt and not _position_is_open(state.get("position")):
            return {
                "action": "recorded_exit",
                "price": clean_float(price, 8),
                "time": stamp,
                "already_flat": True,
            }
        if not state.get("last_exit_candle"):
            state["last_exit_candle"] = state.get("last_closed_candle_open")
        state["position"] = None
        state["highest_close"] = None
        state["stop"] = None
        state["last_processed_candle"] = None
        state["last_check"] = stamp
        save_btc_state(state_dir, state)
        append_btc_log(state_dir, f"## {stamp} exit\n- price: {clean_float(price, 8)}\n\n")
    return {"action": "recorded_exit", "price": clean_float(price, 8), "time": stamp}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="signal_check.py",
        description="Read-only Hyperliquid signal. Prints one JSON line and does not trade.",
    )
    parser.add_argument(
        "--state-dir",
        default=DEFAULT_STATE_DIR,
        help=f"directory for state files (default: {DEFAULT_STATE_DIR})",
    )
    parser.add_argument("command", choices=("hype", "btc"))
    parser.add_argument(
        "--mark-asked",
        action="store_true",
        help="hype only: record that the 2026-11-16 deadline was already asked",
    )
    parser.add_argument(
        "--record-entry",
        nargs=2,
        metavar=("PRICE", "SIZE"),
        help="btc only: persist a fill after the caller entered",
    )
    parser.add_argument(
        "--record-exit",
        metavar="PRICE",
        help="btc only: persist a fill after the caller exited",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    try:
        args = build_parser().parse_args(argv)
        state_dir = Path(args.state_dir)
        if args.command == "hype":
            if args.record_entry is not None or args.record_exit is not None:
                raise ValueError("record-entry and record-exit are btc commands")
            payload = run_hype(state_dir, mark_asked=args.mark_asked)
        else:
            if args.mark_asked:
                raise ValueError("mark-asked is a hype command")
            if args.record_entry is not None and args.record_exit is not None:
                raise ValueError("pass only one of --record-entry and --record-exit")
            if args.record_entry is not None:
                price = float(args.record_entry[0])
                size = float(args.record_entry[1])
                payload = run_record_entry(state_dir, price, size)
            elif args.record_exit is not None:
                payload = run_record_exit(state_dir, float(args.record_exit))
            else:
                payload = run_btc(state_dir)
        emit(payload)
        return 0
    except SystemExit:
        raise
    except CommandError as exc:
        emit({"action": "error", "reason": exc.reason})
        return exc.code
    except Exception as exc:
        emit({"action": "error", "reason": one_line(exc)})
        return 0


if __name__ == "__main__":
    sys.exit(main())
