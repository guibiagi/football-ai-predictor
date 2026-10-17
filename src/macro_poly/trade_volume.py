from __future__ import annotations

import json
import time
from dataclasses import asdict, dataclass
from pathlib import Path

from src.macro_poly.http_util import get_json

DATA_API = "https://data-api.polymarket.com/trades"
PAGE_LIMIT = 100
MAX_OFFSET = 10000
WINDOW_SEC = 30 * 86400
MAX_PAGES_PER_WINDOW = (MAX_OFFSET // PAGE_LIMIT) + 1
GAMMA_COVERAGE_MIN = 0.45  # pre-registered exclusion for trade-based volume


@dataclass
class TradeFetchMeta:
    pages_fetched: int
    trades_kept: int
    trade_volume_total_usd: float
    gamma_volume_usd: float
    gamma_ratio: float | None
    gamma_ratio_below_min: bool
    oldest_trade_ts: int | None
    hit_page_cap: bool
    fetch_error: bool
    complete: bool
    pagination_mode: str


def trade_dedupe_key(t: dict) -> tuple:
    return (
        str(t.get("transactionHash") or ""),
        str(t.get("asset") or ""),
        str(t.get("side") or ""),
        str(t.get("size") or ""),
    )


def _fetch_window(
    condition_id: str,
    token_ids: set[str],
    start: int,
    end: int,
) -> tuple[list[dict], int, bool, bool]:
    """Offset pagination inside [start, end]. Returns rows, pages, hit_cap, fetch_error."""
    rows: list[dict] = []
    pages = 0
    hit_cap = False
    fetch_error = False
    offset = 0
    while pages < MAX_PAGES_PER_WINDOW:
        url = (
            f"{DATA_API}?market={condition_id}&limit={PAGE_LIMIT}&offset={offset}"
            f"&start={start}&end={end}&takerOnly=false"
        )
        try:
            batch = get_json(url)
        except Exception:
            fetch_error = True
            break
        pages += 1
        if not batch:
            break
        for t in batch:
            if str(t.get("asset")) in token_ids:
                rows.append(t)
        if len(batch) < PAGE_LIMIT:
            break
        offset += PAGE_LIMIT
        if offset > MAX_OFFSET:
            hit_cap = True
            break
    return rows, pages, hit_cap, fetch_error


def fetch_trades_paged(
    condition_id: str,
    token_ids: set[str],
    start_ts: int | None = None,
) -> tuple[list[dict], TradeFetchMeta]:
    """
    Fetch trades via data-api ``start``/``end`` windows + ``offset`` (``before`` is ineffective).

    ``complete`` is False when any window hits the offset cap with a full page, on fetch error,
    or when the earliest kept trade is still after ``start_ts`` (truncated early history).
    """
    end_ts = int(time.time())
    floor = int(start_ts or 0)
    merged: list[dict] = []
    seen: set[tuple] = set()
    total_pages = 0
    hit_cap = False
    fetch_error = False

    if floor >= end_ts:
        floor = max(0, end_ts - WINDOW_SEC)

    window_start = floor
    while window_start < end_ts:
        window_end = min(window_start + WINDOW_SEC, end_ts)
        chunk, pages, cap, err = _fetch_window(condition_id, token_ids, window_start, window_end)
        total_pages += pages
        hit_cap = hit_cap or cap
        fetch_error = fetch_error or err
        for t in chunk:
            key = trade_dedupe_key(t)
            if key in seen:
                continue
            seen.add(key)
            merged.append(t)
        window_start = window_end
        if err:
            break

    oldest: int | None = None
    if merged:
        oldest = min(int(t.get("timestamp") or 0) for t in merged)

    complete = (
        not hit_cap
        and not fetch_error
        and oldest is not None
        and (start_ts is None or oldest <= start_ts + 86400)
    )

    total_usd = sum(float(t.get("size") or 0) * float(t.get("price") or 0) for t in merged)
    meta = TradeFetchMeta(
        pages_fetched=total_pages,
        trades_kept=len(merged),
        trade_volume_total_usd=total_usd,
        gamma_volume_usd=0.0,
        gamma_ratio=None,
        gamma_ratio_below_min=False,
        oldest_trade_ts=oldest,
        hit_page_cap=hit_cap,
        fetch_error=fetch_error,
        complete=complete,
        pagination_mode="start_end_offset",
    )
    return merged, meta


def _meta_from_dict(d: dict) -> TradeFetchMeta:
    md = dict(d)
    if "gamma_ratio_below_min" not in md:
        ratio = md.get("gamma_ratio")
        md["gamma_ratio_below_min"] = ratio is not None and ratio < GAMMA_COVERAGE_MIN
    if "oldest_trade_ts" not in md:
        md["oldest_trade_ts"] = None
    if "pagination_mode" not in md:
        md["pagination_mode"] = "legacy"
    # v3 caches used broken `before` pagination — never treat as complete
    if md.get("pagination_mode") == "legacy":
        md["complete"] = False
    return TradeFetchMeta(**md)


def load_trades_cached(
    condition_id: str,
    token_ids: set[str],
    gamma_volume: float,
    cache_dir: Path,
    start_ts: int | None = None,
) -> tuple[list[dict] | None, TradeFetchMeta]:
    cache_dir.mkdir(parents=True, exist_ok=True)
    path = cache_dir / f"{condition_id}_v4.json"
    if path.exists():
        data = json.loads(path.read_text(encoding="utf-8"))
        if data.get("failed"):
            return None, _meta_from_dict(data["meta"])
        meta = _meta_from_dict(data["meta"])
        return data.get("trades") or [], meta

    trades, meta = fetch_trades_paged(condition_id, token_ids, start_ts=start_ts)
    meta.gamma_volume_usd = gamma_volume
    if gamma_volume > 100:
        meta.gamma_ratio = meta.trade_volume_total_usd / gamma_volume
        meta.gamma_ratio_below_min = meta.gamma_ratio < GAMMA_COVERAGE_MIN
        if meta.gamma_ratio_below_min:
            meta.complete = False
    payload = {
        "condition_id": condition_id,
        "failed": False,
        "trades": trades,
        "meta": asdict(meta),
    }
    path.write_text(json.dumps(payload), encoding="utf-8")
    return trades, meta


def cumulative_volume_usd(trades: list[dict], until_ts: int) -> float:
    total = 0.0
    for t in trades:
        ts = int(t.get("timestamp") or 0)
        if ts > until_ts:
            continue
        total += float(t.get("size") or 0) * float(t.get("price") or 0)
    return total


class TradeVolumeStore:
    def __init__(self, cache_dir: Path):
        self.cache_dir = cache_dir
        self._trades: dict[str, list[dict] | None] = {}
        self._meta: dict[str, TradeFetchMeta] = {}
        self._reg: dict[str, tuple[str, set[str], float, int | None]] = {}

    def register(
        self,
        token_yes: str,
        condition_id: str,
        token_no: str | None = None,
        gamma_volume: float = 0.0,
        start_ts: int | None = None,
    ) -> None:
        tokens = {str(token_yes)}
        if token_no:
            tokens.add(str(token_no))
        self._reg[token_yes] = (condition_id, tokens, gamma_volume, start_ts)

    def _ensure(self, token_yes: str) -> None:
        if token_yes in self._trades:
            return
        reg = self._reg.get(token_yes)
        if not reg:
            self._trades[token_yes] = None
            return
        cid, tokens, gvol, start_ts = reg
        trades, meta = load_trades_cached(cid, tokens, gvol, self.cache_dir, start_ts=start_ts)
        self._trades[token_yes] = trades
        self._meta[token_yes] = meta

    def is_incomplete(self, token_yes: str) -> bool:
        self._ensure(token_yes)
        meta = self._meta.get(token_yes)
        return meta is None or not meta.complete

    def meta(self, token_yes: str) -> TradeFetchMeta | None:
        self._ensure(token_yes)
        return self._meta.get(token_yes)

    def trade_volume_usable(self, token_yes: str) -> bool:
        self._ensure(token_yes)
        meta = self._meta.get(token_yes)
        return meta is not None and meta.complete and not meta.gamma_ratio_below_min

    def volume_at(self, token_yes: str, decision_ts: int) -> float | None:
        if not self.trade_volume_usable(token_yes):
            return None
        trades = self._trades.get(token_yes)
        if trades is None:
            return None
        return cumulative_volume_usd(trades, decision_ts)

    def coverage_fraction(self, markets, decision_ts_fn) -> float:
        ok = 0
        n = 0
        for m in markets:
            ts = decision_ts_fn(m)
            if ts is None:
                continue
            n += 1
            if self.trade_volume_usable(m.token_yes):
                v = self.volume_at(m.token_yes, ts)
                if v is not None:
                    ok += 1
        return ok / max(n, 1)

    def segmentation_feasible(self, min_complete_frac: float = 0.8) -> tuple[bool, dict[str, int]]:
        complete = 0
        capped_legacy_100 = 0
        gamma_low = 0
        for tid in self._reg:
            self._ensure(tid)
            meta = self._meta.get(tid)
            if meta is None:
                continue
            if meta.complete:
                complete += 1
            if meta.trades_kept == 100 and meta.pagination_mode == "legacy":
                capped_legacy_100 += 1
            if meta.gamma_ratio_below_min:
                gamma_low += 1
        n = len(self._reg)
        feasible = n > 0 and (complete / n) >= min_complete_frac
        return feasible, {
            "registered": n,
            "trade_history_complete": complete,
            "legacy_capped_at_100": capped_legacy_100,
            "gamma_ratio_below_min": gamma_low,
        }

    def incomplete_summary(self) -> dict[str, int]:
        fetch_inc = 0
        gamma_low = 0
        for tid in self._reg:
            self._ensure(tid)
            meta = self._meta.get(tid)
            if meta is None:
                fetch_inc += 1
                continue
            if not meta.complete:
                fetch_inc += 1
            if meta.gamma_ratio_below_min:
                gamma_low += 1
        return {
            "registered": len(self._reg),
            "fetch_incomplete": fetch_inc,
            "gamma_ratio_below_min": gamma_low,
        }
