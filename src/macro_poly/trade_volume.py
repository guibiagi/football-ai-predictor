from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path

from src.macro_poly.http_util import get_json

DATA_API = "https://data-api.polymarket.com/trades"
PAGE_LIMIT = 100  # API returns at most ~100 rows per request
MAX_PAGES = 500  # safety cap on cursor pages
GAMMA_COVERAGE_MIN = 0.45  # trade USD / gamma volume


@dataclass
class TradeFetchMeta:
    pages_fetched: int
    trades_kept: int
    trade_volume_total_usd: float
    gamma_volume_usd: float
    gamma_ratio: float | None
    gamma_ratio_below_min: bool
    hit_page_cap: bool
    fetch_error: bool
    complete: bool


def fetch_trades_paged(
    condition_id: str,
    token_ids: set[str],
    start_ts: int | None = None,
) -> tuple[list[dict], TradeFetchMeta]:
    """Fetch all trades for a market (newest-first API; cursor via ``before``)."""
    rows: list[dict] = []
    seen_tx: set[str] = set()
    pages = 0
    hit_cap = False
    fetch_error = False
    before: int | None = None
    while pages < MAX_PAGES:
        url = f"{DATA_API}?market={condition_id}&limit={PAGE_LIMIT}"
        if before is not None:
            url += f"&before={before}"
        try:
            batch = get_json(url)
        except Exception:
            fetch_error = True
            break
        pages += 1
        if not batch:
            break
        oldest_ts = min(int(t.get("timestamp") or 0) for t in batch)
        for t in batch:
            tx = str(t.get("transactionHash") or "")
            if tx and tx in seen_tx:
                continue
            if tx:
                seen_tx.add(tx)
            ts = int(t.get("timestamp") or 0)
            if start_ts is not None and ts < start_ts:
                continue
            if str(t.get("asset")) in token_ids:
                rows.append(t)
        if len(batch) < PAGE_LIMIT:
            break
        next_before = oldest_ts - 1
        if before is not None and next_before >= before:
            break
        before = next_before
    else:
        hit_cap = True

    total_usd = sum(float(t.get("size") or 0) * float(t.get("price") or 0) for t in rows)
    meta = TradeFetchMeta(
        pages_fetched=pages,
        trades_kept=len(rows),
        trade_volume_total_usd=total_usd,
        gamma_volume_usd=0.0,
        gamma_ratio=None,
        gamma_ratio_below_min=False,
        hit_page_cap=hit_cap,
        fetch_error=fetch_error,
        complete=not hit_cap and not fetch_error,
    )
    return rows, meta


def _meta_from_dict(d: dict) -> TradeFetchMeta:
    md = dict(d)
    if "gamma_ratio_below_min" not in md:
        ratio = md.get("gamma_ratio")
        md["gamma_ratio_below_min"] = ratio is not None and ratio < GAMMA_COVERAGE_MIN
        if not md.get("hit_page_cap") and not md.get("fetch_error"):
            md["complete"] = True
    return TradeFetchMeta(**md)


def load_trades_cached(
    condition_id: str,
    token_ids: set[str],
    gamma_volume: float,
    cache_dir: Path,
    start_ts: int | None = None,
) -> tuple[list[dict] | None, TradeFetchMeta]:
    cache_dir.mkdir(parents=True, exist_ok=True)
    path = cache_dir / f"{condition_id}_v3.json"
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

    def volume_at(self, token_yes: str, decision_ts: int) -> float | None:
        self._ensure(token_yes)
        if self.is_incomplete(token_yes):
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
            if not self.is_incomplete(m.token_yes):
                v = self.volume_at(m.token_yes, ts)
                if v is not None:
                    ok += 1
        return ok / max(n, 1)

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
