"""Livro na hora do sinal e trades acumulados até o kickoff.

A data-api pública é lida com `takerOnly=true` (o lado é o do taker).
A janela é fatiada para não depender de um offset enorme. Erro, 429 esgotado
ou página vazia no meio da paginação deixam a fita incompleta.

A fita só fica `complete` depois do kickoff. Uma hora depois disso ela congela:
não é consultada de novo e a cobertura não volta para incompleta.

Sem chave, sem ordem. Só leitura.
"""

from __future__ import annotations

import json
import re
import time
import urllib.error
import urllib.request
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from src.queue_fill import (
    assumed_shares,
    best_ask_price,
    queue_ahead_from_book,
    read_tick_size,
    snap_bid_to_tick,
    trade_timestamp,
)

DATA_API = "https://data-api.polymarket.com/trades"
PAGE_LIMIT = 100
MAX_OFFSET = 10000
SUB_WINDOW_SEC = 6 * 3600
FREEZE_BUFFER_SEC = 3600
PAGE_PAUSE_SEC = 0.2
RETRY_DELAYS_SEC = (0.5, 1.0, 2.0, 4.0)
TAPE_DIR = Path(__file__).resolve().parent.parent / "data" / "paper_rn1_tapes"

_UPGRADE_FIELDS = (
    "sinal",
    "placed_at",
    "queue_ahead",
    "our_size",
    "sim_price",
    "tick_size",
    "tick_known",
    "token_no",
    "condition_id",
    "fill_coverage",
    "entry_cross",
    "bid",
    "poly_exec",
    "token",
)


@dataclass(frozen=True)
class TradeFetch:
    trades: list[dict]
    hit_cap: bool
    fetch_error: bool


def _iso(dt: datetime) -> str:
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    else:
        dt = dt.astimezone(timezone.utc)
    return dt.strftime("%Y-%m-%dT%H:%M:%SZ")


def iso_to_ts(value) -> int | None:
    if value is None:
        return None
    text = str(value).strip()
    if not text or text.lower() in {"nan", "none", "nat", "<na>"}:
        return None
    try:
        parsed = datetime.fromisoformat(text.replace("Z", "+00:00"))
    except ValueError:
        return None
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return int(parsed.timestamp())


def _now_ts(now) -> int | None:
    if now is None:
        return None
    if isinstance(now, datetime):
        if now.tzinfo is None:
            now = now.replace(tzinfo=timezone.utc)
        return int(now.timestamp())
    try:
        return int(now)
    except (TypeError, ValueError):
        return None


def _blank(value) -> bool:
    if value is None:
        return True
    try:
        if pd.isna(value):
            return True
    except (TypeError, ValueError):
        pass
    if isinstance(value, str):
        return value.strip() == "" or value.strip().lower() in {"nan", "none", "nat"}
    return False


def _sleep(seconds: float) -> None:
    if seconds > 0:
        time.sleep(seconds)


def capture_entry(row: dict, book: dict | None, now: datetime) -> tuple[dict, dict]:
    """Grava a fila no preço simulado (bid descido ao tick). Sem rede."""
    bid = float(row["bid"])
    tick = read_tick_size(book)
    sim = snap_bid_to_tick(bid, tick)
    tick_known = sim is not None
    queue_ahead = queue_ahead_from_book(book, sim, tick) if tick_known else None
    observed = queue_ahead is not None
    ask = best_ask_price(book)
    entry_cross = bool(tick_known and ask is not None and ask <= sim)
    shares = assumed_shares(sim if sim is not None else bid)
    placed = _iso(now)
    updated = dict(row)
    updated["placed_at"] = placed
    updated["queue_ahead"] = queue_ahead
    updated["our_size"] = shares
    updated["sim_price"] = sim
    updated["tick_size"] = tick
    updated["tick_known"] = tick_known
    updated["entry_cross"] = entry_cross
    updated["fill_coverage"] = "pending"
    tape = {
        "slug": row.get("slug"),
        "token": row.get("token"),
        "token_no": row.get("token_no"),
        "condition_id": row.get("condition_id"),
        "bid": bid,
        "sim_price": sim,
        "tick_size": tick,
        "tick_known": tick_known,
        "entry_cross": entry_cross,
        "placed_at": placed,
        "kickoff": row.get("kickoff"),
        "queue_ahead": queue_ahead,
        "our_size": shares,
        "book_observed": observed,
        "trades": [],
        "coverage": "pending",
        "frozen": False,
        "fetch_error": False,
        "hit_cap": False,
    }
    return updated, tape


def _trade_key(trade: dict) -> tuple:
    outcome_index = trade.get("outcomeIndex")
    return (
        str(trade.get("transactionHash") or ""),
        str(trade.get("asset") or trade.get("asset_id") or ""),
        str(trade.get("side") or ""),
        str(trade.get("price") or ""),
        str(trade.get("size") or ""),
        str(trade_timestamp(trade.get("timestamp"))),
        str(trade.get("proxyWallet") or ""),
        "" if outcome_index is None else str(outcome_index),
    )


def _is_frozen_tape(tape: dict, now_ts: int | None) -> bool:
    if tape.get("frozen"):
        return True
    if str(tape.get("coverage") or "") != "complete":
        return False
    kickoff_ts = iso_to_ts(tape.get("kickoff"))
    if kickoff_ts is None or now_ts is None:
        return False
    return now_ts >= kickoff_ts + FREEZE_BUFFER_SEC


def merge_polled_trades(
    tape: dict,
    fetched: list[dict],
    *,
    hit_cap: bool,
    fetch_error: bool,
    now=None,
) -> dict:
    """Junta trades novos. Fita congelada não muda e não perde a cobertura."""
    if tape.get("frozen"):
        kept = dict(tape)
        if str(kept.get("coverage") or "") == "complete":
            kept["coverage"] = "complete"
        kept["frozen"] = True
        return kept

    placed_ts = iso_to_ts(tape.get("placed_at"))
    kickoff_ts = iso_to_ts(tape.get("kickoff"))
    kept_trades: list[dict] = []
    seen: set[tuple] = set()
    for trade in list(tape.get("trades") or []) + list(fetched or []):
        if not isinstance(trade, dict):
            continue
        ts = trade_timestamp(trade.get("timestamp"))
        if placed_ts is None or ts is None or ts <= placed_ts:
            continue
        if kickoff_ts is not None and ts >= kickoff_ts:
            continue
        key = _trade_key(trade)
        if key in seen:
            continue
        seen.add(key)
        kept_trades.append(trade)

    now_ts = _now_ts(now)
    window_closed = kickoff_ts is not None and now_ts is not None and now_ts >= kickoff_ts
    if fetch_error or hit_cap:
        coverage = "incomplete"
        frozen = False
    elif not window_closed:
        coverage = "partial"
        frozen = False
    else:
        coverage = "complete"
        frozen = now_ts is not None and kickoff_ts is not None and now_ts >= kickoff_ts + FREEZE_BUFFER_SEC

    merged = dict(tape)
    merged["trades"] = kept_trades
    merged["hit_cap"] = bool(hit_cap)
    merged["fetch_error"] = bool(fetch_error)
    merged["coverage"] = coverage
    merged["frozen"] = frozen
    return merged


def _token_set(token_ids) -> set[str]:
    if token_ids is None:
        return set()
    if isinstance(token_ids, str):
        return {token_ids} if token_ids else set()
    return {str(token) for token in token_ids if token}


def _read_batch(opener, req, sleeper):
    """Devolve (batch, erro). 429 espera e tenta de novo; esgotado vira erro."""
    delays = list(RETRY_DELAYS_SEC)
    while True:
        try:
            with opener(req, timeout=60) as resp:
                batch = json.load(resp)
            return batch, False
        except urllib.error.HTTPError as exc:
            if exc.code == 429 and delays:
                sleeper(delays.pop(0))
                continue
            return None, True
        except (urllib.error.URLError, TimeoutError, json.JSONDecodeError, OSError, ValueError):
            return None, True


def _fetch_subwindow(condition_id, token_ids, start_ts, end_ts, opener, sleeper) -> TradeFetch:
    kept: list[dict] = []
    offset = 0
    hit_cap = False
    while True:
        if offset >= MAX_OFFSET:
            hit_cap = True
            break
        url = (
            f"{DATA_API}?market={condition_id}&limit={PAGE_LIMIT}&offset={offset}"
            f"&start={int(start_ts)}&end={int(end_ts)}&takerOnly=true"
        )
        req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
        batch, failed = _read_batch(opener, req, sleeper)
        if failed:
            return TradeFetch(trades=kept, hit_cap=False, fetch_error=True)
        if not isinstance(batch, list):
            return TradeFetch(trades=kept, hit_cap=False, fetch_error=True)
        if not batch:
            if offset > 0:
                hit_cap = True
            break
        for trade in batch:
            if not isinstance(trade, dict):
                continue
            asset = str(trade.get("asset") or trade.get("asset_id") or "")
            if asset in token_ids:
                kept.append(trade)
        if len(batch) < PAGE_LIMIT:
            break
        offset += PAGE_LIMIT
        if offset >= MAX_OFFSET:
            hit_cap = True
            break
        sleeper(PAGE_PAUSE_SEC)
    return TradeFetch(trades=kept, hit_cap=hit_cap, fetch_error=False)


def fetch_trades_window(
    condition_id: str,
    token_ids,
    start_ts: int,
    end_ts: int,
    *,
    urlopen=None,
    sleeper=None,
) -> TradeFetch:
    """Trades taker dos tokens na janela. Falha de rede não vira volume zero medido."""
    opener = urlopen or urllib.request.urlopen
    pause = sleeper or _sleep
    tokens = _token_set(token_ids)
    if end_ts <= start_ts or not tokens:
        return TradeFetch(trades=[], hit_cap=False, fetch_error=False)
    kept: list[dict] = []
    hit_cap = False
    cursor = int(start_ts)
    end_ts = int(end_ts)
    while cursor < end_ts:
        chunk_end = min(cursor + SUB_WINDOW_SEC, end_ts)
        part = _fetch_subwindow(condition_id, tokens, cursor, chunk_end, opener, pause)
        kept.extend(part.trades)
        hit_cap = hit_cap or part.hit_cap
        if part.fetch_error:
            return TradeFetch(trades=kept, hit_cap=hit_cap, fetch_error=True)
        cursor = chunk_end
        if cursor < end_ts:
            pause(PAGE_PAUSE_SEC)
    return TradeFetch(trades=kept, hit_cap=hit_cap, fetch_error=False)


def _slug_path(slug: str, tape_dir: Path) -> Path:
    safe = re.sub(r"[^A-Za-z0-9._-]+", "_", str(slug))[:120] or "sem-slug"
    return tape_dir / f"{safe}.json"


def save_tape(tape: dict, tape_dir: Path | None = None) -> Path:
    directory = tape_dir or TAPE_DIR
    directory.mkdir(parents=True, exist_ok=True)
    path = _slug_path(str(tape.get("slug") or ""), directory)
    path.write_text(json.dumps(tape), encoding="utf-8")
    return path


def load_tape(slug: str, tape_dir: Path | None = None) -> dict | None:
    directory = tape_dir or TAPE_DIR
    path = _slug_path(slug, directory)
    if not path.exists():
        return None
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    return data if isinstance(data, dict) else None


def _tape_from_row(row) -> dict | None:
    placed = row.get("placed_at")
    if _blank(placed):
        return None
    bid = row.get("bid")
    try:
        bid_f = float(bid)
    except (TypeError, ValueError):
        return None
    queue_raw = row.get("queue_ahead")
    observed = not _blank(queue_raw)
    our_raw = row.get("our_size")
    sim = None if _blank(row.get("sim_price")) else row.get("sim_price")
    try:
        sim_f = float(sim) if sim is not None else None
    except (TypeError, ValueError):
        sim_f = None
    try:
        our = float(our_raw) if not _blank(our_raw) else assumed_shares(sim_f or bid_f)
    except (TypeError, ValueError):
        our = assumed_shares(sim_f or bid_f)
    tick_known = row.get("tick_known")
    if tick_known is None and sim_f is not None:
        tick_known = True
    return {
        "slug": row.get("slug"),
        "token": None if _blank(row.get("token")) else str(row.get("token")),
        "token_no": None if _blank(row.get("token_no")) else str(row.get("token_no")),
        "condition_id": None if _blank(row.get("condition_id")) else str(row.get("condition_id")),
        "bid": bid_f,
        "sim_price": sim_f,
        "tick_size": None if _blank(row.get("tick_size")) else row.get("tick_size"),
        "tick_known": bool(tick_known) if tick_known is not None else None,
        "entry_cross": bool(row.get("entry_cross")) if not _blank(row.get("entry_cross")) else False,
        "placed_at": str(placed),
        "kickoff": None if _blank(row.get("kickoff")) else str(row.get("kickoff")),
        "queue_ahead": float(queue_raw) if observed else None,
        "our_size": our,
        "book_observed": observed,
        "trades": [],
        "coverage": "pending",
        "frozen": False,
        "fetch_error": False,
        "hit_cap": False,
    }


def _copy_tape_fields(log, i, tape: dict) -> None:
    log.at[i, "fill_coverage"] = tape.get("coverage")
    if tape.get("book_observed") and tape.get("queue_ahead") is not None:
        log.at[i, "queue_ahead"] = tape["queue_ahead"]
    if tape.get("our_size") is not None:
        log.at[i, "our_size"] = tape["our_size"]
    if "sim_price" in log.columns and tape.get("sim_price") is not None:
        log.at[i, "sim_price"] = tape["sim_price"]


def poll_tapes(log, *, fetch=None, tape_dir: Path | None = None, now: datetime | None = None):
    """Acumula trades até congelar. Fita congelada não gera outro request."""
    directory = tape_dir or TAPE_DIR
    moment = now or datetime.now(timezone.utc)
    now_ts = _now_ts(moment)
    fetch_fn = fetch or (
        lambda condition_id, tokens, start_ts, end_ts: fetch_trades_window(
            condition_id, tokens, start_ts, end_ts
        )
    )
    for col in ("fill_coverage", "queue_ahead", "our_size", "sim_price"):
        if col not in log.columns:
            log[col] = None
    log["fill_coverage"] = log["fill_coverage"].astype(object)
    for i, row in log.iterrows():
        base = load_tape(str(row.get("slug") or ""), directory) or _tape_from_row(row)
        if base is None:
            continue
        placed_ts = iso_to_ts(base.get("placed_at"))
        if placed_ts is None:
            continue
        if now_ts is not None and placed_ts > now_ts:
            continue
        if base.get("frozen") or _is_frozen_tape(base, now_ts):
            base["frozen"] = True
            if str(base.get("coverage") or "") == "complete":
                base["coverage"] = "complete"
            save_tape(base, directory)
            _copy_tape_fields(log, i, base)
            continue
        condition_id = base.get("condition_id") or (
            None if _blank(row.get("condition_id")) else str(row.get("condition_id"))
        )
        token = base.get("token") or (None if _blank(row.get("token")) else str(row.get("token")))
        token_no = base.get("token_no") or (None if _blank(row.get("token_no")) else str(row.get("token_no")))
        kickoff_ts = iso_to_ts(base.get("kickoff") or row.get("kickoff"))
        if not condition_id or not token or kickoff_ts is None:
            continue
        base["condition_id"] = condition_id
        base["token"] = token
        base["token_no"] = token_no
        base["slug"] = row.get("slug")
        if _blank(base.get("kickoff")) and not _blank(row.get("kickoff")):
            base["kickoff"] = str(row.get("kickoff"))
        tokens = {str(token)}
        if token_no:
            tokens.add(str(token_no))
        result = fetch_fn(condition_id, tokens, placed_ts, kickoff_ts)
        merged = merge_polled_trades(
            base,
            result.trades,
            hit_cap=result.hit_cap,
            fetch_error=result.fetch_error,
            now=moment,
        )
        save_tape(merged, directory)
        _copy_tape_fields(log, i, merged)
    return log


def stamp_signals(rows: list[dict], fetch_book, *, now: datetime | None = None, tape_dir: Path | None = None) -> list[dict]:
    """Na primeira vez que o sinal existe, fotografa o livro. Sinal falso não é ordem."""
    moment = now or datetime.now(timezone.utc)
    stamped: list[dict] = []
    for row in rows:
        if row.get("sinal") is not True:
            stamped.append(row)
            continue
        prior = load_tape(str(row.get("slug") or ""), tape_dir)
        if prior and prior.get("placed_at"):
            stamped.append(row)
            continue
        book = None
        token = row.get("token")
        if token:
            try:
                book = fetch_book(token)
            except Exception:
                book = None
        if not isinstance(book, dict):
            book = None
        updated, tape = capture_entry(row, book, moment)
        save_tape(tape, tape_dir)
        stamped.append(updated)
    return stamped


def partition_signals(rows: list[dict], log) -> tuple[list[dict], list[dict]]:
    """Separa slug novo de slug que estava só na faixa e agora virou sinal.

    Slug que já tem placed_at não é regravado: a fila continua a da primeira ordem.
    """
    known: dict[str, object] = {}
    if log is not None and len(getattr(log, "index", [])) and "slug" in log.columns:
        for _, existing in log.iterrows():
            known[str(existing.get("slug"))] = existing
    fresh: list[dict] = []
    upgrades: list[dict] = []
    for row in rows:
        slug = str(row.get("slug") or "")
        prev = known.get(slug)
        if prev is None:
            fresh.append(row)
            continue
        if row.get("sinal") is True and _blank(prev.get("placed_at")):
            upgrades.append(row)
    return fresh, upgrades


def apply_signal_upgrades(log, upgrades: list[dict]):
    """O primeiro registro era só a faixa. O sinal de verdade grava livro e placed_at."""
    if log is None or not upgrades:
        return log
    for col in _UPGRADE_FIELDS:
        if col not in log.columns:
            log[col] = None
    for row in upgrades:
        slug = str(row.get("slug") or "")
        hits = log.index[log["slug"].astype(str) == slug]
        if len(hits) == 0:
            continue
        i = hits[0]
        if not _blank(log.at[i, "placed_at"]):
            continue
        for col in _UPGRADE_FIELDS:
            if col in row:
                log.at[i, col] = row[col]
    return log


def fetch_live_book(token_id: str) -> dict:
    """Livro ao vivo, sem o cache de backtest. Cache velho mentiria a fila."""
    from src.poly_clob_client import fetch_order_book

    book = fetch_order_book(str(token_id), use_cache=False)
    return book if isinstance(book, dict) else {}
