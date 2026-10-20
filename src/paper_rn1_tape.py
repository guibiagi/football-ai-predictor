"""Livro na hora do sinal e trades acumulados até o kickoff.

A data-api pública (`/trades?market=&start=&end=`) é a mesma já usada no repo.
Ela não garante o histórico até o instante do sinal: página estourada ou erro
deixam a fita `incomplete`. Nesse caso o settle não inventa fill. A rodada
seguinte do monitor acumula o que a API devolver.

Sem chave, sem ordem. Só leitura.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from src.queue_fill import assumed_shares, queue_ahead_from_book, trade_timestamp

DATA_API = "https://data-api.polymarket.com/trades"
# Teto observado da data-api: offset alto sem janela volta vazio, e a janela
# start/end também corta por volta de 10 mil linhas. Acima disso a fita fica incompleta.
PAGE_LIMIT = 100
MAX_OFFSET = 10000
TAPE_DIR = Path(__file__).resolve().parent.parent / "data" / "paper_rn1_tapes"


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


def capture_entry(row: dict, book: dict | None, now: datetime) -> tuple[dict, dict]:
    """Grava a fila no preço do bid a partir do livro deste instante. Sem rede."""
    bid = float(row["bid"])
    queue_ahead = queue_ahead_from_book(book, bid)
    observed = queue_ahead is not None
    shares = assumed_shares(bid)
    placed = _iso(now)
    updated = dict(row)
    updated["placed_at"] = placed
    updated["queue_ahead"] = queue_ahead
    updated["our_size"] = shares
    updated["fill_coverage"] = "pending"
    tape = {
        "slug": row.get("slug"),
        "token": row.get("token"),
        "condition_id": row.get("condition_id"),
        "bid": bid,
        "placed_at": placed,
        "kickoff": row.get("kickoff"),
        "queue_ahead": queue_ahead,
        "our_size": shares,
        "book_observed": observed,
        "trades": [],
        "coverage": "pending",
        "fetch_error": False,
        "hit_cap": False,
    }
    return updated, tape


def _trade_key(trade: dict) -> tuple:
    return (
        str(trade.get("transactionHash") or ""),
        str(trade.get("asset") or trade.get("asset_id") or ""),
        str(trade.get("side") or ""),
        str(trade.get("price") or ""),
        str(trade.get("size") or ""),
        str(trade_timestamp(trade.get("timestamp"))),
    )


def merge_polled_trades(
    tape: dict,
    fetched: list[dict],
    *,
    hit_cap: bool,
    fetch_error: bool,
) -> dict:
    """Junta trades novos. Só fica o que aconteceu depois da entrada e antes do kickoff."""
    placed_ts = iso_to_ts(tape.get("placed_at"))
    kickoff_ts = iso_to_ts(tape.get("kickoff"))
    kept: list[dict] = []
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
        kept.append(trade)
    merged = dict(tape)
    merged["trades"] = kept
    merged["hit_cap"] = bool(hit_cap)
    merged["fetch_error"] = bool(fetch_error)
    if fetch_error or hit_cap:
        merged["coverage"] = "incomplete"
    else:
        merged["coverage"] = "complete"
    return merged


def fetch_trades_window(
    condition_id: str,
    token_id: str,
    start_ts: int,
    end_ts: int,
    *,
    urlopen=None,
) -> TradeFetch:
    """Trades do token na janela [start, end]. Falha de rede vira fetch_error, não lista vazia 'medida'."""
    import urllib.error
    import urllib.request

    opener = urlopen or urllib.request.urlopen
    kept: list[dict] = []
    offset = 0
    try:
        while True:
            if offset > MAX_OFFSET:
                return TradeFetch(trades=kept, hit_cap=True, fetch_error=False)
            url = (
                f"{DATA_API}?market={condition_id}&limit={PAGE_LIMIT}&offset={offset}"
                f"&start={int(start_ts)}&end={int(end_ts)}&takerOnly=false"
            )
            req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
            with opener(req, timeout=60) as resp:
                batch = json.load(resp)
            if not isinstance(batch, list):
                return TradeFetch(trades=kept, hit_cap=False, fetch_error=True)
            if not batch:
                break
            for trade in batch:
                if not isinstance(trade, dict):
                    continue
                asset = str(trade.get("asset") or trade.get("asset_id") or "")
                if asset == str(token_id):
                    kept.append(trade)
            if len(batch) < PAGE_LIMIT:
                break
            offset += PAGE_LIMIT
    except (urllib.error.URLError, TimeoutError, json.JSONDecodeError, OSError, ValueError):
        return TradeFetch(trades=[], hit_cap=False, fetch_error=True)
    hit_cap = offset > MAX_OFFSET
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
    try:
        our = float(our_raw) if not _blank(our_raw) else assumed_shares(bid_f)
    except (TypeError, ValueError):
        our = assumed_shares(bid_f)
    return {
        "slug": row.get("slug"),
        "token": None if _blank(row.get("token")) else str(row.get("token")),
        "condition_id": None if _blank(row.get("condition_id")) else str(row.get("condition_id")),
        "bid": bid_f,
        "placed_at": str(placed),
        "kickoff": None if _blank(row.get("kickoff")) else str(row.get("kickoff")),
        "queue_ahead": float(queue_raw) if observed else None,
        "our_size": our,
        "book_observed": observed,
        "trades": [],
        "coverage": "pending",
        "fetch_error": False,
        "hit_cap": False,
    }


def poll_tapes(log, *, fetch=None, tape_dir: Path | None = None, now: datetime | None = None):
    """Acumula trades das linhas que já têm placed_at. Não chama a rede se `fetch` for injetado."""
    directory = tape_dir or TAPE_DIR
    fetch_fn = fetch or (
        lambda condition_id, token, start_ts, end_ts: fetch_trades_window(
            condition_id, token, start_ts, end_ts
        )
    )
    for col in ("fill_coverage", "queue_ahead", "our_size"):
        if col not in log.columns:
            log[col] = None
    log["fill_coverage"] = log["fill_coverage"].astype(object)
    for i, row in log.iterrows():
        base = load_tape(str(row.get("slug") or ""), directory) or _tape_from_row(row)
        if base is None:
            continue
        condition_id = base.get("condition_id") or (
            None if _blank(row.get("condition_id")) else str(row.get("condition_id"))
        )
        token = base.get("token") or (None if _blank(row.get("token")) else str(row.get("token")))
        placed_ts = iso_to_ts(base.get("placed_at"))
        kickoff_ts = iso_to_ts(base.get("kickoff") or row.get("kickoff"))
        if not condition_id or not token or placed_ts is None or kickoff_ts is None:
            continue
        if now is not None and placed_ts > int(now.timestamp()):
            continue
        base["condition_id"] = condition_id
        base["token"] = token
        base["slug"] = row.get("slug")
        result = fetch_fn(condition_id, token, placed_ts, kickoff_ts)
        merged = merge_polled_trades(
            base,
            result.trades,
            hit_cap=result.hit_cap,
            fetch_error=result.fetch_error,
        )
        save_tape(merged, directory)
        log.at[i, "fill_coverage"] = merged["coverage"]
        if merged.get("book_observed") and merged.get("queue_ahead") is not None:
            log.at[i, "queue_ahead"] = merged["queue_ahead"]
        if merged.get("our_size") is not None:
            log.at[i, "our_size"] = merged["our_size"]
    return log


def stamp_signals(rows: list[dict], fetch_book, *, now: datetime | None = None, tape_dir: Path | None = None) -> list[dict]:
    """Na primeira vez que o sinal existe, fotografa o livro. Sinal falso não é ordem."""
    moment = now or datetime.now(timezone.utc)
    stamped: list[dict] = []
    for row in rows:
        if row.get("sinal") is not True:
            stamped.append(row)
            continue
        # A segunda rodada do monitor não pode substituir o livro da entrada.
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


def fetch_live_book(token_id: str) -> dict:
    """Livro ao vivo, sem o cache de backtest. Cache velho mentiria a fila."""
    from src.poly_clob_client import fetch_order_book

    book = fetch_order_book(str(token_id), use_cache=False)
    return book if isinstance(book, dict) else {}
