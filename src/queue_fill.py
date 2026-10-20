"""Estimativa de fill de uma compra limite parada no CLOB.

Três contas, todas simuladas. Nenhuma delas envia ordem:

- touch (otimista): qualquer venda a preço <= bid, depois da entrada, enche a ordem.
- queue (central): essa mesma venda só nos alcança depois de consumir o tamanho
  que já estava no preço no momento do sinal. Fill parcial é possível.
- through (pessimista): a ordem só enche se o preço negociar estritamente abaixo
  do bid. Um print no preço exato não conta.

Preços entram na grade de 4 casas do bid do paper (`round(fair - offset, 4)`).
Só entra venda agressiva (`side` SELL): compra no ask não consome a fila do bid.
Trade sem `side` não entra na conta — não chutamos o lado.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

# Suposição de simulação, a mesma aposta típica já usada no projeto.
# Não é tamanho medido de uma ordem real. Shares = stake / bid.
ASSUMED_STAKE_USD = 25.0


@dataclass(frozen=True)
class FillEstimate:
    model: str
    queue_ahead: float | None
    our_size: float
    filled_size: float | None
    eligible_volume: float | None
    measured: bool


def _num(value) -> float | None:
    if value is None:
        return None
    if isinstance(value, str) and not value.strip():
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    if math.isnan(number) or math.isinf(number):
        return None
    return number


def _price(value) -> float | None:
    number = _num(value)
    if number is None:
        return None
    return round(number, 4)


def trade_timestamp(value) -> int | None:
    """Unix em segundos. A data-api às vezes manda milissegundos."""
    number = _num(value)
    if number is None:
        return None
    ts = int(number)
    if ts >= 10**12:
        ts //= 1000
    return ts


def queue_ahead_from_book(book: dict | None, price: float) -> float | None:
    """Tamanho já descansando no bid `price`.

    None quando o livro não foi observado. 0.0 é medida: o nível existe vazio.
    """
    if not isinstance(book, dict) or "bids" not in book:
        return None
    target = _price(price)
    if target is None:
        return None
    total = 0.0
    for level in book.get("bids") or []:
        if not isinstance(level, dict):
            continue
        level_price = _price(level.get("price"))
        level_size = _num(level.get("size"))
        if level_price is None or level_size is None or level_size <= 0:
            continue
        if level_price == target:
            total += level_size
    return total


def assumed_shares(bid: float, stake_usd: float = ASSUMED_STAKE_USD) -> float:
    price = _num(bid)
    if price is None or price <= 0:
        return 0.0
    return stake_usd / price


def _in_window(ts: int | None, placed_ts: int, kickoff_ts: int | None) -> bool:
    if ts is None or ts <= placed_ts:
        return False
    if kickoff_ts is not None and ts >= kickoff_ts:
        return False
    return True


def _eligible_volume(
    trades: list[dict],
    *,
    price: float,
    placed_ts: int,
    kickoff_ts: int | None,
    strict: bool,
) -> float:
    target = _price(price)
    if target is None:
        return 0.0
    total = 0.0
    for trade in trades:
        if not isinstance(trade, dict):
            continue
        if str(trade.get("side") or "").strip().upper() != "SELL":
            continue
        if not _in_window(trade_timestamp(trade.get("timestamp")), placed_ts, kickoff_ts):
            continue
        trade_price = _price(trade.get("price"))
        trade_size = _num(trade.get("size"))
        if trade_price is None or trade_size is None or trade_size <= 0:
            continue
        if strict:
            if trade_price < target:
                total += trade_size
        elif trade_price <= target:
            total += trade_size
    return total


def estimate_fills(
    *,
    price: float,
    our_size: float,
    queue_ahead: float | None,
    trades: list[dict],
    placed_ts: int,
    kickoff_ts: int | None = None,
) -> dict[str, FillEstimate]:
    """Volume só com timestamp estritamente depois da entrada e antes do kickoff."""
    size = _num(our_size)
    our = size if size is not None and size > 0 else 0.0
    vol_at_or_below = _eligible_volume(
        trades, price=price, placed_ts=placed_ts, kickoff_ts=kickoff_ts, strict=False,
    )
    vol_through = _eligible_volume(
        trades, price=price, placed_ts=placed_ts, kickoff_ts=kickoff_ts, strict=True,
    )
    touch_filled = our if vol_at_or_below > 0 else 0.0
    through_filled = our if vol_through > 0 else 0.0
    ahead = _num(queue_ahead)
    if ahead is None or ahead < 0:
        queue = FillEstimate("queue", None, our, None, vol_at_or_below, False)
    else:
        filled = min(our, max(0.0, vol_at_or_below - ahead))
        queue = FillEstimate("queue", ahead, our, filled, vol_at_or_below, True)
    return {
        "touch": FillEstimate("touch", ahead, our, touch_filled, vol_at_or_below, True),
        "queue": queue,
        "through": FillEstimate("through", ahead, our, through_filled, vol_through, True),
    }


def simulated_return(
    bid: float,
    won: bool,
    filled_size: float | None,
    our_size: float,
) -> float | None:
    """Retorno em unidades de stake, proporcional ao que a simulação encheu.

    None quando o fill não foi medido. 0.0 é simulação de 'não encheu', não um chute.
    """
    if filled_size is None:
        return None
    filled = _num(filled_size)
    size = _num(our_size)
    price = _num(bid)
    if filled is None or size is None or price is None or size <= 0 or price <= 0:
        return None
    if filled <= 0:
        return 0.0
    unit = (1.0 - price) / price if won else -1.0
    return round(unit * (filled / size), 4)
