"""Estimativa de fill de uma compra limite parada no CLOB.

Três contas, todas simuladas. Nenhuma delas envia ordem:

- touch (otimista): fluxo taker que alcança o bid enche a ordem inteira.
- queue (central): esse fluxo só nos alcança depois da fila que já estava no
  preço. Um print estritamente abaixo do bid esvazia o nível, então o fill é cheio.
- through (pessimista): a ordem só enche se o preço negociar estritamente abaixo
  do bid.

O fluxo que consome o bid YES é venda taker do YES a preço <= p, mais compra
taker do NO a preço >= 1-p (mint; o tamanho entra 1:1 em shares do YES).
O bid simulado desce para a grade `tick_size` do livro. Sem tick, a fila não
é zero inventado: fica não medida.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

# Suposição de simulação, a mesma aposta típica já usada no projeto.
# Não é tamanho medido de uma ordem real. Shares = stake / preço simulado.
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


def tick_decimals(tick: float) -> int:
    if tick <= 0:
        return 4
    if tick >= 1:
        return 0
    return max(0, int(round(-math.log10(tick))))


def round_to_tick(price: float | None, tick: float | None) -> float | None:
    number = _num(price)
    step = _num(tick)
    if number is None:
        return None
    if step is None or step <= 0:
        return _price(number)
    snapped = round(round(number / step) * step, tick_decimals(step))
    return snapped


def snap_bid_to_tick(bid: float | None, tick: float | None) -> float | None:
    """Desce o bid para a grade. Preço fora da grade não existe no CLOB."""
    number = _num(bid)
    step = _num(tick)
    if number is None or step is None or step <= 0 or number <= 0:
        return None
    ticks = math.floor((number / step) + 1e-9)
    snapped = round(ticks * step, tick_decimals(step))
    if snapped <= 0 or snapped >= 1:
        return None
    return snapped


def read_tick_size(book: dict | None) -> float | None:
    if not isinstance(book, dict):
        return None
    tick = _num(book.get("tick_size"))
    if tick is None or tick <= 0:
        return None
    return tick


def trade_timestamp(value) -> int | None:
    """Unix em segundos. A data-api às vezes manda milissegundos."""
    number = _num(value)
    if number is None:
        return None
    ts = int(number)
    if ts >= 10**12:
        ts //= 1000
    return ts


def best_ask_price(book: dict | None) -> float | None:
    """Menor ask. A ordem do array do CLOB não é o melhor preço."""
    if not isinstance(book, dict):
        return None
    prices = []
    for level in book.get("asks") or []:
        if not isinstance(level, dict):
            continue
        price = _num(level.get("price"))
        if price is not None and price > 0:
            prices.append(price)
    if not prices:
        return None
    return min(prices)


def queue_ahead_from_book(
    book: dict | None,
    price: float,
    tick_size: float | None = None,
) -> float | None:
    """Tamanho já descansando no bid simulado.

    None se o livro ou o tick não foram observados. 0.0 é medida: o nível existe vazio.
    """
    if not isinstance(book, dict) or "bids" not in book:
        return None
    tick = _num(tick_size if tick_size is not None else book.get("tick_size"))
    if tick is None or tick <= 0:
        return None
    target = snap_bid_to_tick(price, tick)
    if target is None:
        return None
    total = 0.0
    for level in book.get("bids") or []:
        if not isinstance(level, dict):
            continue
        level_price = round_to_tick(_num(level.get("price")), tick)
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


def _asset(trade: dict) -> str:
    return str(trade.get("asset") or trade.get("asset_id") or "")


def _side(trade: dict) -> str:
    return str(trade.get("side") or "").strip().upper()


def _yes_price_and_size(
    trade: dict,
    *,
    yes_token: str | None,
    no_token: str | None,
    tick: float | None,
) -> tuple[float, float] | None:
    """Preço YES e tamanho, se o print taker consome o bid YES.

    Sem os dois tokens, só SELL conta (testes antigos de um único ativo).
    Com tokens, SELL do YES e BUY do NO no complemento.
    """
    side = _side(trade)
    if side not in {"BUY", "SELL"}:
        return None
    size = _num(trade.get("size"))
    raw = _num(trade.get("price"))
    if size is None or size <= 0 or raw is None:
        return None
    price = round_to_tick(raw, tick)
    if price is None:
        return None
    asset = _asset(trade)
    if not yes_token and not no_token:
        if side != "SELL":
            return None
        return price, size
    if yes_token and asset == str(yes_token) and side == "SELL":
        return price, size
    if no_token and asset == str(no_token) and side == "BUY":
        implied = round_to_tick(1.0 - price, tick)
        if implied is None:
            return None
        return implied, size
    return None


def _strictly_below(yes_price: float, sim: float, tick: float | None) -> bool:
    step = _num(tick)
    if step is not None and step > 0:
        return yes_price <= sim - step + step * 1e-6
    return yes_price < sim


def _consuming_prints(
    trades: list[dict],
    *,
    sim: float,
    placed_ts: int,
    kickoff_ts: int | None,
    yes_token: str | None,
    no_token: str | None,
    tick: float | None,
) -> list[tuple[float, float]]:
    """Cada registro taker entra na soma. Com takerOnly=true um mint é só o NO BUY."""
    grouped: dict[str, list[tuple[float, float]]] = {}
    order: list[str] = []
    for index, trade in enumerate(trades):
        if not isinstance(trade, dict):
            continue
        if not _in_window(trade_timestamp(trade.get("timestamp")), placed_ts, kickoff_ts):
            continue
        parsed = _yes_price_and_size(trade, yes_token=yes_token, no_token=no_token, tick=tick)
        if parsed is None:
            continue
        yes_price, size = parsed
        if yes_price > sim:
            continue
        tx = str(trade.get("transactionHash") or "")
        key = tx if tx else f"sem-tx-{index}"
        if key not in grouped:
            grouped[key] = []
            order.append(key)
        grouped[key].append((yes_price, size))
    prints: list[tuple[float, float]] = []
    for key in order:
        legs = grouped[key]
        yes_price = min(price for price, _ in legs)
        size = sum(size for _, size in legs)
        prints.append((yes_price, size))
    return prints


def _estimate(model: str, queue_ahead, our: float, filled, volume, measured: bool) -> FillEstimate:
    return FillEstimate(model, queue_ahead, our, filled, volume, measured)


def estimate_fills(
    *,
    price: float,
    our_size: float,
    queue_ahead: float | None,
    trades: list[dict],
    placed_ts: int,
    kickoff_ts: int | None = None,
    yes_token: str | None = None,
    no_token: str | None = None,
    tick_size: float | None = None,
    marketable: bool = False,
) -> dict[str, FillEstimate]:
    """Volume só com timestamp estritamente depois da entrada e antes do kickoff.

    `marketable` é o caso em que o ask do snapshot já estava <= preço simulado:
    a limite cruzaria na hora. Os três modelos enchem por inteiro.
    """
    size = _num(our_size)
    our = size if size is not None and size > 0 else 0.0
    tick = _num(tick_size)
    sim = snap_bid_to_tick(price, tick) if tick else _price(price)
    ahead = _num(queue_ahead)
    if sim is None:
        empty = _estimate("touch", ahead if ahead is not None and ahead >= 0 else None, our, None, None, False)
        return {"touch": empty, "queue": _estimate("queue", None, our, None, None, False), "through": empty}

    if marketable:
        full = our
        return {
            "touch": _estimate("touch", ahead, our, full, full, True),
            "queue": _estimate("queue", ahead if ahead is not None else None, our, full, full, True),
            "through": _estimate("through", ahead, our, full, full, True),
        }

    prints = _consuming_prints(
        trades,
        sim=sim,
        placed_ts=placed_ts,
        kickoff_ts=kickoff_ts,
        yes_token=yes_token,
        no_token=no_token,
        tick=tick,
    )
    vol_touch = sum(size for _, size in prints)
    vol_through = sum(size for yes_price, size in prints if _strictly_below(yes_price, sim, tick))
    touch_filled = our if vol_touch > 0 else 0.0
    through_filled = our if vol_through > 0 else 0.0
    if ahead is None or ahead < 0:
        queue = _estimate("queue", None, our, None, vol_touch, False)
    elif vol_through > 0:
        queue = _estimate("queue", ahead, our, our, vol_touch, True)
    else:
        filled = min(our, max(0.0, vol_touch - ahead))
        queue = _estimate("queue", ahead, our, filled, vol_touch, True)
    return {
        "touch": _estimate("touch", ahead if ahead is not None and ahead >= 0 else None, our, touch_filled, vol_touch, True),
        "queue": queue,
        "through": _estimate("through", ahead if ahead is not None and ahead >= 0 else None, our, through_filled, vol_through, True),
    }


def simulated_return(
    bid: float,
    won: bool,
    filled_size: float | None,
    our_size: float,
) -> float | None:
    """Retorno em unidades de stake, proporcional ao que a simulação encheu.

    None quando o fill não foi medido. 0.0 é simulação de 'não encheu', não um chute.
    O preço é o simulado na grade, quando ele existe.
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
