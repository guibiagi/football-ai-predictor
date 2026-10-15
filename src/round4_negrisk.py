"""H4-NEG: soma dos YES em eventos multi-resultado ≠ 1."""

from __future__ import annotations

import json
from datetime import timezone

import pandas as pd

from src.hypothesis_engine import HOLDOUT_DAYS
from src.poly_hypothesis_registry import HypothesisRegistry
from src.poly_market_utils import mean_price_in_window, parse_kickoff, parse_token_yes, price_history
from src.polymarket_fees import infer_category, net_return_taker
from src.polymarket_gamma_client import load_merged_cached_events


def run_negrisk_tests(
    registry: HypothesisRegistry,
    events: list[dict] | None = None,
    cache_only: bool = True,
    holdout_cut: pd.Timestamp | None = None,
    sum_lo: float = 0.96,
    sum_hi: float = 1.04,
) -> None:
    events = events or load_merged_cached_events()
    hold_dev: dict[str, list[float]] = {"buy_all_yes": [], "buy_all_no": []}
    hold_hold: dict[str, list[float]] = {k: [] for k in hold_dev}

    for event in events:
        markets = event.get("markets") or []
        if len(markets) < 2 or len(markets) > 8:
            continue
        ko = parse_kickoff(event)
        if ko is None:
            continue
        if ko.tzinfo is None:
            ko = ko.replace(tzinfo=timezone.utc)
        end_ts = int(ko.timestamp())
        prices: list[tuple[float, bool]] = []
        for m in markets:
            tok = parse_token_yes(m)
            if not tok:
                continue
            hist = price_history(tok, end_ts - 7 * 86400, end_ts, cache_only=cache_only)
            p = mean_price_in_window(hist, end_ts, 24.0) if hist else None
            if p is None:
                try:
                    p = float(json.loads(m["outcomePrices"])[0])
                except Exception:
                    continue
            try:
                yw = float(json.loads(m["outcomePrices"])[0]) > 0.9
            except Exception:
                yw = False
            prices.append((p + 0.01, yw))

        if len(prices) < 2:
            continue
        s = sum(p for p, _ in prices)
        cat = infer_category(event)
        kt = pd.Timestamp(ko)
        if kt.tzinfo is not None:
            kt = kt.tz_convert(None)

        def push(key: str, nets: list[float]) -> None:
            if kt >= (holdout_cut or pd.Timestamp("2099-01-01")):
                hold_hold[key].extend(nets)
            else:
                hold_dev[key].extend(nets)

        if s < sum_lo:
            # Comprar YES em todos os resultados mutuamente exclusivos: custo s, liquida 1.
            stake = max(s, 0.05)
            ret = (1.0 - stake) / stake
            net = net_return_taker(ret, stake / len(prices), cat)
            push("buy_all_yes", [net])
        elif s > sum_hi:
            # Vender conjunto completo de YES: recebe s, paga 1 ao vencedor (+ slippage já em p+1¢).
            fee_buffer = 0.02 * len(prices)
            stake = max(s, 0.05)
            ret = (s - 1.0 - fee_buffer) / stake
            net = net_return_taker(ret, stake / len(prices), cat)
            push("buy_all_no", [net])

    registry.add(
        "H4-NEG-sum-low-buy-yes",
        "negrisk",
        f"Comprar todos YES quando soma < {sum_lo}",
        hold_hold["buy_all_yes"],
        hold_dev["buy_all_yes"],
        sum_lo=sum_lo,
        implementation_validated=False,
    )
    registry.add(
        "H4-NEG-sum-high",
        "negrisk",
        f"Arb quando soma > {sum_hi} (proxy basket; não executável sem neg-risk CLOB)",
        hold_hold["buy_all_no"],
        hold_dev["buy_all_no"],
        sum_hi=sum_hi,
        implementation_validated=False,
    )
