"""Forward test papel: taker quando Poly < Pin fair − gap (rodada 4)."""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from src.odds_utils import market_probs
from src.polymarket_prices import execution_price_yes, parse_token_ids

PAPER_LOG = Path(__file__).resolve().parent.parent / "data" / "paper_taker_pin_forward.csv"
GAP = 0.02
SPREAD = 0.01
MIN_VOLUME = 1000.0

COLS = [
    "data", "liga", "slug", "casa", "fora", "pick", "pin_fair", "poly_exec",
    "gap", "volume", "kickoff", "sinal", "resultado", "retorno_sim",
]


def ensure_log(path: Path = PAPER_LOG) -> pd.DataFrame:
    if path.exists():
        df = pd.read_csv(path)
    else:
        df = pd.DataFrame(columns=COLS)
    for c in COLS:
        if c not in df.columns:
            df[c] = None
    return df


def scan_pin_gap_fixture(
    comp: str,
    slug: str,
    home: str,
    away: str,
    markets: list[dict],
    roles: list[str],
    dataset_row: pd.Series | None,
    today: str,
) -> dict | None:
    if dataset_row is None:
        return None
    if dataset_row[["psc_h", "psc_d", "psc_a"]].isna().any():
        return None
    pin = market_probs(dataset_row[["psc_h", "psc_d", "psc_a"]].to_numpy(dtype=float).reshape(1, -1))[0]
    best = None
    for i, lab in enumerate(roles):
        exec_p = execution_price_yes(markets[i]) or markets[i].get("price")
        if exec_p is None:
            continue
        vol = float(markets[i].get("volume") or 0)
        if vol < MIN_VOLUME:
            continue
        edge = pin[i] - (float(exec_p) + SPREAD)
        if edge >= GAP:
            cand = {
                "data": today,
                "liga": comp,
                "slug": slug,
                "casa": home,
                "fora": away,
                "pick": lab,
                "pin_fair": round(pin[i], 4),
                "poly_exec": round(float(exec_p), 4),
                "gap": round(edge, 4),
                "volume": round(vol),
                "kickoff": markets[i].get("endDate") or "",
                "sinal": True,
                "resultado": None,
                "retorno_sim": None,
            }
            if best is None or cand["gap"] > best["gap"]:
                best = cand
    return best
