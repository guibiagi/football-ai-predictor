"""Forward test papel: neg-risk / soma YES (somente sinais)."""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from src.round5_negrisk import SUM_HI, SUM_LO, scan_live_negrisk

PAPER_LOG = Path(__file__).resolve().parent.parent / "data" / "paper_negrisk_forward.csv"

COLS = [
    "data", "slug", "direction", "n_legs", "sum_prices", "size_usdc",
    "net_profit_usdc", "annualized_roi", "neg_risk", "sinal",
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


def scan_live_signals(max_events: int = 35) -> list[dict]:
    today = str(datetime.now(timezone.utc).date())
    out: list[dict] = []
    for opp in scan_live_negrisk(max_events=max_events):
        out.append({
            "data": today,
            "slug": opp["slug"],
            "direction": opp["direction"],
            "n_legs": opp["n_legs"],
            "sum_prices": opp.get("sum_asks") or opp.get("sum_bids"),
            "size_usdc": opp["size_usdc"],
            "net_profit_usdc": opp["net_profit_usdc"],
            "annualized_roi": opp["annualized_roi"],
            "neg_risk": opp.get("neg_risk", True),
            "sinal": True,
        })
    return out
