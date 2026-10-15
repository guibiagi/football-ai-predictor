"""Closing Line Value (CLV) em mercados de probabilidade implícita."""

from __future__ import annotations

import numpy as np


def clv_probability(p_entry: float, p_close: float) -> float:
    """CLV em pontos de probabilidade: positivo = fechamento mais alto que entrada."""
    if p_entry is None or p_close is None or np.isnan(p_entry) or np.isnan(p_close):
        return float("nan")
    return float(p_close - p_entry)


def clv_odds_ratio(odds_entry: float, odds_close: float) -> float:
    """CLV relativo em odds decimais: odds_close/odds_entry - 1."""
    if odds_entry is None or odds_close is None or odds_entry <= 1 or odds_close <= 1:
        return float("nan")
    return float(odds_close / odds_entry - 1.0)
