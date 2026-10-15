"""Testes do monitor: mapeamento, gates, settle, CLV."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from unittest.mock import MagicMock, patch

import pandas as pd
import pytest

from src.clv import clv_probability
from src.monitor_gates import market_allowed
from src.team_matching import build_map, load_aliases, lookup


def test_lookup_fuzzy_wolverhampton():
    teams = ["Wolves", "Bolton"]
    idx = build_map(teams)
    assert lookup(idx, "Wolverhampton Wanderers FC") == "Wolves"


def test_gate_blocks_weak_league():
    gates = {"1X2": {"La Liga": {"alert": False, "log_loss_gain_vs_close": -0.04}}}
    assert not market_allowed(gates, "La Liga", "1X2")
    assert market_allowed(gates, "Serie A", "1X2")


def test_clv_positive_when_close_higher():
    assert clv_probability(0.35, 0.40) == pytest.approx(0.05)


def test_ensure_forward_schema_compat():
    from monitor import _ensure_forward_schema

    old = pd.DataFrame([{"slug": "x", "preco": 0.4, "pick": "casa"}])
    out = _ensure_forward_schema(old)
    assert "clv_poly" in out.columns
    assert "preco_exec" in out.columns


def test_settle_writes_clv_when_close_known():
    from monitor import settle

    log = pd.DataFrame([{
        "slug": "test-slug",
        "casa": "A",
        "fora": "B",
        "pick": "casa",
        "preco": 0.4,
        "preco_exec": 0.4,
        "inicio": "2026-10-12T19:00:00Z",
        "resultado": None,
        "liga": "Premier League",
    }])
    ev = [{
        "markets": [
            {"groupItemTitle": "A", "closed": True, "outcomePrices": '["1","0"]',
             "clobTokenIds": '["t1","t2"]'},
            {"groupItemTitle": "Draw", "closed": True, "outcomePrices": '["0","1"]'},
            {"groupItemTitle": "B", "closed": True, "outcomePrices": '["0","1"]'},
        ],
    }]
    with patch("monitor._get", return_value=ev):
        with patch("monitor.closing_price_before_kickoff", return_value=0.45):
            with patch("monitor.outcome_roles", return_value=["casa", "empate", "fora"]):
                out = settle(log, hist_df=None)
    assert out.loc[0, "clv_poly"] == pytest.approx(0.05)
    assert out.loc[0, "resultado"] == "acerto"
