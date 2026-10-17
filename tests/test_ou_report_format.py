"""Testes do gerador de relatório O/U."""

from __future__ import annotations

import importlib.util
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]


def _load_runner():
    spec = importlib.util.spec_from_file_location(
        "run_ou", ROOT / "scripts" / "run_backtest_over_under.py",
    )
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_format_report_lists_per_league_survivors():
    from src.ou_backtest import VariantResult

    mod = _load_runner()
    survivor = VariantResult(
        variant_id="H1a_edge5_multiplicative@Premier League",
        hypothesis="H1_per_league",
        split="holdout",
        n=89,
        roi=0.1889,
        roi_ci_lo=-0.1,
        roi_ci_hi=0.4,
        clv_mean=-0.01,
        clv_pos_pct=25.0,
        p_raw=0.036,
        log_loss_model=0.7,
        log_loss_pin=0.68,
        brier_model=0.25,
        brier_pin=0.24,
        survives=True,
        meta={"p_bonf": 1.0, "q_bh": 1.0},
    )
    global_ll = {
        "ll_model": 0.6869,
        "ll_pin": 0.6718,
        "br_model": 0.2468,
        "br_pin": 0.2396,
        "n": 12717,
    }
    text = mod.format_report(
        [survivor],
        global_ll,
        pd.DataFrame(),
        0.8276,
        16,
        12717,
        "holdout 2025-07-25 a 2026-01-14",
        24,
        1,
    )
    assert "**Por liga (família de testes):**" in text
    assert "Premier League" in text
    assert "Por liga (família de testes)" in text.split("## Veredito")[1]
