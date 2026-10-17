from __future__ import annotations

from src.macro_poly.stats import (
    benjamini_hochberg,
    bonferroni,
    bootstrap_mean_ci,
    trade_roi_binary,
)
from src.macro_poly.prices import price_at_or_after, price_at_or_before, spread_adjusted_ask
from src.macro_poly.liquidity import compute_terciles, taker_ask_with_slippage


def test_trade_roi_win():
    roi = trade_roi_binary(0.5, True, fee_rate=0.02)
    assert roi > 0
    assert trade_roi_binary(0.5, False) < 0


def test_bootstrap_ci():
    rois = [0.1, -0.05, 0.2, 0.0, 0.15]
    mean, lo, hi = bootstrap_mean_ci(rois, n_boot=500, seed=1)
    assert lo <= mean <= hi


def test_multiple_testing():
    p = [0.01, 0.04, 0.2]
    assert bonferroni(p)[0] <= 0.05
    assert benjamini_hochberg(p)[0] <= 0.05


def test_liquidity_slippage_increases_with_low_volume():
    hi = taker_ask_with_slippage(0.5, volume_gamma_usd=5_000_000, stake_usd=25)
    lo = taker_ask_with_slippage(0.5, volume_gamma_usd=2_000, stake_usd=25)
    assert lo > hi


def test_tercile_labels():
    t = compute_terciles([100, 500, 1000, 5000, 20000])
    assert t.label(50) == "T1_low"
    assert t.label(50000) == "T3_high"


def test_price_lookup():
    hist = [{"t": 100, "p": 0.4}, {"t": 200, "p": 0.6}]
    assert price_at_or_before(hist, 150) == 0.4
    assert price_at_or_after(hist, 150) == 0.6
    assert spread_adjusted_ask(0.5) == 0.505
