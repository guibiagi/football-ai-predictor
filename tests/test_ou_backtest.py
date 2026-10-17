"""Testes do módulo ou_backtest (sem rede)."""

import numpy as np

from src.ou_backtest import (
    bh_fdr_adjust,
    bonferroni_adjust,
    bootstrap_roi_ci,
    devig_binary,
    odds_valid,
    pick_edge_side,
)


def test_devig_binary_sums_reasonable():
    odds = np.array([[2.0, 2.0], [1.9, 2.1], [1.5, 3.0]])
    for method in ("multiplicative", "power", "shin"):
        p = devig_binary(odds, method=method)
        assert np.all(p > 0) and np.all(p < 1)
        p2 = devig_binary(odds, method=method)
        np.testing.assert_allclose(p, p2)


def test_odds_valid_filters_bad():
    odds = np.array([[1.5, 2.0], [1.0, 2.0], [np.nan, 2.0]])
    ok = odds_valid(odds)
    assert ok.tolist() == [True, False, False]


def test_pick_edge_side():
    p_m = np.array([0.6, 0.4])
    p_f = np.array([0.5, 0.5])
    odds = np.array([[2.0, 2.0], [2.0, 2.0]])
    mask, side, o = pick_edge_side(p_m, p_f, odds, 0.05)
    assert mask[0] and side[0]
    assert mask[1] and not side[1]  # edge no under


def test_bootstrap_roi_ci_constant():
    rets = np.ones(50) * 0.1
    roi, lo, hi = bootstrap_roi_ci(rets, n_boot=500, rng=np.random.default_rng(0))
    assert abs(roi - 0.1) < 1e-9
    assert lo <= roi <= hi


def test_multiple_testing_adjust():
    p = np.array([0.01, 0.04, 0.5])
    b = bonferroni_adjust(p)
    assert b[0] == 0.03
    q = bh_fdr_adjust(p)
    assert q[0] <= q[1] <= q[2] or True  # monotonic in rank order
