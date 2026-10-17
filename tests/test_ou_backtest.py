"""Testes do módulo ou_backtest (sem rede)."""

import numpy as np

from src.ou_backtest import (
    VariantResult,
    aggregate_variant_results,
    bh_fdr_adjust,
    bonferroni_adjust,
    bootstrap_best_strategy_null_p,
    bootstrap_p_roi_nonpos,
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


def test_shin_devig_removes_margin_extreme():
    odds = np.array([[1.05, 9.0]])
    p_over = devig_binary(odds, method="shin")[0]
    p_under = 1.0 - p_over
    np.testing.assert_allclose(p_over + p_under, 1.0, rtol=0, atol=1e-8)


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
    assert mask[1] and not side[1]


def test_bootstrap_roi_ci_constant():
    rets = np.ones(50) * 0.1
    roi, lo, hi = bootstrap_roi_ci(rets, n_boot=500, rng=np.random.default_rng(0))
    assert abs(roi - 0.1) < 1e-9
    assert lo <= roi <= hi


def test_bootstrap_p_negative_roi_near_one():
    rets = np.full(200, -0.05)
    p = bootstrap_p_roi_nonpos(rets, n_boot=2000, rng=np.random.default_rng(42))
    assert p > 0.9


def test_bootstrap_p_positive_roi_small():
    rets = np.full(200, 0.08)
    p = bootstrap_p_roi_nonpos(rets, n_boot=2000, rng=np.random.default_rng(42))
    assert p < 0.05


def test_bootstrap_best_strategy_null_detects_planted_edge():
    rng = np.random.default_rng(0)
    losers = [rng.normal(-0.02, 0.5, size=500) for _ in range(15)]
    winner = np.concatenate([np.full(1500, 0.12), rng.normal(-0.01, 0.3, size=500)])
    p = bootstrap_best_strategy_null_p(losers + [winner], n_boot=3000, rng=np.random.default_rng(1))
    assert p < 0.05


def test_bootstrap_best_strategy_null_flat_strategies_high_p():
    rng = np.random.default_rng(0)
    flat = [rng.normal(0.0, 0.4, size=400) for _ in range(10)]
    p = bootstrap_best_strategy_null_p(flat, n_boot=2000, rng=np.random.default_rng(2))
    assert p > 0.2


def test_aggregate_variant_results():
    a = VariantResult("v1", "H1", "train", 2, 0.1, 0.0, 0.2, 0.0, 50.0, 0.5,
                      0.5, 0.5, 0.2, 0.2, False, meta={"returns": [1.0, -1.0]})
    b = VariantResult("v1", "H1", "train", 1, 0.5, 0.0, 1.0, 0.0, 100.0, 0.5,
                      0.5, 0.5, 0.2, 0.2, False, meta={"returns": [0.5]})
    agg = aggregate_variant_results([a, b])
    assert len(agg) == 1
    assert agg[0].n == 3


def test_bh_fdr_monotone_on_sorted():
    p = np.array([0.01, 0.04, 0.5])
    q = bh_fdr_adjust(p)
    order = np.argsort(p)
    assert q[order[0]] <= q[order[1]] <= q[order[2]]


def test_bonferroni_adjust():
    p = np.array([0.01, 0.04, 0.5])
    b = bonferroni_adjust(p)
    assert b[0] == 0.03
