from __future__ import annotations

from src.macro_poly.backtest import deterministic_holdout_events
from src.macro_poly.discovery import MacroMarket, is_fomc_rate_event
from src.macro_poly.fomc_calendar import fomc_statement_ts
from src.macro_poly.liquidity import compute_terciles, taker_ask_with_slippage
from src.macro_poly.prices import price_at_or_after, price_at_or_before, spread_adjusted_ask
from src.macro_poly.stats import (
    benjamini_hochberg,
    bonferroni,
    bootstrap_mean_ci,
    trade_roi_binary,
)


def _m(eid: str, rts: int) -> MacroMarket:
    return MacroMarket(
        event_id=eid,
        event_title="t",
        event_slug="s",
        market_id="1",
        question="q",
        market_slug="ms",
        token_yes="tok",
        token_no=None,
        condition_id="0xabc",
        end_date="2025-01-01T00:00:00Z",
        start_ts=rts - 86400,
        volume=1000.0,
        category="cpi",
        release_ts=rts,
        threshold_op=None,
        threshold_pct=None,
        outcome_yes=1,
    )


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


def test_no_099_entry_cap():
    p = taker_ask_with_slippage(0.995, volume_gamma_usd=1_000_000, stake_usd=25)
    assert p > 0.995


def test_liquidity_slippage_increases_with_low_volume():
    hi = taker_ask_with_slippage(0.5, volume_gamma_usd=5_000_000, stake_usd=25)
    lo = taker_ask_with_slippage(0.5, volume_gamma_usd=2_000, stake_usd=25)
    assert lo > hi


def test_tercile_labels():
    t = compute_terciles([100, 500, 1000, 5000, 20000])
    assert t.label(50) == "T1_low"
    assert t.label(50000) == "T3_high"


def test_price_at_or_after_no_stale_fallback():
    hist = [{"t": 100, "p": 0.4}, {"t": 200, "p": 0.6}]
    assert price_at_or_after(hist, 250, max_lag_sec=60) is None
    assert price_at_or_after(hist, 200, max_lag_sec=60) == 0.6


def test_holdout_tiebreak_deterministic():
    markets = [_m("b", 100), _m("a", 100), _m("c", 200)]
    h1 = deterministic_holdout_events(markets, frac=0.34)
    h2 = deterministic_holdout_events(markets, frac=0.34)
    assert h1 == h2


def test_fomc_march_2024_statement_day():
    ts = fomc_statement_ts(2024, 3)
    from datetime import datetime
    from zoneinfo import ZoneInfo

    dt = datetime.fromtimestamp(ts, tz=ZoneInfo("America/New_York"))
    assert dt.day == 20
    assert dt.hour == 14


def test_fomc_excludes_mention_markets():
    assert not is_fomc_rate_event(
        "Will anyone say Bank during the FED board meeting?",
        "Will anyone say Bank 50+ times?",
    )
    assert is_fomc_rate_event(
        "Fed decision in March?",
        "Fed decreases interest rates by 25 bps after March 2024 meeting?",
    )


def test_h4_fade_direction_via_runner(tmp_path):
    from src.macro_poly.backtest import run_intraday_moves
    from src.macro_poly.trade_volume import TradeVolumeStore

    m = _m("e1", 1_000_000)
    m = MacroMarket(
        **{**m.__dict__, "outcome_yes": 1, "release_ts": 1_000_000, "start_ts": 900_000}
    )
    cache = tmp_path / "prices"
    cache.mkdir()
    token = m.token_yes
    w_start, w_end = 900_000, 1_000_000 + 3 * 86400
    hist = [
        {"t": 820_000, "p": 0.55},
        {"t": 840_000, "p": 0.55},
        {"t": 860_000, "p": 0.55},
        {"t": 880_000, "p": 0.55},
        {"t": 900_000, "p": 0.5},
    ]
    import json

    path = cache / f"{token}_{w_start}_{w_end}_60.json"
    path.write_text(json.dumps({"fetch_failed": False, "history": hist}))
    store = TradeVolumeStore(tmp_path / "trades")
    store.register(token, m.condition_id, None, 10_000.0)
    (tmp_path / "trades").mkdir()
    (tmp_path / "trades" / f"{m.condition_id}_v4.json").write_text(
        json.dumps(
            {
                "failed": False,
                "trades": [
                    {
                        "asset": token,
                        "timestamp": 800_000,
                        "size": 100,
                        "price": 0.5,
                        "transactionHash": "0xabc",
                    }
                ],
                "meta": {
                    "pages_fetched": 1,
                    "trades_kept": 1,
                    "trade_volume_total_usd": 50.0,
                    "gamma_volume_usd": 10000.0,
                    "gamma_ratio": 0.005,
                    "gamma_ratio_below_min": False,
                    "oldest_trade_ts": 800_000,
                    "hit_page_cap": False,
                    "fetch_error": False,
                    "complete": True,
                    "pagination_mode": "start_end_offset",
                },
            }
        )
    )
    out = run_intraday_moves([m], set(), cache, "H4a", [0.03], None, store)
    tr = out[0][1][0]
    assert tr.event_ts == 900_000
    assert tr.won is True  # fade down move -> buy YES


def test_price_lookup():
    hist = [{"t": 100, "p": 0.4}, {"t": 200, "p": 0.6}]
    assert price_at_or_before(hist, 150) == 0.4
    assert spread_adjusted_ask(0.5) == 0.505
