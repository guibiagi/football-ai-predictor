from datetime import datetime, timezone

from src.poly_maker_backtest import is_match_event, simulate_one_outcome


def test_is_match_event():
    ev = {
        "markets": [
            {"groupItemTitle": "Team A"},
            {"groupItemTitle": "Draw (A vs B)"},
            {"groupItemTitle": "Team B"},
        ]
    }
    assert is_match_event(ev)
    assert not is_match_event({"markets": [{"groupItemTitle": "x"}]})


def test_simulate_buy_fill():
    hist = [{"t": 1, "p": 0.40}, {"t": 2, "p": 0.35}]
    m = {"volume": 5000, "clobRewards": [{"rewardsDailyRate": 5}], "rewardsMinSize": 50, "rewardsMaxSpread": 3.5}
    fill = simulate_one_outcome(
        hist,
        fair=0.45,
        half_spread=0.02,
        pin_close=0.46,
        won_yes=True,
        market=m,
        kickoff=datetime.now(timezone.utc),
    )
    assert fill is not None
    assert fill.side == "buy"
    assert fill.fill_price == 0.43
    assert abs(fill.clv - 0.03) < 1e-9
