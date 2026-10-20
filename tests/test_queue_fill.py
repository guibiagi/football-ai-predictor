"""Fill simulado com fila para o paper RN1.

Três modelos, todos simulados (nenhuma ordem vai ao livro):
- touch: otimista, qualquer venda no preço do bid ou melhor enche a ordem;
- queue: central, só o volume que passa da fila que já estava na frente;
- through: pessimista, só conta fill se o preço negocia estritamente através do bid.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from unittest.mock import patch

import pandas as pd
import pytest

from src.queue_fill import (
    ASSUMED_STAKE_USD,
    estimate_fills,
    queue_ahead_from_book,
    simulated_return,
)
from src.paper_rn1 import (
    ensure_log,
    format_fill_model_report,
    scan_fixture,
    settle_paper,
    summarize_fill_models,
)
from src.paper_rn1_tape import (
    capture_entry,
    fetch_trades_window,
    merge_polled_trades,
    poll_tapes,
    stamp_signals,
)
from tests.test_paper_rn1 import BID, KICKOFF, _Resp, _base_row, _event, _float_nan_log

PLACED = "2026-10-18T10:00:00Z"
PLACED_TS = int(datetime(2026, 10, 18, 10, 0, tzinfo=timezone.utc).timestamp())
KICKOFF_TS = int(datetime(2026, 10, 18, 13, 0, tzinfo=timezone.utc).timestamp())
OUR = 50.0


def _sell(ts: int, price: float, size: float, **extra) -> dict:
    trade = {
        "timestamp": ts,
        "price": price,
        "size": size,
        "side": "SELL",
        "asset": "tok-home",
        "transactionHash": f"0x{ts}-{price}-{size}",
    }
    trade.update(extra)
    return trade


def _fills(**kwargs):
    params = dict(
        price=BID,
        our_size=OUR,
        queue_ahead=100.0,
        trades=[],
        placed_ts=PLACED_TS,
        kickoff_ts=KICKOFF_TS,
    )
    params.update(kwargs)
    return estimate_fills(**params)


def test_no_trades_after_placement_is_no_fill_on_all_models():
    out = _fills(trades=[])
    for name in ("touch", "queue", "through"):
        assert out[name].measured is True
        assert out[name].filled_size == 0.0


def test_trades_at_or_before_placement_do_not_count():
    trades = [
        _sell(PLACED_TS - 60, BID, 10_000),
        _sell(PLACED_TS, BID - 0.01, 10_000),
    ]
    out = _fills(trades=trades)
    for name in ("touch", "queue", "through"):
        assert out[name].filled_size == 0.0
        assert out[name].eligible_volume == 0.0


def test_trades_at_or_after_kickoff_do_not_count():
    trades = [
        _sell(KICKOFF_TS, BID - 0.05, 10_000),
        _sell(KICKOFF_TS + 30, BID, 10_000),
    ]
    out = _fills(trades=trades)
    for name in ("touch", "queue", "through"):
        assert out[name].filled_size == 0.0


def test_queue_does_not_fill_when_volume_only_matches_queue_ahead():
    out = _fills(trades=[_sell(PLACED_TS + 10, BID, 100)])
    assert out["queue"].filled_size == 0.0
    assert out["touch"].filled_size == OUR
    assert out["through"].filled_size == 0.0


def test_queue_partial_fill_when_volume_exceeds_queue_but_not_our_size():
    out = _fills(trades=[_sell(PLACED_TS + 10, BID, 80), _sell(PLACED_TS + 20, BID, 40)])
    assert out["queue"].eligible_volume == pytest.approx(120)
    assert out["queue"].filled_size == pytest.approx(20)
    assert out["touch"].filled_size == OUR
    assert out["through"].filled_size == 0.0


def test_queue_full_fill_when_volume_covers_queue_and_our_size():
    out = _fills(trades=[_sell(PLACED_TS + 10, BID, 150)])
    assert out["queue"].filled_size == OUR


def test_touch_ignores_queue_when_price_only_touches_bid():
    out = _fills(queue_ahead=10_000, trades=[_sell(PLACED_TS + 5, BID, 1)])
    assert out["touch"].filled_size == OUR
    assert out["queue"].filled_size == 0.0
    assert out["through"].filled_size == 0.0


def test_through_ignores_prints_at_the_bid_and_fills_when_price_trades_through():
    at_bid = _fills(trades=[_sell(PLACED_TS + 5, BID, 10_000)])
    assert at_bid["through"].filled_size == 0.0
    assert at_bid["touch"].filled_size == OUR

    through = _fills(trades=[_sell(PLACED_TS + 5, BID - 0.01, 5)])
    assert through["through"].filled_size == OUR
    assert through["through"].eligible_volume == pytest.approx(5)
    # O print abaixo do bid também entra no volume <= p, mas 5 não cobre a fila.
    assert through["queue"].filled_size == 0.0
    assert through["touch"].filled_size == OUR


def test_sells_above_the_bid_and_buy_side_do_not_consume_the_queue():
    trades = [
        _sell(PLACED_TS + 5, BID + 0.01, 10_000),
        _sell(PLACED_TS + 6, BID, 10_000, side="BUY"),
        _sell(PLACED_TS + 7, BID - 0.02, 10_000, side="buy"),
    ]
    out = _fills(trades=trades)
    for name in ("touch", "queue", "through"):
        assert out[name].filled_size == 0.0


def test_missing_side_is_not_counted():
    trade = _sell(PLACED_TS + 5, BID - 0.01, 10_000)
    del trade["side"]
    out = _fills(trades=[trade])
    assert out["touch"].filled_size == 0.0
    assert out["through"].filled_size == 0.0


def test_millisecond_timestamps_are_compared_in_seconds():
    inside = _sell((PLACED_TS + 15) * 1000, BID, 1)
    exact_ms = _sell(PLACED_TS * 1000, BID, 10_000)
    out = _fills(queue_ahead=0, trades=[exact_ms, inside])
    assert out["touch"].eligible_volume == pytest.approx(1)
    assert out["queue"].filled_size == pytest.approx(1)


def test_queue_ahead_sums_only_the_bid_level_at_our_price():
    book = {
        "bids": [
            {"price": "0.63", "size": "999"},
            {"price": "0.625", "size": "40"},
            {"price": "0.6250", "size": "15"},
            {"price": "0.62", "size": "100"},
        ],
        "asks": [{"price": "0.625", "size": "500"}],
    }
    assert queue_ahead_from_book(book, BID) == pytest.approx(55)
    assert queue_ahead_from_book({"bids": []}, BID) == 0.0
    assert queue_ahead_from_book({}, BID) is None
    assert queue_ahead_from_book(None, BID) is None


def test_missing_queue_is_unmeasured_not_a_zero_queue():
    out = _fills(queue_ahead=None, trades=[_sell(PLACED_TS + 5, BID, 10)])
    assert out["queue"].measured is False
    assert out["queue"].filled_size is None
    assert out["touch"].measured is True
    assert out["touch"].filled_size == OUR
    assert out["through"].filled_size == 0.0


def test_assumed_stake_is_labelled_and_not_a_measured_size():
    assert ASSUMED_STAKE_USD == 25.0
    ret = simulated_return(BID, won=True, filled_size=25, our_size=50)
    assert ret == pytest.approx(round(((1.0 - BID) / BID) * 0.5, 4))
    assert simulated_return(BID, won=False, filled_size=0, our_size=50) == 0.0
    assert simulated_return(BID, won=True, filled_size=None, our_size=50) is None


def test_capture_entry_stores_queue_from_the_snapshot_only():
    row = {
        "slug": "epl-bou-sun-2026-10-18",
        "token": "tok-home",
        "bid": BID,
        "condition_id": "0xabc",
        "kickoff": KICKOFF,
        "sinal": True,
    }
    book = {
        "bids": [{"price": str(BID), "size": "80"}, {"price": "0.50", "size": "10"}],
        "asks": [{"price": "0.70", "size": "5"}],
    }
    now = datetime(2026, 10, 18, 10, 0, tzinfo=timezone.utc)
    updated, tape = capture_entry(row, book, now)
    assert updated["placed_at"] == PLACED
    assert updated["queue_ahead"] == pytest.approx(80)
    assert updated["our_size"] == pytest.approx(ASSUMED_STAKE_USD / BID)
    assert tape["book_observed"] is True
    assert tape["coverage"] == "pending"
    assert tape["trades"] == []

    blind, blind_tape = capture_entry(row, None, now)
    assert blind["queue_ahead"] is None
    assert blind_tape["book_observed"] is False


def test_stamp_keeps_the_first_book_snapshot(tmp_path):
    row = {
        "slug": "epl-first-book",
        "token": "tok-home",
        "bid": BID,
        "condition_id": "0xabc",
        "kickoff": KICKOFF,
        "sinal": True,
    }
    first = {"bids": [{"price": str(BID), "size": "10"}]}
    later = {"bids": [{"price": str(BID), "size": "999"}]}
    stamp_signals(
        [row], lambda token: first, tape_dir=tmp_path,
        now=datetime(2026, 10, 18, 10, 0, tzinfo=timezone.utc),
    )
    stamp_signals(
        [row], lambda token: later, tape_dir=tmp_path,
        now=datetime(2026, 10, 18, 12, 0, tzinfo=timezone.utc),
    )
    saved = json.loads((tmp_path / "epl-first-book.json").read_text(encoding="utf-8"))
    assert saved["queue_ahead"] == pytest.approx(10)
    assert saved["placed_at"] == PLACED


def test_merge_keeps_only_trades_inside_the_window_and_dedupes():
    _, tape = capture_entry(
        {
            "slug": "s",
            "token": "tok-home",
            "bid": BID,
            "condition_id": "0xabc",
            "kickoff": KICKOFF,
        },
        {"bids": [{"price": str(BID), "size": "1"}]},
        datetime(2026, 10, 18, 10, 0, tzinfo=timezone.utc),
    )
    inside = _sell(PLACED_TS + 30, BID, 4)
    fetched = [
        _sell(PLACED_TS - 5, BID, 99),
        inside,
        dict(inside),
        _sell(KICKOFF_TS, BID, 99),
    ]
    merged = merge_polled_trades(tape, fetched, hit_cap=False, fetch_error=False)
    assert len(merged["trades"]) == 1
    assert merged["trades"][0]["size"] == 4
    assert merged["coverage"] == "complete"

    again = merge_polled_trades(merged, [inside], hit_cap=False, fetch_error=False)
    assert len(again["trades"]) == 1

    capped = merge_polled_trades(tape, [inside], hit_cap=True, fetch_error=False)
    assert capped["coverage"] == "incomplete"
    assert capped["trades"][0]["size"] == 4


class _UrlResp:
    def __init__(self, payload):
        self._raw = json.dumps(payload).encode()

    def read(self):
        return self._raw

    def __enter__(self):
        return self

    def __exit__(self, *args):
        return False


def test_fetch_trades_window_filters_token_and_flags_errors():
    rows = [
        _sell(PLACED_TS + 10, BID, 3, asset="tok-home"),
        _sell(PLACED_TS + 10, BID, 9, asset="tok-other"),
    ]

    def urlopen(req, timeout=60):
        assert "start=" in req.full_url
        assert "end=" in req.full_url
        assert "takerOnly=false" in req.full_url
        return _UrlResp(rows)

    got = fetch_trades_window(
        "0xabc",
        "tok-home",
        PLACED_TS,
        KICKOFF_TS,
        urlopen=urlopen,
    )
    assert got.fetch_error is False
    assert got.hit_cap is False
    assert len(got.trades) == 1
    assert got.trades[0]["asset"] == "tok-home"

    def broken(req, timeout=60):
        raise TimeoutError("slow")

    failed = fetch_trades_window("0xabc", "tok-home", PLACED_TS, KICKOFF_TS, urlopen=broken)
    assert failed.fetch_error is True
    assert failed.trades == []


def _tape_for_settle(*, coverage: str, trades: list[dict], queue_ahead: float | None = 100.0) -> dict:
    return {
        "coverage": coverage,
        "book_observed": queue_ahead is not None,
        "queue_ahead": queue_ahead,
        "our_size": OUR,
        "trades": trades,
        "placed_at": PLACED,
        "bid": BID,
        "kickoff": KICKOFF,
    }


def test_settle_keeps_legacy_touch_and_writes_three_models_from_a_complete_tape():
    log = _float_nan_log(placed_at=PLACED, sinal=True)
    trades = [
        _sell(PLACED_TS - 60, BID, 10_000),
        _sell(PLACED_TS + 15, BID, 120),
        _sell(KICKOFF_TS + 15, BID - 0.05, 10_000),
    ]
    tapes = {_base_row()["slug"]: _tape_for_settle(coverage="complete", trades=trades)}
    with patch("urllib.request.urlopen", return_value=_Resp([_event(["1", "0", "0"])])):
        with patch("src.paper_rn1.closing_price_before_kickoff", return_value=0.70):
            with patch("src.pinnacle_lookup.lookup_1x2_close", return_value=(1.6, 4.0, 6.0)):
                out = settle_paper(log, pd.DataFrame(), tapes=tapes)

    row = out.loc[0]
    assert row["fill_simulado"] == False
    assert row["resultado"] == "sem_fill"
    assert pd.isna(row["retorno_sim"])
    assert row["queue_ahead"] == pytest.approx(100)
    assert row["filled_size"] == pytest.approx(20)
    assert row["fill_model"] == "queue"
    assert row["filled_touch"] == pytest.approx(OUR)
    assert row["filled_queue"] == pytest.approx(20)
    assert row["filled_through"] == pytest.approx(0)
    unit = (1.0 - BID) / BID
    assert row["retorno_touch"] == pytest.approx(round(unit, 4))
    assert row["retorno_queue"] == pytest.approx(round(unit * 20 / OUR, 4))
    assert row["retorno_through"] == pytest.approx(0.0)
    assert row["fill_coverage"] == "complete"


def test_settled_sem_fill_picks_up_a_complete_tape_later():
    """A regra antiga já gravou sem_fill. A fita que chega depois ainda estima a fila."""
    log = pd.DataFrame([_base_row(
        placed_at=PLACED,
        sinal=True,
        resultado="sem_fill",
        fill_simulado=False,
    )])
    tapes = {
        _base_row()["slug"]: _tape_for_settle(
            coverage="complete",
            trades=[_sell(PLACED_TS + 15, BID, 120)],
        )
    }
    with patch("urllib.request.urlopen", return_value=_Resp([_event(["1", "0", "0"])])):
        out = settle_paper(log, pd.DataFrame(), tapes=tapes)
    row = out.loc[0]
    assert row["resultado"] == "sem_fill"
    assert row["fill_simulado"] == False
    assert pd.isna(row["retorno_sim"])
    assert row["filled_queue"] == pytest.approx(20)
    assert row["fill_model"] == "queue"
    assert row["filled_touch"] == pytest.approx(OUR)
    assert row["filled_through"] == pytest.approx(0)


def test_incomplete_tape_does_not_invent_a_fill():
    log = _float_nan_log(placed_at=PLACED, sinal=True)
    tapes = {
        _base_row()["slug"]: _tape_for_settle(
            coverage="incomplete",
            trades=[_sell(PLACED_TS + 15, BID - 0.05, 10_000)],
        )
    }
    with patch("urllib.request.urlopen", return_value=_Resp([_event(["1", "0", "0"])])):
        with patch("src.paper_rn1.closing_price_before_kickoff", return_value=0.50):
            with patch("src.pinnacle_lookup.lookup_1x2_close", return_value=(1.6, 4.0, 6.0)):
                out = settle_paper(log, pd.DataFrame(), tapes=tapes)

    row = out.loc[0]
    assert row["fill_simulado"] == True
    assert row["resultado"] == "acerto"
    assert pd.isna(row["filled_size"])
    assert row["fill_model"] in ("", None) or (isinstance(row["fill_model"], float) and pd.isna(row["fill_model"]))
    assert pd.isna(row["filled_touch"])
    assert pd.isna(row["retorno_queue"])
    assert row["fill_coverage"] == "incomplete"
    assert row["queue_ahead"] == pytest.approx(100)


def test_report_shows_three_models_and_does_not_invent_unmeasured_rates():
    slug = _base_row()["slug"]
    measured = _base_row(
        slug=slug,
        sinal=True,
        resultado="sem_fill",
        fill_simulado=False,
        our_size=OUR,
        filled_size=OUR,
        fill_model="queue",
        filled_touch=OUR,
        filled_queue=OUR,
        filled_through=0.0,
        retorno_touch=0.6,
        retorno_queue=0.6,
        retorno_through=0.0,
        fill_coverage="complete",
    )
    missed = _base_row(
        slug="epl-miss",
        sinal=True,
        resultado="acerto",
        fill_simulado=True,
        retorno_sim=0.6,
        our_size=OUR,
        filled_size=0.0,
        fill_model="queue",
        filled_touch=0.0,
        filled_queue=0.0,
        filled_through=0.0,
        retorno_touch=0.0,
        retorno_queue=0.0,
        retorno_through=0.0,
        fill_coverage="complete",
    )
    unknown = _base_row(
        slug="epl-unknown",
        sinal=True,
        resultado="sem_fill",
        fill_simulado=False,
        fill_coverage="incomplete",
    )
    log = pd.DataFrame([measured, missed, unknown])
    stats = summarize_fill_models(log)
    assert stats["queue"]["medidos"] == 2
    assert stats["queue"]["fill_rate"] == pytest.approx(0.5)
    assert stats["queue"]["fill_rate_tamanho"] == pytest.approx(0.5)
    assert stats["queue"]["pnl_simulado"] == pytest.approx(0.6)
    assert stats["touch"]["fill_rate"] == pytest.approx(0.5)
    assert stats["through"]["fill_rate"] == pytest.approx(0.0)
    assert stats["through"]["pnl_simulado"] == pytest.approx(0.0)
    assert stats["nao_medidos"] == 1

    text = format_fill_model_report(stats)
    assert "simulado" in text.lower()
    assert "touch" in text and "otimista" in text
    assert "queue" in text and "central" in text
    assert "through" in text and "pessimista" in text
    assert "não medido" in text
    assert "suposição" in text
    assert str(int(ASSUMED_STAKE_USD)) in text


def test_empty_log_report_does_not_invent_a_fill_rate():
    text = format_fill_model_report(summarize_fill_models(pd.DataFrame()))
    assert "não medido" in text
    assert "0%" not in text


def test_ensure_log_keeps_new_fill_columns(tmp_path):
    path = tmp_path / "paper_rn1_forward.csv"
    pd.DataFrame([_base_row()]).to_csv(path, index=False)
    out = ensure_log(path)
    for col in (
        "placed_at",
        "condition_id",
        "queue_ahead",
        "our_size",
        "filled_size",
        "fill_model",
        "filled_touch",
        "filled_queue",
        "filled_through",
        "retorno_touch",
        "retorno_queue",
        "retorno_through",
        "fill_coverage",
    ):
        assert col in out.columns


def test_scan_fixture_copies_condition_id():
    odds = pd.Series({"ps_h": 1.50, "ps_d": 4.5, "ps_a": 7.0})
    markets = [
        {
            "title": "Home FC",
            "price": 0.70,
            "volume": 8000,
            "token_yes": "tok-h",
            "conditionId": "0xcond-home",
            "gameStartTime": "2026-10-19 19:00:00+00",
        },
        {"title": "Draw", "price": 0.20, "volume": 8000, "token_yes": "tok-d", "conditionId": "0xdraw"},
        {"title": "Away FC", "price": 0.14, "volume": 8000, "token_yes": "tok-a"},
    ]
    scanned = scan_fixture(
        "Premier League", "epl-h-a", "Home FC", "Away FC",
        markets, ["casa", "empate", "fora"], odds, "2026-10-19",
    )
    assert scanned is not None
    assert scanned["condition_id"] == "0xcond-home"
    assert scanned["placed_at"] is None


def test_poll_tapes_persists_complete_window_without_network_in_settle(tmp_path):
    row = _base_row(sinal=True, condition_id="0xabc", placed_at=PLACED, kickoff=KICKOFF, token="tok-home")
    log = pd.DataFrame([row])
    calls = {"n": 0}

    def fetch(condition_id, token, start_ts, end_ts):
        calls["n"] += 1
        assert condition_id == "0xabc"
        assert token == "tok-home"
        assert start_ts == PLACED_TS
        assert end_ts == KICKOFF_TS
        from src.paper_rn1_tape import TradeFetch

        return TradeFetch(
            trades=[_sell(PLACED_TS + 20, BID, 4)],
            hit_cap=False,
            fetch_error=False,
        )

    out = poll_tapes(log, fetch=fetch, tape_dir=tmp_path, now=datetime(2026, 10, 18, 11, 0, tzinfo=timezone.utc))
    assert calls["n"] == 1
    assert out.loc[0, "fill_coverage"] == "complete"
    saved = json.loads(next(tmp_path.glob("*.json")).read_text(encoding="utf-8"))
    assert len(saved["trades"]) == 1

    def settle_must_not_fetch(*args, **kwargs):
        raise AssertionError("settle não pode chamar a rede para trades")

    with patch("urllib.request.urlopen", return_value=_Resp([_event(["1", "0", "0"])])):
        with patch("src.paper_rn1.closing_price_before_kickoff", return_value=0.70):
            with patch("src.paper_rn1_tape.fetch_trades_window", side_effect=settle_must_not_fetch):
                settled = settle_paper(out, pd.DataFrame(), tape_dir=tmp_path)
    assert settled.loc[0, "fill_coverage"] == "complete"
    assert settled.loc[0, "filled_touch"] == pytest.approx(saved["our_size"])
