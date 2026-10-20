"""Fill simulado com fila para o paper RN1.

Três modelos, todos simulados (nenhuma ordem vai ao livro):
- touch: otimista, qualquer venda no preço do bid ou melhor enche a ordem;
- queue: central, só o volume que passa da fila que já estava na frente;
- through: pessimista, só conta fill se o preço negocia estritamente através do bid.
"""

from __future__ import annotations

import io
import json
import urllib.error
from datetime import datetime, timezone
from pathlib import Path
from unittest.mock import patch

import pandas as pd
import pytest

from src.queue_fill import (
    ASSUMED_STAKE_USD,
    best_ask_price,
    estimate_fills,
    queue_ahead_from_book,
    simulated_return,
    snap_bid_to_tick,
)
from src.paper_rn1 import (
    ensure_log,
    format_fill_model_report,
    scan_fixture,
    settle_paper,
    summarize_fill_models,
)
from src.paper_rn1_tape import (
    FREEZE_BUFFER_SEC,
    INCOMPLETE_GIVE_UP_SEC,
    INCOMPLETE_POLL_CAP,
    MAX_OFFSET,
    PAGE_LIMIT,
    SUB_WINDOW_SEC,
    TAPE_SCHEMA,
    TradeFetch,
    _tape_from_row,
    apply_signal_upgrades,
    capture_entry,
    fetch_trades_window,
    merge_polled_trades,
    partition_signals,
    poll_tapes,
    save_tape,
    stamp_signals,
)
from tests.test_paper_rn1 import BID, KICKOFF, _Resp, _base_row, _event, _float_nan_log

PLACED = "2026-10-18T10:00:00Z"
PLACED_TS = int(datetime(2026, 10, 18, 10, 0, tzinfo=timezone.utc).timestamp())
KICKOFF_TS = int(datetime(2026, 10, 18, 13, 0, tzinfo=timezone.utc).timestamp())
OUR = 50.0
YES = "yes-tok"
NO = "no-tok"
FIXTURE = Path(__file__).resolve().parent / "fixtures" / "lal_rea_sev_2026_10_18_taker_trades.json"


def _assert_fill_order(out: dict) -> None:
    touch, queue, through = out["touch"], out["queue"], out["through"]
    if touch.measured and queue.measured and through.measured:
        assert touch.filled_size >= queue.filled_size >= through.filled_size


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
    # Print estritamente abaixo do bid esvazia o nível inteiro, inclusive a nossa ordem.
    assert through["queue"].filled_size == OUR
    assert through["touch"].filled_size == OUR
    _assert_fill_order(through)


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
        "tick_size": "0.001",
        "bids": [
            {"price": "0.63", "size": "999"},
            {"price": "0.625", "size": "40"},
            {"price": "0.6250", "size": "15"},
            {"price": "0.62", "size": "100"},
        ],
        "asks": [{"price": "0.625", "size": "500"}],
    }
    assert queue_ahead_from_book(book, BID) == pytest.approx(55)
    assert queue_ahead_from_book({"tick_size": "0.001", "bids": []}, BID) == 0.0
    assert queue_ahead_from_book({"bids": []}, BID) is None
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
        "tick_size": "0.001",
        "bids": [{"price": str(BID), "size": "80"}, {"price": "0.50", "size": "10"}],
        "asks": [{"price": "0.70", "size": "5"}],
    }
    now = datetime(2026, 10, 18, 10, 0, tzinfo=timezone.utc)
    updated, tape = capture_entry(row, book, now)
    assert updated["placed_at"] == PLACED
    assert updated["queue_ahead"] == pytest.approx(80)
    assert updated["our_size"] == pytest.approx(ASSUMED_STAKE_USD / BID)
    assert updated["sim_price"] == pytest.approx(BID)
    assert updated["entry_cross"] is False
    assert tape["book_observed"] is True
    assert tape["tick_known"] is True
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
    first = {"tick_size": "0.001", "bids": [{"price": str(BID), "size": "10"}]}
    later = {"tick_size": "0.001", "bids": [{"price": str(BID), "size": "999"}]}
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
        {"tick_size": "0.001", "bids": [{"price": str(BID), "size": "1"}]},
        datetime(2026, 10, 18, 10, 0, tzinfo=timezone.utc),
    )
    inside = _sell(PLACED_TS + 30, BID, 4)
    fetched = [
        _sell(PLACED_TS - 5, BID, 99),
        inside,
        dict(inside),
        _sell(KICKOFF_TS, BID, 99),
    ]
    after_kickoff = datetime(2026, 10, 18, 13, 30, tzinfo=timezone.utc)
    merged = merge_polled_trades(tape, fetched, hit_cap=False, fetch_error=False, now=after_kickoff)
    assert len(merged["trades"]) == 1
    assert merged["trades"][0]["size"] == 4
    assert merged["coverage"] == "complete"
    assert merged["frozen"] is False

    before_kickoff = datetime(2026, 10, 18, 11, 0, tzinfo=timezone.utc)
    early = merge_polled_trades(tape, fetched, hit_cap=False, fetch_error=False, now=before_kickoff)
    assert early["coverage"] == "partial"

    again = merge_polled_trades(merged, [inside], hit_cap=False, fetch_error=False, now=after_kickoff)
    assert len(again["trades"]) == 1

    capped = merge_polled_trades(tape, [inside], hit_cap=True, fetch_error=False, now=after_kickoff)
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
        assert "takerOnly=true" in req.full_url
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
        "sim_price",
        "tick_size",
        "token_no",
        "entry_cross",
        "tick_known",
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
            "token_no": "tok-h-no",
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
    assert scanned["token_no"] == "tok-h-no"
    assert scanned["placed_at"] is None


def test_poll_tapes_persists_complete_window_without_network_in_settle(tmp_path):
    row = _base_row(sinal=True, condition_id="0xabc", placed_at=PLACED, kickoff=KICKOFF, token="tok-home")
    log = pd.DataFrame([row])
    calls = {"n": 0}

    def fetch(condition_id, tokens, start_ts, end_ts):
        calls["n"] += 1
        assert condition_id == "0xabc"
        assert "tok-home" in tokens
        assert start_ts == PLACED_TS
        assert end_ts == KICKOFF_TS
        return TradeFetch(
            trades=[_sell(PLACED_TS + 20, BID, 4)],
            hit_cap=False,
            fetch_error=False,
        )

    out = poll_tapes(log, fetch=fetch, tape_dir=tmp_path, now=datetime(2026, 10, 18, 13, 30, tzinfo=timezone.utc))
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


def test_fill_invariant_touch_is_at_least_queue_and_queue_at_least_through():
    cases = [
        _fills(trades=[]),
        _fills(trades=[_sell(PLACED_TS + 10, BID, 100)]),
        _fills(trades=[_sell(PLACED_TS + 10, BID, 120)]),
        _fills(trades=[_sell(PLACED_TS + 10, BID, 150)]),
        _fills(trades=[_sell(PLACED_TS + 5, BID - 0.01, 5)]),
        _fills(queue_ahead=0, trades=[_sell(PLACED_TS + 5, BID, 1)]),
        _fills(marketable=True, trades=[]),
    ]
    for out in cases:
        _assert_fill_order(out)


def test_no_buy_at_the_complement_consumes_the_yes_bid():
    """Mint: taker compra NO a 1-p. Isso bate no bid YES, 1 share NO = 1 share YES."""
    mint = _sell(PLACED_TS + 8, 1 - BID, 30, asset=NO, side="BUY", transactionHash="0xmint")
    ignored = [
        _sell(PLACED_TS + 9, BID, 10_000, asset=YES, side="BUY"),
        _sell(PLACED_TS + 9, 1 - BID, 10_000, asset=NO, side="SELL"),
    ]
    out = _fills(
        trades=[mint, *ignored],
        yes_token=YES,
        no_token=NO,
        tick_size=0.001,
        queue_ahead=10,
        our_size=50,
    )
    assert out["touch"].eligible_volume == pytest.approx(30)
    assert out["queue"].filled_size == pytest.approx(20)
    assert out["through"].filled_size == 0.0
    assert out["touch"].filled_size == 50
    _assert_fill_order(out)


def test_two_taker_records_in_one_transaction_are_summed():
    """takerOnly=true devolve um registro por taker. Dois no mesmo negócio somam."""
    tx = "0xshared"
    first = _sell(
        PLACED_TS + 8, 1 - BID, 100, asset=NO, side="BUY",
        transactionHash=tx, proxyWallet="0xaaa",
    )
    second = _sell(
        PLACED_TS + 8, 1 - BID, 40, asset=NO, side="BUY",
        transactionHash=tx, proxyWallet="0xbbb",
    )
    out = _fills(
        trades=[first, second],
        yes_token=YES,
        no_token=NO,
        tick_size=0.001,
        queue_ahead=0,
        our_size=200,
    )
    assert out["touch"].eligible_volume == pytest.approx(140)
    assert out["queue"].filled_size == pytest.approx(140)
    assert out["through"].filled_size == 0.0
    _assert_fill_order(out)


def test_print_strictly_below_clears_the_whole_queue():
    no_through = _sell(PLACED_TS + 8, 0.39, 1, asset=NO, side="BUY")
    out = _fills(
        trades=[no_through],
        yes_token=YES,
        no_token=NO,
        tick_size=0.001,
        queue_ahead=10_000,
        our_size=OUR,
    )
    assert out["through"].filled_size == OUR
    assert out["queue"].filled_size == OUR
    assert out["touch"].filled_size == OUR
    _assert_fill_order(out)


def test_off_grid_bid_snaps_down_to_the_book_tick():
    book = {
        "tick_size": "0.001",
        "bids": [
            {"price": "0.738", "size": "5000"},
            {"price": "0.739", "size": "3000"},
        ],
    }
    assert snap_bid_to_tick(0.7385, 0.001) == pytest.approx(0.738)
    assert queue_ahead_from_book(book, 0.7385) == pytest.approx(5000)
    assert queue_ahead_from_book({"bids": book["bids"]}, 0.7385) is None

    now = datetime(2026, 10, 18, 10, 0, tzinfo=timezone.utc)
    row = {
        "slug": "off-grid",
        "token": YES,
        "bid": 0.7385,
        "condition_id": "0xabc",
        "kickoff": KICKOFF,
        "sinal": True,
    }
    updated, tape = capture_entry(row, book, now)
    assert updated["sim_price"] == pytest.approx(0.738)
    assert updated["queue_ahead"] == pytest.approx(5000)
    assert tape["tick_known"] is True

    blind, blind_tape = capture_entry(row, {"bids": book["bids"]}, now)
    assert blind["sim_price"] is None
    assert blind["queue_ahead"] is None
    assert blind_tape["tick_known"] is False

    at_level = _sell(PLACED_TS + 5, 0.738, 1, asset=YES)
    out = estimate_fills(
        price=0.7385,
        our_size=1,
        queue_ahead=5000,
        trades=[at_level],
        placed_ts=PLACED_TS,
        kickoff_ts=KICKOFF_TS,
        yes_token=YES,
        no_token=NO,
        tick_size=0.001,
    )
    assert out["touch"].filled_size == pytest.approx(1)
    assert out["queue"].filled_size == 0.0
    assert out["through"].filled_size == 0.0
    _assert_fill_order(out)


def test_real_taker_sample_counts_no_buy_and_yes_sell():
    fixture = json.loads(FIXTURE.read_text(encoding="utf-8"))
    trades = fixture["trades"]
    yes = fixture["token_yes"]
    no = fixture["token_no"]
    tick = float(fixture["gamma_orderPriceMinTickSize"])
    sim = 0.790
    start = int(fixture["window_start"])
    end = int(fixture["window_end"])

    def on_grid(price: float) -> float:
        return round(round(price / tick) * tick, 3)

    counted: list[tuple[float, float]] = []
    for trade in trades:
        size = float(trade["size"])
        raw = float(trade["price"])
        if trade["asset"] == yes and trade["side"] == "SELL":
            yes_price = on_grid(raw)
        elif trade["asset"] == no and trade["side"] == "BUY":
            yes_price = round(1.0 - on_grid(raw), 3)
        else:
            continue
        if yes_price <= sim:
            counted.append((yes_price, size))
    assert counted, "a amostra pública precisa ter fluxo que consome o bid"
    expected = sum(size for _, size in counted)
    assert any(price < sim for price, _ in counted)

    out = estimate_fills(
        price=sim,
        our_size=OUR,
        queue_ahead=0,
        trades=trades,
        placed_ts=start,
        kickoff_ts=end,
        yes_token=yes,
        no_token=no,
        tick_size=tick,
    )
    assert out["touch"].eligible_volume == pytest.approx(expected)
    assert out["queue"].filled_size == OUR
    assert out["through"].filled_size == OUR
    _assert_fill_order(out)

    mint = max(
        (trade for trade in trades if trade["asset"] == no and trade["side"] == "BUY"),
        key=lambda trade: float(trade["size"]),
    )
    mint_only = estimate_fills(
        price=sim,
        our_size=OUR,
        queue_ahead=1000,
        trades=[mint],
        placed_ts=start,
        kickoff_ts=end,
        yes_token=yes,
        no_token=no,
        tick_size=tick,
    )
    assert float(mint["size"]) - 1000 >= OUR
    assert mint_only["touch"].eligible_volume == pytest.approx(float(mint["size"]))
    assert mint_only["queue"].filled_size == OUR
    assert mint_only["through"].filled_size == 0.0

    ignored = [
        trade for trade in trades
        if (trade["asset"] == yes and trade["side"] == "BUY")
        or (trade["asset"] == no and trade["side"] == "SELL")
    ]
    assert ignored
    quiet = estimate_fills(
        price=sim,
        our_size=OUR,
        queue_ahead=0,
        trades=ignored,
        placed_ts=start,
        kickoff_ts=end,
        yes_token=yes,
        no_token=no,
        tick_size=tick,
    )
    assert quiet["touch"].filled_size == 0.0
    assert quiet["queue"].filled_size == 0.0
    assert quiet["through"].filled_size == 0.0

    book = fixture["open_book_example"]
    assert best_ask_price(book) == pytest.approx(min(float(level["price"]) for level in book["asks"]))
    assert best_ask_price(book) != pytest.approx(float(book["asks"][0]["price"]))
    level = book["bids"][2]
    level_price = float(level["price"])
    off = level_price + 0.4 * float(book["tick_size"])
    assert snap_bid_to_tick(off, float(book["tick_size"])) == pytest.approx(level_price)
    assert queue_ahead_from_book(book, off) == pytest.approx(float(level["size"]))


def test_best_ask_is_the_minimum_and_a_cross_fills_all_models():
    book = {
        "tick_size": "0.001",
        "bids": [{"price": str(BID), "size": "80"}],
        "asks": [{"price": "0.999", "size": "9"}, {"price": "0.620", "size": "3"}],
    }
    assert best_ask_price(book) == pytest.approx(0.620)
    now = datetime(2026, 10, 18, 10, 0, tzinfo=timezone.utc)
    row = {
        "slug": "cross",
        "token": YES,
        "token_no": NO,
        "bid": BID,
        "condition_id": "0xabc",
        "kickoff": KICKOFF,
        "sinal": True,
    }
    updated, tape = capture_entry(row, book, now)
    assert updated["entry_cross"] is True
    assert tape["sim_price"] == pytest.approx(BID)

    log = _float_nan_log(placed_at=PLACED, sinal=True)
    stored = _tape_for_settle(coverage="incomplete", trades=[], queue_ahead=80)
    stored.update(
        entry_cross=True,
        sim_price=BID,
        tick_size=0.001,
        tick_known=True,
        token=YES,
        token_no=NO,
    )
    tapes = {_base_row()["slug"]: stored}
    with patch("urllib.request.urlopen", return_value=_Resp([_event(["1", "0", "0"])])):
        with patch("src.paper_rn1.closing_price_before_kickoff", return_value=0.70):
            with patch("src.pinnacle_lookup.lookup_1x2_close", return_value=(1.6, 4.0, 6.0)):
                out = settle_paper(log, pd.DataFrame(), tapes=tapes)
    row_out = out.loc[0]
    assert row_out["fill_simulado"] == False
    assert row_out["resultado"] == "sem_fill"
    assert row_out["filled_touch"] == pytest.approx(OUR)
    assert row_out["filled_queue"] == pytest.approx(OUR)
    assert row_out["filled_through"] == pytest.approx(OUR)
    assert row_out["fill_model"] == "cross"


def test_unknown_tick_leaves_the_three_models_unmeasured():
    log = _float_nan_log(placed_at=PLACED, sinal=True)
    tape = _tape_for_settle(
        coverage="complete",
        trades=[_sell(PLACED_TS + 5, BID - 0.01, 10_000)],
    )
    tape["tick_known"] = False
    tapes = {_base_row()["slug"]: tape}
    with patch("urllib.request.urlopen", return_value=_Resp([_event(["1", "0", "0"])])):
        with patch("src.paper_rn1.closing_price_before_kickoff", return_value=0.50):
            with patch("src.pinnacle_lookup.lookup_1x2_close", return_value=(1.6, 4.0, 6.0)):
                out = settle_paper(log, pd.DataFrame(), tapes=tapes)
    row = out.loc[0]
    assert row["fill_simulado"] == True
    assert pd.isna(row["filled_touch"])
    assert pd.isna(row["filled_queue"])
    assert pd.isna(row["filled_through"])


def test_partial_and_blank_complete_count_as_unmeasured():
    measured = _base_row(
        slug="medido",
        sinal=True,
        our_size=OUR,
        filled_touch=OUR,
        filled_queue=OUR,
        filled_through=0.0,
        retorno_touch=0.2,
        retorno_queue=0.2,
        retorno_through=0.0,
        fill_coverage="complete",
    )
    partial = _base_row(
        slug="parcial",
        sinal=True,
        our_size=OUR,
        filled_touch=OUR,
        filled_queue=OUR,
        filled_through=OUR,
        fill_coverage="partial",
    )
    blank = _base_row(slug="vazio", sinal=True, fill_coverage="complete")
    crossed = _base_row(
        slug="cruzou",
        sinal=True,
        our_size=OUR,
        filled_touch=OUR,
        filled_queue=0.0,
        filled_through=0.0,
        retorno_touch=0.0,
        retorno_queue=0.0,
        retorno_through=0.0,
        fill_coverage="incomplete",
        entry_cross=True,
    )
    stats = summarize_fill_models(pd.DataFrame([measured, partial, blank, crossed]))
    assert stats["queue"]["medidos"] == 2
    assert stats["nao_medidos"] == 2
    assert stats["sinais"] == 4


def test_merge_does_not_downgrade_a_frozen_tape():
    _, tape = capture_entry(
        {
            "slug": "frozen",
            "token": YES,
            "bid": BID,
            "condition_id": "0xabc",
            "kickoff": KICKOFF,
        },
        {"tick_size": "0.001", "bids": [{"price": str(BID), "size": "1"}]},
        datetime(2026, 10, 18, 10, 0, tzinfo=timezone.utc),
    )
    inside = _sell(PLACED_TS + 30, BID, 4)
    closed = merge_polled_trades(
        tape,
        [inside],
        hit_cap=False,
        fetch_error=False,
        now=datetime(2026, 10, 18, 15, 0, tzinfo=timezone.utc),
    )
    assert closed["coverage"] == "complete"
    assert closed["frozen"] is True
    downgraded = merge_polled_trades(
        closed,
        [],
        hit_cap=True,
        fetch_error=True,
        now=datetime(2026, 10, 18, 16, 0, tzinfo=timezone.utc),
    )
    assert downgraded["coverage"] == "complete"
    assert downgraded["frozen"] is True
    assert downgraded["trades"][0]["size"] == 4


def test_poll_does_not_refetch_a_frozen_tape(tmp_path):
    row = _base_row(sinal=True, condition_id="0xabc", placed_at=PLACED, kickoff=KICKOFF, token="tok-home")
    tape = {
        "slug": row["slug"],
        "coverage": "complete",
        "frozen": False,
        "trades": [_sell(PLACED_TS + 20, BID, 4)],
        "placed_at": PLACED,
        "kickoff": KICKOFF,
        "condition_id": "0xabc",
        "token": "tok-home",
        "bid": BID,
        "sim_price": BID,
        "tick_size": 0.001,
        "tick_known": True,
        "queue_ahead": 1,
        "our_size": OUR,
        "book_observed": True,
        "fetch_error": False,
        "hit_cap": False,
        "fetched_at": "2026-10-18T14:01:00Z",
    }
    save_tape(tape, tmp_path)
    calls = {"n": 0}

    def fetch(*args, **kwargs):
        calls["n"] += 1
        raise AssertionError("fita congelada não pede trades")

    later = datetime(2026, 10, 18, 13, 0, tzinfo=timezone.utc).timestamp() + FREEZE_BUFFER_SEC + 60
    moment = datetime.fromtimestamp(later, tz=timezone.utc)
    out = poll_tapes(pd.DataFrame([row]), fetch=fetch, tape_dir=tmp_path, now=moment)
    assert calls["n"] == 0
    assert out.loc[0, "fill_coverage"] == "complete"
    saved = json.loads(next(tmp_path.glob("*.json")).read_text(encoding="utf-8"))
    assert saved["frozen"] is True
    assert saved["coverage"] == "complete"
    assert saved["schema"] == TAPE_SCHEMA


def test_poll_before_kickoff_stays_partial(tmp_path):
    row = _base_row(sinal=True, condition_id="0xabc", placed_at=PLACED, kickoff=KICKOFF, token="tok-home")
    out = poll_tapes(
        pd.DataFrame([row]),
        fetch=lambda *args, **kwargs: TradeFetch(
            trades=[_sell(PLACED_TS + 20, BID, 4)],
            hit_cap=False,
            fetch_error=False,
        ),
        tape_dir=tmp_path,
        now=datetime(2026, 10, 18, 11, 0, tzinfo=timezone.utc),
    )
    assert out.loc[0, "fill_coverage"] == "partial"
    with patch("urllib.request.urlopen", return_value=_Resp([_event(["1", "0", "0"])])):
        with patch("src.paper_rn1.closing_price_before_kickoff", return_value=0.50):
            settled = settle_paper(out, pd.DataFrame(), tape_dir=tmp_path)
    assert pd.isna(settled.loc[0, "filled_touch"])
    assert pd.isna(settled.loc[0, "filled_queue"])


def test_429_mid_pagination_keeps_the_first_page_and_backs_off():
    page = [
        _sell(PLACED_TS + 10, BID, 1, asset="tok-home", transactionHash=f"0x{i}")
        for i in range(PAGE_LIMIT)
    ]
    state = {"n": 0, "delays": []}

    def urlopen(req, timeout=60):
        state["n"] += 1
        if "offset=0&" in req.full_url or req.full_url.endswith("offset=0"):
            return _UrlResp(page)
        raise urllib.error.HTTPError(
            req.full_url, 429, "Too Many Requests", None, io.BytesIO(b"")
        )

    got = fetch_trades_window(
        "0xabc",
        "tok-home",
        PLACED_TS,
        KICKOFF_TS,
        urlopen=urlopen,
        sleeper=state["delays"].append,
    )
    assert got.fetch_error is True
    assert got.hit_cap is False
    assert len(got.trades) == PAGE_LIMIT
    assert state["n"] > 1
    assert state["delays"]


def test_empty_page_after_an_exact_multiple_of_the_page_is_a_normal_end():
    def trades(n, tag):
        return [
            _sell(PLACED_TS + 10, BID, 1, asset="tok-home", transactionHash=f"0x{tag}-{i}")
            for i in range(n)
        ]

    def run(pages):
        offsets = []

        def urlopen(req, timeout=60):
            offset = int(req.full_url.split("offset=")[1].split("&")[0])
            offsets.append(offset)
            if offset // PAGE_LIMIT < len(pages):
                return _UrlResp(pages[offset // PAGE_LIMIT])
            return _UrlResp([])

        got = fetch_trades_window(
            "0xabc", "tok-home", PLACED_TS, KICKOFF_TS, urlopen=urlopen, sleeper=lambda _s: None,
        )
        return got, offsets

    one, offsets_one = run([trades(PAGE_LIMIT, "a")])
    assert one.hit_cap is False
    assert one.fetch_error is False
    assert len(one.trades) == PAGE_LIMIT
    assert offsets_one == [0, PAGE_LIMIT]

    two, _offsets_two = run([trades(PAGE_LIMIT, "b"), trades(PAGE_LIMIT, "c")])
    assert two.hit_cap is False
    assert len(two.trades) == PAGE_LIMIT * 2

    short, offsets_short = run([trades(PAGE_LIMIT - 1, "d")])
    assert short.hit_cap is False
    assert len(short.trades) == PAGE_LIMIT - 1
    assert offsets_short == [0]


def test_pagination_does_not_request_the_offset_cap():
    offsets: list[int] = []

    def urlopen(req, timeout=60):
        offset = int(req.full_url.split("offset=")[1].split("&")[0])
        offsets.append(offset)
        page = [
            _sell(PLACED_TS + 10, BID, 1, asset="tok-home", transactionHash=f"0x{offset}-{i}")
            for i in range(PAGE_LIMIT)
        ]
        return _UrlResp(page)

    got = fetch_trades_window(
        "0xabc", "tok-home", PLACED_TS, KICKOFF_TS, urlopen=urlopen, sleeper=lambda _s: None,
    )
    assert got.hit_cap is True
    assert MAX_OFFSET not in offsets
    assert max(offsets) == MAX_OFFSET - PAGE_LIMIT
    assert len(offsets) == MAX_OFFSET // PAGE_LIMIT


def test_long_window_is_split_and_stays_taker_only():
    urls: list[str] = []

    def urlopen(req, timeout=60):
        urls.append(req.full_url)
        return _UrlResp([])

    end = PLACED_TS + SUB_WINDOW_SEC + 3600
    got = fetch_trades_window(
        "0xabc", "tok-home", PLACED_TS, end, urlopen=urlopen, sleeper=lambda _s: None,
    )
    assert got.fetch_error is False
    assert got.hit_cap is False
    assert len(urls) >= 2
    assert all("takerOnly=true" in url for url in urls)
    starts = {url.split("start=")[1].split("&")[0] for url in urls}
    assert len(starts) >= 2


def test_dedupe_keeps_distinct_proxy_wallets():
    _, tape = capture_entry(
        {
            "slug": "wallets",
            "token": YES,
            "bid": BID,
            "condition_id": "0xabc",
            "kickoff": KICKOFF,
        },
        {"tick_size": "0.001", "bids": [{"price": str(BID), "size": "1"}]},
        datetime(2026, 10, 18, 10, 0, tzinfo=timezone.utc),
    )
    base = _sell(PLACED_TS + 30, BID, 4, proxyWallet="0xaaa", outcomeIndex=0)
    other = dict(base)
    other["proxyWallet"] = "0xbbb"
    moment = datetime(2026, 10, 18, 13, 30, tzinfo=timezone.utc)
    merged = merge_polled_trades(tape, [base, other], hit_cap=False, fetch_error=False, now=moment)
    assert len(merged["trades"]) == 2
    again = merge_polled_trades(tape, [base, dict(base)], hit_cap=False, fetch_error=False, now=moment)
    assert len(again["trades"]) == 1


def test_signal_that_turns_true_later_gets_a_book_snapshot(tmp_path):
    log = pd.DataFrame([_base_row(sinal=False)])
    incoming = _base_row(
        sinal=True,
        token="tok-home",
        condition_id="0xabc",
        kickoff=KICKOFF,
        bid=BID,
    )
    fresh, upgrades = partition_signals([incoming], log)
    assert fresh == []
    assert len(upgrades) == 1
    book = {
        "tick_size": "0.001",
        "bids": [{"price": str(BID), "size": "12"}],
        "asks": [{"price": "0.80", "size": "1"}],
    }
    stamped = stamp_signals(
        upgrades,
        lambda token: book,
        tape_dir=tmp_path,
        now=datetime(2026, 10, 18, 10, 0, tzinfo=timezone.utc),
    )
    out = apply_signal_upgrades(log, stamped)
    assert out.loc[0, "placed_at"] == PLACED
    assert out.loc[0, "queue_ahead"] == pytest.approx(12)
    assert out.loc[0, "sinal"] == True
    assert out.loc[0, "sim_price"] == pytest.approx(BID)

    later = dict(stamped[0])
    later["queue_ahead"] = 999
    later["placed_at"] = "2026-10-18T12:00:00Z"
    fresh_again, upgrades_again = partition_signals([later], out)
    assert fresh_again == []
    assert upgrades_again == []
    kept = apply_signal_upgrades(out, [later])
    assert kept.loc[0, "placed_at"] == PLACED
    assert kept.loc[0, "queue_ahead"] == pytest.approx(12)


def test_csv_false_flags_do_not_invent_a_full_fill(tmp_path):
    path = tmp_path / "paper_rn1_forward.csv"
    row = _base_row(
        sinal=True,
        placed_at=PLACED,
        kickoff=KICKOFF,
        condition_id="0xabc",
        token="tok-home",
        token_no=NO,
        our_size=OUR,
        queue_ahead=0,
        sim_price=BID,
        fill_coverage="pending",
        entry_cross=False,
        tick_known=False,
    )
    pd.DataFrame([row]).to_csv(path, index=False)
    loaded = ensure_log(path)
    assert str(loaded.loc[0, "entry_cross"]) == "False"
    assert str(loaded.loc[0, "tick_known"]) == "False"
    tape = _tape_from_row(loaded.iloc[0])
    assert tape["entry_cross"] is False
    assert tape["tick_known"] is False

    nan_row = loaded.iloc[0].copy()
    nan_row["tick_known"] = float("nan")
    nan_row["entry_cross"] = "False"
    nan_tape = _tape_from_row(nan_row)
    assert nan_tape["tick_known"] is False
    assert nan_tape["entry_cross"] is False

    with patch("urllib.request.urlopen", return_value=_Resp([_event(["1", "0", "0"])])):
        with patch("src.paper_rn1.closing_price_before_kickoff", return_value=0.70):
            settled = settle_paper(loaded, pd.DataFrame(), tapes={row["slug"]: tape})
    assert pd.isna(settled.loc[0, "filled_touch"])
    assert pd.isna(settled.loc[0, "filled_queue"])
    assert settled.loc[0, "fill_model"] != "cross"
    stats = summarize_fill_models(settled)
    assert stats["queue"]["medidos"] == 0
    assert stats["nao_medidos"] == 1


def test_legacy_tape_without_tick_is_not_measured(tmp_path):
    row = _base_row(
        sinal=True,
        placed_at=PLACED,
        kickoff=KICKOFF,
        condition_id="0xabc",
        token="tok-home",
        queue_ahead=0,
        our_size=OUR,
    )
    tape = {
        "slug": row["slug"],
        "coverage": "complete",
        "trades": [_sell(PLACED_TS + 20, 0.738, 10)],
        "placed_at": PLACED,
        "kickoff": KICKOFF,
        "condition_id": "0xabc",
        "token": "tok-home",
        "bid": 0.7385,
        "queue_ahead": 0.0,
        "our_size": OUR,
        "book_observed": True,
    }
    save_tape(tape, tmp_path)
    calls = {"fetch": 0, "no": 0}

    def fetch(*args, **kwargs):
        calls["fetch"] += 1
        raise AssertionError("fita antiga não é relida como se fosse válida")

    def resolve(slug, yes):
        calls["no"] += 1
        assert yes == "tok-home"
        return "tok-no-backfill"

    moment = datetime(2026, 10, 18, 15, 0, tzinfo=timezone.utc)
    out = poll_tapes(
        pd.DataFrame([row]),
        fetch=fetch,
        tape_dir=tmp_path,
        now=moment,
        resolve_no_token=resolve,
    )
    assert calls["fetch"] == 0
    assert calls["no"] == 1
    assert out.loc[0, "fill_coverage"] == "incomplete"
    assert out.loc[0, "tick_known"] == False
    assert out.loc[0, "token_no"] == "tok-no-backfill"
    saved = json.loads((tmp_path / f"{row['slug']}.json").read_text(encoding="utf-8"))
    assert saved["schema"] == TAPE_SCHEMA
    assert saved["legacy"] is True
    assert saved["tick_known"] is False
    assert saved["trades"] == []
    assert saved["token_no"] == "tok-no-backfill"
    assert saved["poll_stopped"] is True


def test_freeze_waits_for_a_fetch_after_the_buffer(tmp_path):
    row = _base_row(sinal=True, condition_id="0xabc", placed_at=PLACED, kickoff=KICKOFF, token="tok-home")
    calls = {"n": 0}

    def fetch(*args, **kwargs):
        calls["n"] += 1
        return TradeFetch(
            trades=[_sell(PLACED_TS + 20, BID, 4)],
            hit_cap=False,
            fetch_error=False,
        )

    near = datetime(2026, 10, 18, 13, 10, tzinfo=timezone.utc)
    out = poll_tapes(pd.DataFrame([row]), fetch=fetch, tape_dir=tmp_path, now=near)
    assert calls["n"] == 1
    assert out.loc[0, "fill_coverage"] == "complete"
    saved = json.loads(next(tmp_path.glob("*.json")).read_text(encoding="utf-8"))
    assert saved["frozen"] is False

    later = datetime(2026, 10, 18, 15, 0, tzinfo=timezone.utc)
    out = poll_tapes(out, fetch=fetch, tape_dir=tmp_path, now=later)
    assert calls["n"] == 2
    saved = json.loads(next(tmp_path.glob("*.json")).read_text(encoding="utf-8"))
    assert saved["frozen"] is True
    assert saved["fetched_at"] == "2026-10-18T15:00:00Z"

    poll_tapes(out, fetch=fetch, tape_dir=tmp_path, now=later)
    assert calls["n"] == 2


def test_incomplete_tape_stops_after_the_attempt_cap(tmp_path):
    row = _base_row(sinal=True, condition_id="0xabc", placed_at=PLACED, kickoff=KICKOFF, token="tok-home")
    calls = {"n": 0}

    def fetch(*args, **kwargs):
        calls["n"] += 1
        return TradeFetch(trades=[], hit_cap=False, fetch_error=True)

    moment = datetime(2026, 10, 18, 15, 0, tzinfo=timezone.utc)
    log = pd.DataFrame([row])
    for _ in range(INCOMPLETE_POLL_CAP):
        log = poll_tapes(log, fetch=fetch, tape_dir=tmp_path, now=moment)
    assert calls["n"] == INCOMPLETE_POLL_CAP
    assert log.loc[0, "fill_coverage"] == "incomplete"
    log = poll_tapes(log, fetch=fetch, tape_dir=tmp_path, now=moment)
    assert calls["n"] == INCOMPLETE_POLL_CAP
    saved = json.loads(next(tmp_path.glob("*.json")).read_text(encoding="utf-8"))
    assert saved["poll_stopped"] is True
    assert saved["coverage"] == "incomplete"


def test_incomplete_tape_stops_48h_after_kickoff(tmp_path):
    row = _base_row(sinal=True, condition_id="0xabc", placed_at=PLACED, kickoff=KICKOFF, token="tok-home")

    def fetch(*args, **kwargs):
        raise AssertionError("depois de 48h a fita incompleta não é consultada")

    moment = datetime.fromtimestamp(KICKOFF_TS + INCOMPLETE_GIVE_UP_SEC + 60, tz=timezone.utc)
    out = poll_tapes(pd.DataFrame([row]), fetch=fetch, tape_dir=tmp_path, now=moment)
    assert out.loc[0, "fill_coverage"] == "incomplete"
    saved = json.loads(next(tmp_path.glob("*.json")).read_text(encoding="utf-8"))
    assert saved["poll_stopped"] is True
    assert saved["coverage"] == "incomplete"


def test_paper_rn1_monitor_polls_once_per_run():
    source = Path(__file__).resolve().parents[1].joinpath("monitor.py").read_text(encoding="utf-8")
    assert source.count("poll_tapes(") == 4
