"""Paper RN1: dtype do log, kickoff/token e liquidação simulada."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from unittest.mock import patch

import numpy as np
import pandas as pd
import pytest

from src.clv import clv_probability
from src.odds_utils import market_probs
from src.paper_rn1 import (
    ensure_log,
    scan_fixture,
    settle_paper,
    summarize_paper,
)

TOKEN = "58413957650880464582217008625565461775009894754199001914450706042394844512977"
KICKOFF = "2026-10-18T13:00:00Z"
BID = 0.625


class _Resp:
    def __init__(self, payload):
        self._raw = json.dumps(payload).encode()

    def read(self, *args, **kwargs):
        return self._raw

    def __enter__(self):
        return self

    def __exit__(self, *args):
        return False


def _base_row(**overrides) -> dict:
    row = {
        "data": "2026-10-18",
        "liga": "Premier League",
        "slug": "epl-bou-sun-2026-10-18",
        "casa": "Bournemouth",
        "fora": "Sunderland",
        "pick": "casa",
        "fair_pin": 0.65,
        "bid": BID,
        "poly_exec": 0.70,
        "volume": 5000,
        "kickoff": KICKOFF,
        "sinal": False,
        "fill_simulado": np.nan,
        "resultado": np.nan,
        "retorno_sim": np.nan,
        "clv_pin": np.nan,
        "token": "tok-home",
    }
    row.update(overrides)
    return row


def _float_nan_log(**overrides) -> pd.DataFrame:
    """Reproduz o CSV: colunas todas vazias nascem float64 e recusam bool."""
    df = pd.DataFrame([_base_row(**overrides)])
    for col in ("fill_simulado", "resultado", "retorno_sim", "clv_pin"):
        df[col] = df[col].astype("float64")
    return df


def _event(prices, *, token="tok-home", kickoff=KICKOFF, with_token=True, with_time=True, closed=True):
    def market(title, price, token_id):
        raw = {
            "groupItemTitle": title,
            "outcomePrices": json.dumps([price, "0"]),
            "closed": closed,
        }
        if with_time:
            raw["gameStartTime"] = "2026-10-18 13:00:00+00"
            raw["endDate"] = kickoff
        if with_token and token_id:
            raw["clobTokenIds"] = json.dumps([token_id, token_id + "-no"])
        return raw

    ev = {
        "slug": "epl-bou-sun-2026-10-18",
        "markets": [
            market("Bournemouth", prices[0], token),
            market("Draw", prices[1], "tok-draw"),
            market("Sunderland", prices[2], "tok-away"),
        ],
    }
    if with_time:
        ev["startTime"] = kickoff
        ev["endDate"] = kickoff
    return ev


def _settle(log, event, close, pin=(1.6, 4.0, 6.0)):
    with patch("urllib.request.urlopen", return_value=_Resp([event])):
        with patch("src.paper_rn1.closing_price_before_kickoff", return_value=close):
            with patch("src.pinnacle_lookup.lookup_1x2_close", return_value=pin):
                return settle_paper(log, pd.DataFrame())


def test_settle_writes_bool_into_all_nan_float_column():
    log = _float_nan_log()
    assert log["fill_simulado"].dtype == np.dtype("float64")
    assert log["resultado"].dtype == np.dtype("float64")
    out = _settle(log, _event(["1", "0", "0"]), close=0.70)
    assert out.loc[0, "fill_simulado"] == False
    assert out.loc[0, "resultado"] == "sem_fill"
    assert pd.isna(out.loc[0, "retorno_sim"])
    assert pd.isna(out.loc[0, "clv_pin"])


def test_missing_kickoff_and_token_stay_pending():
    log = _float_nan_log(kickoff=np.nan, token=np.nan)
    log["kickoff"] = log["kickoff"].astype("float64")
    log["token"] = log["token"].astype("float64")
    out = _settle(log, _event(["1", "0", "0"], with_token=False, with_time=False), close=0.50)
    assert pd.isna(out.loc[0, "resultado"])
    assert pd.isna(out.loc[0, "fill_simulado"])
    assert out.loc[0, "resultado"] != "sem_fill"


def test_false_sem_fill_without_ids_is_reopened_and_filled():
    log = pd.DataFrame([_base_row(
        kickoff=np.nan,
        token=np.nan,
        resultado="sem_fill",
        fill_simulado=False,
    )])
    log["kickoff"] = log["kickoff"].astype("float64")
    log["token"] = log["token"].astype("float64")
    assert log["token"].dtype == np.dtype("float64")
    out = _settle(log, _event(["1", "0", "0"], token=TOKEN), close=0.50)
    assert out.loc[0, "fill_simulado"] == True
    assert out.loc[0, "resultado"] == "acerto"
    assert out.loc[0, "kickoff"] == KICKOFF
    assert out.loc[0, "token"] == TOKEN
    assert out.loc[0, "retorno_sim"] == pytest.approx(round((1.0 - BID) / BID, 4))
    p_close = market_probs(np.array([[1.6, 4.0, 6.0]], dtype=float))[0][0]
    assert out.loc[0, "clv_pin"] == pytest.approx(round(clv_probability(BID, float(p_close)), 4))


def test_happy_path_fill_and_no_fill():
    filled = _base_row(slug="epl-filled", token="tok-filled")
    missed = _base_row(slug="epl-missed", token="tok-missed")
    log = pd.DataFrame([filled, missed])
    for col in ("fill_simulado", "resultado", "retorno_sim", "clv_pin"):
        log[col] = log[col].astype("float64")

    def close_for(token, kickoff):
        return 0.50 if token == "tok-filled" else 0.90

    events = {
        "epl-filled": _event(["0", "0", "1"], token="tok-filled"),
        "epl-missed": _event(["1", "0", "0"], token="tok-missed"),
    }

    def urlopen(req, timeout=30):
        slug = req.full_url.split("slug=")[-1]
        return _Resp([events[slug]])

    with patch("urllib.request.urlopen", side_effect=urlopen):
        with patch("src.paper_rn1.closing_price_before_kickoff", side_effect=close_for):
            with patch("src.pinnacle_lookup.lookup_1x2_close", return_value=(1.6, 4.0, 6.0)):
                out = settle_paper(log, pd.DataFrame())

    hit = out[out["slug"] == "epl-filled"].iloc[0]
    miss = out[out["slug"] == "epl-missed"].iloc[0]
    assert hit["fill_simulado"] == True
    assert hit["resultado"] == "erro"
    assert hit["retorno_sim"] == pytest.approx(-1.0)
    assert pd.notna(hit["clv_pin"])
    assert miss["fill_simulado"] == False
    assert miss["resultado"] == "sem_fill"
    assert pd.isna(miss["retorno_sim"])
    assert pd.isna(miss["clv_pin"])


def test_missing_price_history_is_a_skip_not_sem_fill():
    log = _float_nan_log()
    out = _settle(log, _event(["1", "0", "0"]), close=None)
    assert pd.isna(out.loc[0, "resultado"])
    assert pd.isna(out.loc[0, "fill_simulado"])


def test_summarize_counts_sem_fill_as_resolved():
    log = pd.DataFrame([
        _base_row(resultado="sem_fill", fill_simulado=False),
        _base_row(slug="pendente", resultado=np.nan, fill_simulado=np.nan),
        _base_row(slug="acerto", resultado="acerto", fill_simulado=True, retorno_sim=0.6, clv_pin=0.02),
    ])
    summary = summarize_paper(log)
    assert summary["registros"] == 3
    assert summary["resolvidos"] == 2
    assert summary["sem_fill"] == 1
    assert summary["fills"] == 1
    assert summary["pendentes"] == 1
    assert summary["retorno"] == pytest.approx(0.6)


def test_scan_fixture_reads_token_yes_and_kickoff_not_enddate_only():
    odds = pd.Series({"ps_h": 1.50, "ps_d": 4.5, "ps_a": 7.0})
    markets = [
        {
            "title": "Home FC",
            "price": 0.70,
            "volume": 8000,
            "token_yes": "tok-h",
            "gameStartTime": "2026-10-19 19:00:00+00",
            "endDate": "2026-10-20T23:00:00Z",
        },
        {"title": "Draw", "price": 0.20, "volume": 8000, "token_yes": "tok-d"},
        {"title": "Away FC", "price": 0.14, "volume": 8000, "token_yes": "tok-a"},
    ]
    roles = ["casa", "empate", "fora"]
    above = scan_fixture(
        "Premier League", "epl-tot-cov-2026-10-19", "Home FC", "Away FC",
        markets, roles, odds, "2026-10-19",
    )
    assert above is not None
    assert above["token"] == "tok-h"
    assert above["kickoff"] == "2026-10-19T19:00:00Z"
    assert above["sinal"] is False

    markets[0] = {**markets[0], "price": 0.55, "clobTokenIds": '["from-clob","no"]'}
    markets[0].pop("token_yes")
    below = scan_fixture(
        "Premier League", "epl-tot-cov-2026-10-19", "Home FC", "Away FC",
        markets, roles, odds, "2026-10-19",
        kickoff="2026-10-19T19:00:00Z",
    )
    assert below["sinal"] is True
    assert below["token"] == "from-clob"
    assert below["kickoff"] == "2026-10-19T19:00:00Z"

    # endDate passado como kickoff não pode vencer um gameStartTime real.
    forced_end = scan_fixture(
        "Premier League", "epl-tot-cov-2026-10-19", "Home FC", "Away FC",
        markets, roles, odds, "2026-10-19",
        kickoff="2026-10-20T23:00:00Z",
    )
    assert forced_end["kickoff"] == "2026-10-19T19:00:00Z"
    assert forced_end["pick"] == "casa"
    assert forced_end["token"] == "from-clob"


def test_token_and_kickoff_survive_csv_roundtrip(tmp_path):
    path = tmp_path / "paper_rn1_forward.csv"
    frame = pd.DataFrame([_base_row(token=TOKEN, kickoff=KICKOFF)])
    frame.to_csv(path, index=False)
    out = ensure_log(path)
    assert out.loc[0, "token"] == TOKEN
    assert out.loc[0, "kickoff"] == KICKOFF
    assert "e" not in out.loc[0, "token"].lower()


def test_fixture_from_event_keeps_start_and_token():
    from monitor import _fixture_from_event, outcome_roles

    now = datetime(2026, 10, 19, 12, 0, tzinfo=timezone.utc)
    event = {
        "slug": "epl-tot-cov-2026-10-19",
        "live": False,
        "ended": False,
        "startTime": "2026-10-19T19:00:00Z",
        "endDate": "2026-10-19T19:00:00Z",
        "markets": [
            {
                "groupItemTitle": "Tottenham Hotspur FC",
                "question": "Tottenham?",
                "bestAsk": 0.62,
                "volume": 20000,
                "acceptingOrders": True,
                "clobTokenIds": json.dumps([TOKEN, "no-home"]),
                "gameStartTime": "2026-10-19 19:00:00+00",
                "outcomePrices": '["0.60","0.40"]',
            },
            {
                "groupItemTitle": "Draw (Tottenham Hotspur FC vs. Coventry City FC)",
                "question": "Draw?",
                "bestAsk": 0.22,
                "volume": 20000,
                "acceptingOrders": True,
                "clobTokenIds": '["tok-draw","no-draw"]',
                "outcomePrices": '["0.22","0.78"]',
            },
            {
                "groupItemTitle": "Coventry City FC",
                "question": "Coventry?",
                "bestAsk": 0.20,
                "volume": 20000,
                "acceptingOrders": True,
                "clobTokenIds": '["tok-away","no-away"]',
                "outcomePrices": '["0.18","0.82"]',
            },
        ],
    }
    fixture = _fixture_from_event(event, "Premier League", now.date(), now)
    assert fixture is not None
    assert fixture["kickoff"] == "2026-10-19T19:00:00Z"
    assert fixture["markets"][0]["token_yes"] == TOKEN
    roles = outcome_roles(fixture["markets"], "Tottenham Hotspur FC", "Coventry City FC")
    assert roles == ["casa", "empate", "fora"]
    odds = pd.Series({"ps_h": 1.50, "ps_d": 4.5, "ps_a": 7.0})
    scanned = scan_fixture(
        "Premier League", fixture["slug"], "Tottenham Hotspur FC", "Coventry City FC",
        fixture["markets"], roles, odds, "2026-10-19", kickoff=fixture["kickoff"],
    )
    assert scanned["token"] == TOKEN
    assert scanned["kickoff"] == "2026-10-19T19:00:00Z"


def test_scan_and_settle_use_role_names_when_markets_are_reversed():
    """Gamma pode devolver [fora, empate, casa]. Fair de casa não pode pegar o token do visitante."""
    odds = pd.Series({"ps_h": 1.50, "ps_d": 4.5, "ps_a": 7.0})
    markets = [
        {"title": "Away FC", "price": 0.14, "volume": 9000, "token_yes": "tok-away"},
        {"title": "Draw (Home FC vs. Away FC)", "price": 0.20, "volume": 9000, "token_yes": "tok-draw"},
        {
            "title": "Home FC",
            "price": 0.62,
            "volume": 9000,
            "token_yes": "tok-home",
            "gameStartTime": "2026-10-19 19:00:00+00",
            "endDate": "2026-10-21T00:00:00Z",
        },
    ]
    # Lista posicional errada, de propósito. O scan tem que reler os nomes.
    scanned = scan_fixture(
        "Premier League", "epl-home-away-2026-10-19", "Home FC", "Away FC",
        markets, ["casa", "empate", "fora"], odds, "2026-10-19",
    )
    assert scanned is not None
    assert scanned["pick"] == "casa"
    assert scanned["token"] == "tok-home"
    assert scanned["poly_exec"] == pytest.approx(0.62)
    assert scanned["kickoff"] == "2026-10-19T19:00:00Z"

    log = _float_nan_log(token=np.nan, pick="casa")
    log["token"] = log["token"].astype("float64")
    event = {
        "startTime": KICKOFF,
        "endDate": "2026-10-21T00:00:00Z",
        "markets": [
            {
                "groupItemTitle": "Sunderland",
                "outcomePrices": '["0","1"]',
                "closed": True,
                "clobTokenIds": '["tok-away","no-away"]',
                "gameStartTime": "2026-10-18 13:00:00+00",
            },
            {
                "groupItemTitle": "Draw (Bournemouth vs. Sunderland)",
                "outcomePrices": '["0","1"]',
                "closed": True,
                "clobTokenIds": '["tok-draw","no-draw"]',
            },
            {
                "groupItemTitle": "Bournemouth",
                "outcomePrices": '["1","0"]',
                "closed": True,
                "clobTokenIds": json.dumps([TOKEN, "no-home"]),
                "gameStartTime": "2026-10-18 13:00:00+00",
            },
        ],
    }
    out = _settle(log, event, close=0.50, pin=(1.6, 4.0, 6.0))
    assert out.loc[0, "token"] == TOKEN
    assert out.loc[0, "fill_simulado"] == True
    assert out.loc[0, "resultado"] == "acerto"
    home_close = float(market_probs(np.array([[1.6, 4.0, 6.0]], dtype=float))[0][0])
    assert out.loc[0, "clv_pin"] == pytest.approx(round(clv_probability(BID, home_close), 4))


def test_open_market_with_high_price_stays_pending():
    log = _float_nan_log()
    event = _event(["0.95", "0.04", "0.01"], closed=False)
    out = _settle(log, event, close=0.50)
    assert pd.isna(out.loc[0, "resultado"])
    assert pd.isna(out.loc[0, "fill_simulado"])
    assert pd.isna(out.loc[0, "retorno_sim"])


def test_scientific_notation_token_is_replaced_from_gamma():
    log = _float_nan_log(token="5.841395765088047e+76")
    out = _settle(log, _event(["1", "0", "0"], token=TOKEN), close=0.50)
    assert out.loc[0, "token"] == TOKEN
    assert out.loc[0, "fill_simulado"] == True
    assert out.loc[0, "resultado"] == "acerto"
