"""Forward test em papel: RN1 0.60–0.80, bid fair Pin − 2.5¢."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from src.clv import clv_probability
from src.odds_utils import market_probs
from src.polymarket_prices import closing_price_before_kickoff, execution_price_yes, parse_token_ids

PAPER_LOG = Path(__file__).resolve().parent.parent / "data" / "paper_rn1_forward.csv"
FAIR_LO, FAIR_HI = 0.60, 0.80
OFFSET = 0.025
MIN_VOLUME = 1000.0

COLS = [
    "data", "liga", "slug", "casa", "fora", "pick", "fair_pin", "bid", "poly_exec",
    "volume", "kickoff", "sinal", "fill_simulado", "resultado", "retorno_sim", "clv_pin", "token",
]


def ensure_log(path: Path = PAPER_LOG) -> pd.DataFrame:
    if path.exists():
        df = pd.read_csv(path)
    else:
        df = pd.DataFrame(columns=COLS)
    for c in COLS:
        if c not in df.columns:
            df[c] = None
    return df


def pin_fair_for_fixture(row: pd.Series, pick_idx: int) -> float | None:
    if row[["ps_h", "ps_d", "ps_a"]].isna().any():
        return None
    p = market_probs(row[["ps_h", "ps_d", "ps_a"]].to_numpy(dtype=float).reshape(1, -1))[0]
    return float(p[pick_idx])


def scan_fixture(
    comp: str,
    slug: str,
    home: str,
    away: str,
    markets: list[dict],
    roles: list[str],
    dataset_row: pd.Series | None,
    today: str,
) -> dict | None:
    if dataset_row is None:
        return None
    pick_idx = {"casa": 0, "empate": 1, "fora": 2}
    probs_pin = market_probs(dataset_row[["ps_h", "ps_d", "ps_a"]].to_numpy(dtype=float).reshape(1, -1))[0]
    fav_i = int(probs_pin.argmax())
    fair = float(probs_pin[fav_i])
    if not (FAIR_LO <= fair <= FAIR_HI):
        return None
    bid = round(fair - OFFSET, 4)
    mi = fav_i
    m = markets[mi]
    exec_p = execution_price_yes(m) or m.get("price")
    vol = float(m.get("volume") or 0)
    if exec_p is None or vol < MIN_VOLUME:
        return None
    label = roles[mi]
    sinal = exec_p <= bid + 0.01  # preço atual próximo ou abaixo do bid
    return {
        "data": today,
        "liga": comp,
        "slug": slug,
        "casa": home,
        "fora": away,
        "fair_pin": round(fair, 4),
        "bid": bid,
        "poly_exec": round(float(exec_p), 4),
        "volume": round(vol),
        "kickoff": m.get("endDate") or "",
        "sinal": bool(sinal),
        "fill_simulado": None,
        "resultado": None,
        "retorno_sim": None,
        "clv_pin": None,
        "pick": label,
        "token": parse_token_ids(m)[0],
    }


def settle_paper(log: pd.DataFrame, hist_df: pd.DataFrame) -> pd.DataFrame:
    pick_idx = {"casa": 0, "empate": 1, "fora": 2}
    for i, r in log[log["resultado"].isna()].iterrows():
        try:
            import urllib.request
            ev = json.load(
                urllib.request.urlopen(
                    urllib.request.Request(
                        f"https://gamma-api.polymarket.com/events?slug={r['slug']}",
                        headers={"User-Agent": "Mozilla/5.0"},
                    ),
                    timeout=30,
                )
            )
        except Exception:
            continue
        if not ev:
            continue
        ms = ev[0].get("markets", [])
        if len(ms) != 3:
            continue
        prices = []
        for m in ms:
            try:
                prices.append(float(json.loads(m["outcomePrices"])[0]))
            except Exception:
                prices.append(0.0)
        if max(prices) < 0.9:
            continue
        # roles from pick
        pi = pick_idx.get(r.get("pick") or r.get("pick_idx"), 0)
        winner_idx = prices.index(max(prices))
        acertou = winner_idx == pi
        bid = float(r["bid"])
        kickoff = r.get("kickoff")
        fill_sim = False
        token = r.get("token")
        if token and kickoff:
            try:
                ko = datetime.fromisoformat(str(kickoff).replace("Z", "+00:00"))
                close_p = closing_price_before_kickoff(str(token), ko)
                if close_p is not None and close_p <= bid:
                    fill_sim = True
            except Exception:
                pass
        if not fill_sim:
            log.at[i, "fill_simulado"] = False
            log.at[i, "resultado"] = "sem_fill"
            continue
        log.at[i, "fill_simulado"] = True
        log.at[i, "resultado"] = "acerto" if acertou else "erro"
        log.at[i, "retorno_sim"] = round((1.0 - bid) / bid if acertou else -1.0, 4)
        from src.pinnacle_lookup import lookup_1x2_close
        pin = lookup_1x2_close(hist_df, r["liga"], r["casa"], r["fora"], str(r.get("kickoff") or ""))
        if pin:
            p_devig = market_probs([[pin[0], pin[1], pin[2]]])[0][pi]
            log.at[i, "clv_pin"] = round(clv_probability(bid, float(p_devig)), 4)
    return log
