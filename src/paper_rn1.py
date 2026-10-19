"""Forward test em papel: RN1 0.60–0.80, bid fair Pin − 2.5¢."""

from __future__ import annotations

import json
import re
import urllib.request
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from src.clv import clv_probability
from src.odds_utils import market_probs
from src.polymarket_prices import closing_price_before_kickoff, execution_price_yes, parse_token_ids

PAPER_LOG = Path(__file__).resolve().parent.parent / "data" / "paper_rn1_forward.csv"
FAIR_LO, FAIR_HI = 0.60, 0.80
OFFSET = 0.025
MIN_VOLUME = 1000.0
GAMMA_EVENTS = "https://gamma-api.polymarket.com/events"

COLS = [
    "data", "liga", "slug", "casa", "fora", "pick", "fair_pin", "bid", "poly_exec",
    "volume", "kickoff", "sinal", "fill_simulado", "resultado", "retorno_sim", "clv_pin", "token",
]

# Colunas que recebem bool/texto. Se nascerem float64 (CSV todo NaN), .at recusa False.
_FLEX_COLS = (
    "data", "liga", "slug", "casa", "fora", "pick", "kickoff",
    "sinal", "fill_simulado", "resultado", "token",
)
_SCI_TOKEN = re.compile(r"[+-]?(?:\d+\.?\d*|\.\d+)[eE][+-]?\d+")
_ROLE_ORDER = ("casa", "empate", "fora")

_TEXT_DTYPES = {
    "data": str,
    "liga": str,
    "slug": str,
    "casa": str,
    "fora": str,
    "pick": str,
    "kickoff": str,
    "resultado": str,
    "token": str,
}


def _blank(value) -> bool:
    if value is None:
        return True
    if isinstance(value, str):
        s = value.strip()
        return s == "" or s.lower() in {"nan", "none", "nat", "<na>"}
    try:
        if pd.isna(value):
            return True
    except (TypeError, ValueError):
        return False
    return False


def _is_scientific_token(value) -> bool:
    """Token lido como float vira `5.84e+76` e não serve no CLOB."""
    if _blank(value):
        return False
    return _SCI_TOKEN.fullmatch(str(value).strip()) is not None


def _token_ok(value) -> bool:
    return not _blank(value) and not _is_scientific_token(value)


def _clean_token(value) -> str | None:
    if not _token_ok(value):
        return None
    return str(value).strip()


def _is_true(value) -> bool:
    if value is True:
        return True
    if isinstance(value, str) and value.strip().lower() == "true":
        return True
    try:
        if pd.isna(value):
            return False
    except (TypeError, ValueError):
        pass
    return False


def coerce_log_dtypes(log: pd.DataFrame) -> pd.DataFrame:
    """Garante que bool/texto caibam numa coluna que o CSV leu como float64.

    `read_csv` tipa coluna toda vazia como float64. `log.at[i, col] = False`
    então levanta TypeError: Invalid value 'False' for dtype 'float64'.
    """
    for c in COLS:
        if c not in log.columns:
            log[c] = None
    for c in _FLEX_COLS:
        if c in log.columns and not pd.api.types.is_object_dtype(log[c]):
            log[c] = log[c].astype(object)
    for c in ("kickoff", "resultado", "pick", "slug"):
        if c not in log.columns:
            continue
        log[c] = log[c].map(lambda v: None if _blank(v) else str(v).strip())
    if "token" in log.columns:
        log["token"] = log["token"].map(_clean_token)
    return log


def ensure_log(path: Path = PAPER_LOG) -> pd.DataFrame:
    if path.exists():
        # token id do CLOB tem ~77 dígitos; sem dtype=str o pandas lê como float e corrompe.
        header = pd.read_csv(path, nrows=0).columns
        dtype = {c: _TEXT_DTYPES[c] for c in header if c in _TEXT_DTYPES}
        df = pd.read_csv(path, dtype=dtype or None)
    else:
        df = pd.DataFrame(columns=COLS)
    for c in COLS:
        if c not in df.columns:
            df[c] = None
    return coerce_log_dtypes(df)


def normalize_kickoff(raw) -> str:
    """ISO UTC (`...Z`). Aceita gameStartTime (`2026-10-19 19:00:00+00`)."""
    if _blank(raw):
        return ""
    s = str(raw).strip()
    try:
        dt = datetime.fromisoformat(s.replace("Z", "+00:00"))
    except ValueError:
        return ""
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    else:
        dt = dt.astimezone(timezone.utc)
    return dt.strftime("%Y-%m-%dT%H:%M:%SZ")


def resolve_token(market: dict | None) -> str | None:
    """YES token do outcome. O monitor simplifica o mercado para `token_yes`."""
    if not market:
        return None
    for key in ("token_yes", "token"):
        cleaned = _clean_token(market.get(key))
        if cleaned:
            return cleaned
    yes, _ = parse_token_ids(market)
    return _clean_token(yes)


def _iso_list(raws) -> list[str]:
    found: list[str] = []
    for raw in raws:
        iso = normalize_kickoff(raw)
        if iso and iso not in found:
            found.append(iso)
    return found


def resolve_kickoff(
    market: dict | None = None,
    event: dict | None = None,
    kickoff: str | None = None,
) -> str:
    """Horário de início. `endDate` só entra quando não há gameStartTime/startTime."""
    market = market or {}
    event = event or {}
    sources: list[dict] = []
    if market:
        sources.append(market)
    for extra in event.get("markets") or []:
        if isinstance(extra, dict) and extra not in sources:
            sources.append(extra)
    sources.append(event)
    starts: list = []
    ends: list = []
    for src in sources:
        for key in ("gameStartTime", "startTime", "kickoff"):
            starts.append(src.get(key))
        ends.append(src.get("endDate"))
    start_isos = _iso_list(starts)
    if start_isos:
        return start_isos[0]
    explicit = normalize_kickoff(kickoff)
    end_isos = _iso_list(ends)
    if explicit and explicit not in end_isos:
        return explicit
    if end_isos:
        return end_isos[0]
    return explicit


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
    kickoff: str | None = None,
) -> dict | None:
    if dataset_row is None:
        return None
    probs_pin = market_probs(dataset_row[["ps_h", "ps_d", "ps_a"]].to_numpy(dtype=float).reshape(1, -1))[0]
    fav_i = int(probs_pin.argmax())
    fair = float(probs_pin[fav_i])
    if not (FAIR_LO <= fair <= FAIR_HI):
        return None
    bid = round(fair - OFFSET, 4)
    # fair_i é casa/empate/fora. A lista da Gamma não segue essa ordem.
    # `roles` do chamador é ignorado de propósito: pode ter sido montado por posição.
    mapped = _outcome_roles(markets, home, away)
    if not mapped:
        return None
    label = _ROLE_ORDER[fav_i]
    if label not in mapped:
        return None
    m = markets[mapped.index(label)]
    exec_p = execution_price_yes(m) or m.get("price")
    vol = float(m.get("volume") or 0)
    if exec_p is None or vol < MIN_VOLUME:
        return None
    # sinal = preço já encostou no bid. Faixa 0.60–0.80 sozinha deixa sinal False.
    sinal = float(exec_p) <= bid + 0.01
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
        "kickoff": resolve_kickoff(m, kickoff=kickoff),
        "sinal": bool(sinal),
        "fill_simulado": None,
        "resultado": None,
        "retorno_sim": None,
        "clv_pin": None,
        "pick": label,
        "token": resolve_token(m),
    }


def _outcome_roles(markets: list[dict], home: str, away: str) -> list[str] | None:
    """Mesmo mapa do monitor: nome do mercado, não a posição na lista."""
    from monitor import outcome_roles

    return outcome_roles(markets, home, away)


def _needs_settle(row: pd.Series) -> bool:
    resultado = row.get("resultado")
    if _blank(resultado):
        return True
    # sem_fill gravado sem kickoff/token não foi medido: o preço de fechamento nem foi consultado.
    if str(resultado).strip() == "sem_fill" and (
        _blank(row.get("kickoff")) or not _token_ok(row.get("token"))
    ):
        return True
    return False


def _fetch_event(slug: str) -> dict | None:
    if _blank(slug):
        return None
    try:
        payload = json.load(
            urllib.request.urlopen(
                urllib.request.Request(
                    f"{GAMMA_EVENTS}?slug={slug}",
                    headers={"User-Agent": "Mozilla/5.0"},
                ),
                timeout=30,
            )
        )
    except Exception:
        return None
    if not payload:
        return None
    return payload[0]


def _yes_prices(markets: list[dict]) -> list[float]:
    prices = []
    for m in markets:
        try:
            prices.append(float(json.loads(m["outcomePrices"])[0]))
        except Exception:
            prices.append(0.0)
    return prices


def _clear_false_sem_fill(log: pd.DataFrame, i) -> None:
    log.at[i, "resultado"] = None
    log.at[i, "fill_simulado"] = None


def settle_paper(log: pd.DataFrame, hist_df: pd.DataFrame) -> pd.DataFrame:
    """Liquida jogos já resolvidos na Polymarket.

    Fill simulado: último preço CLOB do token até o kickoff ≤ bid.
    Sem kickoff, sem token ou sem histórico de preço, a linha fica pendente
    (não vira `sem_fill` — isso seria 'não deu para medir').
    Resultado final só com os três mercados `closed`, como `monitor.settle`.
    Preço ≥ 0.9 no meio do jogo não liquida.
    """
    log = coerce_log_dtypes(log)
    pick_idx = {"casa": 0, "empate": 1, "fora": 2}
    pending = [i for i, r in log.iterrows() if _needs_settle(r)]
    for i in pending:
        r = log.loc[i]
        invalid_sem_fill = str(r.get("resultado") or "").strip() == "sem_fill"
        ev = _fetch_event(str(r.get("slug") or ""))
        if not ev:
            continue
        # sem_fill sem kickoff/token não foi medido. Tira o rótulo antes de decidir de novo;
        # se o CLOB ou o mercado ainda não derem para liquidar, a linha fica pendente.
        if invalid_sem_fill:
            _clear_false_sem_fill(log, i)
        ms = ev.get("markets") or []
        home = str(r.get("casa") or "")
        away = str(r.get("fora") or "")
        pick = str(r.get("pick") or "").strip()
        mapped = _outcome_roles(ms, home, away) if len(ms) == 3 else None
        picked = ms[mapped.index(pick)] if mapped and pick in mapped else {}
        if not _token_ok(log.at[i, "token"]) and picked:
            tok = resolve_token(picked)
            if tok:
                log.at[i, "token"] = tok
        if _blank(log.at[i, "kickoff"]):
            ko_s = resolve_kickoff(picked or None, event=ev)
            if ko_s:
                log.at[i, "kickoff"] = ko_s
        token = _clean_token(log.at[i, "token"])
        kickoff = "" if _blank(log.at[i, "kickoff"]) else str(log.at[i, "kickoff"])

        if (
            len(ms) != 3
            or mapped is None
            or pick not in mapped
            or not all(m.get("closed") for m in ms)
        ):
            continue
        prices = _yes_prices(ms)
        if max(prices) < 0.9:
            continue
        if not token or not kickoff:
            continue
        try:
            ko = datetime.fromisoformat(kickoff.replace("Z", "+00:00"))
        except ValueError:
            continue
        if ko.tzinfo is None:
            ko = ko.replace(tzinfo=timezone.utc)
        if ko > datetime.now(timezone.utc):
            continue
        try:
            bid = float(r["bid"])
        except (TypeError, ValueError):
            continue
        close_p = closing_price_before_kickoff(token, ko)
        if close_p is None:
            # CLOB sem histórico: não inventar sem_fill.
            continue
        winner = mapped[prices.index(max(prices))]
        acertou = winner == pick
        if close_p > bid:
            log.at[i, "fill_simulado"] = False
            log.at[i, "resultado"] = "sem_fill"
            continue
        log.at[i, "fill_simulado"] = True
        log.at[i, "resultado"] = "acerto" if acertou else "erro"
        log.at[i, "retorno_sim"] = round((1.0 - bid) / bid if acertou else -1.0, 4)
        pi = pick_idx.get(pick)
        pin = None
        if pi is not None:
            try:
                from src.pinnacle_lookup import lookup_1x2_close

                pin = lookup_1x2_close(
                    hist_df, r["liga"], r["casa"], r["fora"], kickoff,
                )
            except Exception:
                pin = None
        if pin and pi is not None:
            p_devig = market_probs(np.array([[pin[0], pin[1], pin[2]]], dtype=float))[0][pi]
            log.at[i, "clv_pin"] = round(clv_probability(bid, float(p_devig)), 4)
    return log


def summarize_paper(log: pd.DataFrame) -> dict:
    """`resolvidos` inclui `sem_fill`. PnL só entra em fill de verdade (acerto/erro)."""
    if log is None or log.empty:
        return {
            "registros": 0, "resolvidos": 0, "fills": 0, "sem_fill": 0,
            "pendentes": 0, "retorno": 0.0, "hit": float("nan"),
        }
    work = log
    if "resultado" not in work.columns:
        work = coerce_log_dtypes(work.copy())
    resolved_mask = work["resultado"].map(lambda v: not _blank(v))
    resolvidos = work[resolved_mask]
    fills = int(work["fill_simulado"].map(_is_true).sum()) if "fill_simulado" in work.columns else 0
    bets = resolvidos[resolvidos["resultado"].isin(["acerto", "erro"])] if len(resolvidos) else resolvidos
    retorno = float(pd.to_numeric(bets["retorno_sim"], errors="coerce").sum()) if len(bets) else 0.0
    hit = float((bets["resultado"] == "acerto").mean()) if len(bets) else float("nan")
    return {
        "registros": int(len(work)),
        "resolvidos": int(resolved_mask.sum()),
        "fills": fills,
        "sem_fill": int((resolvidos["resultado"] == "sem_fill").sum()) if len(resolvidos) else 0,
        "pendentes": int((~resolved_mask).sum()),
        "retorno": retorno,
        "hit": hit,
    }
