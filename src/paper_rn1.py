"""Forward test em papel: RN1 0.60–0.80, bid fair Pin − 2.5¢.

`fill_simulado` continua a regra antiga: último preço CLOB até o kickoff <= bid.
Os três modelos novos (touch, queue, through) são simulação em cima da fita
de livro + trades e não substituem essa coluna.
"""

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
    "placed_at", "condition_id", "queue_ahead", "our_size", "filled_size", "fill_model",
    "filled_touch", "filled_queue", "filled_through",
    "retorno_touch", "retorno_queue", "retorno_through", "fill_coverage",
    "sim_price", "tick_size", "token_no", "entry_cross", "tick_known",
]

# Colunas que recebem bool/texto. Se nascerem float64 (CSV todo NaN), .at recusa False.
_FLEX_COLS = (
    "data", "liga", "slug", "casa", "fora", "pick", "kickoff",
    "sinal", "fill_simulado", "resultado", "token",
    "placed_at", "condition_id", "fill_model", "fill_coverage",
    "token_no", "entry_cross", "tick_known",
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
    "placed_at": str,
    "condition_id": str,
    "fill_model": str,
    "fill_coverage": str,
    "token_no": str,
    "entry_cross": str,
    "tick_known": str,
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
    if value is False:
        return False
    # pandas guarda bool como np.True_. Isso não é `is True`, mas é verdadeiro.
    if isinstance(value, (bool, np.bool_)):
        return bool(value)
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


def _condition_id(market: dict | None) -> str | None:
    if not market:
        return None
    raw = market.get("conditionId") or market.get("condition_id")
    if _blank(raw):
        return None
    text = str(raw).strip()
    return text or None


def _token_no(market: dict | None) -> str | None:
    if not market:
        return None
    cleaned = _clean_token(market.get("token_no"))
    if cleaned:
        return cleaned
    _, no = parse_token_ids(market)
    return _clean_token(no)


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
        "token_no": _token_no(m),
        "condition_id": _condition_id(m),
        "placed_at": None,
        "sim_price": None,
        "entry_cross": None,
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


def _cell_number(value) -> float | None:
    if _blank(value):
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    if number != number:
        return None
    return number


def _mark_fill_unmeasured(log: pd.DataFrame, i) -> None:
    """Legado não entra na taxa: sem fill, cobertura incompleta, não congelada."""
    if "fill_coverage" in log.columns:
        log.at[i, "fill_coverage"] = "incomplete"
    for name in ("touch", "queue", "through"):
        if f"filled_{name}" in log.columns:
            log.at[i, f"filled_{name}"] = None
        if f"retorno_{name}" in log.columns:
            log.at[i, f"retorno_{name}"] = None
    if "filled_size" in log.columns:
        log.at[i, "filled_size"] = None
    if "fill_model" in log.columns:
        log.at[i, "fill_model"] = None
    if "entry_cross" in log.columns:
        log.at[i, "entry_cross"] = False
    if "queue_ahead" in log.columns:
        log.at[i, "queue_ahead"] = None


def _apply_fill_tape(log: pd.DataFrame, i, tape: dict | None, *, bid: float, won: bool) -> None:
    """Escreve os três modelos quando a fita cobre a janela inteira. Fita cortada não vira zero."""
    if not tape:
        return
    from src.paper_rn1_tape import _stored_flag, iso_to_ts
    from src.queue_fill import estimate_fills, simulated_return

    our = _cell_number(tape.get("our_size"))
    if our is not None and "our_size" in log.columns:
        log.at[i, "our_size"] = our
    sim = _cell_number(tape.get("sim_price"))
    known = _stored_flag(tape.get("tick_known"))
    # tick ausente, NaN ou preço fora da grade: a linha antiga não é medida,
    # mesmo que a cobertura tenha sido gravada como complete.
    if bool(tape.get("legacy")) or known is not True or sim is None:
        _mark_fill_unmeasured(log, i)
        return
    coverage = str(tape.get("coverage") or "pending")
    log.at[i, "fill_coverage"] = coverage
    observed = bool(tape.get("book_observed"))
    queue_ahead = _cell_number(tape.get("queue_ahead")) if observed else None
    if observed and queue_ahead is not None:
        log.at[i, "queue_ahead"] = queue_ahead
    if "sim_price" in log.columns:
        log.at[i, "sim_price"] = sim
    marketable = _is_true(tape.get("entry_cross"))
    if marketable and "entry_cross" in log.columns:
        log.at[i, "entry_cross"] = True
    if coverage != "complete" and not marketable:
        return
    placed_ts = iso_to_ts(tape.get("placed_at") or log.at[i, "placed_at"])
    kickoff_ts = iso_to_ts(tape.get("kickoff") or log.at[i, "kickoff"])
    if placed_ts is None or our is None:
        return
    price = sim if sim is not None else bid
    yes_token = tape.get("token")
    no_token = tape.get("token_no")
    if _blank(yes_token):
        yes_token = None
    if _blank(no_token):
        no_token = None
    estimates = estimate_fills(
        price=price,
        our_size=our,
        queue_ahead=queue_ahead,
        trades=list(tape.get("trades") or []),
        placed_ts=placed_ts,
        kickoff_ts=kickoff_ts,
        yes_token=None if yes_token is None else str(yes_token),
        no_token=None if no_token is None else str(no_token),
        tick_size=_cell_number(tape.get("tick_size")),
        marketable=marketable,
    )
    for name in ("touch", "queue", "through"):
        estimate = estimates[name]
        filled = estimate.filled_size if estimate.measured else None
        log.at[i, f"filled_{name}"] = filled
        log.at[i, f"retorno_{name}"] = simulated_return(price, won, filled, our)
    if marketable and estimates["queue"].measured:
        log.at[i, "filled_size"] = estimates["queue"].filled_size
        log.at[i, "fill_model"] = "cross"
    elif estimates["queue"].measured:
        log.at[i, "filled_size"] = estimates["queue"].filled_size
        log.at[i, "fill_model"] = "queue"


def settle_paper(
    log: pd.DataFrame,
    hist_df: pd.DataFrame,
    tapes: dict | None = None,
    tape_dir: Path | None = None,
) -> pd.DataFrame:
    """Liquida jogos já resolvidos na Polymarket.

    Fill simulado (regra antiga): último preço CLOB do token até o kickoff ≤ bid.
    Sem kickoff, sem token ou sem histórico de preço, a linha fica pendente
    (não vira `sem_fill` — isso seria 'não deu para medir').
    Resultado final só com os três mercados `closed`, como `monitor.settle`.
    Preço ≥ 0.9 no meio do jogo não liquida.

    Se houver fita completa (livro na entrada + trades da janela), grava também
    filled_touch / filled_queue / filled_through. Fita incompleta não vira fill zero.
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
        if _blank(log.at[i, "condition_id"]) and picked:
            cid = _condition_id(picked)
            if cid:
                log.at[i, "condition_id"] = cid
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
        winner = mapped[prices.index(max(prices))]
        acertou = winner == pick
        _apply_fill_tape(
            log,
            i,
            _tape_for_slug(str(r.get("slug") or ""), tapes, tape_dir),
            bid=bid,
            won=acertou,
        )
        if close_p is None:
            # CLOB sem histórico: não inventar sem_fill.
            continue
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
    _backfill_fill_models(log, tapes, tape_dir)
    return log


def _pick_won_after_settle(row: pd.Series) -> bool | None:
    """acerto/erro já dizem o vencedor. sem_fill precisa reler o evento, sem mexer no resultado."""
    resultado = str(row.get("resultado") or "").strip()
    if resultado == "acerto":
        return True
    if resultado == "erro":
        return False
    if resultado != "sem_fill":
        return None
    ev = _fetch_event(str(row.get("slug") or ""))
    if not ev:
        return None
    markets = ev.get("markets") or []
    home = str(row.get("casa") or "")
    away = str(row.get("fora") or "")
    pick = str(row.get("pick") or "").strip()
    mapped = _outcome_roles(markets, home, away) if len(markets) == 3 else None
    if mapped is None or pick not in mapped or not all(m.get("closed") for m in markets):
        return None
    prices = _yes_prices(markets)
    if max(prices) < 0.9:
        return None
    return mapped[prices.index(max(prices))] == pick


def _backfill_fill_models(log: pd.DataFrame, tapes: dict | None, tape_dir: Path | None) -> None:
    """Linha já liquidada pela regra antiga ainda recebe a fila quando a fita completa chega."""
    for i, row in log.iterrows():
        resultado = str(row.get("resultado") or "").strip()
        if resultado not in {"acerto", "erro", "sem_fill"}:
            continue
        if str(row.get("fill_coverage") or "").strip() == "complete" and _cell_number(row.get("filled_touch")) is not None:
            continue
        tape = _tape_for_slug(str(row.get("slug") or ""), tapes, tape_dir)
        if not tape or str(tape.get("coverage") or "") != "complete":
            continue
        won = _pick_won_after_settle(row)
        if won is None:
            continue
        try:
            bid = float(row["bid"])
        except (TypeError, ValueError):
            continue
        _apply_fill_tape(log, i, tape, bid=bid, won=won)


def _tape_for_slug(slug: str, tapes: dict | None, tape_dir: Path | None) -> dict | None:
    if tapes is not None:
        return tapes.get(slug)
    from src.paper_rn1_tape import TAPE_DIR, load_tape

    directory = TAPE_DIR if tape_dir is None else tape_dir
    if not Path(directory).exists():
        return None
    return load_tape(slug, directory)


def _model_block() -> dict:
    return {
        "medidos": 0,
        "com_fill": 0,
        "fill_rate": None,
        "fill_rate_tamanho": None,
        "pnl_simulado": None,
    }


def summarize_fill_models(log: pd.DataFrame) -> dict:
    """Taxa de fill e PnL simulado por modelo.

    Um modelo só entra na taxa quando `filled_*` está preenchido e a fita está
    `complete`, ou quando a entrada já cruzou o ask (`entry_cross`). Cobertura
    `partial` / `pending`, ou `complete` com fill vazio, conta como não medido.
    """
    empty = {
        "touch": _model_block(),
        "queue": _model_block(),
        "through": _model_block(),
        "nao_medidos": 0,
        "sinais": 0,
    }
    if log is None or log.empty or "sinal" not in log.columns:
        return empty
    signals = log[log["sinal"].map(_is_true)]
    empty["sinais"] = int(len(signals))
    if signals.empty:
        return empty
    if "fill_coverage" in signals.columns:
        complete = signals["fill_coverage"].map(lambda v: str(v).strip() == "complete")
    else:
        complete = pd.Series(False, index=signals.index)
    if "entry_cross" in signals.columns:
        crossed = signals["entry_cross"].map(_is_true)
    else:
        crossed = pd.Series(False, index=signals.index)
    eligible = complete | crossed
    queue_measured = pd.Series(False, index=signals.index)
    for name in ("touch", "queue", "through"):
        col = f"filled_{name}"
        ret_col = f"retorno_{name}"
        block = _model_block()
        if col not in signals.columns:
            empty[name] = block
            continue
        filled = signals[col].map(_cell_number)
        usable_mask = filled.notna() & eligible
        if name == "queue":
            queue_measured = usable_mask
        usable = signals[usable_mask]
        filled = filled[usable_mask]
        block["medidos"] = int(len(filled))
        if filled.empty:
            empty[name] = block
            continue
        block["com_fill"] = int((filled > 0).sum())
        block["fill_rate"] = block["com_fill"] / block["medidos"]
        our = usable["our_size"].map(_cell_number) if "our_size" in usable.columns else None
        our_sum = float(our.sum()) if our is not None else 0.0
        if our_sum > 0:
            block["fill_rate_tamanho"] = float(filled.sum()) / our_sum
        if ret_col in usable.columns:
            pnl = usable[ret_col].map(_cell_number)
            block["pnl_simulado"] = float(pnl.dropna().sum()) if pnl.notna().any() else 0.0
        else:
            block["pnl_simulado"] = 0.0
        empty[name] = block
    empty["nao_medidos"] = int((~queue_measured).sum())
    return empty


def format_fill_model_report(stats: dict) -> str:
    """Texto do relatório. Tudo aqui é simulado; o que não foi medido não vira porcentagem."""
    from src.queue_fill import ASSUMED_STAKE_USD

    labels = {
        "touch": "touch (otimista: fluxo taker que alcança o bid enche a ordem, sem fila)",
        "queue": "queue (central: volume a preço ≤ bid precisa passar da fila; print abaixo do bid esvazia o nível)",
        "through": "through (pessimista: só enche se o preço negociar estritamente abaixo do bid)",
    }
    lines = [
        "Fill simulado (nenhuma ordem enviada; taxa de fill e PnL abaixo são simulação, não execução):",
    ]
    for key, label in labels.items():
        block = (stats or {}).get(key) or _model_block()
        medidos = int(block.get("medidos") or 0)
        if medidos <= 0 or block.get("fill_rate") is None:
            lines.append(f"  {label}: não medido")
            continue
        size_rate = block.get("fill_rate_tamanho")
        size_txt = "não medido" if size_rate is None else f"{size_rate:.0%}"
        pnl = block.get("pnl_simulado")
        pnl_txt = "não medido" if pnl is None else f"{pnl:+.2f}u"
        lines.append(
            f"  {label}: fill {block['fill_rate']:.0%} dos sinais medidos "
            f"({block['com_fill']}/{medidos}); "
            f"fill ponderado por tamanho {size_txt}; "
            f"PnL simulado {pnl_txt}"
        )
    nao = int((stats or {}).get("nao_medidos") or 0)
    if nao:
        lines.append(
            f"  {nao} sinal(is) sem fita completa até o kickoff: não medido "
            "(a data-api pode não devolver trades até o horário do sinal)."
        )
    lines.append(
        "  A coluna fill_simulado continua sendo o último preço CLOB ≤ bid cru, separada destes três modelos. "
        "O retorno dela usa esse bid; os três modelos usam o preço simulado na grade."
    )
    lines.append(
        "  Limitação: a fila é o tamanho no livro no preço simulado (bid descido ao tick_size) no instante "
        "do sinal. Sem tick no livro a fila fica não medida, não zero. Os trades são a janela pública da "
        "data-api entre a entrada e o kickoff, com takerOnly: venda YES a preço ≤ bid e compra NO a preço "
        "≥ 1−bid (mint, shares 1:1; cada taker do mesmo negócio soma). A fita só congela depois de uma "
        "leitura limpa feita no mínimo uma hora após o kickoff. Fita incompleta para de ser consultada "
        "depois de algumas tentativas ou 48h após o kickoff e fica não medido. Erro, 429 ou teto de offset "
        "também deixam o fill não medido — não colocamos zero no lugar. Página vazia é fim de janela, não "
        "corte. Uma linha de CSV sem tick_known confirmado ou sem sim_price é legado: não medida, "
        "sem fill no bid cru e sem congelar. O monitor grava o livro na entrada e acumula trades numa consulta por rodada. "
        "Fill parcial conta como fill na taxa por sinal; a taxa ponderada por tamanho está ao lado."
    )
    lines.append(
        f"  Tamanho da ordem: suposição de US$ {ASSUMED_STAKE_USD:.0f} de stake "
        "(não é tamanho medido); shares = stake / preço simulado."
    )
    return "\n".join(lines)


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
