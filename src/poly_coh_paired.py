"""H-COH com eventos pareados (1X2 + O/U + margem) por data/slug."""

from __future__ import annotations

import pandas as pd

from backtest_markets import MODEL_PARAMS, PoissonModel, _over25
from src.hypothesis_engine import HOLDOUT_DAYS
from src.poly_coherence_backtest import _model_derived_probs, _poly_yes_price_at_kickoff
from src.poly_event_index import build_event_index, classify_event, event_date_key
from src.poly_maker_backtest import _parse_kickoff, fetch_closed_soccer_events, is_match_event, match_event_to_row
from src.poly_market_utils import parse_token_yes, pnl_viable, summarize_returns, yes_won
from src.polymarket_fees import net_return_taker
from src.polymarket_gamma_client import fetch_events_closed
from src.poisson_model import build_score_matrix


def fetch_soccer_universe(max_pages: int = 40) -> list[dict]:
    from src.polymarket_gamma_client import load_merged_cached_events

    cached = load_merged_cached_events()
    if cached:
        soccer = [e for e in cached if is_match_event(e)]
        ou_margin = [
            e for e in cached
            if classify_event(e) in ("ou", "margin")
            and event_date_key(e)
        ]
        return soccer + ou_margin

    tags = ["soccer", "epl", "bundesliga", "la-liga", "ligue-1", "premier-league"]
    seen: set[str] = set()
    out: list[dict] = []
    for tag in tags:
        for e in fetch_events_closed(tag_slug=tag, max_pages=max_pages):
            slug = e.get("slug")
            if slug in seen:
                continue
            seen.add(slug)
            out.append(e)
    return out


def run_coh_paired(df: pd.DataFrame, edge_min: float = 0.04, cache_only: bool = True) -> dict:
    events = fetch_soccer_universe()
    idx = build_event_index(events)
    holdout_cut = df.date.max() - pd.Timedelta(days=HOLDOUT_DAYS)
    teams_idx: dict[str, dict[str, str]] = {}
    trades: list[dict] = []
    pairs_found = 0
    model_cache: dict[str, PoissonModel] = {}

    for key, group in idx.items():
        kinds = {classify_event(e): e for e in group}
        if "1x2" not in kinds and not any(is_match_event(e) for e in group):
            continue
        ev_1x2 = kinds.get("1x2") or next((e for e in group if is_match_event(e)), None)
        ev_ou = kinds.get("ou")
        if ev_1x2 is None:
            continue
        if ev_ou is not None:
            pairs_found += 1
        kickoff = _parse_kickoff(ev_1x2)
        if kickoff is None:
            continue
        row = match_event_to_row(ev_1x2, df, teams_idx)
        if row is None:
            continue
        comp = row.competition
        sub = df[(df.competition == comp) & (df.date < row.date)]
        if len(sub) < 80:
            continue
        if comp not in model_cache:
            try:
                model_cache[comp] = PoissonModel(**MODEL_PARAMS).fit(sub)
            except Exception:
                continue
        try:
            pred = model_cache[comp].predict_match(row.home_team, row.away_team, neutral=False)
            lh, la = float(pred["expected_goals_home"]), float(pred["expected_goals_away"])
            rho = float(model_cache[comp]._rho)
        except Exception:
            lh, la, rho = 1.35, 1.15, 0.0
        mp = _model_derived_probs(lh, la, rho)

        # 1X2 legs
        for m in ev_1x2.get("markets") or []:
            _maybe_trade(m, kickoff, mp, edge_min, trades, leg="1x2", cache_only=cache_only)

        # O/U pareado
        if ev_ou:
            for m in ev_ou.get("markets") or []:
                q = (m.get("question") or "").lower()
                tok = parse_token_yes(m)
                if not tok:
                    continue
                p_poly = _poly_yes_price_at_kickoff(tok, kickoff, cache_only=cache_only)
                if p_poly is None:
                    continue
                yw = yes_won(m)
                if yw is None:
                    continue
                model_p = mp["over_2_5"] if "over" in q else (1.0 - mp["over_2_5"])
                edge = model_p - p_poly
                if edge < edge_min:
                    continue
                gross = (1.0 - p_poly) / p_poly if yw else -1.0
                net = net_return_taker(gross, p_poly, "sports")
                trades.append({"kickoff": kickoff, "net": net, "type": "ou_paired"})

        # Coerência estrutural: P(win by 2+) <= P(win)
        for e in group:
            if classify_event(e) != "margin":
                continue
            for m in e.get("markets") or []:
                _maybe_trade(m, kickoff, mp, edge_min, trades, leg="margin", cache_only=cache_only)

    trades.sort(key=lambda t: t["kickoff"])

    def _kt(kickoff) -> pd.Timestamp:
        kt = pd.Timestamp(kickoff)
        if kt.tzinfo is not None:
            kt = kt.tz_convert(None)
        return kt

    dev = [t for t in trades if _kt(t["kickoff"]) < holdout_cut]
    hold = [t for t in trades if _kt(t["kickoff"]) >= holdout_cut]
    hold_s = summarize_returns([t["net"] for t in hold])
    return {
        "hypothesis": "H-COH_paired",
        "pairs_1x2_ou": pairs_found,
        "n_trades": len(trades),
        "dev": summarize_returns([t["net"] for t in dev]),
        "holdout": hold_s,
        "pnl_viable_holdout": pnl_viable(hold_s),
    }


def _maybe_trade(m, kickoff, mp, edge_min, trades, leg, cache_only: bool = False):
    tok = parse_token_yes(m)
    if not tok:
        return
    p_poly = _poly_yes_price_at_kickoff(tok, kickoff, cache_only=cache_only)
    if p_poly is None:
        return
    yw = yes_won(m)
    if yw is None:
        return
    q = (m.get("question") or "").lower()
    if "draw" in q:
        model_p = mp["draw"]
    elif "over" in q:
        model_p = mp["over_2_5"]
    elif "2 goal" in q or "by 2" in q:
        model_p = mp["home_win_by_2plus"]
    else:
        model_p = mp["home_win"]
    edge = model_p - p_poly
    if edge < edge_min:
        return
    gross = (1.0 - p_poly) / p_poly if yw else -1.0
    net = net_return_taker(gross, p_poly, "sports")
    trades.append({"kickoff": kickoff, "net": net, "type": leg})
