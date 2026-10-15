"""H-COH: coerência Dixon-Coles entre mercados de futebol na Polymarket."""

from __future__ import annotations

import json
import re
from datetime import datetime, timezone

import numpy as np
import pandas as pd

from backtest_markets import MODEL_PARAMS, PoissonModel, _over25
from src.hypothesis_engine import HOLDOUT_DAYS
from src.poly_market_utils import (
    parse_token_yes,
    pnl_viable,
    price_history,
    summarize_returns,
    yes_won,
)
from src.polymarket_fees import net_return_taker
from src.poly_maker_backtest import (
    fetch_closed_soccer_events,
    match_event_to_row,
    is_match_event,
)
from src.polymarket_gamma_client import fetch_events_closed
from src.poisson_model import build_score_matrix


def _model_derived_probs(lh: float, la: float, rho: float) -> dict[str, float]:
    m = build_score_matrix(float(lh), float(la), rho=rho)
    n = m.shape[0]
    p_home = sum(m[i, j] for i in range(n) for j in range(n) if i > j)
    p_draw = sum(m[i, i] for i in range(n))
    p_away = sum(m[i, j] for i in range(n) for j in range(n) if i < j)
    p_home_2plus = sum(m[i, j] for i in range(n) for j in range(n) if i - j >= 2)
    p_over25 = float(_over25(np.array([lh]), np.array([la]), rho)[0])
    return {
        "home_win": float(p_home),
        "draw": float(p_draw),
        "away_win": float(p_away),
        "home_win_by_2plus": float(p_home_2plus),
        "over_2_5": p_over25,
    }


def _norm_teams(title: str) -> tuple[str, str] | None:
    t = title.lower()
    m = re.search(r"([\w\s]+?)\s+vs\.?\s+([\w\s]+)", t)
    if m:
        return m.group(1).strip(), m.group(2).strip()
    return None


def _fetch_auxiliary_soccer_markets(max_pages: int = 25) -> list[dict]:
    out: list[dict] = []
    for tag in ("soccer", "epl", "bundesliga"):
        ev = fetch_events_closed(tag_slug=tag, max_pages=max_pages)
        for e in ev:
            title = (e.get("title") or "").lower()
            if any(k in title for k in ("2 goals", "3 goals", "by 2", "win by", "over", "under", "o/u")):
                out.append(e)
            for m in e.get("markets") or []:
                q = (m.get("question") or "").lower()
                if any(k in q for k in ("2 goals", "by 2", "over 2.5", "under 2.5", "total goals")):
                    out.append(e)
                    break
    return out


def _poly_yes_price_at_kickoff(token: str, kickoff: datetime) -> float | None:
    if kickoff.tzinfo is None:
        kickoff = kickoff.replace(tzinfo=timezone.utc)
    end_ts = int(kickoff.timestamp())
    hist = price_history(token, end_ts - 72 * 3600, end_ts)
    if not hist:
        return None
    last = max(hist, key=lambda h: h.get("t", 0))
    return float(last.get("p", 0))


def run_coherence_backtest(df: pd.DataFrame, edge_min: float = 0.05) -> dict:
    """Taker nas pernas subprecificadas vs Dixon-Coles (λ do Poisson no dataset)."""
    events_1x2 = fetch_closed_soccer_events(["epl", "bundesliga", "soccer"], max_per_tag=50)
    events_1x2 = [e for e in events_1x2 if is_match_event(e)]
    aux = _fetch_auxiliary_soccer_markets()
    aux_by_key: dict[str, list[dict]] = {}
    for e in aux:
        pair = _norm_teams(e.get("title") or "")
        if pair:
            key = f"{pair[0]}|{pair[1]}|{(e.get('endDate') or '')[:10]}"
            aux_by_key.setdefault(key, []).append(e)

    holdout_cut = df.date.max() - pd.Timedelta(days=HOLDOUT_DAYS)
    teams_idx: dict[str, dict[str, str]] = {}
    trades: list[dict] = []
    violations: list[dict] = []

    for event in events_1x2:
        kickoff = _parse_kickoff_ev(event)
        if kickoff is None:
            continue
        row = match_event_to_row(event, df, teams_idx)
        if row is None:
            continue
        try:
            sub = df[(df.competition == row.competition) & (df.date < row.date)]
            rho = PoissonModel(**MODEL_PARAMS).fit(sub)._rho if len(sub) > 100 else 0.0
        except Exception:
            rho = 0.0
        try:
            pred = PoissonModel(**MODEL_PARAMS).fit(
                df[(df.competition == row.competition) & (df.date < row.date)]
            ).predict_match(row.home_team, row.away_team, neutral=False)
            lh = float(pred["expected_goals_home"])
            la = float(pred["expected_goals_away"])
        except Exception:
            lh, la = 1.35, 1.15
        mp = _model_derived_probs(lh, la, rho)

        markets = event.get("markets") or []
        poly_1x2 = {}
        for m in markets:
            tok = parse_token_yes(m)
            if not tok:
                continue
            p = _poly_yes_price_at_kickoff(tok, kickoff)
            if p is None:
                continue
            q = (m.get("groupItemTitle") or m.get("question") or "").lower()
            if "draw" in q:
                poly_1x2["draw"] = (p, m, tok)
            elif "beat" in q or "win" in q:
                poly_1x2.setdefault("home", (p, m, tok))

        # regra estrutural: P(vitória 2+) ≤ P(vitória)
        if poly_1x2.get("home"):
            p_hw, _, _ = poly_1x2["home"]
            if mp["home_win_by_2plus"] > mp["home_win"] + 0.001:
                violations.append({"type": "model_internal", "slug": event.get("slug")})

        pair = _norm_teams(event.get("title") or f"{row.home_team} vs {row.away_team}")
        key = f"{pair[0]}|{pair[1]}|{str(kickoff.date())}" if pair else ""
        for aux_e in aux_by_key.get(key, []):
            for m in aux_e.get("markets") or []:
                q = (m.get("question") or "").lower()
                tok = parse_token_yes(m)
                if not tok:
                    continue
                p_poly = _poly_yes_price_at_kickoff(tok, kickoff)
                if p_poly is None:
                    continue
                yw = yes_won(m)
                if yw is None:
                    continue
                model_p = None
                if "2 goal" in q or "by 2" in q:
                    model_p = mp["home_win_by_2plus"]
                elif "over 2.5" in q or "over 2" in q:
                    model_p = mp["over_2_5"]
                if model_p is None:
                    continue
                edge = model_p - p_poly
                if edge < edge_min:
                    continue
                gross = (1.0 - p_poly) / p_poly if yw else -1.0
                net = net_return_taker(gross, p_poly, "sports")
                trades.append({
                    "date": str(kickoff.date()),
                    "kickoff": kickoff,
                    "edge": edge,
                    "net": net,
                    "type": q[:40],
                })

        # 1X2 vs Dixon-Coles (todas as pernas com preço Poly)
        role_map = [("home", "home_win"), ("draw", "draw"), ("away", "away_win")]
        for mi, m in enumerate(markets):
            tok = parse_token_yes(m)
            if not tok:
                continue
            p_poly = _poly_yes_price_at_kickoff(tok, kickoff)
            if p_poly is None:
                continue
            yw = yes_won(m)
            if yw is None:
                continue
            title = (m.get("groupItemTitle") or m.get("question") or "").lower()
            if "draw" in title:
                model_p = mp["draw"]
                leg = "draw"
            elif row.home_team.lower()[:4] in title or title[:8] in row.home_team.lower():
                model_p = mp["home_win"]
                leg = "home"
            else:
                model_p = mp["away_win"]
                leg = "away"
            edge = model_p - p_poly
            if edge < edge_min:
                continue
            gross = (1.0 - p_poly) / p_poly if yw else -1.0
            net = net_return_taker(gross, p_poly, "sports")
            trades.append({
                "date": str(kickoff.date()),
                "kickoff": kickoff,
                "edge": edge,
                "net": net,
                "type": f"1x2_{leg}",
            })

    trades.sort(key=lambda t: t["date"])
    dev = [t for t in trades if pd.Timestamp(t["kickoff"]).tz_localize(None) < holdout_cut]
    hold = [t for t in trades if pd.Timestamp(t["kickoff"]).tz_localize(None) >= holdout_cut]
    hold_s = summarize_returns([t["net"] for t in hold])
    return {
        "hypothesis": "H-COH_dixon_coles_poly",
        "edge_min": edge_min,
        "n_trades": len(trades),
        "n_aux_events_indexed": len(aux_by_key),
        "structural_violations_model": len(violations),
        "dev": summarize_returns([t["net"] for t in dev]),
        "holdout": hold_s,
        "pnl_viable_holdout": pnl_viable(hold_s),
    }


def _parse_kickoff_ev(event: dict):
    from src.poly_maker_backtest import _parse_kickoff
    return _parse_kickoff(event)
