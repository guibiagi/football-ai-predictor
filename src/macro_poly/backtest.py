from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from src.macro_poly.discovery import MacroMarket
from src.macro_poly.liquidity import (
    LiquidityTerciles,
    compute_terciles,
    taker_ask_with_slippage,
)
from src.macro_poly.prices import (
    load_or_fetch_history,
    price_at_or_after,
    price_at_or_before,
    iso_to_ts,
)
from src.macro_poly.stats import (
    benjamini_hochberg,
    binary_brier,
    binary_log_loss,
    bonferroni,
    bootstrap_mean_ci,
    bootstrap_p_value_roi,
    trade_roi_binary,
)

FEE_RATE = 0.02
LIQUIDITY_SEGMENTS = ("T1_low", "T2_mid", "T3_high")


@dataclass
class Trade:
    hypothesis: str
    market_id: str
    event_ts: int
    p_entry: float
    won: bool
    roi: float
    clv: float | None
    holdout: bool
    volume_gamma: float
    liquidity_tercile: str
    pre_prob: float | None = None
    log_loss: float | None = None
    brier: float | None = None


def _build_trade(
    hid: str,
    m: MacroMarket,
    event_ts: int,
    mid_yes: float,
    buy_yes: bool,
    holdout_events: set[str],
    terciles: LiquidityTerciles,
    pre_prob: float | None = None,
    clv: float | None = None,
) -> Trade:
    p_entry = taker_ask_with_slippage(mid_yes if buy_yes else 1.0 - mid_yes, m.volume)
    won = (m.outcome_yes == 1) if buy_yes else (m.outcome_yes == 0)
    ll = binary_log_loss(pre_prob, m.outcome_yes) if pre_prob is not None else None
    br = binary_brier(pre_prob, m.outcome_yes) if pre_prob is not None else None
    return Trade(
        hypothesis=hid,
        market_id=m.market_id,
        event_ts=event_ts,
        p_entry=p_entry,
        won=won,
        roi=trade_roi_binary(p_entry, won, FEE_RATE),
        clv=clv,
        holdout=m.event_id in holdout_events,
        volume_gamma=m.volume,
        liquidity_tercile=terciles.label(m.volume),
        pre_prob=pre_prob,
        log_loss=ll,
        brier=br,
    )


@dataclass
class HypothesisResult:
    hypothesis_id: str
    description: str
    liquidity_segment: str
    n: int
    roi_mean: float
    roi_ci_low: float
    roi_ci_high: float
    hit_rate: float
    p_raw: float
    log_loss: float | None
    brier: float | None
    clv_mean: float | None
    holdout_n: int
    holdout_roi: float | None


def _split_holdout(events: list[int], frac: float = 0.3) -> set[int]:
    uniq = sorted(set(events))
    if not uniq:
        return set()
    cut = int(len(uniq) * (1 - frac))
    train_events = set(uniq[:cut])
    return set(uniq) - train_events


def _histories_for_market(
    m: MacroMarket,
    cache_dir: Path,
    pad_days: int = 21,
) -> list[dict]:
    if not m.release_ts:
        end = iso_to_ts(m.end_date)
        if not end:
            return []
        start = end - pad_days * 86400
        end_ts = end + 2 * 86400
    else:
        start = m.release_ts - pad_days * 86400
        end_ts = m.release_ts + 3 * 86400
    return load_or_fetch_history(m.token_yes, start, end_ts, cache_dir)


def run_post_release_drift(
    markets: list[MacroMarket],
    category: str,
    deltas_min: list[int],
    holdout_events: set[str],
    cache_dir: Path,
    hypothesis_prefix: str,
    terciles: LiquidityTerciles,
) -> list[tuple[str, list[Trade], list[float], list[float]]]:
    """Compra lado vencedor conhecido pós-release (teste de drift / mispricing)."""
    results = []
    subset = [m for m in markets if m.category == category and m.release_ts and m.outcome_yes is not None]
    for delta in deltas_min:
        trades: list[Trade] = []
        hid = f"{hypothesis_prefix}_{delta}m"
        for m in subset:
            hist = _histories_for_market(m, cache_dir)
            if not hist:
                continue
            entry_ts = m.release_ts + delta * 60
            mid = price_at_or_after(hist, entry_ts)
            if mid is None:
                continue
            buy_yes = bool(m.outcome_yes)
            pre = price_at_or_before(hist, m.release_ts - 3600)
            close_ts = m.release_ts + max(deltas_min) * 60 * 2
            p_later = price_at_or_after(hist, close_ts) or mid
            p_entry = taker_ask_with_slippage(mid if buy_yes else 1.0 - mid, m.volume)
            clv = (p_later - p_entry) if buy_yes else None
            trades.append(
                _build_trade(
                    hid,
                    m,
                    m.release_ts,
                    mid,
                    buy_yes,
                    holdout_events,
                    terciles,
                    pre_prob=pre,
                    clv=clv,
                )
            )
        log_losses = [t.log_loss for t in trades if t.log_loss is not None]
        briers = [t.brier for t in trades if t.brier is not None]
        results.append((hid, trades, log_losses, briers))
    return results


def run_fomc_divergence(
    markets: list[MacroMarket],
    holdout_events: set[str],
    cache_dir: Path,
    thresholds_pp: list[float],
    windows_h: list[int],
    terciles: LiquidityTerciles,
) -> list[tuple[str, list[Trade], list[float], list[float]]]:
    """H2a: momentum externo = resultado do encontro FOMC anterior no mesmo slug family."""
    by_event: dict[str, list[MacroMarket]] = {}
    for m in markets:
        if m.category != "fomc":
            continue
        by_event.setdefault(m.event_id, []).append(m)

    # ordenar eventos por release
    event_order = sorted(
        ((eid, max(x.release_ts or 0 for x in ms)) for eid, ms in by_event.items()),
        key=lambda x: x[1],
    )
    prev_winner: str | None = None  # cut | hold | hike
    results_map: dict[str, list[Trade]] = {}
    ll_map: dict[str, list[float]] = {}
    br_map: dict[str, list[float]] = {}

    for eid, _ in event_order:
        ms = by_event[eid]
        # classificar outcomes
        winner_label = None
        for m in ms:
            q = m.question.lower()
            if m.outcome_yes != 1:
                continue
            if "50" in q or "25" in q and "decrease" in q or "cut" in q:
                winner_label = "cut"
            elif "increase" in q or "hike" in q:
                winner_label = "hike"
            elif "no change" in q or "unchanged" in q:
                winner_label = "hold"
        p_ext = {"cut": 0.55, "hold": 0.5, "hike": 0.45}.get(prev_winner or "hold", 0.5)

        for th in thresholds_pp:
            for wh in windows_h:
                hid = f"H2a_{int(th*100)}pp_{wh}h"
                results_map.setdefault(hid, [])
                ll_map.setdefault(hid, [])
                br_map.setdefault(hid, [])
                for m in ms:
                    if m.release_ts is None or m.outcome_yes is None:
                        continue
                    hist = _histories_for_market(m, cache_dir)
                    if not hist:
                        continue
                    entry_ts = m.release_ts - wh * 3600
                    mid = price_at_or_before(hist, entry_ts)
                    if mid is None:
                        continue
                    q = m.question.lower()
                    is_cut = "decrease" in q or "cut" in q
                    is_hike = "increase" in q or "hike" in q
                    p_side = mid if is_cut or is_hike else (1 - mid)
                    if is_cut:
                        p_implied = p_side
                    elif is_hike:
                        p_implied = p_side
                    else:
                        continue
                    if abs(p_implied - p_ext) < th:
                        continue
                    # apostar no lado com prob externa maior
                    bet_yes = p_ext > p_implied
                    if not bet_yes and not is_cut:
                        continue
                    results_map[hid].append(
                        _build_trade(
                            hid,
                            m,
                            entry_ts,
                            mid,
                            bet_yes,
                            holdout_events,
                            terciles,
                        )
                    )
        prev_winner = winner_label or prev_winner

    return [(hid, results_map[hid], ll_map[hid], br_map[hid]) for hid in sorted(results_map)]


def run_cpi_divergence(
    markets: list[MacroMarket],
    holdout_events: set[str],
    cache_dir: Path,
    surprise_thresholds: list[float],
    terciles: LiquidityTerciles,
) -> list[tuple[str, list[Trade], list[float], list[float]]]:
    """H2b: surpresa naive = último mercado CPI resolvido com mesmo threshold family."""
    cpi = sorted(
        [m for m in markets if m.category == "cpi" and m.release_ts],
        key=lambda x: x.release_ts or 0,
    )
    last_mom: float | None = None
    out: list[tuple[str, list[Trade], list[float], list[float]]] = []
    for th in surprise_thresholds:
        trades: list[Trade] = []
        hid = f"H2b_{th}pp"
        for m in cpi:
            if m.threshold_pct is None or m.outcome_yes is None:
                continue
            if last_mom is None:
                last_mom = m.threshold_pct
                continue
            surprise = abs(last_mom - m.threshold_pct)
            if surprise < th:
                continue
            hist = _histories_for_market(m, cache_dir)
            if not hist:
                continue
            entry_ts = m.release_ts - 4 * 3600
            mid = price_at_or_before(hist, entry_ts)
            if mid is None:
                continue
            # se surpresa alta, apostar YES se último print implied > threshold
            bet_yes = last_mom > (m.threshold_pct or 0)
            trades.append(
                _build_trade(
                    hid, m, entry_ts, mid, bet_yes, holdout_events, terciles
                )
            )
            last_mom = m.threshold_pct
        out.append((hid, trades, [], []))
    return out


def run_favorite_longshot(
    markets: list[MacroMarket],
    holdout_events: set[str],
    cache_dir: Path,
    category: str,
    fav_min: float,
    long_max: float,
    label: str,
    side: str,
    terciles: LiquidityTerciles,
) -> tuple[str, list[Trade], list[float], list[float]]:
    trades: list[Trade] = []
    hid = label
    for m in markets:
        if m.category != category or m.outcome_yes is None:
            continue
        hist = _histories_for_market(m, cache_dir, pad_days=14)
        if not hist or not m.release_ts:
            continue
        open_ts = int(hist[0]["t"])
        mid = price_at_or_after(hist, open_ts + 3600)
        if mid is None:
            continue
        if side == "fav" and mid < fav_min:
            continue
        if side == "long" and mid > long_max:
            continue
        trades.append(
            _build_trade(hid, m, open_ts, mid, True, holdout_events, terciles)
        )
    return hid, trades, [], []


def run_intraday_moves(
    markets: list[MacroMarket],
    holdout_events: set[str],
    cache_dir: Path,
    mode: str,
    thresholds_pp: list[float],
    terciles: LiquidityTerciles,
) -> list[tuple[str, list[Trade], list[float], list[float]]]:
    out = []
    for th in thresholds_pp:
        trades: list[Trade] = []
        hid = f"{mode}_{int(th*100)}pp"
        for m in markets:
            if m.release_ts is None or m.outcome_yes is None:
                continue
            hist = _histories_for_market(m, cache_dir)
            if len(hist) < 5:
                continue
            for i in range(1, len(hist)):
                t0, p0 = int(hist[i - 1]["t"]), float(hist[i - 1]["p"])
                t1, p1 = int(hist[i]["t"]), float(hist[i]["p"])
                if abs(p1 - p0) < th:
                    continue
                if mode == "H4a" and m.release_ts - 3600 <= t1 <= m.release_ts + 3600:
                    continue  # sem release window
                if mode == "H4b" and not (m.release_ts - 48 * 3600 <= t1 <= m.release_ts):
                    continue
                fade = p1 < p0  # comprar YES se preço subiu (fade down move)
                bet_yes = not fade if mode == "H4a" else fade
                trades.append(
                    _build_trade(hid, m, t1, p1, bet_yes, holdout_events, terciles)
                )
                break
        out.append((hid, trades, [], []))
    return out


def _segment_bundles(
    bundles: list[tuple[str, list[Trade], list[float], list[float]]],
) -> list[tuple[str, str, list[Trade], list[float], list[float]]]:
    """Expande cada hipótese base em tercis de liquidez (T1/T2/T3)."""
    out: list[tuple[str, str, list[Trade], list[float], list[float]]] = []
    for hid, trades, ll, br in bundles:
        for seg in LIQUIDITY_SEGMENTS:
            seg_trades = [t for t in trades if t.liquidity_tercile == seg]
            seg_ll = [t.log_loss for t in seg_trades if t.log_loss is not None]
            seg_br = [t.brier for t in seg_trades if t.brier is not None]
            out.append((hid, seg, seg_trades, seg_ll, seg_br))
    return out


def summarize_hypothesis(
    hid: str,
    liquidity_segment: str,
    trades: list[Trade],
    log_losses: list[float],
    briers: list[float],
) -> HypothesisResult:
    rois = [t.roi for t in trades]
    hold = [t for t in trades if t.holdout]
    mean, lo, hi = bootstrap_mean_ci(rois)
    p_raw = bootstrap_p_value_roi(rois)
    clvs = [t.clv for t in trades if t.clv is not None]
    full_id = f"{hid}__{liquidity_segment}"
    return HypothesisResult(
        hypothesis_id=full_id,
        description=hid,
        liquidity_segment=liquidity_segment,
        n=len(trades),
        roi_mean=mean,
        roi_ci_low=lo,
        roi_ci_high=hi,
        hit_rate=float(sum(t.won for t in trades) / len(trades)) if trades else float("nan"),
        p_raw=p_raw,
        log_loss=float(sum(log_losses) / len(log_losses)) if log_losses else None,
        brier=float(sum(briers) / len(briers)) if briers else None,
        clv_mean=float(sum(clvs) / len(clvs)) if clvs else None,
        holdout_n=len(hold),
        holdout_roi=float(sum(t.roi for t in hold) / len(hold)) if hold else None,
    )


def run_full_backtest(
    markets: list[MacroMarket],
    cache_dir: Path,
    holdout_frac: float = 0.3,
) -> dict[str, Any]:
    event_ids = [m.event_id for m in markets if m.release_ts]
    holdout_events = _split_holdout(
        [m.release_ts for m in markets if m.release_ts],
        frac=holdout_frac,
    )
    # map release ts to event - use event_id from markets with release in holdout set
    by_release = sorted(
        {(m.event_id, m.release_ts) for m in markets if m.release_ts},
        key=lambda x: x[1],
    )
    cut = int(len(by_release) * (1 - holdout_frac))
    holdout_event_ids = {eid for eid, _ in by_release[cut:]}
    holdout_events = holdout_event_ids
    terciles = compute_terciles([m.volume for m in markets])

    bundles: list[tuple[str, list[Trade], list[float], list[float]]] = []
    bundles.extend(
        run_post_release_drift(
            markets, "cpi", [1, 5, 15, 60], holdout_events, cache_dir, "H1a", terciles
        )
    )
    bundles.extend(
        run_post_release_drift(
            markets, "pce", [1, 5, 15, 60], holdout_events, cache_dir, "H1b", terciles
        )
    )
    bundles.extend(
        run_post_release_drift(
            markets, "fomc", [1, 5, 15, 60], holdout_events, cache_dir, "H1c", terciles
        )
    )
    bundles.extend(
        run_fomc_divergence(
            markets, holdout_events, cache_dir, [0.05, 0.10], [24, 4], terciles
        )
    )
    bundles.extend(
        run_cpi_divergence(markets, holdout_events, cache_dir, [0.1, 0.2], terciles)
    )
    bundles.append(
        run_favorite_longshot(
            markets, holdout_events, cache_dir, "cpi", 0.55, 0.25, "H3a_fav", "fav", terciles
        )
    )
    bundles.append(
        run_favorite_longshot(
            markets, holdout_events, cache_dir, "cpi", 0.55, 0.25, "H3a_long", "long", terciles
        )
    )
    bundles.append(
        run_favorite_longshot(
            markets, holdout_events, cache_dir, "fomc", 0.60, 0.25, "H3b_fav", "fav", terciles
        )
    )
    bundles.extend(
        run_intraday_moves(markets, holdout_events, cache_dir, "H4a", [0.03, 0.05], terciles)
    )
    bundles.extend(
        run_intraday_moves(markets, holdout_events, cache_dir, "H4b", [0.03, 0.05], terciles)
    )

    segmented = _segment_bundles(bundles)
    summaries = [
        summarize_hypothesis(hid, seg, tr, ll, br) for hid, seg, tr, ll, br in segmented
    ]
    p_raws = [s.p_raw for s in summaries]
    bonf = bonferroni(p_raws)
    bh = benjamini_hochberg(p_raws)

    rows = []
    for s, pb, ph in zip(summaries, bonf, bh):
        survives = (
            s.holdout_n >= 3
            and (s.holdout_roi or 0) > 0
            and pb < 0.05
        )
        survives_bh = (
            s.holdout_n >= 3
            and (s.holdout_roi or 0) > 0
            and ph < 0.05
        )
        rows.append(
            {
                **s.__dict__,
                "p_bonferroni": pb,
                "p_bh": ph,
                "survives_bonferroni": survives,
                "survives_bh": survives_bh,
            }
        )

    return {
        "holdout_event_count": len(holdout_event_ids),
        "market_count": len(markets),
        "liquidity_terciles_usd": {"q33": terciles.q33, "q66": terciles.q66},
        "stake_usd_assumed": 25.0,
        "n_base_hypotheses": len(bundles),
        "hypotheses": rows,
        "n_tests": len(rows),
    }
