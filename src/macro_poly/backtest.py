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
    ENTRY_MAX_LAG_SEC,
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
from src.macro_poly.volume import volume_at_decision

FEE_RATE = 0.02
LIQUIDITY_SEGMENTS = ("T1_low", "T2_mid", "T3_high")
DIAGNOSTIC_PREFIXES = ("H1a", "H1b", "H1c")


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
    volume_at_decision: float
    liquidity_tercile: str
    pre_prob: float | None = None
    log_loss: float | None = None
    brier: float | None = None
    tradeable: bool = True


def deterministic_holdout_events(
    markets: list[MacroMarket],
    frac: float = 0.3,
) -> set[str]:
    by_event = sorted(
        {(m.event_id, m.release_ts or 0) for m in markets if m.release_ts},
        key=lambda x: (x[1], x[0]),
    )
    cut = int(len(by_event) * (1 - frac))
    return {eid for eid, _ in by_event[cut:]}


def terciles_from_training(
    markets: list[MacroMarket],
    holdout_events: set[str],
    decision_ts_fn,
) -> LiquidityTerciles:
    vols = []
    for m in markets:
        if m.event_id in holdout_events or m.release_ts is None:
            continue
        vols.append(volume_at_decision(m, decision_ts_fn(m)))
    return compute_terciles(vols)


def _build_trade(
    hid: str,
    m: MacroMarket,
    event_ts: int,
    mid_yes: float,
    buy_yes: bool,
    holdout_events: set[str],
    terciles: LiquidityTerciles,
    decision_ts: int,
    pre_prob: float | None = None,
    clv: float | None = None,
    tradeable: bool = True,
) -> Trade:
    vol_d = volume_at_decision(m, decision_ts)
    side_mid = mid_yes if buy_yes else 1.0 - mid_yes
    p_entry = taker_ask_with_slippage(side_mid, vol_d)
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
        volume_at_decision=vol_d,
        liquidity_tercile=terciles.label(vol_d),
        pre_prob=pre_prob,
        log_loss=ll,
        brier=br,
        tradeable=tradeable,
    )


@dataclass
class HypothesisResult:
    hypothesis_id: str
    description: str
    liquidity_segment: str
    tradeable: bool
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


def _histories_for_market(
    m: MacroMarket,
    cache_dir: Path,
    window_start_ts: int,
    window_end_ts: int,
) -> list[dict]:
    return load_or_fetch_history(m.token_yes, window_start_ts, window_end_ts, cache_dir)


def _market_window(m: MacroMarket, anchor_ts: int, before_sec: int, after_sec: int) -> tuple[int, int]:
    start = anchor_ts - before_sec
    if m.start_ts is not None:
        start = max(start, m.start_ts)
    return start, anchor_ts + after_sec


def run_post_release_drift(
    markets: list[MacroMarket],
    category: str,
    deltas_min: list[int],
    holdout_events: set[str],
    cache_dir: Path,
    hypothesis_prefix: str,
    terciles: LiquidityTerciles,
) -> list[tuple[str, list[Trade], list[float], list[float]]]:
    """Diagnóstico: compra vencedor conhecido — não entra em testes de ROI tradeáveis."""
    results = []
    subset = [m for m in markets if m.category == category and m.release_ts and m.outcome_yes is not None]
    for delta in deltas_min:
        trades: list[Trade] = []
        hid = f"{hypothesis_prefix}_{delta}m"
        for m in subset:
            entry_ts = m.release_ts + delta * 60
            w_start, w_end = _market_window(m, m.release_ts, 21 * 86400, 3 * 86400)
            hist = _histories_for_market(m, cache_dir, w_start, w_end)
            if not hist:
                continue
            mid = price_at_or_after(hist, entry_ts, max_lag_sec=ENTRY_MAX_LAG_SEC)
            if mid is None:
                continue
            buy_yes = bool(m.outcome_yes)
            pre = price_at_or_before(hist, m.release_ts - 3600)
            close_ts = m.release_ts + max(deltas_min) * 60 * 2
            p_later = price_at_or_after(hist, close_ts, max_lag_sec=ENTRY_MAX_LAG_SEC)
            p_entry_side = taker_ask_with_slippage(
                mid if buy_yes else 1.0 - mid,
                volume_at_decision(m, entry_ts),
            )
            clv = (p_later - p_entry_side) if buy_yes and p_later is not None else None
            trades.append(
                _build_trade(
                    hid,
                    m,
                    entry_ts,
                    mid,
                    buy_yes,
                    holdout_events,
                    terciles,
                    decision_ts=entry_ts,
                    pre_prob=pre,
                    clv=clv,
                    tradeable=False,
                )
            )
        log_losses = [t.log_loss for t in trades if t.log_loss is not None]
        briers = [t.brier for t in trades if t.brier is not None]
        results.append((hid, trades, log_losses, briers))
    return results


def _classify_fomc_winner(ms: list[MacroMarket]) -> str | None:
    for m in ms:
        if m.outcome_yes != 1:
            continue
        q = m.question.lower()
        if ("decrease" in q or "cut" in q) and "bps" in q:
            return "cut"
        if "increase" in q or "hike" in q:
            return "hike"
        if "no change" in q or "unchanged" in q:
            return "hold"
    return None


def run_fomc_divergence(
    markets: list[MacroMarket],
    holdout_events: set[str],
    cache_dir: Path,
    thresholds_pp: list[float],
    windows_h: list[int],
    terciles: LiquidityTerciles,
) -> list[tuple[str, list[Trade], list[float], list[float]]]:
    by_event: dict[str, list[MacroMarket]] = {}
    for m in markets:
        if m.category != "fomc":
            continue
        by_event.setdefault(m.event_id, []).append(m)

    event_order = sorted(
        ((eid, max(x.release_ts or 0 for x in ms)) for eid, ms in by_event.items()),
        key=lambda x: (x[1], x[0]),
    )
    prev_winner: str | None = None
    results_map: dict[str, list[Trade]] = {}
    ll_map: dict[str, list[float]] = {}
    br_map: dict[str, list[float]] = {}

    for eid, _ in event_order:
        ms = by_event[eid]
        winner_label = _classify_fomc_winner(ms)
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
                    entry_ts = m.release_ts - wh * 3600
                    w_start, w_end = _market_window(m, m.release_ts, 21 * 86400, 3 * 86400)
                    hist = _histories_for_market(m, cache_dir, w_start, w_end)
                    if not hist:
                        continue
                    mid = price_at_or_before(hist, entry_ts)
                    if mid is None:
                        continue
                    q = m.question.lower()
                    is_cut = "decrease" in q or ("cut" in q and "bps" in q)
                    is_hike = "increase" in q or "hike" in q
                    if not is_cut and not is_hike:
                        continue
                    p_implied = mid
                    if abs(p_implied - p_ext) < th:
                        continue
                    bet_yes = p_ext > p_implied if is_cut else p_ext < p_implied
                    results_map[hid].append(
                        _build_trade(
                            hid,
                            m,
                            entry_ts,
                            mid,
                            bet_yes,
                            holdout_events,
                            terciles,
                            decision_ts=entry_ts,
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
    cpi = sorted(
        [m for m in markets if m.category == "cpi" and m.release_ts],
        key=lambda x: (x.release_ts or 0, x.market_id),
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
            entry_ts = m.release_ts - 4 * 3600
            w_start, w_end = _market_window(m, m.release_ts, 21 * 86400, 3 * 86400)
            hist = _histories_for_market(m, cache_dir, w_start, w_end)
            if not hist:
                continue
            mid = price_at_or_before(hist, entry_ts)
            if mid is None:
                continue
            bet_yes = last_mom > (m.threshold_pct or 0)
            trades.append(
                _build_trade(
                    hid, m, entry_ts, mid, bet_yes, holdout_events, terciles, entry_ts
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
        if m.category != category or m.outcome_yes is None or m.release_ts is None:
            continue
        if m.start_ts is None:
            continue
        open_ts = m.start_ts + 3600
        w_start, w_end = _market_window(m, m.release_ts, 14 * 86400, 3 * 86400)
        hist = _histories_for_market(m, cache_dir, w_start, w_end)
        if not hist:
            continue
        mid = price_at_or_after(hist, open_ts, max_lag_sec=ENTRY_MAX_LAG_SEC)
        if mid is None:
            continue
        if side == "fav" and mid < fav_min:
            continue
        if side == "long" and mid > long_max:
            continue
        trades.append(
            _build_trade(hid, m, open_ts, mid, True, holdout_events, terciles, open_ts)
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
            w_start, w_end = _market_window(m, m.release_ts, 21 * 86400, 3 * 86400)
            hist = _histories_for_market(m, cache_dir, w_start, w_end)
            if len(hist) < 5:
                continue
            for i in range(1, len(hist)):
                t0, p0 = int(hist[i - 1]["t"]), float(hist[i - 1]["p"])
                t1, p1 = int(hist[i]["t"]), float(hist[i]["p"])
                if abs(p1 - p0) < th:
                    continue
                if mode == "H4a" and m.release_ts - 3600 <= t1 <= m.release_ts + 3600:
                    continue
                if mode == "H4b" and not (m.release_ts - 48 * 3600 <= t1 <= m.release_ts):
                    continue
                # H4a fade: reverte movimento; H4b continuação: segue movimento
                moved_up = p1 > p0
                bet_yes = (not moved_up) if mode == "H4a" else moved_up
                trades.append(
                    _build_trade(hid, m, t1, p1, bet_yes, holdout_events, terciles, t1)
                )
                break
        out.append((hid, trades, [], []))
    return out


def run_h5_calibration(
    markets: list[MacroMarket],
    holdout_events: set[str],
    cache_dir: Path,
) -> dict[str, Any]:
    """Calibração no holdout: preço 1h antes do release vs outcome."""
    ll: list[float] = []
    br: list[float] = []
    for m in markets:
        if m.release_ts is None or m.outcome_yes is None:
            continue
        if m.event_id not in holdout_events:
            continue
        w_start, w_end = _market_window(m, m.release_ts, 21 * 86400, 3 * 86400)
        hist = _histories_for_market(m, cache_dir, w_start, w_end)
        if not hist:
            continue
        pre = price_at_or_before(hist, m.release_ts - 3600)
        if pre is None:
            continue
        ll.append(binary_log_loss(pre, m.outcome_yes))
        br.append(binary_brier(pre, m.outcome_yes))
    return {
        "n": len(ll),
        "log_loss": sum(ll) / len(ll) if ll else None,
        "brier": sum(br) / len(br) if br else None,
    }


def _segment_bundles(
    bundles: list[tuple[str, list[Trade], list[float], list[float]]],
) -> list[tuple[str, str, list[Trade], list[float], list[float]]]:
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
    tradeable: bool,
) -> HypothesisResult:
    rois = [t.roi for t in trades]
    hold = [t for t in trades if t.holdout]
    mean, lo, hi = bootstrap_mean_ci(rois)
    p_raw = bootstrap_p_value_roi(rois) if tradeable else float("nan")
    clvs = [t.clv for t in trades if t.clv is not None]
    full_id = f"{hid}__{liquidity_segment}"
    return HypothesisResult(
        hypothesis_id=full_id,
        description=hid,
        liquidity_segment=liquidity_segment,
        tradeable=tradeable,
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
    holdout_events = deterministic_holdout_events(markets, frac=holdout_frac)

    def _decision_ts(m: MacroMarket) -> int:
        return m.release_ts or 0

    terciles = terciles_from_training(markets, holdout_events, _decision_ts)

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

    h5 = run_h5_calibration(markets, holdout_events, cache_dir)

    segmented = _segment_bundles(bundles)
    summaries = []
    for hid, seg, tr, ll, br in segmented:
        tradeable = not any(hid.startswith(p) for p in DIAGNOSTIC_PREFIXES)
        summaries.append(summarize_hypothesis(hid, seg, tr, ll, br, tradeable))

    tradeable_summaries = [s for s in summaries if s.tradeable]
    p_raws = [s.p_raw for s in tradeable_summaries]
    bonf = bonferroni(p_raws)
    bh = benjamini_hochberg(p_raws)

    rows = []
    ti = 0
    for s in summaries:
        if s.tradeable:
            pb, ph = bonf[ti], bh[ti]
            ti += 1
        else:
            pb, ph = float("nan"), float("nan")
        survives = (
            s.tradeable
            and s.holdout_n >= 3
            and (s.holdout_roi or 0) > 0
            and pb < 0.05
        )
        survives_bh = (
            s.tradeable
            and s.holdout_n >= 3
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
        "holdout_event_count": len(holdout_events),
        "market_count": len(markets),
        "liquidity_terciles_usd": {"q33": terciles.q33, "q66": terciles.q66},
        "liquidity_terciles_note": "Cortes só no treino; volume interpolado até decision_ts",
        "stake_usd_assumed": 25.0,
        "n_base_hypotheses": len(bundles),
        "n_tradeable_tests": len(tradeable_summaries),
        "h5_calibration_holdout": h5,
        "hypotheses": rows,
        "n_tests": len(tradeable_summaries),
    }
