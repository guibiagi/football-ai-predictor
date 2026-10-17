from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
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
from src.macro_poly.market_preprocess import dedupe_fomc_events
from src.macro_poly.trade_volume import TradeVolumeStore
from src.macro_poly.volume import gamma_liquidity_usd, volume_at_decision

FEE_RATE = 0.02
LIQUIDITY_SEGMENTS_TERCILE = ("T1_low", "T2_mid", "T3_high")
MIN_TRADES_FOR_TEST = 3
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
    store: TradeVolumeStore,
) -> LiquidityTerciles | None:
    vols: list[float] = []
    for m in markets:
        if m.event_id in holdout_events or m.release_ts is None:
            continue
        if not store.trade_volume_usable(m.token_yes):
            continue
        ts = decision_ts_fn(m)
        if ts is None:
            continue
        v = store.volume_at(m.token_yes, ts)
        if v is not None and v > 0:
            vols.append(v)
    if len(vols) < 9:
        return None
    return compute_terciles(vols)


def terciles_from_training_gamma(
    markets: list[MacroMarket],
    holdout_events: set[str],
) -> LiquidityTerciles | None:
    vols = [
        gamma_liquidity_usd(m)
        for m in markets
        if m.event_id not in holdout_events and m.release_ts is not None
    ]
    if len(vols) < 9:
        return None
    return compute_terciles(vols)


def _build_trade(
    hid: str,
    m: MacroMarket,
    event_ts: int,
    mid_yes: float,
    buy_yes: bool,
    holdout_events: set[str],
    terciles: LiquidityTerciles | None,
    store: TradeVolumeStore,
    decision_ts: int,
    pre_prob: float | None = None,
    clv: float | None = None,
    tradeable: bool = True,
) -> Trade | None:
    if terciles is not None:
        if not store.trade_volume_usable(m.token_yes):
            return None
        raw_vol = store.volume_at(m.token_yes, decision_ts)
        if raw_vol is None:
            return None
        if raw_vol <= 0:
            vol_d = 1.0
            liq_seg = "T1_low"
        else:
            vol_d = raw_vol
            liq_seg = terciles.label(vol_d)
    else:
        vol_d = volume_at_decision(m, decision_ts, store)
        liq_seg = "ALL"
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
        liquidity_tercile=liq_seg,
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
    terciles: LiquidityTerciles | None,
    store: TradeVolumeStore,
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
            vol_e = volume_at_decision(m, entry_ts, store) or 1.0
            p_entry_side = taker_ask_with_slippage(mid if buy_yes else 1.0 - mid, vol_e)
            clv = (p_later - p_entry_side) if buy_yes and p_later is not None else None
            tr = _build_trade(
                hid,
                m,
                entry_ts,
                mid,
                buy_yes,
                holdout_events,
                terciles,
                store,
                decision_ts=entry_ts,
                pre_prob=pre,
                clv=clv,
                tradeable=False,
            )
            if tr:
                trades.append(tr)
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
    terciles: LiquidityTerciles | None,
    store: TradeVolumeStore,
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
                    tr = _build_trade(
                        hid,
                        m,
                        entry_ts,
                        mid,
                        bet_yes,
                        holdout_events,
                        terciles,
                        store,
                        decision_ts=entry_ts,
                    )
                    if tr:
                        results_map[hid].append(tr)
        prev_winner = winner_label or prev_winner

    return [(hid, results_map[hid], ll_map[hid], br_map[hid]) for hid in sorted(results_map)]


def run_cpi_divergence(
    markets: list[MacroMarket],
    holdout_events: set[str],
    cache_dir: Path,
    surprise_thresholds: list[float],
    terciles: LiquidityTerciles | None,
    store: TradeVolumeStore,
) -> list[tuple[str, list[Trade], list[float], list[float]]]:
    cpi = sorted(
        [m for m in markets if m.category == "cpi" and m.release_ts],
        key=lambda x: (x.release_ts or 0, x.market_id),
    )
    out: list[tuple[str, list[Trade], list[float], list[float]]] = []
    for th in surprise_thresholds:
        last_mom: float | None = None
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
            tr = _build_trade(
                hid, m, entry_ts, mid, bet_yes, holdout_events, terciles, store, entry_ts
            )
            if tr:
                trades.append(tr)
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
    terciles: LiquidityTerciles | None,
    store: TradeVolumeStore,
) -> tuple[str, list[Trade], list[float], list[float], dict[str, int]]:
    trades: list[Trade] = []
    hid = label
    stats = {
        "eligible": 0,
        "traded": 0,
        "dropped_no_history": 0,
        "dropped_no_price": 0,
        "dropped_incomplete_trade_history": 0,
        "zero_volume_assigned_t1": 0,
    }
    for m in markets:
        if m.category != category or m.outcome_yes is None or m.release_ts is None:
            continue
        if m.start_ts is None:
            continue
        stats["eligible"] += 1
        open_ts = m.start_ts + 3600
        w_start = m.start_ts
        w_end = max(m.release_ts + 3 * 86400, open_ts + 86400)
        hist = _histories_for_market(m, cache_dir, w_start, w_end)
        if not hist:
            stats["dropped_no_history"] += 1
            continue
        mid = price_at_or_after(hist, open_ts, max_lag_sec=ENTRY_MAX_LAG_SEC)
        if mid is None:
            stats["dropped_no_price"] += 1
            continue
        if side == "fav" and mid < fav_min:
            continue
        if side == "long" and mid > long_max:
            continue
        if terciles is not None:
            if not store.trade_volume_usable(m.token_yes):
                stats["dropped_incomplete_trade_history"] += 1
                continue
            vol0 = store.volume_at(m.token_yes, open_ts)
            if vol0 is not None and vol0 <= 0:
                stats["zero_volume_assigned_t1"] += 1
        tr = _build_trade(hid, m, open_ts, mid, True, holdout_events, terciles, store, open_ts)
        if tr:
            trades.append(tr)
            stats["traded"] += 1
    return hid, trades, [], [], stats


def first_intraday_entry_ts(
    m: MacroMarket,
    cache_dir: Path,
    mode: str,
    th: float,
) -> int | None:
    if m.release_ts is None:
        return None
    w_start, w_end = _market_window(m, m.release_ts, 21 * 86400, 3 * 86400)
    hist = _histories_for_market(m, cache_dir, w_start, w_end)
    if len(hist) < 5:
        return None
    for i in range(1, len(hist)):
        t0, p0 = int(hist[i - 1]["t"]), float(hist[i - 1]["p"])
        t1, p1 = int(hist[i]["t"]), float(hist[i]["p"])
        if abs(p1 - p0) < th:
            continue
        if mode == "H4a" and m.release_ts - 3600 <= t1 <= m.release_ts + 3600:
            continue
        if mode == "H4b" and not (m.release_ts - 48 * 3600 <= t1 <= m.release_ts):
            continue
        return t1
    return None


def terciles_from_intraday_training(
    markets: list[MacroMarket],
    holdout_events: set[str],
    cache_dir: Path,
    mode: str,
    move_threshold: float,
    store: TradeVolumeStore,
) -> LiquidityTerciles | None:
    vols: list[float] = []
    for m in markets:
        if m.event_id in holdout_events or m.outcome_yes is None:
            continue
        t1 = first_intraday_entry_ts(m, cache_dir, mode, move_threshold)
        if t1 is None or not store.trade_volume_usable(m.token_yes):
            continue
        v = store.volume_at(m.token_yes, t1)
        if v is not None and v > 0:
            vols.append(v)
    if len(vols) < 9:
        return None
    return compute_terciles(vols)


def run_intraday_moves(
    markets: list[MacroMarket],
    holdout_events: set[str],
    cache_dir: Path,
    mode: str,
    thresholds_pp: list[float],
    terciles: LiquidityTerciles | None,
    store: TradeVolumeStore,
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
                moved_up = p1 > p0
                bet_yes = (not moved_up) if mode == "H4a" else moved_up
                tr = _build_trade(hid, m, t1, p1, bet_yes, holdout_events, terciles, store, t1)
                if tr:
                    trades.append(tr)
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
    segments: tuple[str, ...],
) -> list[tuple[str, str, list[Trade], list[float], list[float]]]:
    out: list[tuple[str, str, list[Trade], list[float], list[float]]] = []
    for hid, trades, ll, br in bundles:
        for seg in segments:
            seg_trades = [t for t in trades if t.liquidity_tercile == seg]
            seg_ll = [t.log_loss for t in seg_trades if t.log_loss is not None]
            seg_br = [t.brier for t in seg_trades if t.brier is not None]
            out.append((hid, seg, seg_trades, seg_ll, seg_br))
    return out


def _trade_fingerprint(trades: list[Trade]) -> tuple:
    return tuple(sorted((t.market_id, t.event_ts, t.won, round(t.p_entry, 6)) for t in trades))


def dedupe_identical_bundles(
    bundles: list[tuple[str, list[Trade], list[float], list[float]]],
) -> list[tuple[str, list[Trade], list[float], list[float]]]:
    seen: set[tuple] = set()
    out = []
    for hid, trades, ll, br in bundles:
        fp = _trade_fingerprint(trades)
        if fp in seen:
            continue
        seen.add(fp)
        out.append((hid, trades, ll, br))
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
    markets = dedupe_fomc_events(markets)
    holdout_events = deterministic_holdout_events(markets, frac=holdout_frac)

    trade_cache = cache_dir.parent / "trades"
    store = TradeVolumeStore(trade_cache)
    for m in markets:
        if getattr(m, "condition_id", None):
            store.register(
                m.token_yes,
                m.condition_id,
                getattr(m, "token_no", None),
                float(m.volume or 0.0),
                m.start_ts,
            )

    def _warm(tid: str) -> None:
        store.volume_at(tid, 2_000_000_000)

    unique_tokens = {m.token_yes for m in markets if m.condition_id}
    with ThreadPoolExecutor(max_workers=10) as pool:
        list(pool.map(_warm, unique_tokens))

    def vol_decision_h2a(m: MacroMarket) -> int:
        return (m.release_ts or 0) - 24 * 3600

    def vol_decision_h2b(m: MacroMarket) -> int:
        return (m.release_ts or 0) - 4 * 3600

    def vol_decision_h3(m: MacroMarket) -> int:
        return (m.start_ts or 0) + 3600

    train_markets = [m for m in markets if m.event_id not in holdout_events]
    trade_seg_feasible, trade_seg_stats = store.segmentation_feasible()
    coverage = store.coverage_fraction(train_markets, vol_decision_h2a)
    liquidity_enabled = trade_seg_feasible and coverage >= 0.35
    segments: tuple[str, ...] = (
        LIQUIDITY_SEGMENTS_TERCILE if liquidity_enabled else ("ALL",)
    )

    terciles_h2a = (
        terciles_from_training(markets, holdout_events, vol_decision_h2a, store)
        if liquidity_enabled
        else None
    )
    terciles_h2b = (
        terciles_from_training(markets, holdout_events, vol_decision_h2b, store)
        if liquidity_enabled
        else None
    )
    terciles_h3 = (
        terciles_from_training(markets, holdout_events, vol_decision_h3, store)
        if liquidity_enabled
        else None
    )
    terciles_h4a = (
        terciles_from_intraday_training(
            markets, holdout_events, cache_dir, "H4a", 0.03, store
        )
        if liquidity_enabled
        else None
    )
    terciles_h4b = (
        terciles_from_intraday_training(
            markets, holdout_events, cache_dir, "H4b", 0.03, store
        )
        if liquidity_enabled
        else None
    )
    terciles_h1 = (
        terciles_from_training(
            markets,
            holdout_events,
            lambda m: (m.release_ts or 0) + 60 * 60,
            store,
        )
        if liquidity_enabled
        else None
    )
    gamma_terciles_ref = terciles_from_training_gamma(markets, holdout_events)

    bundles: list[tuple[str, list[Trade], list[float], list[float]]] = []
    h3_stats: dict[str, dict[str, int]] = {}

    bundles.extend(
        run_post_release_drift(
            markets, "cpi", [60], holdout_events, cache_dir, "H1a", terciles_h1, store
        )
    )
    bundles.extend(
        run_post_release_drift(
            markets, "pce", [60], holdout_events, cache_dir, "H1b", terciles_h1, store
        )
    )
    bundles.extend(
        run_post_release_drift(
            markets, "fomc", [60], holdout_events, cache_dir, "H1c", terciles_h1, store
        )
    )
    bundles.extend(
        run_fomc_divergence(
            markets, holdout_events, cache_dir, [0.10], [24], terciles_h2a, store
        )
    )
    bundles.extend(
        run_cpi_divergence(markets, holdout_events, cache_dir, [0.1, 0.2], terciles_h2b, store)
    )
    h3_variants = [
        ("H3a_fav", "fav", "cpi", terciles_h3, 0.55, 0.25),
        ("H3a_long", "long", "cpi", terciles_h3, 0.55, 0.25),
        ("H3b_fav", "fav", "fomc", terciles_h3, 0.60, 0.25),
    ]
    for label, side, cat, terc, fav_min, long_max in h3_variants:
        hid, tr, ll, br, st = run_favorite_longshot(
            markets,
            holdout_events,
            cache_dir,
            cat,
            fav_min,
            long_max,
            label,
            side,
            terc,
            store,
        )
        h3_stats[hid] = st
        bundles.append((hid, tr, ll, br))
    bundles.extend(
        run_intraday_moves(
            markets, holdout_events, cache_dir, "H4a", [0.03, 0.05], terciles_h4a, store
        )
    )
    bundles.extend(
        run_intraday_moves(
            markets, holdout_events, cache_dir, "H4b", [0.03, 0.05], terciles_h4b, store
        )
    )

    tradeable_bundles = [
        b for b in bundles if not any(b[0].startswith(p) for p in DIAGNOSTIC_PREFIXES)
    ]
    tradeable_bundles = dedupe_identical_bundles(tradeable_bundles)
    diag_bundles = [b for b in bundles if any(b[0].startswith(p) for p in DIAGNOSTIC_PREFIXES)]
    bundles = diag_bundles + tradeable_bundles

    h5 = run_h5_calibration(markets, holdout_events, cache_dir)

    segmented = _segment_bundles(tradeable_bundles, segments)
    diag_segmented = _segment_bundles(diag_bundles, segments)

    summaries = []
    for hid, seg, tr, ll, br in diag_segmented + segmented:
        tradeable = not any(hid.startswith(p) for p in DIAGNOSTIC_PREFIXES)
        summaries.append(summarize_hypothesis(hid, seg, tr, ll, br, tradeable))

    if liquidity_enabled:
        h3_bases = {b[0] for b in tradeable_bundles if b[0].startswith("H3")}
        for base in h3_bases:
            has_family_cell = any(
                s.description == base and s.n >= MIN_TRADES_FOR_TEST and s.tradeable
                for s in summaries
            )
            if has_family_cell:
                continue
            bundle = next((b for b in tradeable_bundles if b[0] == base), None)
            if not bundle:
                continue
            _, trades, ll, br = bundle
            if len(trades) < MIN_TRADES_FOR_TEST:
                continue
            summaries.append(
                summarize_hypothesis(base, "ALL", trades, ll, br, tradeable=True)
            )

    tradeable_summaries = [
        s for s in summaries if s.tradeable and s.n >= MIN_TRADES_FOR_TEST
    ]
    if not liquidity_enabled:
        seen_base: set[str] = set()
        deduped: list[HypothesisResult] = []
        for s in tradeable_summaries:
            if s.liquidity_segment != "ALL":
                continue
            if s.description in seen_base:
                continue
            seen_base.add(s.description)
            deduped.append(s)
        tradeable_summaries = deduped
    p_raws = [s.p_raw for s in tradeable_summaries]
    bonf = bonferroni(p_raws) if p_raws else []
    bh = benjamini_hochberg(p_raws) if p_raws else []

    rows = []
    ti = 0
    for s in summaries:
        in_family = s.tradeable and s.n >= MIN_TRADES_FOR_TEST
        if in_family:
            pb, ph = bonf[ti], bh[ti]
            ti += 1
        else:
            pb, ph = float("nan"), float("nan")
        survives = (
            in_family
            and s.holdout_n >= MIN_TRADES_FOR_TEST
            and (s.holdout_roi or 0) > 0
            and pb < 0.05
        )
        survives_bh = (
            in_family
            and s.holdout_n >= MIN_TRADES_FOR_TEST
            and (s.holdout_roi or 0) > 0
            and ph < 0.05
        )
        rows.append(
            {
                **s.__dict__,
                "in_test_family": in_family,
                "p_bonferroni": pb,
                "p_bh": ph,
                "survives_bonferroni": survives,
                "survives_bh": survives_bh,
            }
        )

    tercile_ref = terciles_h2a or terciles_h4a or gamma_terciles_ref
    trade_meta = store.incomplete_summary()
    return {
        "holdout_event_count": len(holdout_events),
        "market_count": len(markets),
        "liquidity_segmentation_enabled": liquidity_enabled,
        "trade_segmentation_feasible": trade_seg_feasible,
        "trade_segmentation_stats": trade_seg_stats,
        "liquidity_trade_coverage_train": coverage,
        "trade_history_incomplete": trade_meta,
        "liquidity_proxy": "gamma_volume" if not liquidity_enabled else "data_api_trades",
        "liquidity_terciles_usd": (
            {"q33": tercile_ref.q33, "q66": tercile_ref.q66} if tercile_ref else None
        ),
        "liquidity_terciles_note": (
            "Volume cumulativo data-api (YES+NO) até decision_ts; cortes no treino por hipótese "
            "(H2a 24h, H2b 4h, H3 abertura+1h, H4 no instante da entrada do sinal)"
            if liquidity_enabled
            else (
                "Segmentação por tercil de trades **inviável** (API truncada / gamma_ratio); "
                "família múltipla em agregado ALL; slippage com volume Gamma do manifest"
            )
        ),
        "h3_coverage_stats": h3_stats,
        "stake_usd_assumed": 25.0,
        "n_base_hypotheses": len(tradeable_bundles),
        "n_tradeable_tests": len(tradeable_summaries),
        "h5_calibration_holdout": h5,
        "hypotheses": rows,
        "n_tests": len(tradeable_summaries),
    }
