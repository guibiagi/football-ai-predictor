#!/usr/bin/env python3
"""Executa o backtest Over/Under 2.5 pré-registrado em reports/HIPOTESES_OVER_UNDER.md."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from backtest_markets import (  # noqa: E402
    INITIAL_TRAIN_YEARS,
    MODEL_PARAMS,
    PoissonModel,
    STEP_MONTHS,
    _over25,
    _odds_validas,
)
from backtest_skill import MIN_MATCHES  # noqa: E402
from src.blend_model import BlendModel  # noqa: E402
from src.ou_backtest import (  # noqa: E402
    aggregate_variant_results,
    apply_survival_flags,
    bh_fdr_adjust,
    bonferroni_adjust,
    bootstrap_roi_ci,
    calibration_bins,
    devig_binary,
    flat_bet_returns,
    pick_edge_side,
    save_variant_log,
    summarize_bets,
    white_reality_check_p,
)
from src.validation import expanding_window_backtest  # noqa: E402

REPORT = ROOT / "reports" / "RELATORIO_OVER_UNDER.md"
VARIANT_LOG = ROOT / "reports" / "ou_variants_log.json"
CACHE = ROOT / "reports" / "ou_predictions_cache.pkl"


def _roda(df: pd.DataFrame, comp: str, model_factory) -> pd.DataFrame | None:
    sub = df[df["competition"] == comp].dropna(
        subset=["home_goals", "away_goals", "psc_h", "b365c_h"]
    )
    if len(sub) < MIN_MATCHES:
        return None
    preds = expanding_window_backtest(
        sub, model_factory,
        initial_train_years=INITIAL_TRAIN_YEARS, step_months=STEP_MONTHS,
    )
    cols = [
        "date", "home_team", "away_team", "season",
        "odds_open_h", "odds_open_d", "odds_open_a",
        "psc_h", "psc_d", "psc_a", "b365c_h", "b365c_d", "b365c_a",
        "ou_open_over", "ou_open_under",
        "ou_close_over", "ou_close_under",
        "ou_ps_over", "ou_ps_under",
        "ou_psc_over", "ou_psc_under",
        "ou_max_over", "ou_max_under",
        "ou_avg_over", "ou_avg_under",
        "xg_h", "xg_a",
    ]
    preds = preds.merge(
        sub[[c for c in cols if c in sub.columns]],
        on=["date", "home_team", "away_team"], how="inner",
    )
    return preds if len(preds) >= 100 else None


def _rho(df: pd.DataFrame, comp: str, p: pd.DataFrame) -> float:
    corte = p.date.min()
    try:
        return PoissonModel(**MODEL_PARAMS).fit(
            df[(df.competition == comp) & (df.date < corte)]
        )._rho
    except ValueError:
        return 0.0


def _holdout_mask(p: pd.DataFrame) -> np.ndarray:
    last = p.groupby("competition")["season"].transform("max")
    return (p["season"] == last).to_numpy()


def _build_frame(
    p: pd.DataFrame, rho: float, p_over: np.ndarray, devig_method: str,
) -> dict:
    open_pin = p[["ou_ps_over", "ou_ps_under"]].to_numpy()
    close_pin = p[["ou_psc_over", "ou_psc_under"]].to_numpy()
    ok = _odds_validas(open_pin) & _odds_validas(close_pin)
    y = (p.home_goals + p.away_goals > 2.5).astype(float).to_numpy()
    p_open = np.full(len(p), np.nan)
    p_close = np.full(len(p), np.nan)
    if ok.any():
        p_open[ok] = devig_binary(open_pin[ok], method=devig_method)
        p_close[ok] = devig_binary(close_pin[ok], method="multiplicative")
    return {
        "ok": ok,
        "y": y,
        "p_over": p_over,
        "open_pin": open_pin,
        "close_pin": close_pin,
        "p_open_fair": p_open,
        "p_close_fair": p_close,
    }


def _split_masks(p: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    ho = _holdout_mask(p)
    return ~ho, ho


def run_h1(frame: dict, p: pd.DataFrame, edge: float, devig: str,
           variant: str, hypothesis: str, use_close: bool = False) -> list:
    results = []
    open_odds = frame["open_pin"] if not use_close else frame["close_pin"]
    p_mkt = frame["p_open_fair"] if not use_close else frame["p_close_fair"]
    for split_name, mask_split in (("train", _split_masks(p)[0]), ("holdout", _split_masks(p)[1])):
        m = frame["ok"] & mask_split
        if m.sum() == 0:
            continue
        bet_m, side_over, odds_side = pick_edge_side(
            frame["p_over"], p_mkt, open_odds, edge,
        )
        bet_m &= m
        vid = f"{variant}_edge{int(edge*100)}_{devig}" + ("_close" if use_close else "")
        results.append(summarize_bets(
            vid, hypothesis, split_name, bet_m, odds_side, side_over,
            frame["y"], frame["p_over"], frame["p_close_fair"], frame["p_open_fair"],
        ))
    return results


def run_h2(frame: dict, p: pd.DataFrame, w: float, edge: float = 0.05) -> list:
    results = []
    p_blend = w * frame["p_over"] + (1 - w) * frame["p_open_fair"]
    p_blend = np.clip(p_blend, 1e-6, 1 - 1e-6)
    for split_name, mask_split in (("train", _split_masks(p)[0]), ("holdout", _split_masks(p)[1])):
        m = frame["ok"] & mask_split
        bet_m, side_over, odds_side = pick_edge_side(
            p_blend, frame["p_open_fair"], frame["open_pin"], edge,
        )
        bet_m &= m
        vid = f"H2_w{w:.2f}"
        results.append(summarize_bets(
            vid, "H2_shrink", split_name, bet_m, odds_side, side_over,
            frame["y"], p_blend, frame["p_close_fair"], frame["p_open_fair"],
        ))
    return results


def run_h3(frame: dict, p: pd.DataFrame, soft_cols: tuple[str, str],
           edge: float, book: str) -> list:
    results = []
    soft = p[list(soft_cols)].to_numpy()
    ok_soft = _odds_validas(soft)
    for split_name, mask_split in (("train", _split_masks(p)[0]), ("holdout", _split_masks(p)[1])):
        m = frame["ok"] & ok_soft & mask_split
        if m.sum() == 0:
            continue
        p_fair = frame["p_open_fair"]
        ev_over = p_fair * soft[:, 0] - 1
        ev_under = (1 - p_fair) * soft[:, 1] - 1
        bet_over = ev_over >= edge
        bet_under = ev_under >= edge
        both = bet_over & bet_under
        side_over = np.where(both, ev_over >= ev_under, bet_over)
        mask = (bet_over | bet_under) & m
        odds_side = np.where(side_over, soft[:, 0], soft[:, 1])
        won = np.where(side_over[mask], frame["y"][mask] == 1, frame["y"][mask] == 0)
        rets = flat_bet_returns(odds_side[mask], won)
        roi, lo, hi = bootstrap_roi_ci(rets)
        from src.ou_backtest import bootstrap_p_roi_nonpos, clv_probability
        clv = []
        for i, is_over in zip(np.where(mask)[0], side_over[mask]):
            pe = p_fair[i] if is_over else 1 - p_fair[i]
            pc = frame["p_close_fair"][i] if is_over else 1 - frame["p_close_fair"][i]
            clv.append(clv_probability(pe, pc))
        clv_arr = np.array(clv)
        from src.ou_backtest import VariantResult, log_loss_binary, brier_binary
        pm = frame["p_over"][mask]
        ym = frame["y"][mask]
        pc = frame["p_close_fair"][mask]
        results.append(VariantResult(
            variant_id=f"H3_{book}_e{int(edge*100)}",
            hypothesis="H3_soft",
            split=split_name,
            n=int(mask.sum()),
            roi=roi, roi_ci_lo=lo, roi_ci_hi=hi,
            clv_mean=float(np.nanmean(clv_arr)),
            clv_pos_pct=float(np.nanmean(clv_arr > 0) * 100),
            p_raw=bootstrap_p_roi_nonpos(rets),
            log_loss_model=log_loss_binary(pm, ym),
            log_loss_pin=log_loss_binary(pc, ym),
            brier_model=brier_binary(pm, ym),
            brier_pin=brier_binary(pc, ym),
            survives=False,
            meta={
                "returns": rets.tolist(),
                "clv": clv_arr.tolist(),
                "p_model": pm.tolist(),
                "y": ym.tolist(),
                "p_close": pc.tolist(),
            },
        ))
    return results


def run_h4(frame: dict, p: pd.DataFrame, fade: bool = False) -> list:
    """Drift histórico expansivo por liga (sem fechamento do jogo atual)."""
    results = []
    open_imp = 1.0 / frame["open_pin"][:, 0]
    close_imp = 1.0 / frame["close_pin"][:, 0]
    drift = close_imp - open_imp
    signal_over = np.zeros(len(p), dtype=bool)
    has_hist = np.zeros(len(p), dtype=bool)
    for comp in p.competition.unique():
        idx = np.where(p.competition.to_numpy() == comp)[0]
        order = idx[np.argsort(p.date.iloc[idx].values)]
        for j, i in enumerate(order):
            if j == 0:
                continue
            past = drift[order[:j]].mean()
            has_hist[i] = True
            if fade:
                signal_over[i] = past <= 0
            else:
                signal_over[i] = past > 0
    for split_name, mask_split in (("train", _split_masks(p)[0]), ("holdout", _split_masks(p)[1])):
        m = frame["ok"] & mask_split & has_hist
        side_over = signal_over
        mask = m
        odds_side = np.where(side_over, frame["open_pin"][:, 0], frame["open_pin"][:, 1])
        won = np.where(side_over[mask], frame["y"][mask] == 1, frame["y"][mask] == 0)
        rets = flat_bet_returns(odds_side[mask], won)
        roi, lo, hi = bootstrap_roi_ci(rets)
        from src.ou_backtest import VariantResult, bootstrap_p_roi_nonpos, brier_binary, log_loss_binary
        results.append(VariantResult(
            variant_id="H4b_fade" if fade else "H4a_follow",
            hypothesis="H4_drift",
            split=split_name,
            n=int(mask.sum()),
            roi=roi, roi_ci_lo=lo, roi_ci_hi=hi,
            clv_mean=float("nan"), clv_pos_pct=float("nan"),
            p_raw=bootstrap_p_roi_nonpos(rets),
            log_loss_model=float("nan"),
            log_loss_pin=float("nan"),
            brier_model=float("nan"),
            brier_pin=float("nan"),
            survives=False,
            meta={"returns": rets.tolist()},
        ))
    return results


def run_h5(frame: dict, p: pd.DataFrame, segment_mask: np.ndarray, tag: str) -> list:
    edge = 0.05
    out = []
    for split_name, mask_split in (("train", _split_masks(p)[0]), ("holdout", _split_masks(p)[1])):
        m = frame["ok"] & mask_split & segment_mask
        bet_m, side_over, odds_side = pick_edge_side(
            frame["p_over"], frame["p_open_fair"], frame["open_pin"], edge,
        )
        bet_m &= m
        out.append(summarize_bets(
            f"H5_{tag}", "H5_segment", split_name, bet_m, odds_side, side_over,
            frame["y"], frame["p_over"], frame["p_close_fair"], frame["p_open_fair"],
        ))
    return out


def _early_season_mask(p: pd.DataFrame) -> np.ndarray:
    p = p.copy()
    p["season_start"] = p.groupby(["competition", "season"])["date"].transform("min")
    return (p["date"] - p["season_start"]).dt.days <= 60


def _low_scoring_leagues(df: pd.DataFrame) -> set[str]:
    """Ligas com média de gols abaixo da mediana global (fixo no dataset)."""
    g = df.groupby("competition").apply(
        lambda x: (x.home_goals + x.away_goals).mean(), include_groups=False,
    )
    return set(g[g <= g.median()].index)


def _low_scoring_mask(p: pd.DataFrame, low_leagues: set[str]) -> np.ndarray:
    return p.competition.isin(low_leagues).to_numpy()


def _calib_table(calib: pd.DataFrame) -> str:
    hdr = "| bin | n | pred | actual |\n|-----|---|------|--------|"
    rows = [
        f"| {r.bin_lo:.2f}-{r.bin_hi:.2f} | {r.n} | {r.pred:.3f} | {r.actual:.3f} |"
        for r in calib.itertuples()
    ]
    return hdr + "\n" + "\n".join(rows)


def format_report(
    results: list,
    global_ll: dict,
    calib: pd.DataFrame,
    white_p: float,
    n_leagues: int,
    n_matches: int,
) -> str:
    lines = [
        "# Relatório — Backtest Over/Under 2,5 vs Pinnacle",
        "",
        "**Aviso:** simulação histórica apenas; sem apostas reais.",
        "",
        f"- Ligas com Pinnacle O/U: **{n_leagues}**",
        f"- Partidas com predição + odds abertura Pinnacle: **{n_matches}**",
        f"- Pré-registro: `reports/HIPOTESES_OVER_UNDER.md`",
        "",
        "## Métricas globais (modelo vs Pinnacle fechamento, sem vig)",
        "",
        f"| Métrica | Modelo | Pinnacle close |",
        f"|---------|--------|----------------|",
        f"| Log-loss | {global_ll['ll_model']:.4f} | {global_ll['ll_pin']:.4f} |",
        f"| Brier | {global_ll['br_model']:.4f} | {global_ll['br_pin']:.4f} |",
        f"| n jogos | {global_ll['n']} | {global_ll['n']} |",
        "",
        f"White reality check (melhor ROI vs permutação): **p = {white_p:.4f}**",
        "",
        "## Resultados por hipótese (agregado em todas as ligas; treino vs holdout)",
        "",
        "| Hipótese | Variante | Split | n | ROI | IC95% | CLV médio | %CLV+ | p raw | p Bonf | q BH | Sobrevive |",
        "|----------|----------|-------|---|-----|-------|-----------|-------|-------|--------|------|-----------|",
    ]
    for r in results:
        pb = r.meta.get("p_bonf", np.nan)
        qb = r.meta.get("q_bh", np.nan)
        ci = f"[{r.roi_ci_lo:+.2%}, {r.roi_ci_hi:+.2%}]" if r.n else "—"
        lines.append(
            f"| {r.hypothesis} | {r.variant_id} | {r.split} | {r.n} | "
            f"{r.roi:+.2%} | {ci} | {r.clv_mean:+.4f} | {r.clv_pos_pct:.1f}% | "
            f"{r.p_raw:.3f} | {pb:.3f} | {qb:.3f} | {'sim' if r.survives else 'não'} |"
        )
    survivors = [r for r in results if r.survives]
    lines += [
        "",
        "## Veredito",
        "",
    ]
    if survivors:
        lines.append("Variantes que passam os critérios pré-registrados (ver nota de amostra):")
        for r in survivors:
            caveat = " ⚠️ amostra pequena" if r.n < 100 else ""
            lines.append(
                f"- **{r.variant_id}** ({r.split}): n={r.n}, ROI {r.roi:+.2%}, "
                f"p={r.p_raw:.3f}, q_BH={r.meta.get('q_bh', float('nan')):.3f}{caveat}"
            )
        lines.append(
            "\n**Nota:** sobrevivência estatística com `n` muito baixo (ex.: holdout) "
            "não implica edge real — interpretar com extrema cautela."
        )
    else:
        lines.append(
            "**Nenhuma variante sobrevive** aos critérios (ROI>0, p bootstrap <0,05 e "
            "ajuste múltiplo Bonferroni ou BH-FDR) em treino ou holdout."
        )
    lines += [
        "",
        "## Calibração (modelo, decílicos)",
        "",
        _calib_table(calib) if len(calib) else "_Sem dados._",
        "",
        "## Limitações",
        "",
        "- Holdout = última temporada por liga; amostra pode ser pequena.",
        "- Ligas sem `P>2.5` excluídas.",
        "- xG só em subconjunto (H1c).",
        "- Múltiplos testes; Bonferroni é conservador.",
        "- Não modela execução, limites ou comissão.",
        "",
    ]
    return "\n".join(lines)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--skip-download", action="store_true")
    ap.add_argument("--use-cache", action="store_true")
    args = ap.parse_args()

    data_path = ROOT / "data" / "matches_xg.csv"
    df = pd.read_csv(data_path, parse_dates=["date"], low_memory=False)
    if "ou_ps_over" not in df.columns or df["ou_ps_over"].notna().sum() < 1000:
        if not args.skip_download:
            import importlib.util
            spec = importlib.util.spec_from_file_location(
                "merge_ou", ROOT / "scripts" / "merge_ou_open_columns.py",
            )
            mod = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(mod)
            print("Colunas Pinnacle abertura ausentes — merge...")
            mod.main()
            df = pd.read_csv(data_path, parse_dates=["date"], low_memory=False)

    has_ps_open = "ou_ps_over" in df.columns and df["ou_ps_over"].notna().sum() > 1000
    base_mask = df.psc_h.notna() & df.ou_psc_over.notna()
    if has_ps_open:
        base_mask &= df.ou_ps_over.notna()
    ligas = sorted(df[base_mask].competition.unique())
    if not has_ps_open:
        print("AVISO: sem ou_ps_over — rode scripts/merge_ou_open_columns.py")

    all_results: list = []
    log_entries: list[dict] = []
    frames_poisson = []
    frames_blend = []

    import pickle

    if args.use_cache and CACHE.exists():
        with CACHE.open("rb") as f:
            cached = pickle.load(f)
        poisson_parts = cached["poisson"]
        blend_parts = cached.get("blend", [])
    else:
        poisson_parts, blend_parts = [], []
        for comp in ligas:
            p = _roda(df, comp, lambda: PoissonModel(**MODEL_PARAMS))
            if p is None:
                continue
            rho = _rho(df, comp, p)
            poisson_parts.append((comp, p, rho))
            if p.xg_h.notna().sum() > 200:
                pb = _roda(df, comp, lambda: BlendModel(**MODEL_PARAMS, pesos={"gols": 0.5, "xg": 0.5}))
                if pb is not None:
                    blend_parts.append((comp, pb, rho))
            print(f"  {comp}: {len(p)} preds", flush=True)
        CACHE.parent.mkdir(parents=True, exist_ok=True)
        with CACHE.open("wb") as f:
            pickle.dump({"poisson": poisson_parts, "blend": blend_parts}, f)

    # Global skill metrics
    ys, pm, pc = [], [], []
    for comp, p, rho in poisson_parts:
        fr = _build_frame(p, rho, _over25(
            p.expected_goals_home.to_numpy(), p.expected_goals_away.to_numpy(), rho,
        ), "multiplicative")
        m = fr["ok"]
        ys.append(fr["y"][m])
        pm.append(fr["p_over"][m])
        pc.append(fr["p_close_fair"][m])
    ys = np.concatenate(ys)
    pm = np.concatenate(pm)
    pc = np.concatenate(pc)
    from src.ou_backtest import brier_binary, log_loss_binary
    global_ll = {
        "n": len(ys),
        "ll_model": log_loss_binary(pm, ys),
        "ll_pin": log_loss_binary(pc, ys),
        "br_model": brier_binary(pm, ys),
        "br_pin": brier_binary(pc, ys),
    }
    calib = calibration_bins(pm, ys)
    low_leagues = _low_scoring_leagues(df)

    for comp, p, rho in poisson_parts:
        p = p.copy()
        p["competition"] = comp
        p_over = _over25(
            p.expected_goals_home.to_numpy(), p.expected_goals_away.to_numpy(), rho,
        )
        for devig in ("multiplicative", "power"):
            frame = _build_frame(p, rho, p_over, devig)
            edges = (0.03, 0.05, 0.07, 0.10) if devig == "multiplicative" else (0.05,)
            for e in edges:
                all_results.extend(run_h1(
                    frame, p, e, devig, "H1a" if devig == "multiplicative" else "H1b", "H1_model",
                ))
        all_results.extend(run_h1(
            frame, p, 0.05, "multiplicative", "H1d", "H1_close_sanity", use_close=True,
        ))
        for w in (0.75, 0.50, 0.25):
            all_results.extend(run_h2(frame, p, w))
        if "ou_max_over" in p.columns:
            all_results.extend(run_h3(frame, p, ("ou_open_over", "ou_open_under"), 0.02, "B365"))
            all_results.extend(run_h3(frame, p, ("ou_open_over", "ou_open_under"), 0.05, "B365"))
            all_results.extend(run_h3(frame, p, ("ou_max_over", "ou_max_under"), 0.02, "Max"))
            all_results.extend(run_h3(frame, p, ("ou_max_over", "ou_max_under"), 0.05, "Max"))
        all_results.extend(run_h4(frame, p, fade=False))
        all_results.extend(run_h4(frame, p, fade=True))
        early = _early_season_mask(p)
        low = _low_scoring_mask(p, low_leagues)
        all_results.extend(run_h5(frame, p, early, "early60d"))
        all_results.extend(run_h5(frame, p, low, "low_goals"))
        all_results.extend(run_h5(frame, p, ~early, "not_early"))

    for comp, p, rho in blend_parts:
        p = p.copy()
        p["competition"] = comp
        p_over = _over25(
            p.expected_goals_home.to_numpy(), p.expected_goals_away.to_numpy(), rho,
        )
        frame = _build_frame(p, rho, p_over, "multiplicative")
        for e in (0.05, 0.10):
            all_results.extend(run_h1(
                frame, p, e, "multiplicative", "H1c_blend", "H1_blend_xg",
            ))

    all_results = aggregate_variant_results(all_results)

    # Multiple testing (uma linha por variante × split, agregada em todas as ligas)
    p_raw = np.array([r.p_raw for r in all_results])
    p_bonf = bonferroni_adjust(p_raw)
    q_bh = bh_fdr_adjust(p_raw)
    for r, pb, qb in zip(all_results, p_bonf, q_bh):
        r.meta["p_bonf"] = float(pb)
        r.meta["q_bh"] = float(qb)
    apply_survival_flags(all_results, p_bonf, q_bh)

    train_return_series = [
        np.array(r.meta["returns"])
        for r in all_results
        if r.split == "train" and r.meta.get("returns")
    ]
    white_p = white_reality_check_p(train_return_series) if train_return_series else 1.0

    for r in all_results:
        log_entries.append({
            "variant_id": r.variant_id,
            "hypothesis": r.hypothesis,
            "split": r.split,
            "n": r.n,
            "roi": r.roi,
            "p_raw": r.p_raw,
            "survives": bool(r.survives),
        })
    save_variant_log(VARIANT_LOG, log_entries)

    n_matches = global_ll["n"]
    report = format_report(all_results, global_ll, calib, white_p, len(poisson_parts), n_matches)
    REPORT.write_text(report, encoding="utf-8")
    print(f"\nRelatório: {REPORT}")
    print(f"Variantes testadas: {len(all_results)}")
    print(f"Sobreviventes: {sum(r.survives for r in all_results)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
