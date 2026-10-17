#!/usr/bin/env python3
"""Executa o backtest Over/Under 2.5 pré-registrado em reports/HIPOTESES_OVER_UNDER.md."""

from __future__ import annotations

import argparse
import json
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
    bootstrap_best_strategy_null_p,
    bootstrap_roi_ci,
    calibration_bins,
    devig_binary,
    flat_bet_returns,
    pick_edge_side,
    save_variant_log,
    summarize_bets,
)
from src.validation import expanding_window_backtest  # noqa: E402

REPORT = ROOT / "reports/RELATORIO_OVER_UNDER.md"
VARIANT_LOG = ROOT / "reports" / "ou_variants_log.json"
CHANGELOG = ROOT / "reports/OU_BACKTEST_CHANGELOG.md"
CACHE = ROOT / "reports" / "ou_predictions_cache.pkl"

LAST_RUN_JSON = ROOT / "reports" / "ou_backtest_last_run.json"


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
        "ou_open_over", "ou_open_under", "ou_close_over", "ou_close_under",
        "ou_ps_over", "ou_ps_under", "ou_psc_over", "ou_psc_under",
        "ou_max_over", "ou_max_under", "ou_avg_over", "ou_avg_under",
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


def _season_layout(df: pd.DataFrame, comp: str) -> tuple[str | None, str | None]:
    """Retorna (holdout season, temporada em andamento excluída) para a liga."""
    sub = df[df.competition == comp]
    if sub.empty:
        return None, None
    g = sub.groupby("season").agg(n=("date", "count"), end=("date", "max"))
    seasons = g.sort_values("end").index.tolist()
    if not seasons:
        return None, None
    if len(seasons) == 1:
        return seasons[0], None
    last = seasons[-1]
    med_n = float(g["n"].median())
    dataset_max = df["date"].max()
    incomplete = (
        g.loc[last, "end"] >= dataset_max - pd.Timedelta(days=21)
        and g.loc[last, "n"] < 0.85 * med_n
    )
    if incomplete:
        return seasons[-2], last
    return last, None


def _holdout_mask(p: pd.DataFrame, holdout_by_comp: dict[str, str | None]) -> np.ndarray:
    return np.array([
        holdout_by_comp.get(c) is not None and s == holdout_by_comp[c]
        for c, s in zip(p.competition, p.season)
    ])


def _in_progress_mask(p: pd.DataFrame, inprog_by_comp: dict[str, str | None]) -> np.ndarray:
    return np.array([
        inprog_by_comp.get(c) is not None and s == inprog_by_comp[c]
        for c, s in zip(p.competition, p.season)
    ])


def _split_masks(
    p: pd.DataFrame,
    holdout_by_comp: dict[str, str | None],
    inprog_by_comp: dict[str, str | None],
) -> tuple[np.ndarray, np.ndarray]:
    ho = _holdout_mask(p, holdout_by_comp)
    prog = _in_progress_mask(p, inprog_by_comp)
    train = ~ho & ~prog
    return train, ho


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
        p_close[ok] = devig_binary(close_pin[ok], method=devig_method)
    return {
        "ok": ok,
        "y": y,
        "p_over": p_over,
        "open_pin": open_pin,
        "close_pin": close_pin,
        "p_open_fair": p_open,
        "p_close_fair": p_close,
        "devig": devig_method,
    }


def _masks_for(p: pd.DataFrame, holdout_by_comp, inprog_by_comp):
    return _split_masks(p, holdout_by_comp, inprog_by_comp)


def run_h1(
    frame: dict, p: pd.DataFrame, edge: float, devig: str,
    variant: str, hypothesis: str, holdout_by_comp, inprog_by_comp,
    use_close: bool = False, league_tag: str = "",
) -> list:
    results = []
    open_odds = frame["open_pin"] if not use_close else frame["close_pin"]
    p_mkt = frame["p_open_fair"] if not use_close else frame["p_close_fair"]
    suffix = f"@{league_tag}" if league_tag else ""
    for split_name, mask_split in zip(
        ("train", "holdout"), _masks_for(p, holdout_by_comp, inprog_by_comp),
    ):
        m = frame["ok"] & mask_split
        if m.sum() == 0:
            continue
        bet_m, side_over, odds_side = pick_edge_side(
            frame["p_over"], p_mkt, open_odds, edge,
        )
        bet_m &= m
        vid = f"{variant}_edge{int(edge * 100)}_{devig}{suffix}" + (
            "_close" if use_close else ""
        )
        results.append(summarize_bets(
            vid, hypothesis, split_name, bet_m, odds_side, side_over,
            frame["y"], frame["p_over"], frame["p_close_fair"], frame["p_open_fair"],
        ))
    return results


def run_h3(
    frame: dict, p: pd.DataFrame, soft_cols: tuple[str, str],
    edge: float, book: str, holdout_by_comp, inprog_by_comp,
    league_tag: str = "",
) -> list:
    from src.ou_backtest import (
        VariantResult, brier_binary, bootstrap_p_roi_nonpos,
        clv_probability, log_loss_binary,
    )
    results = []
    soft = p[list(soft_cols)].to_numpy()
    ok_soft = _odds_validas(soft)
    suffix = f"@{league_tag}" if league_tag else ""
    for split_name, mask_split in zip(
        ("train", "holdout"), _masks_for(p, holdout_by_comp, inprog_by_comp),
    ):
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
        clv = []
        for i, is_over in zip(np.where(mask)[0], side_over[mask]):
            p_entry = 1.0 / odds_side[i]
            pc = frame["p_close_fair"][i] if is_over else 1 - frame["p_close_fair"][i]
            clv.append(clv_probability(p_entry, pc))
        clv_arr = np.array(clv)
        pm = frame["p_over"][mask]
        ym = frame["y"][mask]
        pc = frame["p_close_fair"][mask]
        results.append(VariantResult(
            variant_id=f"H3_{book}_e{int(edge * 100)}{suffix}",
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


def run_h4(
    frame: dict, p: pd.DataFrame, fade: bool,
    holdout_by_comp, inprog_by_comp,
) -> list:
    from src.ou_backtest import VariantResult, bootstrap_p_roi_nonpos
    results = []
    with np.errstate(divide="ignore", invalid="ignore"):
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
            past = float(np.nanmean(drift[order[:j]]))
            if not np.isfinite(past):
                continue
            has_hist[i] = True
            signal_over[i] = (past <= 0) if fade else (past > 0)
    for split_name, mask_split in zip(
        ("train", "holdout"), _masks_for(p, holdout_by_comp, inprog_by_comp),
    ):
        m = frame["ok"] & mask_split & has_hist
        side_over = signal_over
        odds_side = np.where(side_over, frame["open_pin"][:, 0], frame["open_pin"][:, 1])
        won = np.where(side_over[m], frame["y"][m] == 1, frame["y"][m] == 0)
        rets = flat_bet_returns(odds_side[m], won)
        roi, lo, hi = bootstrap_roi_ci(rets)
        results.append(VariantResult(
            variant_id="H4b_fade" if fade else "H4a_follow",
            hypothesis="H4_drift",
            split=split_name,
            n=int(m.sum()),
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


def run_h5(
    frame: dict, p: pd.DataFrame, segment_mask: np.ndarray, tag: str,
    holdout_by_comp, inprog_by_comp,
) -> list:
    edge = 0.05
    out = []
    for split_name, mask_split in zip(
        ("train", "holdout"), _masks_for(p, holdout_by_comp, inprog_by_comp),
    ):
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
    starts = p.groupby(["competition", "season"])["date"].transform("min")
    return (p["date"] - starts).dt.days <= 60


def _low_scoring_leagues_train(
    df: pd.DataFrame,
    holdout_by_comp: dict[str, str | None],
    inprog_by_comp: dict[str, str | None],
) -> set[str]:
    rows = []
    for comp in df.competition.unique():
        ho = holdout_by_comp.get(comp)
        prog = inprog_by_comp.get(comp)
        sub = df[df.competition == comp]
        sub = sub[(sub.season != ho) & (sub.season != prog)]
        rows.append(sub)
    train = pd.concat(rows, ignore_index=True) if rows else df.iloc[0:0]
    g = train.groupby("competition").apply(
        lambda x: (x.home_goals + x.away_goals).mean(), include_groups=False,
    )
    if g.empty:
        return set()
    return set(g[g <= g.median()].index)


def _holdout_odds_date_range(
    poisson_parts: list,
    holdout_by_comp: dict[str, str | None],
    inprog_by_comp: dict[str, str | None],
) -> tuple[pd.Timestamp | None, pd.Timestamp | None, int]:
    """Datas reais dos jogos no holdout com odds Pinnacle abertura válidas."""
    dates: list[pd.Timestamp] = []
    for comp, p, _rho in poisson_parts:
        p = p.copy()
        p["competition"] = comp
        open_pin = p[["ou_ps_over", "ou_ps_under"]].to_numpy()
        ok = _odds_validas(open_pin)
        _, ho = _split_masks(p, holdout_by_comp, inprog_by_comp)
        sel = ok & ho
        if sel.any():
            dates.extend(p.loc[sel, "date"].tolist())
    if not dates:
        return None, None, 0
    return min(dates), max(dates), len(dates)


def _calib_table(calib: pd.DataFrame) -> str:
    hdr = "| bin | n | pred | actual |\n|-----|---|------|--------|"
    rows = [
        f"| {r.bin_lo:.2f}-{r.bin_hi:.2f} | {r.n} | {r.pred:.3f} | {r.actual:.3f} |"
        for r in calib.itertuples()
    ]
    return hdr + "\n" + ("\n".join(rows) if rows else "")


def format_report(
    results: list,
    global_ll: dict,
    calib: pd.DataFrame,
    null_p: float,
    n_leagues: int,
    n_matches: int,
    holdout_info: str,
    n_variants: int,
    n_tests: int,
) -> str:
    lines = [
        "# Relatório — Backtest Over/Under 2,5 vs Pinnacle",
        "",
        "**Aviso:** simulação histórica apenas; sem apostas reais.",
        "",
        f"- Ligas com Pinnacle O/U: **{n_leagues}**",
        f"- Partidas OOS com odds abertura+fechamento Pinnacle: **{n_matches}**",
        f"- Holdout: {holdout_info}",
        f"- Variantes distintas: **{n_variants}**; linhas testadas (× treino/holdout): **{n_tests}**",
        f"- Pré-registro: `reports/HIPOTESES_OVER_UNDER.md`",
        f"- Changelog vs relatório anterior: `reports/OU_BACKTEST_CHANGELOG.md`",
        "",
        "## Métricas globais (modelo vs Pinnacle fechamento, devig multiplicativo)",
        "",
        "| Métrica | Modelo | Pinnacle close |",
        "|---------|--------|----------------|",
        f"| Log-loss | {global_ll['ll_model']:.4f} | {global_ll['ll_pin']:.4f} |",
        f"| Brier | {global_ll['br_model']:.4f} | {global_ll['br_pin']:.4f} |",
        f"| n jogos | {global_ll['n']} | {global_ll['n']} |",
        "",
        "Teste nulo bootstrap do melhor ROI (não é White/Hansen SPA): "
        f"**p = {null_p:.4f}**. Com dezenas de estratégias de treino e muitas "
        "com `n` pequeno (H3), o teste tem **baixo poder** para separar um "
        "único vencedor de ruído — interpretar junto com ROI, `n` e q-values.",
        "",
        "H2 (shrinkage) **não** é estimado à parte: equivale a H1 com limiar "
        "`edge/w` (ver changelog).",
        "",
        "## Resultados agregados (todas as ligas)",
        "",
        "| Hipótese | Variante | Split | n | ROI | IC95% | CLV médio | %CLV+ | "
        "p raw | p Bonf | q BH | Sobrevive |",
        "|----------|----------|-------|---|-----|-------|-----------|-------|"
        "-------|--------|------|-----------|",
    ]
    agg = [r for r in results if "@" not in r.variant_id]
    for r in sorted(agg, key=lambda x: (x.hypothesis, x.variant_id, x.split)):
        pb = r.meta.get("p_bonf", np.nan)
        qb = r.meta.get("q_bh", np.nan)
        ci = f"[{r.roi_ci_lo:+.2%}, {r.roi_ci_hi:+.2%}]" if r.n else "—"
        clv = f"{r.clv_mean:+.4f}" if np.isfinite(r.clv_mean) else "—"
        clvp = f"{r.clv_pos_pct:.1f}%" if np.isfinite(r.clv_pos_pct) else "—"
        lines.append(
            f"| {r.hypothesis} | {r.variant_id} | {r.split} | {r.n} | "
            f"{r.roi:+.2%} | {ci} | {clv} | {clvp} | "
            f"{r.p_raw:.3f} | {pb:.3f} | {qb:.3f} | {'sim' if r.survives else 'não'} |"
        )
    survivors_agg = [r for r in agg if r.survives]
    per_liga_all = [r for r in results if "@" in r.variant_id]
    survivors_pl = [r for r in per_liga_all if r.survives]
    lines += ["", "## Veredito", ""]
    if survivors_agg or survivors_pl:
        if survivors_agg:
            lines.append("**Agregado (todas as ligas):**")
            for r in survivors_agg:
                lines.append(
                    f"- **{r.variant_id}** ({r.split}): n={r.n}, ROI {r.roi:+.2%}, "
                    f"p={r.p_raw:.3f}, q_BH={r.meta.get('q_bh', float('nan')):.3f}"
                )
        if survivors_pl:
            lines += ["", "**Por liga (família de testes):**"]
            for r in survivors_pl:
                lines.append(
                    f"- **{r.variant_id}** ({r.split}): n={r.n}, ROI {r.roi:+.2%}, "
                    f"p={r.p_raw:.3f}, q_BH={r.meta.get('q_bh', float('nan')):.3f}"
                )
        if not survivors_agg:
            lines.append(
                "\nNenhuma variante **agregada** sobrevive; ver linhas por liga acima se houver."
            )
    else:
        lines.append(
            "**Nenhuma variante sobrevive** (agregado nem por liga) aos critérios pré-registrados "
            "(ROI>0, p bootstrap <0,05 e Bonferroni ou BH-FDR)."
        )
    per_liga = [r for r in results if "@" in r.variant_id and r.split == "holdout"]
    lines += [
        "",
        "## Por liga (holdout, H1a edge 5% multiplicativo)",
        "",
        "| Liga | n | ROI | p raw | q BH | Sobrevive |",
        "|------|---|-----|-------|------|-----------|",
    ]
    for r in sorted(per_liga, key=lambda x: x.variant_id):
        qb = r.meta.get("q_bh", float("nan"))
        lines.append(
            f"| {r.variant_id.split('@', 1)[-1]} | {r.n} | {r.roi:+.2%} | {r.p_raw:.3f} | "
            f"{qb:.3f} | {'sim' if r.survives else 'não'} |"
        )
    lines += [
        "",
        "## Calibração (modelo, decílicos)",
        "",
        _calib_table(calib) if len(calib) else "_Sem dados._",
        "",
        "## Limitações",
        "",
        "- Temporada em andamento excluída do treino e do holdout.",
        "- H4 ainda usa fechamentos de jogos anteriores (viés de timing fino).",
        "- `Max`/`Avg` não são totalmente executáveis.",
        "- Família de testes inclui variantes agregadas e por liga.",
        "",
    ]
    return "\n".join(lines)


def write_changelog(current: dict) -> None:
    prev: dict | None = None
    if LAST_RUN_JSON.exists():
        try:
            prev = json.loads(LAST_RUN_JSON.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            prev = None
    lines = [
        "# Changelog — execução vs `reports/ou_backtest_last_run.json` anterior",
        "",
    ]
    if prev is None:
        lines.append("_Sem execução anterior registrada; métricas atuais abaixo._")
    else:
        lines += [
            "| Métrica | Anterior | Atual |",
            "|---------|----------|-------|",
        ]
        keys = [
            ("n_matches", "n jogos OOS", ".0f"),
            ("ll_model", "Log-loss modelo", ".4f"),
            ("ll_pin", "Log-loss Pinnacle", ".4f"),
            ("survivors_agg", "Sobreviventes agregado", ".0f"),
            ("survivors_per_league", "Sobreviventes por liga", ".0f"),
            ("null_p", "Teste nulo melhor ROI", ".4f"),
            ("holdout_n", "Jogos holdout (com P abertura)", ".0f"),
        ]
        for key, label, fmt in keys:
            a, b = prev.get(key), current.get(key)
            if a is None or b is None:
                continue
            lines.append(f"| {label} | {a:{fmt}} | {b:{fmt}} |")
        lines.append("")
    lines += [
        "## Snapshot atual",
        "",
        "```json",
        json.dumps(current, indent=2, ensure_ascii=False),
        "```",
        "",
    ]
    CHANGELOG.write_text("\n".join(lines), encoding="utf-8")
    LAST_RUN_JSON.write_text(json.dumps(current, indent=2, ensure_ascii=False), encoding="utf-8")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--skip-download", action="store_true")
    ap.add_argument("--use-cache", action="store_true")
    args = ap.parse_args()

    data_path = ROOT / "data" / "matches_xg.csv"
    df = pd.read_csv(data_path, parse_dates=["date"], low_memory=False)

    has_ps_open = "ou_ps_over" in df.columns and df["ou_ps_over"].notna().sum() > 1000
    base_mask = df.psc_h.notna() & df.ou_psc_over.notna()
    if has_ps_open:
        base_mask &= df.ou_ps_over.notna()
    ligas = sorted(df[base_mask].competition.unique())

    holdout_by_comp: dict[str, str | None] = {}
    inprog_by_comp: dict[str, str | None] = {}
    for comp in ligas:
        ho, prog = _season_layout(df, comp)
        holdout_by_comp[comp] = ho
        inprog_by_comp[comp] = prog

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

    per_league_raw: list = []
    all_results: list = []

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
    low_leagues = _low_scoring_leagues_train(df, holdout_by_comp, inprog_by_comp)

    for comp, p, rho in poisson_parts:
        p = p.copy()
        p["competition"] = comp
        p_over = _over25(
            p.expected_goals_home.to_numpy(), p.expected_goals_away.to_numpy(), rho,
        )
        frame_mult = _build_frame(p, rho, p_over, "multiplicative")
        frame_power = _build_frame(p, rho, p_over, "power")
        frame_shin = _build_frame(p, rho, p_over, "shin")

        for e in (0.03, 0.05, 0.07, 0.10):
            all_results.extend(run_h1(
                frame_mult, p, e, "multiplicative", "H1a", "H1_model",
                holdout_by_comp, inprog_by_comp,
            ))
        for e in (0.03, 0.05, 0.07, 0.10):
            all_results.extend(run_h1(
                frame_power, p, e, "power", "H1b", "H1_model",
                holdout_by_comp, inprog_by_comp,
            ))
        for e in (0.05, 0.10):
            all_results.extend(run_h1(
                frame_shin, p, e, "shin", "H1_shin", "H1_model",
                holdout_by_comp, inprog_by_comp,
            ))
        all_results.extend(run_h1(
            frame_mult, p, 0.05, "multiplicative", "H1d", "H1_close_sanity",
            holdout_by_comp, inprog_by_comp, use_close=True,
        ))

        for book_cols, book in (
            (("ou_open_over", "ou_open_under"), "B365"),
            (("ou_max_over", "ou_max_under"), "Max"),
            (("ou_avg_over", "ou_avg_under"), "Avg"),
        ):
            if book_cols[0] not in p.columns:
                continue
            for edge in (0.02, 0.05):
                all_results.extend(run_h3(
                    frame_mult, p, book_cols, edge, book,
                    holdout_by_comp, inprog_by_comp,
                ))

        all_results.extend(run_h4(frame_mult, p, False, holdout_by_comp, inprog_by_comp))
        all_results.extend(run_h4(frame_mult, p, True, holdout_by_comp, inprog_by_comp))

        early = _early_season_mask(p)
        low = _low_scoring_mask(p, low_leagues)
        all_results.extend(run_h5(frame_mult, p, early, "early60d", holdout_by_comp, inprog_by_comp))
        all_results.extend(run_h5(frame_mult, p, low, "low_goals", holdout_by_comp, inprog_by_comp))
        all_results.extend(run_h5(frame_mult, p, ~early, "not_early", holdout_by_comp, inprog_by_comp))

        per_league_raw.extend(run_h1(
            frame_mult, p, 0.05, "multiplicative", "H1a", "H1_per_league",
            holdout_by_comp, inprog_by_comp, league_tag=comp,
        ))

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
                holdout_by_comp, inprog_by_comp,
            ))

    all_results = aggregate_variant_results(all_results)
    per_league = aggregate_variant_results(per_league_raw)
    combined = all_results + per_league

    p_raw = np.array([r.p_raw for r in combined])
    p_bonf = bonferroni_adjust(p_raw)
    q_bh = bh_fdr_adjust(p_raw)
    for r, pb, qb in zip(combined, p_bonf, q_bh):
        r.meta["p_bonf"] = float(pb)
        r.meta["q_bh"] = float(qb)
    apply_survival_flags(combined, p_bonf, q_bh)

    train_returns = [
        np.array(r.meta["returns"])
        for r in combined
        if r.split == "train" and r.meta.get("returns") and "@" not in r.variant_id
    ]
    null_p = bootstrap_best_strategy_null_p(train_returns) if train_returns else 1.0

    log_entries = [{
        "variant_id": r.variant_id,
        "hypothesis": r.hypothesis,
        "split": r.split,
        "n": r.n,
        "roi": r.roi,
        "p_raw": r.p_raw,
        "p_bonf": r.meta.get("p_bonf"),
        "q_bh": r.meta.get("q_bh"),
        "survives": bool(r.survives),
    } for r in combined]
    save_variant_log(VARIANT_LOG, log_entries)

    ho_min, ho_max, ho_n = _holdout_odds_date_range(
        poisson_parts, holdout_by_comp, inprog_by_comp,
    )
    n_inprog = sum(1 for v in inprog_by_comp.values() if v)
    if ho_min is not None and ho_max is not None:
        holdout_info = (
            f"código season holdout = última temporada completa por liga "
            f"({n_inprog} ligas com season em curso excluída do treino/holdout); "
            f"jogos com Pinnacle abertura no holdout: {ho_n} partidas, "
            f"{ho_min.date().isoformat()} a {ho_max.date().isoformat()} "
            f"(cobertura parcial do calendário 2025/26 — odds abertura só em parte da temporada)"
        )
    else:
        holdout_info = "sem jogos de holdout com odds de abertura Pinnacle"

    n_variants = len({r.variant_id for r in all_results})
    survivors_agg = sum(1 for r in all_results if r.survives)
    survivors_pl = sum(1 for r in per_league if r.survives)
    snapshot = {
        "n_matches": global_ll["n"],
        "ll_model": global_ll["ll_model"],
        "ll_pin": global_ll["ll_pin"],
        "survivors_agg": survivors_agg,
        "survivors_per_league": survivors_pl,
        "null_p": null_p,
        "holdout_n": ho_n,
        "holdout_date_min": ho_min.isoformat() if ho_min is not None else None,
        "holdout_date_max": ho_max.isoformat() if ho_max is not None else None,
    }
    write_changelog(snapshot)
    report = format_report(
        combined, global_ll, calib, null_p, len(poisson_parts),
        global_ll["n"], holdout_info, n_variants, len(combined),
    )
    REPORT.write_text(report, encoding="utf-8")
    print(f"\nRelatório: {REPORT}")
    print(f"Linhas na família de testes: {len(combined)}")
    print(f"Sobreviventes (agregado): {survivors_agg}; por liga: {survivors_pl}")
    return 0


def _low_scoring_mask(p: pd.DataFrame, low_leagues: set[str]) -> np.ndarray:
    return p.competition.isin(low_leagues).to_numpy()


if __name__ == "__main__":
    raise SystemExit(main())
