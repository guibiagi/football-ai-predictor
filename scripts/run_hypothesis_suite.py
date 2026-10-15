#!/usr/bin/env python3
"""Executa hipóteses de edge (CLV vs Pinnacle close) com holdout final.

Gera reports/hypothesis_suite_<date>.json e reports/RELATORIO_HIPOTESES.md
"""

from __future__ import annotations

import json
import sys
from dataclasses import asdict
from datetime import date, datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from backtest_markets import _over25
from src.hypothesis_engine import (
    ENTRY_DEFAULT,
    HOLDOUT_DAYS,
    CLV_REF,
    LOWER_TIER,
    HypothesisResult,
    bonferroni_adjusted_alpha,
    load_prediction_panel,
    run_hypothesis_on_splits,
    sim_b365_vs_pin_close_discrepancy_1x2,
    sim_model_edge_1x2,
    sim_model_edge_ou,
    sim_movement_gbm_1x2_v2,
    sim_odds_band,
    sim_pin_open_vs_b365_1x2,
    sim_xg_boost_ou,
    split_holdout,
    _blend_1x2_probs,
)

PANEL_CACHE = Path("data/panel_oos_predictions.csv")
EDGE_MODEL = 0.08
EDGE_PIN = 0.03
EDGE_GBM = 0.015
N_MIN = 300


def _build_panel(df: pd.DataFrame) -> pd.DataFrame:
    if PANEL_CACHE.exists():
        mtime = PANEL_CACHE.stat().st_mtime
        src_mtime = Path("data/matches_xg.csv").stat().st_mtime
        if mtime >= src_mtime:
            return pd.read_csv(PANEL_CACHE, parse_dates=["date"], low_memory=False)
    panel = load_prediction_panel(df)
    panel.to_csv(PANEL_CACHE, index=False)
    return panel


def _attach_over_probs(panel: pd.DataFrame) -> pd.DataFrame:
    lam_h = panel.expected_goals_home.to_numpy()
    lam_a = panel.expected_goals_away.to_numpy()
    panel = panel.copy()
    panel["p_over25"] = _over25(lam_h, lam_a, rho=0.0)
    return panel


def main() -> int:
    df = pd.read_csv("data/matches_xg.csv", parse_dates=["date"], low_memory=False)
    ligas = sorted(df[df.psc_h.notna() & df.odds_open_h.notna()].competition.unique())
    print(f"Carregando painel OOS ({len(ligas)} ligas)...")
    panel = _attach_over_probs(_build_panel(df))
    dev, hold = split_holdout(panel, HOLDOUT_DAYS)
    print(f"Painel: {len(panel)} jogos | dev={len(dev)} | holdout={len(hold)} ({HOLDOUT_DAYS}d)")

    pm_dev = dev[
        ["home_win_probability", "draw_probability", "away_win_probability"]
    ].to_numpy()
    pm_hold = hold[
        ["home_win_probability", "draw_probability", "away_win_probability"]
    ].to_numpy()
    blend_dev = _blend_1x2_probs(dev)
    blend_hold = _blend_1x2_probs(hold)
    p_over_dev = dev.p_over25.to_numpy()
    p_over_hold = hold.p_over25.to_numpy()

    results: list[HypothesisResult] = []

    specs = [
        (
            "H1_poisson_1x2_edge8",
            "Poisson 1X2 vs B365 abertura, edge≥8%",
            lambda sub, pm: sim_model_edge_1x2(sub, pm, EDGE_MODEL),
        ),
        (
            "H2_blend_isotonic_1x2_edge8",
            "Blend isotônico + mercado (1X2), edge≥8%",
            lambda sub, pm: sim_model_edge_1x2(sub, pm, EDGE_MODEL),
        ),
        (
            "H3_poisson_ou_edge8",
            "Poisson O/U 2.5 vs B365 abertura, edge≥8%",
            lambda sub, po: sim_model_edge_ou(sub, po, EDGE_MODEL),
        ),
        (
            "H4_pin_open_vs_b365_1x2",
            "Pinnacle abertura > B365 abertura ≥3pp (1X2), execução B365",
            lambda sub, _: sim_pin_open_vs_b365_1x2(sub, EDGE_PIN),
        ),
        (
            "H5_poly_proxy_pin_open_b365",
            "Proxy Polymarket: fair Pinnacle abertura vs exec B365 ≥4pp",
            lambda sub, _: sim_b365_vs_pin_close_discrepancy_1x2(sub, 0.04),
        ),
        (
            "H6_pin_open_vs_b365_edge5",
            "Pinnacle abertura > B365 ≥5pp (1X2)",
            lambda sub, _: sim_pin_open_vs_b365_1x2(sub, 0.05),
        ),
        (
            "H7_lower_tier_poisson_ou_edge8",
            "O/U Poisson, edge≥8%, só divisões inferiores / ligas menores",
            lambda sub, po: sim_model_edge_ou(
                sub, po, EDGE_MODEL, league_filter=lambda c: c in LOWER_TIER,
            ),
        ),
        (
            "H8_brasileirao_poisson_1x2_edge8",
            "Brasileirão A: Poisson 1X2 edge≥8%",
            lambda sub, pm: sim_model_edge_1x2(
                sub, pm, EDGE_MODEL, league_filter=lambda c: c == "Brasileirao Serie A",
            ),
        ),
        (
            "H9_odds_band_18_35_poisson",
            "Poisson 1X2, edge≥8%, odds decimais 1.8–3.5 no pick",
            lambda sub, pm: sim_odds_band(sub, pm, EDGE_MODEL, 1.8, 3.5),
        ),
        (
            "H10_poisson_ou_edge5",
            "Poisson O/U edge≥5% (filtro mais solto)",
            lambda sub, po: sim_model_edge_ou(sub, po, 0.05),
        ),
        (
            "H11_movement_gbm_1x2",
            "GBM prevê CLV (lado pin−b365), mensal walk-forward",
            lambda sub, _: sim_movement_gbm_1x2_v2(sub, edge_min=EDGE_GBM),
        ),
        (
            "H12_xg_boost_ou",
            "GBM O/U (λ Poisson + mercado abertura) prevê CLV",
            lambda sub, _: sim_xg_boost_ou(sub, df, edge_min=0.02),
        ),
        (
            "H4b_pin_open_vs_b365_1pp",
            "Pinnacle abertura > B365 abertura ≥1pp (1X2) — timing cross-book",
            lambda sub, _: sim_pin_open_vs_b365_1x2(sub, 0.01),
        ),
    ]

    model_map = {
        "H1_poisson_1x2_edge8": (pm_dev, pm_hold),
        "H2_blend_isotonic_1x2_edge8": (blend_dev, blend_hold),
        "H3_poisson_ou_edge8": (p_over_dev, p_over_hold),
        "H7_lower_tier_poisson_ou_edge8": (p_over_dev, p_over_hold),
        "H8_brasileirao_poisson_1x2_edge8": (pm_dev, pm_hold),
        "H9_odds_band_18_35_poisson": (pm_dev, pm_hold),
        "H10_poisson_ou_edge5": (p_over_dev, p_over_hold),
    }

    for hid, desc, fn_template in specs:
        extra = model_map.get(hid)
        if extra:
            m_dev, m_hold = extra
            fn_dev = lambda d, t=fn_template, m=m_dev: t(d, m)
            fn_hold = lambda h, t=fn_template, m=m_hold: t(h, m)
        else:
            fn_dev = lambda d, t=fn_template: t(d, None)
            fn_hold = fn_dev

        res = run_hypothesis_on_splits(
            hid, desc, dev, hold, fn_dev, fn_hold, n_min=N_MIN,
        )
        results.append(res)
        print(
            f"  {hid}: holdout n={res.n_holdout} CLV={res.clv_mean_holdout} "
            f"CI={res.clv_ci95_holdout} viable={res.viable}"
        )

    n_tests = len(results)
    alpha_adj = bonferroni_adjusted_alpha(n_tests)
    payload = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "meta": {
            "holdout_days": HOLDOUT_DAYS,
            "clv_reference": CLV_REF,
            "entry_odds": ENTRY_DEFAULT,
            "n_hypotheses": n_tests,
            "bonferroni_alpha_per_test": round(alpha_adj, 6),
            "viability_rule": f"holdout n>={N_MIN}, CLV mean>0, CI95 lower>0",
        },
        "hypotheses": [asdict(r) for r in results],
    }

    reports = Path("reports")
    reports.mkdir(exist_ok=True)
    out_json = reports / f"hypothesis_suite_{date.today().strftime('%Y%m%d')}.json"
    out_json.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")

    # Markdown report
    lines = [
        "# Relatório de hipóteses (CLV vs fechamento Pinnacle)",
        "",
        f"Gerado em: {payload['generated_at']}",
        f"Holdout: últimos **{HOLDOUT_DAYS}** dias ({len(hold)} jogos no painel).",
        f"Entrada: **{ENTRY_DEFAULT}** | CLV ref: **{CLV_REF}**",
        "",
        f"Critério de viabilidade: n≥{N_MIN} no holdout, CLV médio > 0, IC 95% (bootstrap) inteiro acima de 0.",
        f"Bonferroni (referência): α ajustado ≈ {alpha_adj:.4f} para {n_tests} testes.",
        "",
        "| Hipótese | n (dev) | CLV dev | IC CLV dev | ROI dev | n (hold) | CLV hold | IC CLV hold | ROI hold | Viável? |",
        "|----------|---------|---------|------------|---------|----------|----------|-------------|----------|---------|",
    ]
    for r in results:
        lines.append(
            f"| {r.id} | {r.n_dev} | {r.clv_mean_dev:.4f} | [{r.clv_ci95_dev[0]:.4f}, {r.clv_ci95_dev[1]:.4f}] | "
            f"{r.roi_dev:.3f} | {r.n_holdout} | {r.clv_mean_holdout:.4f} | "
            f"[{r.clv_ci95_holdout[0]:.4f}, {r.clv_ci95_holdout[1]:.4f}] | {r.roi_holdout:.3f} | "
            f"{'**sim**' if r.viable else 'não'} |"
        )
    lines.extend(["", "## Descrições", ""])
    for r in results:
        lines.append(f"- **{r.id}**: {r.description}")

    viable = [r for r in results if r.viable]
    lines.extend(["", "## Conclusão", ""])
    if viable:
        lines.append(
            "Estratégia(s) passaram o critério de holdout: "
            + ", ".join(r.id for r in viable)
            + "."
        )
        if any(r.id.startswith("H4b") for r in viable):
            lines.append(
                "**Atenção:** H4b executa na **Bet365 abertura**, não na Polymarket. "
                "É sinal de *cross-book* (Pinnacle abertura como fair), alinhado com a literatura "
                "de que edge direcional vs Pinnacle na Poly é fraco; não ativa o monitor Polymarket "
                "automaticamente."
            )
    else:
        lines.append(
            "Nenhuma hipótese atingiu viabilidade no holdout (CLV>0 com IC 95% acima de zero e n≥300)."
        )
        promising = sorted(
            results,
            key=lambda r: (r.clv_ci95_holdout[1] if r.clv_ci95_holdout else -999, r.clv_mean_holdout),
            reverse=True,
        )[:3]
        lines.append("Mais promissoras (pelo limite superior do IC de CLV no holdout):")
        for r in promising:
            lines.append(
                f"- {r.id}: CLV hold={r.clv_mean_holdout:.4f}, IC=[{r.clv_ci95_holdout[0]:.4f}, {r.clv_ci95_holdout[1]:.4f}], n={r.n_holdout}"
            )

    lines.extend(["", "## Recomendação de stake", ""])
    if any(r.viable for r in results):
        lines.append(
            "Para **H4b** (se operar manualmente na B365): stake fixa 0,5–1% da banca por aposta, "
            "máximo 3 apostas simultâneas; edge mínimo 1pp Pinnacle abertura vs B365 abertura no "
            "mesmo resultado; parar se CLV rolling 100 apostas ≤ 0. **Não** escalar sem forward test."
        )
    else:
        lines.append(
            "Sem edge validado no holdout na Polymarket, **não há base para stake real**. "
            "Maker: ver seção Polymarket abaixo após `scripts/backtest_poly_maker.py`."
        )

    md_path = reports / "RELATORIO_HIPOTESES.md"
    md_path.write_text("\n".join(lines), encoding="utf-8")
    try:
        import subprocess
        subprocess.run([sys.executable, "scripts/merge_hypothesis_reports.py"], check=False)
    except Exception:
        pass
    print(f"\nJSON: {out_json}\nMD: {md_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
