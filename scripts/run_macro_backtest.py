#!/usr/bin/env python3
"""Executa backtest macro e grava JSON + relatório markdown."""

from __future__ import annotations

import argparse
import json
from dataclasses import fields
from pathlib import Path

from src.macro_poly.backtest import run_full_backtest
from src.macro_poly.discovery import MacroMarket

CHANGELOG = """
## Changelog vs v3 (re-review Opus 5559976794)

| Correção | Efeito |
|----------|--------|
| Trades data-api: paginação `before`, YES+NO, meta vs Gamma | Volume completo; mercados incompletos sinalizados |
| H3: volume zero → T1_low (não drop silencioso) | Cobertura H3 em `h3_coverage_stats` |
| Tercis H2b/H4 no instante de entrada (treino) | Segmentos alinhados à regra de trade |
| H3b favorito pré-reg **0,60** (FOMC) | Restaura limiar documentado |
| Calendário FOMC 2023 | Releases 2023 quando existirem no manifest |

## Changelog vs v2 (abb273a)

| Correção | Efeito |
|----------|--------|
| FOMC exige Fed/FOMC; exclui ECB/BoJ/derivativos | Menos contaminação em H2/H4/H5 |
| Dedupe um evento FOMC por reunião (ex. set/2026) | Um outcome por meeting |
| Volume = trades data-api cumulativos até `decision_ts` | Tercis alinhados ao instante da entrada |
| H3 janela desde `start_ts` | Cobertura H3 reportada (`h3_coverage_stats`) |
| H2b estado resetado por limiar; H2a deduplicado | Família de testes menor |
| Células com n&lt;3 fora da família múltipla | `in_test_family` no JSON |

## Changelog vs relatório anterior (PR #4 v1)

| Correção | Efeito esperado nos números |
|----------|----------------------------|
| Removido teto `min(0.99, …)` na entrada | ROI H1 deixa de colapsar em ~0,99% artificial |
| H1 excluído de sobreviventes / família de testes ROI | Nenhum "survivor" tradeável vem de H1 |
| Timestamps FOMC oficiais (14:00 ET, calendário Fed) + filtro de mercados | H1c/H2 FOMC perdem eventos de menção e derivativos |
| Sem fallback de preço stale em `price_at_or_after` | Menos trades; entradas só com candle dentro da tolerância |
| Holdout determinístico `(release_ts, event_id)` | `holdout_n` / ROI holdout estáveis entre runs |
| Tercis: volume interpolado + cortes só no treino | Segmentos T1/T2/T3 realocados |
| H4a/H4b direção corrigida (fade vs continuação) | ROI H4 alterado |
| Cache de preços por janela `(token, start, end)` | H3 usa abertura real do mercado |
| H5 calibração no holdout | Nova seção de log-loss/Brier agregado |
"""


def _load_manifest(path: Path) -> list[MacroMarket]:
    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    field_names = {f.name for f in fields(MacroMarket)}
    markets = []
    for row in data["markets"]:
        clean = {k: row[k] for k in field_names if k in row}
        if "start_ts" not in clean:
            clean["start_ts"] = None
        if "condition_id" not in clean:
            clean["condition_id"] = ""
        if "token_no" not in clean:
            clean["token_no"] = None
        markets.append(MacroMarket(**clean))
    return markets


def _write_report(results: dict, path: Path) -> None:
    n_trade = results["n_tests"]
    lines = [
        "# Relatório — Backtest Polymarket macro (Fed / inflação)",
        "",
        "**Modo:** somente simulação (sem ordens reais).  ",
        f"**Mercados no manifest:** {results['market_count']}  ",
        f"**Eventos no holdout (~30%):** {results['holdout_event_count']}  ",
        f"**Liquidez por trades:** {'sim' if results.get('liquidity_segmentation_enabled') else 'não'} "
        f"(cobertura treino={results.get('liquidity_trade_coverage_train', '—')}; "
        f"histórico incompleto={results.get('trade_history_incomplete', {})})  ",
        f"**Testes ROI na família múltipla:** {n_trade}  ",
        f"**Tercis (volume trades cumulativo, treino):** "
        f"q33={results.get('liquidity_terciles_usd', {}).get('q33', '—') if results.get('liquidity_terciles_usd') else '—'}, "
        f"q66={results.get('liquidity_terciles_usd', {}).get('q66', '—') if results.get('liquidity_terciles_usd') else '—'}  ",
        f"**Stake simulado:** US$ {results.get('stake_usd_assumed', 25)}  ",
        "",
        "Pré-registro: [HIPOTESES_MACRO_POLY_PREREG.md](HIPOTESES_MACRO_POLY_PREREG.md)",
        "",
        CHANGELOG.strip(),
        "",
        "### H5 — Calibração (holdout, preço 1h antes do release)",
        "",
    ]
    h5 = results.get("h5_calibration_holdout") or {}
    lines.append(
        f"n={h5.get('n', 0)}, log-loss={h5.get('log_loss')}, Brier={h5.get('brier')}"
    )
    h3s = results.get("h3_coverage_stats") or {}
    if h3s:
        lines.append("")
        lines.append("### H3 — cobertura (mercados elegíveis vs trades simulados)")
        for k, st in h3s.items():
            lines.append(
                f"- **{k}**: elegíveis={st.get('eligible', 0)}, "
                f"trades={st.get('traded', 0)}, sem histórico={st.get('dropped_no_history', 0)}, "
                f"sem preço na abertura={st.get('dropped_no_price', 0)}, "
                f"trades API incompleto={st.get('dropped_incomplete_trade_history', 0)}, "
                f"volume zero→T1={st.get('zero_volume_assigned_t1', 0)}"
            )
    lines.extend(
        [
            "",
            "## H1 — Diagnóstico (não tradeável; não entra em sobreviventes)",
            "",
            "| ID | tercil | n | ROI médio | IC 95% | CLV médio | log-loss | Brier |",
            "|----|--------|---|-----------|--------|-----------|----------|-------|",
        ]
    )
    for h in results["hypotheses"]:
        if h.get("tradeable", True):
            continue
        ci = f"[{h['roi_ci_low']:.3f}, {h['roi_ci_high']:.3f}]" if h["n"] else "—"
        lines.append(
            f"| {h['hypothesis_id']} | {h.get('liquidity_segment', '—')} | {h['n']} | "
            f"{h['roi_mean']:.4f} | {ci} | {h['clv_mean'] if h['clv_mean'] is not None else '—'} | "
            f"{h['log_loss'] if h['log_loss'] is not None else '—'} | "
            f"{h['brier'] if h['brier'] is not None else '—'} |"
        )

    lines.extend(
        [
            "",
            "## H2–H4 — Testes tradeáveis por tercil",
            "",
            "| ID | tercil | n | ROI médio | IC 95% bootstrap | hit rate | p raw | p Bonf. | p BH | sobrevive | log-loss | Brier | n holdout | ROI holdout |",
            "|----|--------|---|-----------|------------------|----------|-------|---------|------|-----------|----------|-------|-----------|-------------|",
        ]
    )
    for h in results["hypotheses"]:
        if not h.get("tradeable", True):
            continue
        ci = f"[{h['roi_ci_low']:.3f}, {h['roi_ci_high']:.3f}]" if h["n"] else "—"
        surv = "sim" if (h["survives_bonferroni"] or h["survives_bh"]) else "não"
        lines.append(
            f"| {h['hypothesis_id']} | {h.get('liquidity_segment', '—')} | {h['n']} | {h['roi_mean']:.4f} | {ci} | "
            f"{h['hit_rate']:.3f} | {h['p_raw']:.4f} | {h['p_bonferroni']:.4f} | {h['p_bh']:.4f} | {surv} | "
            f"{h['log_loss'] if h['log_loss'] is not None else '—'} | "
            f"{h['brier'] if h['brier'] is not None else '—'} | "
            f"{h['holdout_n']} | "
            f"{h['holdout_roi'] if h['holdout_roi'] is not None else '—'} |"
        )

    tradeable_survivors = [
        h
        for h in results["hypotheses"]
        if h.get("tradeable") and (h["survives_bonferroni"] or h["survives_bh"])
    ]
    lines.extend(["", "## Veredito", ""])
    lines.append(
        "**H1** reportado apenas como diagnóstico de mispricing residual pós-release (compra do vencedor "
        "já conhecido). **Não** entra na família de testes ROI nem em sobreviventes."
    )
    if not tradeable_survivors:
        lines.append(
            "\n**H2–H4:** nenhuma variante tradeável passou holdout (n≥3, ROI>0) com p Bonferroni/BH < 0,05. "
            "Não há evidência de edge operável pré-release nesta amostra e proxies."
        )
    else:
        lines.append("\n**H2–H4 — sobreviventes:**")
        for h in tradeable_survivors:
            lines.append(
                f"- {h['hypothesis_id']}: ROI holdout={h['holdout_roi']}, p_BH={h['p_bh']:.4f}"
            )

    lines.extend(
        [
            "",
            "## Limitações",
            "",
            "- Volume = soma size×price (tokens YES+NO) até decision_ts; mercados com fetch incompleto excluídos dos tercis.",
            "- H3: ver cobertura acima — maioria dos mercados não gera trade (filtro fav/longshot).",
            "- Fidelidade CLOB 60 min → variantes H1 1/5/15 min frequentemente idênticas.",
            "- H2 usa proxies fracos (não FedWatch/consenso).",
            "- Adendo de liquidez no pré-registro é **pós-hoc** (ver arquivo).",
            "",
            "*Gerado por `scripts/run_macro_backtest.py` — números medidos, não inventados.*",
        ]
    )
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", default="data/macro/markets_manifest.json")
    parser.add_argument("--cache", default="data/macro/prices")
    parser.add_argument("--results", default="reports/macro_backtest_results.json")
    parser.add_argument("--report", default="reports/RELATORIO_MACRO_POLY.md")
    args = parser.parse_args()

    markets = _load_manifest(Path(args.manifest))
    markets = [m for m in markets if m.outcome_yes is not None and m.release_ts]
    results = run_full_backtest(markets, Path(args.cache))
    Path(args.results).parent.mkdir(parents=True, exist_ok=True)
    with open(args.results, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    _write_report(results, Path(args.report))
    print(f"Wrote {args.results} and {args.report}")


if __name__ == "__main__":
    main()
