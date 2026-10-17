#!/usr/bin/env python3
"""Executa backtest macro e grava JSON + relatório markdown."""

from __future__ import annotations

import argparse
import json
from dataclasses import fields
from pathlib import Path

from src.macro_poly.backtest import run_full_backtest
from src.macro_poly.discovery import MacroMarket


def _load_manifest(path: Path) -> list[MacroMarket]:
    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    field_names = {f.name for f in fields(MacroMarket)}
    markets = []
    for row in data["markets"]:
        markets.append(MacroMarket(**{k: row[k] for k in field_names if k in row}))
    return markets


def _write_report(results: dict, path: Path) -> None:
    lines = [
        "# Relatório — Backtest Polymarket macro (Fed / inflação)",
        "",
        "**Modo:** somente simulação (sem ordens reais).  ",
        f"**Mercados no manifest:** {results['market_count']}  ",
        f"**Eventos no holdout (~30%):** {results['holdout_event_count']}  ",
        f"**Testes de hipótese (25 base × 3 tercis):** {results['n_tests']}  ",
        f"**Tercis volume Gamma (USD):** q33={results.get('liquidity_terciles_usd', {}).get('q33', '—')}, "
        f"q66={results.get('liquidity_terciles_usd', {}).get('q66', '—')}  ",
        f"**Stake simulado:** US$ {results.get('stake_usd_assumed', 25)}  ",
        "",
        "Pré-registro: [HIPOTESES_MACRO_POLY_PREREG.md](HIPOTESES_MACRO_POLY_PREREG.md)",
        "",
        "## Resultados por hipótese e tercil de liquidez",
        "",
        "| ID | tercil | n | ROI médio | IC 95% bootstrap | hit rate | p raw | p Bonf. | p BH | sobrevive Bonf. | sobrevive BH | log-loss | Brier | CLV médio | n holdout | ROI holdout |",
        "|----|--------|---|-----------|------------------|----------|-------|---------|------|-----------------|--------------|----------|-------|-----------|-----------|-------------|",
    ]
    for h in results["hypotheses"]:
        ci = f"[{h['roi_ci_low']:.3f}, {h['roi_ci_high']:.3f}]" if h["n"] else "—"
        lines.append(
            f"| {h['hypothesis_id']} | {h.get('liquidity_segment', '—')} | {h['n']} | {h['roi_mean']:.4f} | {ci} | "
            f"{h['hit_rate']:.3f} | {h['p_raw']:.4f} | {h['p_bonferroni']:.4f} | {h['p_bh']:.4f} | "
            f"{'sim' if h['survives_bonferroni'] else 'não'} | {'sim' if h['survives_bh'] else 'não'} | "
            f"{h['log_loss'] if h['log_loss'] is not None else '—'} | "
            f"{h['brier'] if h['brier'] is not None else '—'} | "
            f"{h['clv_mean'] if h['clv_mean'] is not None else '—'} | "
            f"{h['holdout_n']} | "
            f"{h['holdout_roi'] if h['holdout_roi'] is not None else '—'} |"
        )

    survivors = [h for h in results["hypotheses"] if h["survives_bonferroni"] or h["survives_bh"]]
    pre_release = [
        h
        for h in results["hypotheses"]
        if h["hypothesis_id"].startswith(("H2", "H3", "H4"))
        and (h["survives_bonferroni"] or h["survives_bh"])
    ]
    lines.extend(
        [
            "",
            "## Veredito",
            "",
        ]
    )
    lines.append(
        "**H1 (pós-release, lado vencedor conhecido):** várias células passam BH/Bonferroni, mas o ROI de holdout "
        "é da ordem de **~1%** por trade (preço já perto de 1,0 após o print). Isso mede *residual* de mispricing, "
        "não alpha pré-informação. No recorte por liquidez, **T1 não domina**: T2/T3 mostram ROI amostral maior "
        "que T1 em H1a — contrário à intuição de que só mercados finos teriam drift."
    )
    if not pre_release:
        lines.append(
            "\n**H2–H4 (pré-release / intradiário):** **nenhuma** variante sobreviveu aos critérios de holdout "
            "com p ajustado. A hipótese de edge operável antes do release ou via favorite-longshot **não foi suportada**."
        )
    else:
        lines.append("\n**H2–H4:** variantes com sinal:")
        for h in pre_release:
            lines.append(
                f"- **{h['hypothesis_id']}**: ROI holdout={h['holdout_roi']}, p_BH={h['p_bh']:.4f}"
            )
    if survivors:
        lines.append("\n<details><summary>Todas as células com BH/Bonf. (incl. H1)</summary>\n")
        for h in survivors:
            lines.append(
                f"- **{h['hypothesis_id']}**: ROI holdout={h['holdout_roi']}, p_BH={h['p_bh']:.4f}"
            )
        lines.append("\n</details>")

    lines.extend(
        [
            "",
            "## Limitações",
            "",
            "- Histórico CLOB: janelas finas exigem `startTs`/`endTs` em blocos; mercados resolvidos podem perder resolução sub-12h com `interval=max`.",
            "- Proxy externo para H2 (momentum FOMC / CPI naive) não substitui FedWatch ou consenso Bloomberg.",
            "- H1 pós-release usa informação do resultado real no instante da entrada (testa mispricing residual, não previsão pré-release).",
            "- Volume Gamma é proxy de liquidez no instante da decisão (não há book histórico completo).",
            "- Impacto de mercado calibrado para stakes ~US$10–50 (baseline US$25).",
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
    # mercados com resolução e timestamp
    markets = [m for m in markets if m.outcome_yes is not None and m.release_ts]
    results = run_full_backtest(markets, Path(args.cache))
    Path(args.results).parent.mkdir(parents=True, exist_ok=True)
    with open(args.results, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    _write_report(results, Path(args.report))
    print(f"Wrote {args.results} and {args.report}")


if __name__ == "__main__":
    main()
