#!/usr/bin/env python3
"""Suíte Polymarket além de futebol: clima, crypto, politics, BTC/Deribit, rewards."""

from __future__ import annotations

import json
import sys
from datetime import date, datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.poly_btc_deribit_backtest import run_btc_above_markets
from src.poly_favorite_bias_backtest import run_favorite_maker_sim, run_favorite_taker
from src.poly_football_rewards_audit import rewards_on_closed_football_sample, scan_open_soccer_rewards
from src.poly_coherence_backtest import run_coherence_backtest
from src.poly_rn1_backtest import run_rn1_offset_grid
from src.poly_weather_backtest import run_weather_buy_no
import pandas as pd

REPORTS = Path("reports")


def main() -> int:
    results = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "geoblock_brazil": {
            "status": "close-only (frontend e API)",
            "code": "BR",
            "doc": "https://docs.polymarket.com/api-reference/geoblock",
            "note": "Usuários no Brasil podem fechar posições existentes, mas não abrir novas.",
        },
        "fee_doc": "https://docs.polymarket.com/polymarket-learn/trading/fees",
        "hypotheses": [],
    }

    print("W1 clima Buy NO favoritos (spread + profundidade)...")
    for spread in (0.04, 0.06, 0.10):
        w = run_weather_buy_no(spread_cents=spread, min_depth_usd=60.0)
        w["variant"] = f"W1_spread_{int(spread*100)}c_depth60"
        results["hypotheses"].append(w)

    df = pd.read_csv("data/matches_xg.csv", parse_dates=["date"], low_memory=False)
    print("H-RN1 maker favorito...")
    results["hypotheses"].extend(run_rn1_offset_grid(df))

    print("H-COH Dixon-Coles...")
    results["hypotheses"].append(run_coherence_backtest(df))

    print("C/P favoritos taker/maker...")
    for tag, label in (("crypto", "C1"), ("politics", "P1")):
        t = run_favorite_taker(tag)
        t["hypothesis"] = f"{label}_favorite_taker_yes_90"
        m = run_favorite_maker_sim(tag)
        m["hypothesis"] = f"{label}_favorite_maker_bid_90"
        results["hypotheses"].extend([t, m])

    print("BTC vs Deribit...")
    results["hypotheses"].append(run_btc_above_markets())

    print("Rewards futebol...")
    results["football_rewards"] = {
        "open_scan": scan_open_soccer_rewards(),
        "closed_sample": rewards_on_closed_football_sample(),
    }

    out = REPORTS / f"poly_expansion_{date.today().strftime('%Y%m%d')}.json"
    REPORTS.mkdir(exist_ok=True)
    out.write_text(json.dumps(results, indent=2, ensure_ascii=False), encoding="utf-8")

    md = _markdown_report(results)
    (REPORTS / "RELATORIO_EXPANSAO_POLY.md").write_text(md, encoding="utf-8")
    print(f"JSON: {out}")
    print(f"MD: {REPORTS / 'RELATORIO_EXPANSAO_POLY.md'}")
    return 0


def _markdown_report(data: dict) -> str:
    lines = [
        "# Expansão Polymarket (além de futebol)",
        "",
        f"Gerado: {data['generated_at']}",
        "",
        "## Geobloqueio Brasil",
        "",
        f"- **Status:** {data['geoblock_brazil']['status']} (código `{data['geoblock_brazil']['code']}`)",
        f"- **Documentação oficial:** [{data['geoblock_brazil']['doc']}]({data['geoblock_brazil']['doc']})",
        f"- {data['geoblock_brazil']['note']}",
        "",
        "## Taxas",
        f"Referência: [{data['fee_doc']}]({data['fee_doc']}) — retornos **líquidos** usam fórmula taker por categoria.",
        "",
        "## Tabela por hipótese (holdout)",
        "",
        "| ID | n hold | ROI líquido | IC 95% ROI | Viável PnL (n≥300, IC>0)? |",
        "|----|--------|-------------|------------|---------------------------|",
    ]
    for h in data["hypotheses"]:
        hid = h.get("hypothesis") or h.get("variant") or h.get("tag", "?")
        if hid == "H-RN1_favorite_bid_maker":
            ho = h.get("aggregate", {}).get("holdout", {})
            if ho.get("n_fills", 0) > 0:
                ci = ho.get("roi_ci95", [None, None])
                lines.append(
                    f"| {hid} off={h.get('offset')} | fills {ho.get('n_fills')} "
                    f"| {ho.get('roi_mean')} | [{ci[0]}, {ci[1]}] | "
                    f"{'sim' if h.get('pnl_viable_holdout') else 'não'} |"
                )
            continue
        if "holdout" not in h:
            continue
        ho = h["holdout"]
        if ho.get("n", 0) == 0:
            lines.append(f"| {hid} | 0 | — | — | não |")
            continue
        ci = ho.get("roi_ci95", [None, None])
        viable = "sim" if h.get("pnl_viable_holdout") else "não"
        lines.append(
            f"| {hid} | {ho['n']} | {ho.get('roi_mean')} | [{ci[0]}, {ci[1]}] | {viable} |"
        )

    fr = data.get("football_rewards", {})
    lines.extend([
        "",
        "## Maker futebol — rewards na API (hoje)",
        "",
        f"- Mercados abertos escaneados: {fr.get('open_scan', {}).get('open_markets_scanned', '?')}",
        f"- Com `clobRewards` ativo: {fr.get('open_scan', {}).get('markets_with_rewards', '?')}",
        f"- Amostra fechada: {fr.get('closed_sample', {}).get('pct', '?')}% com rewards "
        f"(mediana daily rate USDC: {fr.get('closed_sample', {}).get('median_daily_rate')})",
        "",
        "## Conclusão honesta",
        "",
        "Ver JSON completo para gross vs net e variantes de horizonte (clima). "
        "Critério operacional: **PnL líquido** no holdout com IC 95% acima de zero e n≥300 quando possível.",
    ])
    return "\n".join(lines) + "\n"


if __name__ == "__main__":
    raise SystemExit(main())
