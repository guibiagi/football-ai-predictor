#!/usr/bin/env python3
"""Atualiza RELATORIO_HIPOTESES.md com resultados do maker Polymarket."""

from __future__ import annotations

import json
from pathlib import Path


def main() -> int:
    reports = Path("reports")
    md = reports / "RELATORIO_HIPOTESES.md"
    poly = sorted(reports.glob("poly_maker_*.json"))
    if not poly:
        print("Sem poly_maker_*.json")
        return 0
    data = json.loads(poly[-1].read_text(encoding="utf-8"))
    block = [
        "",
        "## Hipóteses maker Polymarket (CLOB prices-history)",
        "",
        f"Fonte: `{data['meta'].get('events_cache', '')}` | Holdout desde {data['meta'].get('holdout_cut')}",
        "",
        "| Cenário | Fair | Spread | n hold | CLV hold | IC CLV | ROI hold | IC ROI | CLV ok | PnL ok |",
        "|---------|------|--------|--------|----------|--------|----------|--------|--------|--------|",
    ]
    for s in data["scenarios"]:
        h = s["holdout"]
        ci = h.get("clv_ci95") or [None, None]
        ri = h.get("roi_ci95") or [None, None]
        block.append(
            f"| {s['id']} | {s['fair']} | {s['half_spread']:.2f} | {h['n']} | "
            f"{h.get('clv_mean')} | [{ci[0]}, {ci[1]}] | {h.get('roi')} | "
            f"[{ri[0]}, {ri[1]}] | {'sim' if s.get('clv_viable') else 'não'} | "
            f"{'sim' if s.get('pnl_viable') else 'não'} |"
        )
    block.extend([
        "",
        "**Nota metodológica:** fair por abertura Pinnacle (`ps_*`) é o cenário honesto para cotação "
        "pré-jogo; fair por fechamento (`psc_*`) é cenário-oráculo (sensibilidade). "
        "Fill: primeiro candle onde preço cruza bid/ask. Rewards: fração conservadora do "
        "`rewardsDailyRate` quando spread ≤ `rewardsMaxSpread`.",
        "",
        "**Leitura vs pesquisa:** confirma o consenso — taker direcional vs Pinnacle na Poly "
        "não passou; maker tem **CLV positivo** nas cotações preenchidas, mas **PnL negativo** "
        "no holdout (IC do ROI abaixo de zero em spreads apertados), típico de *adverse selection*. "
        "Sem modelo de rewards observado no histórico, **não há estratégia Polymarket operável** "
        "pelo critério PnL; continuar só com forward test / coleta de rebates reais.",
        "",
        "### Síntese final",
        "",
        "| Abordagem | CLV holdout | PnL holdout | Operável? |",
        "|-----------|-------------|-------------|-----------|",
        "| Poisson / blend vs Poly (proxy B365) | ≤ 0 | negativo | não |",
        "| H4b cross-book Pin→B365 (taker) | +1,3pp | ~0 | só B365, fora do monitor |",
        "| Maker Poly vs fair Pin (passivo) | +0,8–1,2pp | negativo | não (sem rewards) |",
    ])
    text = md.read_text(encoding="utf-8") if md.exists() else ""
    marker = "## Hipóteses maker Polymarket"
    if marker in text:
        text = text.split(marker)[0].rstrip()
    md.write_text(text + "\n".join(block) + "\n", encoding="utf-8")
    print(f"Atualizado {md}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
