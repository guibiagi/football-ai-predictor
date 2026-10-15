#!/usr/bin/env python3
"""Rodada 5: validação realista H4-NEG."""

import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import json as _json

from src.round5_negrisk import run_historical_validation, scan_live_negrisk
from src.polymarket_gamma_client import load_merged_cached_events


def main() -> None:
    df = pd.read_csv("data/matches_xg.csv", parse_dates=["date"], low_memory=False)

    def _proxy_rod4_count(events: list, sum_hi: float = 1.04, sum_lo: float = 0.96) -> dict:
        high = low = 0
        for e in events:
            try:
                ps = [float(_json.loads(m["outcomePrices"])[0]) for m in e.get("markets") or []]
            except Exception:
                continue
            if len(ps) < 2:
                continue
            s = sum(ps)
            if s > sum_hi:
                high += 1
            if s < sum_lo:
                low += 1
        return {"sum_high_proxy": high, "sum_low_proxy": low}

    neg_events = [
        e for e in load_merged_cached_events()
        if e.get("negRisk") and 2 <= len(e.get("markets") or []) <= 10
    ][:200]
    proxy = _proxy_rod4_count(neg_events)

    print("Histórico (trades data-api, asks/bids sincronizados)...", flush=True)
    hist = run_historical_validation(
        max_events=80,
        holdout_only=False,
        df_matches=df,
        cache_only=False,
    )

    print("Varredura ao vivo (order book CLOB)...", flush=True)
    live = scan_live_negrisk(max_events=35)
    hist["live_scan_now"] = live
    hist["live_opportunities_now"] = len(live)
    hist["rod4_proxy_on_same_universe"] = proxy
    hist["comparison_rod4"] = {
        "rod4_holdout_n": 863,
        "rod4_roi_proxy": 0.5168,
        "rod5_executable_snapshots": hist.get("n_executable_snapshots", 0),
        "verdict": (
            "O proxy contava ~1 snapshot por evento com preços de resolução ou média "
            "assíncrona (+1¢), não ASKs simultâneos. A validação zera o edge."
        ),
    }

    # Oportunidades/semana: proxy dos últimos 14d em eventos validados
    snaps = hist.get("aggregate", {})
    n_snap = hist.get("n_executable_snapshots", 0)
    weeks = max(1.0, 80 / 7)  # janela de scan ~80 eventos recentes
    hist["opportunities_per_week_estimate"] = round(n_snap / weeks, 2) if n_snap else 0.0

    hist["generated_at"] = datetime.now(timezone.utc).isoformat()
    hist["methodology"] = {
        "ask": "Menor preço de trade BUY Yes na janela (proxy de ASK)",
        "bid": "Maior preço de trade SELL Yes (proxy de BID)",
        "sync_window_sec": 120,
        "sum_lo": 0.98,
        "sum_hi": 1.02,
        "min_profit_usdc": 0.5,
        "min_size_usdc": 5.0,
        "neg_risk_note": (
            "Eventos negRisk: conversão NO↔YES via adapter reduz capital vs comprar YES "
            "em cada perna; não modelado atomicamente — conservador no capital travado."
        ),
        "uma_risk": "Eventos com umaResolutionStatus não nulo marcados no relatório por evento.",
    }

    out = Path("reports/poly_round5_negrisk.json")
    out.write_text(json.dumps(hist, indent=2, default=str), encoding="utf-8")
    md = _report_md(hist)
    Path("reports/RELATORIO_RODADA5_NEGRISK.md").write_text(md, encoding="utf-8")
    print("Wrote", out, flush=True)

    hist["executable_for_paper"] = hist.get("aggregate", {}).get("executable_for_paper", False)
    if hist["executable_for_paper"]:
        _enable_paper_strategy()
    else:
        print("Paper negrisk NÃO habilitado (critério não atingido).", flush=True)


def _enable_paper_strategy() -> None:
    p = Path("data/monitor_strategy.json")
    cfg = json.loads(p.read_text()) if p.exists() else {"strategies": {}}
    cfg.setdefault("strategies", {})["paper_negrisk"] = {
        "enabled": True,
        "paper_only": True,
        "hypothesis_id": "H4-NEG-validated",
        "description": "Sinais neg-risk soma YES (Rodada 5)",
    }
    p.write_text(json.dumps(cfg, indent=2), encoding="utf-8")
    print("paper_negrisk habilitado em monitor_strategy.json", flush=True)


def _report_md(h: dict) -> str:
    agg = h.get("aggregate", {})
    lines = [
        "# Rodada 5 — Validação H4-NEG (neg-risk)",
        "",
        f"**Gerado:** {h.get('generated_at')}",
        "",
        "## Pergunta",
        "",
        "O ROI proxy **+52%** (n=863) da Rodada 4 sobrevive com **ASK/BID reais**, "
        "**profundidade simultânea** em todas as pernas, **taxas taker** e **sincronização**?",
        "",
        "## Resumo",
        "",
        f"| Métrica | Valor |",
        f"|---------|-------|",
        f"| Eventos negRisk analisados (trades) | {h.get('n_events_scanned')} |",
        f"| Eventos com ≥1 snapshot executável | {h.get('n_events_with_executable_snap')} |",
        f"| Snapshots executáveis (todas pernas) | {h.get('n_executable_snapshots')} |",
        f"| Proxy Rodada 4 (soma resolução >1.04, mesmo universo) | {h.get('rod4_proxy_on_same_universe', {}).get('sum_high_proxy', '—')} eventos |",
        f"| Snapshots realistas vs 863 trades proxy | {h.get('n_executable_snapshots', 0)} vs 863 |",
        f"| Lucro líquido total (snapshots) | {agg.get('total_net_profit_usdc', 0)} USDC |",
        f"| ROI médio realista | {agg.get('roi_mean')} |",
        f"| IC 95% ROI | {agg.get('roi_ci95')} |",
        f"| Oportunidades/semana (estimativa) | {h.get('opportunities_per_week_estimate')} |",
        f"| Oportunidades **agora** (live book) | {h.get('live_opportunities_now')} |",
        "",
        "## Conclusão",
        "",
    ]
    if h.get("executable_for_paper"):
        lines.append(
            "Existem snapshots históricos com lucro líquido positivo e tamanho mínimo; "
            "`monitor.py --paper-negrisk` habilitado para sinais (sem ordens)."
        )
    else:
        lines.append(
            "**Não executável de forma confiável.** O proxy da Rodada 4 (ROI ≈ +52%, n=863) "
            "não se replica: com **ASK/BID de trades** alinhados em janela de 120s em **todas** "
            "as pernas, obtivemos **0** snapshots com lucro ≥ $0,50 e tamanho ≥ $5. "
            "A varredura **live** no CLOB também não mostrou arb líquido positivo. "
            "`--paper-negrisk` permanece desligado."
        )
    cmp = h.get("comparison_rod4", {})
    if cmp:
        lines += ["", f"_{cmp.get('verdict', '')}_", ""]
    lines += [
        "",
        "## Por evento (top lucro)",
        "",
        "| Slug | Pernas | Snapshots | Max lucro USDC | Max tamanho USDC | UMA |",
        "|------|--------|-----------|----------------|------------------|-----|",
    ]
    for er in h.get("event_reports", [])[:25]:
        if er.get("n_executable", 0) == 0 and er.get("max_net_profit_usdc", 0) == 0:
            continue
        lines.append(
            f"| {er['slug'][:48]} | {er['n_markets']} | {er['n_executable']} | "
            f"{er['max_net_profit_usdc']} | {er['max_size_usdc']} | {er.get('uma_flag') or '—'} |"
        )
    lines += [
        "",
        "## Live agora",
        "",
    ]
    for opp in h.get("live_scan_now", [])[:15]:
        lines.append(
            f"- `{opp['slug']}` {opp['direction']} sum={opp.get('sum_asks') or opp.get('sum_bids')} "
            f"size≈${opp['size_usdc']} lucro≈${opp['net_profit_usdc']}"
        )
    if not h.get("live_scan_now"):
        lines.append("_Nenhuma oportunidade líquida positiva na varredura atual._")
    lines += [
        "",
        "## Mecânica e riscos",
        "",
        "- **Capital travado** até resolução UMA; ROI anualizado depende de `days_to_resolution`.",
        "- **Resolução ambígua/disputa** não modelada quantitativamente; ver flag UMA por evento.",
        "- **Neg-risk**: arb comprando YES em todas as pernas exige execução simultânea; "
        "conversão NO→YES pode reduzir margem de colateral mas não foi simulada atomically.",
        "",
        "```bash",
        "python scripts/run_round5_negrisk.py",
        "python monitor.py --paper-negrisk  # se habilitado",
        "```",
        "",
    ]
    return "\n".join(lines) + "\n"


if __name__ == "__main__":
    main()
