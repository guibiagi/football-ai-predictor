#!/usr/bin/env python3
"""Rodada 2: RN1+rewards ampliado, COH pareado, clima taker trades."""

import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.hypothesis_engine import HOLDOUT_DAYS
from src.poly_coh_paired import run_coh_paired
from src.poly_rn1_expanded import run_rn1_expanded_offsets
from src.poly_weather_taker_backtest import run_weather_taker_trades
from src.polymarket_gamma_client import fetch_events_closed, load_merged_cached_events


def _load_events() -> list[dict]:
    fetched = load_merged_cached_events()
    cache = Path("data/polymarket_closed_matches.json")
    if cache.exists():
        seen = {e.get("slug") for e in fetched}
        for e in json.loads(cache.read_text()):
            if e.get("slug") not in seen:
                fetched.append(e)
    if len(fetched) < 500:
        extra = []
        for tag in ("soccer", "epl", "nba", "nfl", "ufc", "atp", "wta"):
            try:
                extra.extend(fetch_events_closed(tag_slug=tag, max_pages=80))
            except Exception:
                pass
        seen = {e.get("slug") for e in fetched}
        for e in extra:
            s = e.get("slug")
            if s and s not in seen:
                seen.add(s)
                fetched.append(e)
    return fetched


def main():
    # Ampliar cache de temperatura (uma vez; usa gamma_cache).
    try:
        fetch_events_closed(tag_slug="temperature", max_pages=150, title_filter="Highest temperature")
    except Exception as exc:
        print("Aviso: fetch temperatura:", exc)

    df = pd.read_csv("data/matches_xg.csv", parse_dates=["date"], low_memory=False)
    fetched = _load_events()
    print("Eventos no pool:", len(fetched), flush=True)

    results = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "n_events": len(fetched),
        "hypotheses": [],
        "notes": [],
    }

    print("RN1 expanded (cache histórico, offsets 2/2.5/3¢)...", flush=True)
    raws, rn1_results = run_rn1_expanded_offsets(
        df,
        offsets=(0.02, 0.025, 0.03),
        events=fetched,
        cache_only=True,
        include_rewards=True,
        fair_lo=0.60,
        fair_hi=0.80,
    )
    print(f"  oportunidades com histórico: {len(raws)}", flush=True)
    results["rn1_raw_opportunities"] = len(raws)
    for r in rn1_results:
        ho = r.get("aggregate", {}).get("holdout", {})
        print(
            f"  off={r.get('offset')} holdout fills={ho.get('n_fills')} "
            f"roi={ho.get('roi_mean')} ci={ho.get('roi_ci95')}",
            flush=True,
        )
        results["hypotheses"].append(r)

    print("COH paired...", flush=True)
    coh = run_coh_paired(df)
    print(
        f"  pairs={coh.get('pairs_1x2_ou')} trades={coh.get('n_trades')} "
        f"holdout n={coh.get('holdout', {}).get('n')}",
        flush=True,
    )
    results["hypotheses"].append(coh)

    print("Weather taker trades...", flush=True)
    holdout_cut = df.date.max() - pd.Timedelta(days=HOLDOUT_DAYS)
    for spread in (0.04, 0.07, 0.10):
        run_weather_taker_trades(
            spread_cents=spread, max_events=100, cache_only=False, holdout_cut=holdout_cut,
        )
    for spread in (0.04, 0.07, 0.10):
        w = run_weather_taker_trades(
            spread_cents=spread, max_events=400, cache_only=True, holdout_cut=holdout_cut,
        )
        w["variant"] = f"W2_spread_{int(spread * 100)}c"
        ho = w.get("holdout", {})
        print(f"  {w['variant']} holdout n={ho.get('n')} roi={ho.get('roi_mean')}", flush=True)
        results["hypotheses"].append(w)

    results["notes"].append(
        "RN1: fair Pin (futebol) ou preço CLOB ~24h antes (NBA/NFL/tênis/MMA). "
        "PnL maker inclui clobRewards estimados."
    )
    results["notes"].append(
        "Meta n≥300 fills no holdout; amostra limitada pelo cache de prices-history e matching Pin."
    )

    out = Path("reports/poly_round2.json")
    out.write_text(json.dumps(results, indent=2, default=str), encoding="utf-8")
    md = _md(results)
    Path("reports/RELATORIO_RODADA2_POLY.md").write_text(md, encoding="utf-8")
    print("Wrote", out, "and RELATORIO_RODADA2_POLY.md", flush=True)


def _md(data: dict) -> str:
    lines = [
        "# Rodada 2 — Polymarket (pesquisa e sinais)",
        "",
        f"**Gerado:** {data['generated_at']}  ",
        f"**Eventos no pool:** {data['n_events']}  ",
        f"**Oportunidades RN1 com histórico:** {data.get('rn1_raw_opportunities', '—')}",
        "",
        "Critério PnL viável: últimos **365 dias** (holdout), **n ≥ 300**, ROI médio > 0, **IC 95% inferior > 0**.",
        "",
        "| Hipótese | n holdout | ROI líquido | IC 95% ROI | PnL viável? | Detalhe |",
        "|----------|-----------|-------------|------------|-------------|---------|",
    ]
    for h in data["hypotheses"]:
        hid = h.get("hypothesis", h.get("variant", "?"))
        if "RN1" in hid or h.get("offset") is not None:
            ho = h.get("aggregate", {}).get("holdout", {})
            n = ho.get("n_fills", 0)
            ci = ho.get("roi_ci95", [None, None])
            br = h.get("n_sports_breakdown", {})
            notes = (
                f"offset {h.get('offset')}; fill_rate {ho.get('fill_rate')}; "
                f"soccer opps {br.get('soccer_opps')}; outros {br.get('other_opps')}; rewards sim"
            )
            viable = "sim" if h.get("pnl_viable_holdout") else "não"
            lines.append(
                f"| {hid} | {n} | {ho.get('roi_mean')} | [{ci[0]}, {ci[1]}] | {viable} | {notes} |"
            )
        elif h.get("holdout") and "roi_mean" in h["holdout"]:
            ho = h["holdout"]
            n = ho.get("n", 0)
            ci = ho.get("roi_ci95", [None, None])
            viable = "sim" if h.get("pnl_viable_holdout") else "não"
            extra = h.get("pairs_1x2_ou", h.get("variant", ""))
            lines.append(
                f"| {hid} | {n} | {ho.get('roi_mean')} | [{ci[0]}, {ci[1]}] | {viable} | {extra} |"
            )
    lines += [
        "",
        "## Forward test em papel (RN1)",
        "",
        "- Estratégia: favorito **0,60–0,80** (fair Pinnacle `ps`), bid **2,5¢** abaixo do fair.",
        "- `python monitor.py --paper-rn1` — registra sinais em `data/paper_rn1_forward.csv`.",
        "- `python monitor.py --paper-rn1-report` — liquida jogos encerrados (fill simulado se preço ≤ bid antes do KO) e resume PnL.",
        "",
        "## Geoblock Brasil",
        "",
        "Trading restrito no BR (código `BR`): [documentação Polymarket](https://docs.polymarket.com/api-reference/geoblock).",
        "",
        "## Notas metodológicas",
        "",
    ]
    for note in data.get("notes", []):
        lines.append(f"- {note}")
    lines += [
        "",
        "## Conclusão",
        "",
        "- **H-RN1** (maker + `clobRewards`): holdout com poucos fills (meta n≥300 não atingida); IC 95% cruza zero.",
        "- **H-COH** (taker pareado): amostra grande, ROI líquido negativo; IC inclui zero — não viável.",
        "- **W2 clima** (taker histórico): amostra limitada pelos trades em cache; spreads 4–10¢ degradam PnL.",
        "",
        "Nenhuma hipótese atende **ROI > 0 com IC 95% inferior > 0** no holdout. Use `monitor.py --paper-rn1` para forward test em papel (offset 2,5¢).",
        "",
    ]
    return "\n".join(lines) + "\n"


if __name__ == "__main__":
    main()
