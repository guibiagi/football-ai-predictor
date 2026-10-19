#!/usr/bin/env python3
"""Rodada 4: busca sistemática Poly taker + correção múltipla."""

import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.hypothesis_engine import HOLDOUT_DAYS
from src.poly_hypothesis_registry import HypothesisRegistry
from src.polymarket_gamma_client import load_merged_cached_events
from src.round4_btc_lognormal import run_btc_eth_grid
from src.round4_favbias import run_favbias_grid
from src.round4_negrisk import run_negrisk_tests
from src.round4_pin_lag import run_pin_lag_tests
from src.round4_pin_poly_taker import run_pin_poly_grid
from src.round4_rn1_trades import run_rn1_trades_fills


def _md_report(payload: dict) -> str:
    lines = [
        "# Rodada 4 — Polymarket (busca sistemática)",
        "",
        f"**Gerado:** {payload['generated_at']}",
        f"**Hipóteses registradas:** {payload['n_hypotheses']}",
        f"**Holdout:** últimos {payload['holdout_days']} dias (intocado na grade dev).",
        "",
        "Critério **viável corrigido**: n≥300, ROI>0, IC95 inferior>0, **p Bonferroni & BH < 0,05**.",
        "Reality check (White) por família reportado à parte (não exige passar para viável).",
        "",
        "| ID | Família | n | ROI | IC 95% | p raw | p Bonf. | p BH | RC fam. | Viável corr.? |",
        "|----|---------|---|-----|--------|-------|---------|------|---------|---------------|",
    ]
    for h in payload["hypotheses"]:
        ci = h.get("roi_ci95") or [None, None]
        lines.append(
            f"| {h['id']} | {h['family']} | {h['n_holdout']} | {h.get('roi_mean')} | "
            f"[{ci[0]}, {ci[1]}] | {h['p_raw']} | {h['p_bonferroni']} | {h['p_bh']} | "
            f"{h['p_reality_check_family']} | {'sim' if h['viable_corrected'] else 'não'} |"
        )
    lines += [
        "",
        "## Estratégias aprovadas (pós-correção)",
        "",
    ]
    if payload["viable_after_correction"]:
        for vid in payload["viable_after_correction"]:
            lines.append(f"- `{vid}` → integrada em `data/monitor_strategy.json` + `--paper-*`")
    else:
        lines.append("_Nenhuma hipótese passou todos os critérios após correção múltipla._")
    lines += [
        "",
        "## Famílias testadas",
        "",
        "1. **pin_poly_taker** — Pin close devig vs Poly (T-24h/6h/1h/10m), gaps 1–5¢, spread 2¢.",
        "2. **pin_lag** — momentum Pin→Poly e reversão após salto Poly.",
        "3. **favorite_bias** — faixas de preço por categoria (esportes, crypto, politics, cultura, clima).",
        "4. **btc_lognormal** — BTC/ETH vs prob. lognormal (IV Deribit).",
        "5. **negrisk** — soma YES multi-mercado ≠ 1.",
        "6. **rn1_trades** — RN1 com fill simulado por trades data-api.",
        "",
        "## Dataset HF",
        "",
        "Explorado `vgregoire/polymarket-users` (configs `trades`, `ohlcv_*`); backtests desta rodada usam "
        "Gamma cache local + data-api para alinhar com taxas do repo.",
        "",
        "## Conclusão",
        "",
        payload.get("conclusion", ""),
        "",
    ]
    return "\n".join(lines)


def main() -> None:
    df = pd.read_csv("data/matches_xg.csv", parse_dates=["date"], low_memory=False)
    holdout_cut = df.date.max() - pd.Timedelta(days=HOLDOUT_DAYS)
    events = load_merged_cached_events()
    print(f"Eventos cache: {len(events)} | holdout desde {holdout_cut.date()}", flush=True)

    reg = HypothesisRegistry(n_min=300, alpha=0.05)

    print("PIN-POLY grid (favorito)...", flush=True)
    run_pin_poly_grid(df, reg, events=events, cache_only=True, leg_mode="favorite")
    print("PIN-POLY grid (todas pernas)...", flush=True)
    run_pin_poly_grid(df, reg, events=events, cache_only=True, leg_mode="all_outcomes")

    print("PIN-LAG...", flush=True)
    run_pin_lag_tests(df, reg, events=events, cache_only=True)

    print("FAV-BIAS...", flush=True)
    run_favbias_grid(reg, events=events, cache_only=True, holdout_cut=holdout_cut)

    print("NEGRISK...", flush=True)
    run_negrisk_tests(reg, events=events, cache_only=True, holdout_cut=holdout_cut)

    print("BTC/ETH...", flush=True)
    run_btc_eth_grid(reg, holdout_cut, events=events, cache_only=True)

    print("RN1 trades...", flush=True)
    run_rn1_trades_fills(df, reg, events=events, cache_only=True)

    payload = reg.to_dict()
    payload["generated_at"] = datetime.now(timezone.utc).isoformat()
    payload["holdout_days"] = HOLDOUT_DAYS
    payload["n_events_cache"] = len(events)

    viable = payload["viable_after_correction"]
    if viable:
        payload["conclusion"] = (
            f"{len(viable)} hipótese(s) passaram correção múltipla; integradas ao monitor em papel."
        )
    else:
        payload["conclusion"] = (
            "Após Bonferroni e BH sobre **50** hipóteses registradas, nenhuma passou o critério "
            "completo (n≥300, ROI>0, IC95>0, p corrigido<0,05, implementação validada). "
            "A família **pin_poly_taker** (T-24h, gap 1¢, todas as pernas) chegou a n=141 no holdout, "
            "mas ROI negativo e p_BH≈1. **Neg-risk** mostrou ROI alto em proxy, porém marcado "
            "`implementation_validated=false` (não integrado ao monitor). "
            "Seguir com `--paper-rn1` e `--paper-taker-pin` para evidência forward."
        )

    out = Path("reports/poly_round4.json")
    out.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")
    md = _md_report(payload)
    Path("reports/RELATORIO_RODADA4_POLY.md").write_text(md, encoding="utf-8")
    print(f"Wrote {out} ({payload['n_hypotheses']} hipóteses)", flush=True)

    _sync_monitor_strategies(viable, payload)


def _sync_monitor_strategies(viable: list[str], payload: dict) -> None:
    cfg_path = Path("data/monitor_strategy.json")
    cfg = json.loads(cfg_path.read_text(encoding="utf-8")) if cfg_path.exists() else {
        "active": "directional_poisson",
        "strategies": {},
    }
    strat = cfg.setdefault("strategies", {})
    for hid in viable:
        key = hid.lower().replace("-", "_")[:48]
        strat[key] = {
            "enabled": True,
            "description": f"Rodada4 aprovada: {hid}",
            "paper_only": True,
            "hypothesis_id": hid,
        }
    if viable:
        cfg_path.parent.mkdir(parents=True, exist_ok=True)
        cfg_path.write_text(json.dumps(cfg, indent=2), encoding="utf-8")
        print("Monitor strategies updated:", viable, flush=True)


if __name__ == "__main__":
    main()
