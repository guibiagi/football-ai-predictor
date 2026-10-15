"""Estratégias ativáveis no monitor (somente após passar holdout)."""

from __future__ import annotations

import json
from pathlib import Path

CONFIG_PATH = Path(__file__).resolve().parent.parent / "data" / "monitor_strategy.json"

# Backtest 2026-10-15: única hipótese direcional com holdout viável usa B365, não Poly.
DEFAULT = {
    "active": "directional_poisson",
    "strategies": {
        "directional_poisson": {
            "enabled": True,
            "description": "Poisson vs preço Polymarket (alerta edge≥8%)",
            "edge_min": 0.08,
        },
        "pin_b365_cross_book": {
            "enabled": False,
            "description": "Research: Pinnacle abertura vs B365 (H4b) — sem feed B365 no monitor",
            "edge_min": 0.01,
        },
        "maker_pin_fair": {
            "enabled": False,
            "description": "Cotação passiva ±spread em torno do fair Pinnacle (ver --maker)",
            "half_spread": 0.02,
            "min_volume": 1000.0,
        },
    },
}


def load_strategy_config(path: Path | None = None) -> dict:
    p = path or CONFIG_PATH
    if not p.exists():
        return DEFAULT.copy()
    return json.loads(p.read_text(encoding="utf-8"))


def strategy_enabled(name: str, cfg: dict | None = None) -> bool:
    cfg = cfg or load_strategy_config()
    block = cfg.get("strategies", {}).get(name, {})
    return bool(block.get("enabled"))
