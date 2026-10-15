"""Gates por liga/mercado: só alertar onde log-loss vs fechamento >= -0.005."""

from __future__ import annotations

import json
from pathlib import Path

DEFAULT_GAIN_MIN = -0.005
GATES_PATH = Path(__file__).resolve().parent.parent / "data" / "monitor_gates.json"


def load_gates(path: Path | None = None) -> dict:
    p = path or GATES_PATH
    if not p.exists():
        return {}
    return json.loads(p.read_text(encoding="utf-8"))


def market_allowed(gates: dict, liga: str, mercado: str = "1X2") -> bool:
    """True se não há gate ou se ganho_ll >= limiar."""
    if not gates:
        return True
    block = gates.get(mercado, {})
    if liga not in block:
        return True
    entry = block[liga]
    if "alert" in entry:
        return bool(entry["alert"])
    gain = entry.get("log_loss_gain_vs_close")
    if gain is None:
        return True
    return gain >= DEFAULT_GAIN_MIN


def save_gates(gates: dict, path: Path | None = None) -> Path:
    p = path or GATES_PATH
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(gates, indent=2, ensure_ascii=False), encoding="utf-8")
    return p
