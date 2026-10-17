#!/usr/bin/env python3
"""Descobre mercados macro fechados na Polymarket e grava manifest em data/macro/."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from src.macro_poly.discovery import discover_macro_markets, markets_to_manifest


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--out",
        default="data/macro/markets_manifest.json",
        help="Caminho do manifest JSON",
    )
    args = parser.parse_args()
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    markets = discover_macro_markets()
    manifest = {
        "generated_at_utc": __import__("datetime")
        .datetime.now(__import__("datetime").timezone.utc)
        .isoformat()
        .replace("+00:00", "Z"),
        "count": len(markets),
        "markets": markets_to_manifest(markets),
    }
    with open(out, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)
    print(f"Wrote {len(markets)} markets to {out}")


if __name__ == "__main__":
    main()
