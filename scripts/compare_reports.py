#!/usr/bin/env python3
"""Compara dois reports/metrics_*.json."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.reporting import compare_metrics_reports


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("current")
    ap.add_argument("previous")
    args = ap.parse_args()
    diff = compare_metrics_reports(Path(args.current), Path(args.previous))
    import json
    print(json.dumps(diff, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
