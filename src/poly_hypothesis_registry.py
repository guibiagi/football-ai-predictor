"""Registro central de hipóteses Rodada 4+ com correção múltipla."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np

from src.multiple_testing import (
    benjamini_hochberg,
    bonferroni_adjust,
    one_sided_pvalue_mean_gt_zero,
    white_reality_check_pvalue,
)
from src.poly_market_utils import summarize_returns
from src.reporting import bootstrap_ci


@dataclass
class HypothesisRecord:
    hid: str
    family: str
    description: str
    holdout_returns: list[float] = field(default_factory=list)
    dev_returns: list[float] = field(default_factory=list)
    params: dict[str, Any] = field(default_factory=dict)
    p_raw: float | None = None
    p_bonferroni: float | None = None
    p_bh: float | None = None
    p_reality_check: float | None = None
    viable_raw: bool = False
    viable_corrected: bool = False
    holdout_summary: dict = field(default_factory=dict)

    def finalize_holdout(self, n_min: int = 300, alpha: float = 0.05) -> None:
        self.holdout_summary = summarize_returns(self.holdout_returns)
        n = self.holdout_summary.get("n", 0)
        ci = self.holdout_summary.get("roi_ci95")
        roi = self.holdout_summary.get("roi_mean", 0) or 0
        self.p_raw = one_sided_pvalue_mean_gt_zero(np.asarray(self.holdout_returns))
        self.viable_raw = (
            n >= n_min and roi > 0 and ci is not None and ci[0] > 0
        )


class HypothesisRegistry:
    def __init__(self, n_min: int = 300, alpha: float = 0.05) -> None:
        self.n_min = n_min
        self.alpha = alpha
        self.records: list[HypothesisRecord] = []

    def add(
        self,
        hid: str,
        family: str,
        description: str,
        holdout_returns: list[float],
        dev_returns: list[float] | None = None,
        **params,
    ) -> HypothesisRecord:
        rec = HypothesisRecord(
            hid=hid,
            family=family,
            description=description,
            holdout_returns=list(holdout_returns),
            dev_returns=list(dev_returns or []),
            params=params,
        )
        rec.finalize_holdout(self.n_min, self.alpha)
        self.records.append(rec)
        return rec

    def apply_multiple_testing(self) -> None:
        p_raw = [r.p_raw or 1.0 for r in self.records]
        p_bonf = bonferroni_adjust(p_raw)
        p_bh = benjamini_hochberg(p_raw)
        # Reality check por família (max ROI dentro da família)
        fam_returns: dict[str, list[np.ndarray]] = {}
        for r in self.records:
            if len(r.holdout_returns) >= 5:
                fam_returns.setdefault(r.family, []).append(np.asarray(r.holdout_returns))
        fam_rc: dict[str, float] = {}
        for fam, arrs in fam_returns.items():
            fam_rc[fam] = white_reality_check_pvalue(arrs)

        for i, r in enumerate(self.records):
            r.p_bonferroni = p_bonf[i]
            r.p_bh = p_bh[i]
            r.p_reality_check = fam_rc.get(r.family, 1.0)
            ci = r.holdout_summary.get("roi_ci95")
            roi = r.holdout_summary.get("roi_mean", 0) or 0
            n = r.holdout_summary.get("n", 0)
            impl_ok = r.params.get("implementation_validated", True)
            r.viable_corrected = (
                impl_ok
                and n >= self.n_min
                and roi > 0
                and ci is not None
                and ci[0] > 0
                and p_bonf[i] < self.alpha
                and p_bh[i] < self.alpha
            )

    def to_dict(self) -> dict:
        self.apply_multiple_testing()
        return {
            "n_hypotheses": len(self.records),
            "n_min": self.n_min,
            "alpha": self.alpha,
            "hypotheses": [
                {
                    "id": r.hid,
                    "family": r.family,
                    "description": r.description,
                    "params": r.params,
                    "n_holdout": r.holdout_summary.get("n", 0),
                    "roi_mean": r.holdout_summary.get("roi_mean"),
                    "roi_ci95": r.holdout_summary.get("roi_ci95"),
                    "p_raw": round(r.p_raw or 1.0, 6),
                    "p_bonferroni": round(r.p_bonferroni or 1.0, 6),
                    "p_bh": round(r.p_bh or 1.0, 6),
                    "p_reality_check_family": round(r.p_reality_check or 1.0, 6),
                    "viable_raw": r.viable_raw,
                    "viable_corrected": r.viable_corrected,
                }
                for r in self.records
            ],
            "viable_after_correction": [
                r.hid for r in self.records if r.viable_corrected
            ],
        }
