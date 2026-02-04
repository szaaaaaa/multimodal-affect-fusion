"""
CCC metric — concordance correlation coefficient.

CCC 指标 — 一致性相关系数。
"""

from __future__ import annotations

import torch

from src.core.registry import METRICS


@METRICS.register("ccc")
class CCCMetric:
    """
    Compute CCC between predictions and targets.

    Supports multi-dimensional outputs (averages across dims).
    """

    def __init__(self, cfg=None):
        self.eps = 1e-8

    def __call__(self, pred: torch.Tensor, target: torch.Tensor) -> float:
        pred = pred.float()
        target = target.float()
        if pred.ndim == 1:
            pred = pred.unsqueeze(1)
        if target.ndim == 1:
            target = target.unsqueeze(1)

        ccc_vals = []
        for i in range(pred.size(1)):
            x = pred[:, i]
            y = target[:, i]
            mean_x = x.mean()
            mean_y = y.mean()
            var_x = x.var(unbiased=False)
            var_y = y.var(unbiased=False)
            cov = ((x - mean_x) * (y - mean_y)).mean()
            ccc = (2 * cov) / (var_x + var_y + (mean_x - mean_y) ** 2 + self.eps)
            ccc_vals.append(ccc.item())

        return float(sum(ccc_vals) / len(ccc_vals))
