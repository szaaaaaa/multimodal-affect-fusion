"""
Concordance Correlation Coefficient (CCC) loss.

CCC 损失函数 — 情感计算中的标准损失。
"""

from __future__ import annotations

import torch
from torch import nn

from src.core.registry import LOSSES


@LOSSES.register("ccc")
class CCCLoss(nn.Module):
    """
    CCC Loss = 1 - CCC.

    Minimising this maximises the concordance correlation coefficient.
    Supports multi-dimensional targets (averages CCC across dims).
    """

    def __init__(self, cfg=None):
        super().__init__()
        self.eps = 1e-8

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        if pred.ndim == 1:
            pred = pred.unsqueeze(1)
        if target.ndim == 1:
            target = target.unsqueeze(1)

        loss = torch.tensor(0.0, device=pred.device, dtype=pred.dtype)
        for i in range(pred.size(1)):
            x = pred[:, i]
            y = target[:, i]
            mean_x = x.mean()
            mean_y = y.mean()
            var_x = x.var(unbiased=False)
            var_y = y.var(unbiased=False)
            cov = ((x - mean_x) * (y - mean_y)).mean()
            ccc = (2 * cov) / (var_x + var_y + (mean_x - mean_y) ** 2 + self.eps)
            loss = loss + (1.0 - ccc)

        return loss / pred.size(1)
