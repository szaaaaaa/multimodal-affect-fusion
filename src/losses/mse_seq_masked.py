"""
Masked sequence MSE loss for temporal regression.
"""

from __future__ import annotations

import torch
from torch import nn

from src.core.registry import LOSSES


@LOSSES.register("mse_seq_masked")
class MaskedSequenceMSELoss(nn.Module):
    def __init__(self, cfg=None):
        super().__init__()

    def forward(
        self,
        pred: torch.Tensor,
        target: torch.Tensor,
        mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if pred.ndim == 2:
            pred = pred.unsqueeze(-1)
        if target.ndim == 2:
            target = target.unsqueeze(-1)

        if mask is None:
            return torch.mean((pred - target) ** 2)

        if mask.ndim == 2:
            mask = mask.unsqueeze(-1)

        mask_f = mask.float()
        loss = (pred - target) ** 2
        denom = mask_f.sum().clamp(min=1.0)
        return (loss * mask_f).sum() / denom
