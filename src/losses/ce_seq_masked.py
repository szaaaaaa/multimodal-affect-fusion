"""
Masked sequence cross-entropy loss for temporal classification.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn

from src.core.registry import LOSSES


@LOSSES.register("ce_seq_masked")
class MaskedSequenceCELoss(nn.Module):
    def __init__(self, cfg=None):
        super().__init__()
        weight = None
        if cfg is not None:
            _g = cfg.get if isinstance(cfg, dict) else (lambda k, d=None: getattr(cfg, k, d))
            class_weights = _g("class_weights", None)
            if class_weights is not None:
                weight = torch.tensor(class_weights, dtype=torch.float32)
        self.register_buffer("weight", weight)

    def forward(
        self,
        pred: torch.Tensor,
        target: torch.Tensor,
        mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        # pred: [B, T, C], target: [B, T] long
        B, T, C = pred.shape

        if mask is None:
            return F.cross_entropy(
                pred.reshape(-1, C), target.reshape(-1),
                weight=self.weight,
            )

        mask_flat = mask.reshape(-1).bool()
        pred_flat = pred.reshape(-1, C)[mask_flat]
        target_flat = target.reshape(-1)[mask_flat]

        if pred_flat.shape[0] == 0:
            return pred.sum() * 0.0  # no valid steps

        return F.cross_entropy(pred_flat, target_flat, weight=self.weight)
