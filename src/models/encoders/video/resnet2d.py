"""
ResNet-50 frame feature encoder with temporal mean pooling.

ResNet-50 逐帧特征 + 时间均值池化的视觉 encoder。

Works on pre-extracted per-frame features (default). This keeps the training
loop light while matching the new "2D CNN + temporal pooling" design.
"""

from __future__ import annotations

from typing import Optional

import torch
from torch import nn

from src.core.registry import get_encoder_registry
from src.core.types import BaseEncoder, EncoderOut


@get_encoder_registry("video").register("resnet2d")
class VideoResNet2dEncoder(BaseEncoder):
    """
    Temporal mean pooling over per-frame ResNet-50 features.

    Input:  [B, T, feature_dim] or [B, feature_dim]
    Output: tokens=[B, 1, D] (clip-level token), pooled=[B, D], mask=[B, 1]
    """

    def __init__(self, cfg):
        super().__init__()
        feature_dim = cfg.get("feature_dim", 2048)
        d_model = cfg.get("d_model", 512)
        dropout = cfg.get("dropout", 0.1)
        self.temporal_pool = cfg.get("temporal_pool", "mean")

        if self.temporal_pool != "mean":
            raise ValueError("resnet2d encoder only supports temporal_pool='mean'.")

        self.proj = nn.Sequential(
            nn.Linear(feature_dim, d_model),
            nn.LayerNorm(d_model),
            nn.Dropout(dropout),
        )

    def forward(
        self,
        x: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> EncoderOut:
        if x.dim() == 2:
            x = x.unsqueeze(1)
        if x.dim() != 3:
            raise ValueError(f"Expected x with shape [B, T, D] or [B, D], got {tuple(x.shape)}")

        B, T = x.shape[0], x.shape[1]

        if mask is None:
            mask = torch.ones(B, T, dtype=torch.bool, device=x.device)
        else:
            if mask.dim() == 1:
                mask = mask.unsqueeze(1)
            if mask.shape[0] != B or mask.shape[1] != T:
                raise ValueError(f"Mask shape {tuple(mask.shape)} does not match input {B, T}")

        mask_f = mask.float().unsqueeze(-1)
        denom = mask_f.sum(dim=1).clamp(min=1.0)
        pooled_raw = (x * mask_f).sum(dim=1) / denom

        pooled = self.proj(pooled_raw)

        valid = mask_f.sum(dim=1) > 0
        tokens = pooled.unsqueeze(1)
        mask_out = valid.squeeze(-1).unsqueeze(1)

        return EncoderOut(tokens=tokens, pooled=pooled, mask=mask_out)

