"""
Video frame feature encoder with per-frame projection.

逐帧特征投影的视觉 encoder。支持 temporal_pool="mean" 或 "none"。
temporal_pool="none" 时保留帧级 tokens，由下游 fusion 层处理时序。

Works on pre-extracted per-frame features (ResNet-50, CLIP, etc.).
"""

from __future__ import annotations

from typing import Optional

import torch
from torch import nn

from src.core.registry import get_encoder_registry
from src.core.types import BaseEncoder, EncoderOut
from src.models.components.fusion_utils import masked_mean_pool


@get_encoder_registry("video").register("resnet2d")
class VideoResNet2dEncoder(BaseEncoder):
    """
    Per-frame projection encoder.

    temporal_pool="mean": pooled = masked mean of all tokens (default)
    temporal_pool="none": pooled = masked mean, but tokens preserved as-is for fusion

    Input:  [B, T, feature_dim] or [B, feature_dim]
    Output: tokens=[B, T, D], pooled=[B, D], mask=[B, T]
    """

    def __init__(self, cfg):
        super().__init__()
        feature_dim = cfg.get("feature_dim", 2048)
        d_model = cfg.get("d_model", 512)
        dropout = cfg.get("dropout", 0.1)
        self.temporal_pool = cfg.get("temporal_pool", "mean")

        if self.temporal_pool not in ("mean", "none"):
            raise ValueError(f"temporal_pool must be 'mean' or 'none', got '{self.temporal_pool}'")

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

        tokens = self.proj(x)
        pooled = masked_mean_pool(tokens, mask)

        return EncoderOut(tokens=tokens, pooled=pooled, mask=mask)

