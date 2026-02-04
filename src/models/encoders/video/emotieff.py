"""
EmotiEffLib video token encoder — BaseEncoder wrapper.

EmotiEffLib 视频 token encoder，包装为 BaseEncoder 接口。

Wraps the original EmotiEffTokenEncoder (encoder/face/emotieff_encoder.py)
with the frozen EncoderOut interface. Works on pre-extracted features.
"""

from __future__ import annotations

from typing import Optional

import torch
from torch import nn

from src.core.registry import get_encoder_registry
from src.core.types import BaseEncoder, EncoderOut


@get_encoder_registry("video").register("emotieff")
class VideoEmotiEffEncoder(BaseEncoder):
    """
    Linear projection + LayerNorm for pre-extracted EmotiEffLib features.

    输入: [B, T, feature_dim] → 输出: EncoderOut
    """

    def __init__(self, cfg):
        super().__init__()
        feature_dim = cfg.get("feature_dim", 1280)
        d_model = cfg.get("d_model", 256)
        dropout = cfg.get("dropout", 0.1)

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
        B, T = x.shape[0], x.shape[1]
        tokens = self.proj(x)  # [B, T, D]

        if mask is None:
            mask = torch.ones(B, T, dtype=torch.bool, device=x.device)

        mask_f = mask.float().unsqueeze(-1)
        pooled = (tokens * mask_f).sum(dim=1) / mask_f.sum(dim=1).clamp(min=1.0)

        return EncoderOut(tokens=tokens, pooled=pooled, mask=mask)
