"""
KM statistical token encoder — BaseEncoder wrapper.

键鼠统计特征 encoder，包装为 BaseEncoder 接口。

Wraps the original KMStatTokenEncoder (encoder/km/km_encoder_stat.py)
with the frozen EncoderOut interface.
"""

from __future__ import annotations

from typing import Optional

import torch
from torch import nn

from src.core.registry import get_encoder_registry
from src.core.types import BaseEncoder, EncoderOut


@get_encoder_registry("km").register("stat")
class KMStatEncoder(BaseEncoder):
    """
    Lightweight linear projection for pre-computed KM statistical features.

    输入: [B, T, D_in] → 输出: EncoderOut
    """

    def __init__(self, cfg):
        super().__init__()
        d_in = cfg.get("d_in", cfg.get("feature_dim", 25))
        d_model = cfg.get("d_model", 256)
        self.proj = nn.Linear(d_in, d_model)
        self.ln = nn.LayerNorm(d_model)

    def forward(
        self,
        x: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> EncoderOut:
        B, T = x.shape[0], x.shape[1]
        tokens = self.ln(self.proj(x))  # [B, T, D]

        if mask is None:
            mask = torch.ones(B, T, dtype=torch.bool, device=x.device)

        # Masked mean pooling
        mask_f = mask.float().unsqueeze(-1)  # [B, T, 1]
        pooled = (tokens * mask_f).sum(dim=1) / mask_f.sum(dim=1).clamp(min=1.0)

        return EncoderOut(tokens=tokens, pooled=pooled, mask=mask)
