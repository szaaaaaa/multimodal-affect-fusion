"""
Telemetry statistical-pool encoder — BaseEncoder wrapper.

Pre-computed telemetry stat features -> linear projection -> EncoderOut.
"""

from __future__ import annotations

from typing import Optional

import torch
from torch import nn

from src.core.registry import get_encoder_registry
from src.core.types import BaseEncoder, EncoderOut


@get_encoder_registry("telem").register("stat_pool")
class TelemStatPoolEncoder(BaseEncoder):
    """
    Linear projection for pre-computed telemetry statistical features.

    Input: [B, T, D_in] -> Output: EncoderOut
    """

    def __init__(self, cfg):
        super().__init__()
        d_in = cfg.get("d_in", cfg.get("feature_dim", 109))
        d_model = cfg.get("d_model", 256)
        self.proj = nn.Linear(d_in, d_model)
        self.ln = nn.LayerNorm(d_model)

    def forward(
        self,
        x: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> EncoderOut:
        B, T = x.shape[0], x.shape[1]
        tokens = self.ln(self.proj(x))

        if mask is None:
            mask = torch.ones(B, T, dtype=torch.bool, device=x.device)

        mask_f = mask.float().unsqueeze(-1)
        pooled = (tokens * mask_f).sum(dim=1) / mask_f.sum(dim=1).clamp(min=1.0)

        return EncoderOut(tokens=tokens, pooled=pooled, mask=mask)
