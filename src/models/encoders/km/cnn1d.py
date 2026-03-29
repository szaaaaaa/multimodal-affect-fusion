"""
KM 1D-CNN encoder — BaseEncoder wrapper.

键鼠 1D-CNN encoder，包装为 BaseEncoder 接口。

Wraps the original KM1DCNNEncoder (encoder/km/km_encoder_1dCNN.py)
with the frozen EncoderOut interface.
"""

from __future__ import annotations

from typing import Optional

import torch
from torch import nn

from src.core.registry import get_encoder_registry
from src.core.types import BaseEncoder, EncoderOut
from src.models.components.fusion_utils import masked_mean_pool


@get_encoder_registry("km").register("cnn1d")
class KMCnn1dEncoder(BaseEncoder):
    """
    2-layer 1D CNN with residual connection for KM features.

    输入: [B, T, D_in] → 输出: EncoderOut
    """

    def __init__(self, cfg):
        super().__init__()
        d_in = cfg.get("d_in", cfg.get("feature_dim", 25))
        d_model = cfg.get("d_model", 256)
        kernel_size = cfg.get("kernel_size", 5)
        dropout = cfg.get("dropout", 0.1)

        padding = kernel_size // 2
        self.conv1 = nn.Conv1d(d_in, d_model, kernel_size=kernel_size, padding=padding)
        self.conv2 = nn.Conv1d(d_model, d_model, kernel_size=kernel_size, padding=padding)
        self.act = nn.GELU()
        self.drop = nn.Dropout(dropout)
        self.ln = nn.LayerNorm(d_model)

    def forward(
        self,
        x: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> EncoderOut:
        B, T = x.shape[0], x.shape[1]

        # x: [B, T, D] -> [B, D, T]
        h = x.transpose(1, 2)
        y1 = self.drop(self.act(self.conv1(h)))
        y2 = self.act(self.conv2(y1))
        tokens = self.ln((y2 + y1).transpose(1, 2))  # [B, T, D]

        if mask is None:
            mask = torch.ones(B, T, dtype=torch.bool, device=x.device)

        pooled = masked_mean_pool(tokens, mask)

        return EncoderOut(tokens=tokens, pooled=pooled, mask=mask)
