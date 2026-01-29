"""
Minimal 1D-CNN encoder for KM features.

键鼠特征的 1D-CNN 编码器（最小实现）。
"""

from __future__ import annotations

import torch
from torch import nn


class KM1DCNNEncoder(nn.Module):
    """
    1D-CNN encoder for KM tokens.

    Input: x [B, L, D] -> Output: [B, L, d_model]
    """

    def __init__(self, d_in: int, d_model: int, kernel_size: int = 5, dropout: float = 0.1):
        super().__init__()
        padding = kernel_size // 2
        self.conv1 = nn.Conv1d(d_in, d_model, kernel_size=kernel_size, padding=padding)
        self.conv2 = nn.Conv1d(d_model, d_model, kernel_size=kernel_size, padding=padding)
        self.act = nn.GELU()
        self.drop = nn.Dropout(dropout)
        self.ln = nn.LayerNorm(d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, L, D] -> [B, D, L]
        x = x.transpose(1, 2)
        y1 = self.conv1(x)
        y1 = self.act(y1)
        y1 = self.drop(y1)
        y2 = self.conv2(y1)
        y2 = self.act(y2)
        y = y2 + y1
        y = y.transpose(1, 2)
        return self.ln(y)
