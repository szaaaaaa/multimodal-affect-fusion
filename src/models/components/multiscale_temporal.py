"""
Multi-scale temporal encoder (Direction D).

Dilated causal convolutions at multiple scales capture event patterns
at different time granularities:
  - scale 1  (dilation=1):  ~0.6s @ 5Hz — single keypress / mouse click
  - scale 5  (dilation=5):  ~2.2s @ 5Hz — one firefight
  - scale 25 (dilation=25): ~10.2s @ 5Hz — one round

多尺度膨胀因果卷积，在编码器输出后、融合前增强时序特征。
"""

from __future__ import annotations

from typing import List, Optional

import torch
from torch import nn


class MultiScaleTemporalEncoder(nn.Module):
    """
    Multi-scale dilated 1D convolution block with residual connection.

    Parameters
    ----------
    d_model : int
        Token dimension.
    scales : list[int]
        Dilation rates for each branch.
    kernel_size : int
        Convolution kernel size (same for all branches).
    dropout : float
        Dropout rate inside each branch.
    """

    def __init__(
        self,
        d_model: int = 512,
        scales: Optional[List[int]] = None,
        kernel_size: int = 3,
        dropout: float = 0.1,
    ):
        super().__init__()
        if scales is None:
            scales = [1, 5, 25]

        self.branches = nn.ModuleList()
        for dilation in scales:
            padding = dilation * (kernel_size - 1) // 2
            self.branches.append(nn.Sequential(
                nn.Conv1d(d_model, d_model, kernel_size,
                          padding=padding, dilation=dilation),
                nn.GELU(),
                nn.Dropout(dropout),
            ))

        self.proj = nn.Sequential(
            nn.Linear(d_model * len(scales), d_model),
            nn.LayerNorm(d_model),
        )

    def forward(self, tokens: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Parameters
        ----------
        tokens : [B, T, D]
        mask : [B, T] (unused, kept for interface compatibility)

        Returns
        -------
        [B, T, D] — residual-enhanced tokens
        """
        x = tokens.transpose(1, 2)  # [B, D, T]
        branch_outs = [branch(x).transpose(1, 2) for branch in self.branches]
        multi = torch.cat(branch_outs, dim=-1)  # [B, T, D*num_scales]
        return tokens + self.proj(multi)  # residual connection
