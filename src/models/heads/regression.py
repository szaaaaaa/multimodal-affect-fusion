"""
Regression prediction head.

回归预测头 — 用于 VA 预测。
"""

from __future__ import annotations

import torch
from torch import nn

from src.core.registry import HEADS
from src.core.types import BaseHead, FusionOut


@HEADS.register("regression")
class RegressionHead(BaseHead):
    """
    MLP regression head: pooled → hidden → output.

    Parameters (via cfg dict)
    -------------------------
    d_model : int (default 256)
    hidden_dim : int (default 128)
    out_dim : int (default 1)
    dropout : float (default 0.1)
    """

    def __init__(self, cfg):
        super().__init__()
        if isinstance(cfg, dict):
            _g = cfg.get
        else:
            _g = lambda k, d=None: getattr(cfg, k, d)

        d_model = _g("d_model", 256)
        hidden_dim = _g("hidden_dim", 128)
        out_dim = _g("out_dim", 1)
        dropout = _g("dropout", 0.1)

        self.mlp = nn.Sequential(
            nn.Linear(d_model, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, out_dim),
        )

    def forward(self, h: FusionOut) -> torch.Tensor:
        return self.mlp(h["pooled"])  # [B, out_dim]
