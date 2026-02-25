"""
Sequence regression head for strict temporal regression.
"""

from __future__ import annotations

import torch
from torch import nn

from src.core.registry import HEADS
from src.core.types import BaseHead, FusionOut


@HEADS.register("regression_seq")
class RegressionSeqHead(BaseHead):
    """
    Token-wise MLP head.

    Input: h["tokens"] [B, T, D]
    Output: [B, T, out_dim]
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
        tokens = h.get("tokens", None)
        if tokens is None:
            raise ValueError("regression_seq head requires h['tokens']")
        return self.mlp(tokens)
