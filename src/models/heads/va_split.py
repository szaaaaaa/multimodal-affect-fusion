"""
Valence/Arousal split head for video-valence fusion.

Expects pooled features concatenated as [video_only, fused], each d_model.
"""

from __future__ import annotations

import torch
from torch import nn

from src.core.registry import HEADS
from src.core.types import BaseHead, FusionOut


@HEADS.register("va_split")
class VASplitHead(BaseHead):
    """
    Split head: valence from first half, arousal from second half.

    Parameters (via cfg dict)
    -------------------------
    d_model : int (default 256)
    hidden_dim : int (default 128)
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
        dropout = _g("dropout", 0.1)
        self.d_model = d_model

        def _make_head() -> nn.Sequential:
            return nn.Sequential(
                nn.Linear(d_model, hidden_dim),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim, 1),
            )

        self.head_valence = _make_head()
        self.head_arousal = _make_head()

    def forward(self, h: FusionOut) -> torch.Tensor:
        pooled = h["pooled"]
        if pooled.size(1) != self.d_model * 2:
            raise ValueError(
                f"VASplitHead expects pooled dim {self.d_model * 2}, got {pooled.size(1)}"
            )
        v = pooled[:, : self.d_model]
        a = pooled[:, self.d_model :]
        valence = self.head_valence(v)
        arousal = self.head_arousal(a)
        return torch.cat([valence, arousal], dim=1)
