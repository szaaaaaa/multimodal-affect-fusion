"""
Single-modality pass-through fusion.

单模态直通融合 — 当只有一个模态时，直接输出该模态的 encoder 结果。
也支持多模态输入（简单 concat + mean pool）。
"""

from __future__ import annotations

from typing import Dict, Optional

import torch
from torch import nn

from src.core.registry import FUSIONS
from src.core.types import BaseFusion, EncoderOut, FusionOut


@FUSIONS.register("single")
class SingleFusion(BaseFusion):
    """
    Pass-through / simple concatenation fusion.

    If one modality: pass through.
    If multiple: concatenate tokens, mean-pool.
    """

    def __init__(self, cfg=None):
        super().__init__()

    def forward(
        self,
        z_dict: Dict[str, EncoderOut],
        mask_dict: Dict[str, torch.Tensor],
    ) -> FusionOut:
        if len(z_dict) == 1:
            z = next(iter(z_dict.values()))
            return FusionOut(tokens=z["tokens"], pooled=z["pooled"])

        # Multiple modalities: concat tokens, masked mean pool
        all_tokens = []
        all_masks = []
        for mod in sorted(z_dict.keys()):
            all_tokens.append(z_dict[mod]["tokens"])
            all_masks.append(mask_dict[mod])

        tokens = torch.cat(all_tokens, dim=1)    # [B, T_total, D]
        masks = torch.cat(all_masks, dim=1)       # [B, T_total]

        mask_f = masks.float().unsqueeze(-1)      # [B, T_total, 1]
        pooled = (tokens * mask_f).sum(dim=1) / mask_f.sum(dim=1).clamp(min=1.0)

        return FusionOut(tokens=tokens, pooled=pooled)
