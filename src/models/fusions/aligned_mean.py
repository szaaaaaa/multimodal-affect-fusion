"""
Time-aligned multimodal fusion by per-step masked mean across modalities.
"""

from __future__ import annotations

from typing import Dict

import torch

from src.core.registry import FUSIONS
from src.core.types import BaseFusion, EncoderOut, FusionOut


@FUSIONS.register("aligned_mean")
class AlignedMeanFusion(BaseFusion):
    """
    Fuse aligned modality tokens at each timestep.

    Input modality tokens are truncated to the minimum shared T and fused by
    masked mean across modality dimension.
    """

    def __init__(self, cfg=None):
        super().__init__()

    def forward(
        self,
        z_dict: Dict[str, EncoderOut],
        mask_dict: Dict[str, torch.Tensor],
    ) -> FusionOut:
        if not z_dict:
            raise ValueError("aligned_mean fusion received empty z_dict")

        mods = sorted(z_dict.keys())
        min_t = min(z_dict[m]["tokens"].shape[1] for m in mods)

        tokens = []
        masks = []
        for m in mods:
            tok = z_dict[m]["tokens"][:, :min_t, :]
            msk = mask_dict[m][:, :min_t]
            tokens.append(tok)
            masks.append(msk)

        tok_stack = torch.stack(tokens, dim=0)  # [M, B, T, D]
        mask_stack = torch.stack(masks, dim=0)  # [M, B, T]

        w = mask_stack.float().unsqueeze(-1)
        fused_tokens = (tok_stack * w).sum(dim=0) / w.sum(dim=0).clamp(min=1.0)
        fused_mask = mask_stack.any(dim=0)

        mask_f = fused_mask.float().unsqueeze(-1)
        pooled = (fused_tokens * mask_f).sum(dim=1) / mask_f.sum(dim=1).clamp(min=1.0)
        return FusionOut(tokens=fused_tokens, pooled=pooled)
