"""
Late Fusion Transformer with video-only valence and fused arousal.

This reproduces the legacy va_mode="video_valence":
- Valence head sees video-only transformer output
- Arousal head sees video+km fused output

The fusion returns pooled = concat([pooled_video, pooled_fused]) so that
the head can split and predict VA.
"""

from __future__ import annotations

from typing import Dict

import torch

from src.core.registry import FUSIONS
from src.core.types import EncoderOut, FusionOut
from src.models.components.fusion_utils import pool_tokens
from src.models.fusions.lft import LFTFusion


@FUSIONS.register("lft_video_valence")
class LFTVideoValenceFusion(LFTFusion):
    """
    LFT fusion that outputs [pooled_video, pooled_fused] concatenation.

    Inherits all configuration and components from LFTFusion.
    Only the forward pass differs: it runs the transformer twice
    (video-only for valence, all modalities for arousal).
    """

    def forward(
        self,
        z_dict: Dict[str, EncoderOut],
        mask_dict: Dict[str, torch.Tensor],
    ) -> FusionOut:
        if "video" not in z_dict:
            raise ValueError("lft_video_valence requires 'video' modality.")

        modality_names = sorted(z_dict.keys())
        mod_emb = self._get_modality_emb(modality_names)

        tokens_by_mod: Dict[str, torch.Tensor] = {}
        masks_by_mod: Dict[str, torch.Tensor] = {}
        for mod in modality_names:
            tok = self.pos_encoding(z_dict[mod]["tokens"])
            tok = mod_emb(tok, modality=mod)
            tokens_by_mod[mod] = tok
            masks_by_mod[mod] = mask_dict[mod]

        # Video-only path (valence)
        v_tokens, v_masks = self._add_cls_if_needed(
            tokens_by_mod["video"], masks_by_mod["video"],
        )
        v_fused = self.transformer(v_tokens, src_key_padding_mask=~v_masks)
        pooled_video = pool_tokens(v_fused, v_masks, self.pooling_type)

        # Fused path (arousal)
        all_tokens = [tokens_by_mod[mod] for mod in modality_names]
        all_masks = [masks_by_mod[mod] for mod in modality_names]
        f_tokens = torch.cat(all_tokens, dim=1)
        f_masks = torch.cat(all_masks, dim=1)
        f_tokens, f_masks = self._add_cls_if_needed(f_tokens, f_masks)
        f_fused = self.transformer(f_tokens, src_key_padding_mask=~f_masks)
        pooled_fused = pool_tokens(f_fused, f_masks, self.pooling_type)

        pooled = torch.cat([pooled_video, pooled_fused], dim=1)
        return FusionOut(tokens=f_fused, pooled=pooled)
