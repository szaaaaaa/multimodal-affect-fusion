"""
Late Fusion Transformer with video-only valence and fused arousal.

This reproduces the legacy va_mode="video_valence":
- Valence head sees video-only transformer output
- Arousal head sees video+km fused output

The fusion returns pooled = concat([pooled_video, pooled_fused]) so that
the head can split and predict VA.
"""

from __future__ import annotations

from typing import Dict, Optional

import torch
from torch import nn

from src.core.registry import FUSIONS
from src.core.types import BaseFusion, EncoderOut, FusionOut
from src.models.components import (
    SinusoidalPositionalEncoding,
    LearnablePositionalEncoding,
    ModalityEmbedding,
)


@FUSIONS.register("lft_video_valence")
class LFTVideoValenceFusion(BaseFusion):
    """
    LFT fusion that outputs [pooled_video, pooled_fused] concatenation.

    Parameters (via cfg dict)
    -------------------------
    d_model : int (default 256)
    nhead : int (default 8)
    num_layers : int (default 4)
    dim_feedforward : int (default 1024)
    dropout : float (default 0.1)
    max_seq_len : int (default 1000)
    pos_encoding_type : str (default "sinusoidal")
    pooling : str (default "mean")  -- "mean", "max", or "cls"
    """

    def __init__(self, cfg):
        super().__init__()
        if isinstance(cfg, dict):
            _g = cfg.get
        else:
            _g = lambda k, d=None: getattr(cfg, k, d)

        d_model = _g("d_model", 256)
        nhead = _g("nhead", 8)
        num_layers = _g("num_layers", 4)
        dim_feedforward = _g("dim_feedforward", 1024)
        dropout = _g("dropout", 0.1)
        max_seq_len = _g("max_seq_len", 1000)
        pos_type = _g("pos_encoding_type", "sinusoidal")
        self.pooling_type = _g("pooling", "mean")
        self.d_model = d_model

        if pos_type == "learnable":
            self.pos_encoding = LearnablePositionalEncoding(
                d_model=d_model, max_len=max_seq_len, dropout=dropout,
            )
        else:
            self.pos_encoding = SinusoidalPositionalEncoding(
                d_model=d_model, max_len=max_seq_len, dropout=dropout,
            )

        self._modality_emb: Optional[ModalityEmbedding] = None
        self._modality_names: Optional[list] = None
        self._d_model = d_model

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True,
            norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(
            encoder_layer,
            num_layers=num_layers,
            enable_nested_tensor=False,
        )

        if self.pooling_type == "cls":
            self.cls_token = nn.Parameter(torch.randn(1, 1, d_model) * 0.02)

    def _get_modality_emb(self, modality_names: list) -> ModalityEmbedding:
        if self._modality_emb is None or self._modality_names != modality_names:
            self._modality_names = modality_names
            self._modality_emb = ModalityEmbedding(
                d_model=self._d_model,
                num_modalities=len(modality_names),
                modality_names=modality_names,
            ).to(next(self.parameters()).device)
        return self._modality_emb

    def _add_cls(self, tokens: torch.Tensor, masks: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        if self.pooling_type != "cls":
            return tokens, masks
        B = tokens.size(0)
        cls_tokens = self.cls_token.expand(B, -1, -1)
        tokens = torch.cat([cls_tokens, tokens], dim=1)
        cls_mask = torch.ones(B, 1, dtype=torch.bool, device=tokens.device)
        masks = torch.cat([cls_mask, masks], dim=1)
        return tokens, masks

    def _pool(self, fused: torch.Tensor, masks: torch.Tensor) -> torch.Tensor:
        if self.pooling_type == "cls":
            return fused[:, 0, :]
        padding_mask = ~masks
        if self.pooling_type == "max":
            fused_masked = fused.masked_fill(padding_mask.unsqueeze(-1), float("-inf"))
            return fused_masked.max(dim=1)[0]
        mask_f = masks.float().unsqueeze(-1)
        return (fused * mask_f).sum(dim=1) / mask_f.sum(dim=1).clamp(min=1.0)

    def _prepare_tokens(
        self,
        z_dict: Dict[str, EncoderOut],
        mask_dict: Dict[str, torch.Tensor],
    ) -> tuple[Dict[str, torch.Tensor], Dict[str, torch.Tensor], list]:
        modality_names = sorted(z_dict.keys())
        mod_emb = self._get_modality_emb(modality_names)

        tokens_by_mod: Dict[str, torch.Tensor] = {}
        masks_by_mod: Dict[str, torch.Tensor] = {}
        for mod in modality_names:
            z = z_dict[mod]
            tok = z["tokens"]
            tok = self.pos_encoding(tok)
            tok = mod_emb(tok, modality=mod)
            tokens_by_mod[mod] = tok
            masks_by_mod[mod] = mask_dict[mod]
        return tokens_by_mod, masks_by_mod, modality_names

    def forward(
        self,
        z_dict: Dict[str, EncoderOut],
        mask_dict: Dict[str, torch.Tensor],
    ) -> FusionOut:
        tokens_by_mod, masks_by_mod, modality_names = self._prepare_tokens(z_dict, mask_dict)

        if "video" not in tokens_by_mod:
            raise ValueError("lft_video_valence requires 'video' modality.")

        # Video-only path (valence)
        video_tokens = tokens_by_mod["video"]
        video_masks = masks_by_mod["video"]
        v_tokens, v_masks = self._add_cls(video_tokens, video_masks)
        v_padding = ~v_masks
        v_fused = self.transformer(v_tokens, src_key_padding_mask=v_padding)
        pooled_video = self._pool(v_fused, v_masks)

        # Fused path (arousal)
        all_tokens = [tokens_by_mod[mod] for mod in modality_names]
        all_masks = [masks_by_mod[mod] for mod in modality_names]
        fused_tokens = torch.cat(all_tokens, dim=1)
        fused_masks = torch.cat(all_masks, dim=1)
        f_tokens, f_masks = self._add_cls(fused_tokens, fused_masks)
        f_padding = ~f_masks
        f_fused = self.transformer(f_tokens, src_key_padding_mask=f_padding)
        pooled_fused = self._pool(f_fused, f_masks)

        pooled = torch.cat([pooled_video, pooled_fused], dim=1)
        return FusionOut(tokens=f_fused, pooled=pooled)

