"""
Early Fusion — concatenate modality features along feature dim, then shared Transformer.

早期融合 — 沿特征维度拼接各模态，再用共享 Transformer 处理。

Architecture:
    Align all modalities to shortest temporal length
    → Concatenate along feature dim: [B, T, M * d_model]
    → Linear projection back to d_model
    → Positional encoding
    → Shared Transformer
    → Pool

Key difference from LFT:
    - LFT concatenates along TIME axis (different modality tokens at different positions)
    - Early Fusion concatenates along FEATURE axis (single merged token per timestep)
    Cross-modal information is mixed at input level, not learned through attention.
"""

from __future__ import annotations

from typing import Dict

import torch
from torch import nn

from src.core.registry import FUSIONS
from src.core.types import BaseFusion, EncoderOut, FusionOut
from src.models.components import (
    SinusoidalPositionalEncoding,
    LearnablePositionalEncoding,
)
from src.models.components.fusion_utils import cfg_get, pool_tokens


@FUSIONS.register("early")
class EarlyFusion(BaseFusion):
    """
    Early fusion: concat features along dim, project, then shared Transformer.

    Parameters (via cfg dict)
    -------------------------
    d_model : int (default 512)
    nhead : int (default 8)
    num_layers : int (default 2)
    dim_feedforward : int (default 512)
    dropout : float (default 0.1)
    pos_encoding_type : str (default "sinusoidal")
    max_seq_len : int (default 1000)
    pooling : str (default "mean")
    """

    def __init__(self, cfg=None):
        super().__init__()
        cfg = cfg or {}
        self.d_model = cfg_get(cfg, "d_model", 512)
        nhead = cfg_get(cfg, "nhead", 8)
        num_layers = cfg_get(cfg, "num_layers", 2)
        dim_feedforward = cfg_get(cfg, "dim_feedforward", 512)
        dropout = cfg_get(cfg, "dropout", 0.1)
        pos_type = cfg_get(cfg, "pos_encoding_type", "sinusoidal")
        max_seq_len = cfg_get(cfg, "max_seq_len", 1000)
        self.pooling_type = cfg_get(cfg, "pooling", "mean")

        # Positional encoding
        if pos_type == "learnable":
            self.pos_encoding = LearnablePositionalEncoding(
                d_model=self.d_model, max_len=max_seq_len, dropout=dropout,
            )
        else:
            self.pos_encoding = SinusoidalPositionalEncoding(
                d_model=self.d_model, max_len=max_seq_len, dropout=dropout,
            )

        # Projection from M*d_model → d_model (lazy init, modality count unknown)
        self._proj: nn.Module | None = None
        self._num_modalities: int | None = None

        # Shared Transformer
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=self.d_model,
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

    def _init_proj(self, num_modalities: int, device: torch.device) -> None:
        self._proj = nn.Sequential(
            nn.Linear(num_modalities * self.d_model, self.d_model),
            nn.LayerNorm(self.d_model),
            nn.GELU(),
            nn.Dropout(0.1),
        ).to(device)
        self._num_modalities = num_modalities

    def forward(
        self,
        z_dict: Dict[str, EncoderOut],
        mask_dict: Dict[str, torch.Tensor],
    ) -> FusionOut:

        # Single modality: skip concat, go straight to transformer
        if len(z_dict) == 1:
            z = next(iter(z_dict.values()))
            mask = z["mask"]
            tokens = self.pos_encoding(z["tokens"])
            tokens = self.transformer(tokens, src_key_padding_mask=~mask)
            pooled = pool_tokens(tokens, mask, self.pooling_type)
            return FusionOut(tokens=tokens, pooled=pooled)

        modality_names = sorted(z_dict.keys())

        # Lazy init projection layer
        if self._proj is None or self._num_modalities != len(modality_names):
            device = z_dict[modality_names[0]]["tokens"].device
            self._init_proj(len(modality_names), device)

        # Align to shortest temporal length
        min_t = min(z_dict[mod]["tokens"].shape[1] for mod in modality_names)

        trimmed_tokens = []
        trimmed_masks = []
        for mod in modality_names:
            trimmed_tokens.append(z_dict[mod]["tokens"][:, :min_t, :])  # [B, T, D]
            trimmed_masks.append(mask_dict[mod][:, :min_t])              # [B, T]

        # Concat along feature dim: [B, T, M * D]
        concat_features = torch.cat(trimmed_tokens, dim=-1)

        # Union mask: valid if any modality is valid
        fused_mask = torch.stack(trimmed_masks, dim=0).any(dim=0)  # [B, T]

        # Project M*D → D, add pos encoding, run transformer
        tokens = self._proj(concat_features)           # [B, T, D]
        tokens = self.pos_encoding(tokens)
        tokens = self.transformer(tokens, src_key_padding_mask=~fused_mask)

        pooled = pool_tokens(tokens, fused_mask, self.pooling_type)

        return FusionOut(tokens=tokens, pooled=pooled)
