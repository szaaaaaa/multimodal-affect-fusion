"""
Late Fusion Transformer (LFT) — each modality independently encoded by its own
Transformer, then fused via attention-weighted combination.

晚期融合 Transformer — 各模态由独立 Transformer 编码，最后通过注意力加权融合。

Architecture:
    For each modality:
        tokens → pos_encoding → Independent Transformer (N layers) → pool → [B, D]
    Stack modality representations → [B, M, D]
    → Attention-weighted fusion → [B, D]

Key property:
    NO cross-modal interaction during feature extraction. Each modality's
    Transformer runs independently. Cross-modal information only flows
    at the final fusion stage.
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
)
from src.models.components.fusion_utils import cfg_get, pool_tokens


@FUSIONS.register("lft")
class LFTFusion(BaseFusion):
    """
    Late Fusion Transformer.

    Each modality is processed by its own independent Transformer encoder.
    After independent encoding, modality representations are fused via
    multi-head attention with a learned query token.

    Parameters (via cfg dict)
    -------------------------
    d_model : int (default 512)
    nhead : int (default 8)
    num_layers : int (default 4)
    dim_feedforward : int (default 1024)
    dropout : float (default 0.1)
    pos_encoding_type : str (default "sinusoidal")
    max_seq_len : int (default 1000)
    pooling : str (default "mean") — per-modality pooling before fusion
    """

    def __init__(self, cfg=None):
        super().__init__()
        cfg = cfg or {}
        self.d_model = cfg_get(cfg, "d_model", 512)
        self._nhead = cfg_get(cfg, "nhead", 8)
        self._num_layers = cfg_get(cfg, "num_layers", 4)
        self._dim_feedforward = cfg_get(cfg, "dim_feedforward", 1024)
        self._dropout = cfg_get(cfg, "dropout", 0.1)
        pos_type = cfg_get(cfg, "pos_encoding_type", "sinusoidal")
        max_seq_len = cfg_get(cfg, "max_seq_len", 1000)
        self.pooling_type = cfg_get(cfg, "pooling", "mean")

        # Positional encoding (shared params, but each modality uses independently)
        if pos_type == "learnable":
            self.pos_encoding = LearnablePositionalEncoding(
                d_model=self.d_model, max_len=max_seq_len, dropout=self._dropout,
            )
        else:
            self.pos_encoding = SinusoidalPositionalEncoding(
                d_model=self.d_model, max_len=max_seq_len, dropout=self._dropout,
            )

        # Per-modality Transformers — lazy-initialized
        self._transformers: Optional[nn.ModuleDict] = None

        # Attention-weighted fusion: learned query attends to modality representations
        self.fusion_query = nn.Parameter(torch.randn(1, 1, self.d_model) * 0.02)
        self.fusion_attn = nn.MultiheadAttention(
            embed_dim=self.d_model,
            num_heads=self._nhead,
            dropout=self._dropout,
            batch_first=True,
        )
        self.fusion_norm = nn.LayerNorm(self.d_model)

    def _make_transformer(self, device: torch.device) -> nn.TransformerEncoder:
        layer = nn.TransformerEncoderLayer(
            d_model=self.d_model,
            nhead=self._nhead,
            dim_feedforward=self._dim_feedforward,
            dropout=self._dropout,
            batch_first=True,
            norm_first=True,
        )
        return nn.TransformerEncoder(
            layer,
            num_layers=self._num_layers,
            enable_nested_tensor=False,
        ).to(device)

    def _ensure_transformers(self, modality_names: list, device: torch.device) -> None:
        if self._transformers is None:
            self._transformers = nn.ModuleDict()
        for mod in modality_names:
            if mod not in self._transformers:
                self._transformers[mod] = self._make_transformer(device)

    def init_for_modalities(self, modality_names: list, device: torch.device) -> None:
        self._ensure_transformers(modality_names, device)

    def forward(
        self,
        z_dict: Dict[str, EncoderOut],
        mask_dict: Dict[str, torch.Tensor],
    ) -> FusionOut:

        modality_names = sorted(z_dict.keys())

        # Ensure per-modality Transformers exist (incremental, never replaces)
        self._ensure_transformers(modality_names, z_dict[modality_names[0]]["tokens"].device)

        # Single modality: independent Transformer + pool
        if len(modality_names) == 1:
            mod = modality_names[0]
            tokens = self.pos_encoding(z_dict[mod]["tokens"])
            mask = z_dict[mod]["mask"]
            tokens = self._transformers[mod](tokens, src_key_padding_mask=~mask)
            pooled = pool_tokens(tokens, mask, self.pooling_type)
            return FusionOut(tokens=tokens, pooled=pooled)

        # Each modality through its independent Transformer
        mod_encoded = {}
        mod_masks = {}
        mod_pooled = []
        for mod in modality_names:
            tokens = self.pos_encoding(z_dict[mod]["tokens"])
            mask = z_dict[mod]["mask"]
            encoded = self._transformers[mod](tokens, src_key_padding_mask=~mask)
            mod_encoded[mod] = encoded
            mod_masks[mod] = mask
            mod_pooled.append(pool_tokens(encoded, mask, self.pooling_type))  # [B, D]

        # Attention-weighted fusion for pooled representation
        mod_stack = torch.stack(mod_pooled, dim=1)             # [B, M, D]
        B = mod_stack.size(0)
        query = self.fusion_query.expand(B, -1, -1)            # [B, 1, D]
        fused, _ = self.fusion_attn(query, mod_stack, mod_stack)  # [B, 1, D]
        pooled = self.fusion_norm(fused.squeeze(1))             # [B, D]

        # Align tokens to min_t, average across modalities for seq head
        min_t = min(mod_encoded[mod].shape[1] for mod in modality_names)
        stacked = torch.stack(
            [mod_encoded[mod][:, :min_t, :] for mod in modality_names], dim=0,
        )  # [M, B, T, D]
        avg_tokens = stacked.mean(dim=0)  # [B, T, D]

        return FusionOut(tokens=avg_tokens, pooled=pooled)
