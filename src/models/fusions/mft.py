"""
Mid Fusion Transformer (MFT) — private per-modality Transformer layers followed
by cross-attention layers for inter-modality interaction.

中期融合 Transformer — 各模态先经过独立 Transformer 层，再通过交叉注意力交互。

Architecture:
    For each modality:
        tokens → pos_encoding → modality_embedding
    Private phase: N_private independent Transformer layers per modality
    Cross phase:   N_cross cross-attention layers (each modality attends to others)
    Concat along time → Pool

Key property:
    Cross-modal interaction starts at the MIDDLE of the network — after each
    modality has built modality-specific representations in its private layers.
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
from src.models.components.fusion_utils import cfg_get, pool_tokens


class CrossAttentionBlock(nn.Module):
    """Pre-norm cross-attention block: Q attends to KV from other modalities."""

    def __init__(self, d_model: int, nhead: int, dim_feedforward: int, dropout: float):
        super().__init__()
        self.norm_q = nn.LayerNorm(d_model)
        self.norm_kv = nn.LayerNorm(d_model)
        self.cross_attn = nn.MultiheadAttention(
            embed_dim=d_model, num_heads=nhead, dropout=dropout, batch_first=True,
        )
        self.dropout1 = nn.Dropout(dropout)
        self.norm2 = nn.LayerNorm(d_model)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, dim_feedforward),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim_feedforward, d_model),
            nn.Dropout(dropout),
        )

    def forward(
        self,
        query: torch.Tensor,
        key_value: torch.Tensor,
        query_mask: torch.Tensor,
        kv_mask: torch.Tensor,
    ) -> torch.Tensor:
        """
        Parameters
        ----------
        query : [B, T_q, D]
        key_value : [B, T_kv, D]
        query_mask : [B, T_q] (bool, True = valid)
        kv_mask : [B, T_kv] (bool, True = valid)

        Returns
        -------
        [B, T_q, D]
        """
        q = self.norm_q(query)
        kv = self.norm_kv(key_value)
        attn_out, _ = self.cross_attn(
            q, kv, kv, key_padding_mask=~kv_mask,
        )
        query = query + self.dropout1(attn_out)
        query = query + self.ffn(self.norm2(query))
        return query


@FUSIONS.register("mft")
class MFTFusion(BaseFusion):
    """
    Mid Fusion Transformer.

    Each modality first goes through private Transformer layers for
    modality-specific feature extraction, then cross-attention layers
    enable inter-modality information exchange.

    Parameters (via cfg dict)
    -------------------------
    d_model : int (default 512)
    nhead : int (default 8)
    num_private_layers : int (default 2)
    num_cross_layers : int (default 2)
    dim_feedforward : int (default 1024)
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
        self._num_private_layers = cfg_get(cfg, "num_private_layers", 2)
        num_cross_layers = cfg_get(cfg, "num_cross_layers", 2)
        dim_feedforward = cfg_get(cfg, "dim_feedforward", 1024)
        dropout = cfg_get(cfg, "dropout", 0.1)
        pos_type = cfg_get(cfg, "pos_encoding_type", "sinusoidal")
        max_seq_len = cfg_get(cfg, "max_seq_len", 1000)
        self.pooling_type = cfg_get(cfg, "pooling", "mean")
        self._nhead = nhead
        self._dim_feedforward = dim_feedforward
        self._dropout = dropout

        # Positional encoding
        if pos_type == "learnable":
            self.pos_encoding = LearnablePositionalEncoding(
                d_model=self.d_model, max_len=max_seq_len, dropout=dropout,
            )
        else:
            self.pos_encoding = SinusoidalPositionalEncoding(
                d_model=self.d_model, max_len=max_seq_len, dropout=dropout,
            )

        # Modality embedding — lazy-initialized
        self._modality_emb: Optional[ModalityEmbedding] = None
        self._modality_names: Optional[list] = None

        # Per-modality private Transformers — lazy-initialized
        self._private_transformers: Optional[nn.ModuleDict] = None

        # Cross-attention layers (shared across modalities)
        self.cross_layers = nn.ModuleList([
            CrossAttentionBlock(self.d_model, nhead, dim_feedforward, dropout)
            for _ in range(num_cross_layers)
        ])

    def _get_modality_emb(self, modality_names: list) -> ModalityEmbedding:
        if self._modality_emb is None or self._modality_names != modality_names:
            self._modality_names = modality_names
            self._modality_emb = ModalityEmbedding(
                d_model=self.d_model,
                num_modalities=len(modality_names),
                modality_names=modality_names,
            ).to(next(self.parameters()).device)
        return self._modality_emb

    def _make_private_transformer(self, device: torch.device) -> nn.TransformerEncoder:
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
            num_layers=self._num_private_layers,
            enable_nested_tensor=False,
        ).to(device)

    def _ensure_private_transformers(
        self, modality_names: list, device: torch.device,
    ) -> None:
        if self._private_transformers is None:
            self._private_transformers = nn.ModuleDict()
        for mod in modality_names:
            if mod not in self._private_transformers:
                self._private_transformers[mod] = self._make_private_transformer(device)

    def init_for_modalities(self, modality_names: list, device: torch.device) -> None:
        self._ensure_private_transformers(modality_names, device)
        self._get_modality_emb(sorted(modality_names))

    def forward(
        self,
        z_dict: Dict[str, EncoderOut],
        mask_dict: Dict[str, torch.Tensor],
    ) -> FusionOut:

        modality_names = sorted(z_dict.keys())

        # Ensure per-modality Transformers exist (incremental, never replaces)
        self._ensure_private_transformers(modality_names, z_dict[modality_names[0]]["tokens"].device)

        # Prepare: pos_encoding + modality_embedding
        mod_emb = self._get_modality_emb(modality_names)
        mod_tokens: Dict[str, torch.Tensor] = {}
        mod_masks: Dict[str, torch.Tensor] = {}
        for mod in modality_names:
            tok = self.pos_encoding(z_dict[mod]["tokens"])
            tok = mod_emb(tok, modality=mod)
            mod_tokens[mod] = tok
            mod_masks[mod] = mask_dict[mod]

        # Private phase: independent Transformer layers per modality
        for mod in modality_names:
            mod_tokens[mod] = self._private_transformers[mod](
                mod_tokens[mod], src_key_padding_mask=~mod_masks[mod],
            )

        # Cross phase: each modality attends to all other modalities
        if len(modality_names) > 1:
            for cross_layer in self.cross_layers:
                updated = {}
                for mod in modality_names:
                    # KV = concatenation of all OTHER modalities' tokens
                    other_tokens = [mod_tokens[m] for m in modality_names if m != mod]
                    other_masks = [mod_masks[m] for m in modality_names if m != mod]
                    kv = torch.cat(other_tokens, dim=1)
                    kv_mask = torch.cat(other_masks, dim=1)

                    updated[mod] = cross_layer(
                        mod_tokens[mod], kv, mod_masks[mod], kv_mask,
                    )
                mod_tokens = updated

        # Align to min_t, average across modalities → [B, T, D]
        min_t = min(mod_tokens[mod].shape[1] for mod in modality_names)
        stacked = torch.stack(
            [mod_tokens[mod][:, :min_t, :] for mod in modality_names], dim=0,
        )  # [M, B, T, D]
        tokens = stacked.mean(dim=0)  # [B, T, D]

        stacked_masks = torch.stack(
            [mod_masks[mod][:, :min_t] for mod in modality_names], dim=0,
        )  # [M, B, T]
        masks = stacked_masks.any(dim=0)  # [B, T]

        pooled = pool_tokens(tokens, masks, self.pooling_type)

        return FusionOut(tokens=tokens, pooled=pooled)
