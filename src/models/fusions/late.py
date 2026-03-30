"""
Late Fusion — each modality processed independently, then average representations.

晚期融合 — 各模态独立处理，无跨模态交互，最后平均表示。

Architecture:
    For each modality:
        tokens → pos_encoding → independent Transformer → modality_repr
    Align all to shortest temporal length
    → Average token representations across modalities
    → Pool

Key difference from LFT:
    - LFT: all modality tokens are concatenated and processed by a SHARED
      Transformer, where cross-modal attention is possible.
    - Late Fusion: each modality has its OWN Transformer. No cross-modal
      information flow at any stage. This serves as the "no-interaction"
      baseline to demonstrate whether cross-modal fusion adds value.
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


@FUSIONS.register("late")
class LateFusion(BaseFusion):
    """
    Late (decision-level-like) fusion with independent per-modality Transformers.

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
        self._nhead = cfg_get(cfg, "nhead", 8)
        self._num_layers = cfg_get(cfg, "num_layers", 2)
        self._dim_feedforward = cfg_get(cfg, "dim_feedforward", 512)
        self._dropout = cfg_get(cfg, "dropout", 0.1)
        pos_type = cfg_get(cfg, "pos_encoding_type", "sinusoidal")
        max_seq_len = cfg_get(cfg, "max_seq_len", 1000)
        self.pooling_type = cfg_get(cfg, "pooling", "mean")

        # Positional encoding (shared across modalities — no learnable cross-modal info)
        if pos_type == "learnable":
            self.pos_encoding = LearnablePositionalEncoding(
                d_model=self.d_model, max_len=max_seq_len, dropout=self._dropout,
            )
        else:
            self.pos_encoding = SinusoidalPositionalEncoding(
                d_model=self.d_model, max_len=max_seq_len, dropout=self._dropout,
            )

        # Per-modality Transformers are lazy-initialized (modality set unknown at init)
        self._transformers: Optional[nn.ModuleDict] = None
        self._modality_names: Optional[list] = None

    def _init_transformers(self, modality_names: list, device: torch.device) -> None:
        transformers = {}
        for mod in modality_names:
            layer = nn.TransformerEncoderLayer(
                d_model=self.d_model,
                nhead=self._nhead,
                dim_feedforward=self._dim_feedforward,
                dropout=self._dropout,
                batch_first=True,
                norm_first=True,
            )
            transformers[mod] = nn.TransformerEncoder(
                layer,
                num_layers=self._num_layers,
                enable_nested_tensor=False,
            )
        self._transformers = nn.ModuleDict(transformers).to(device)
        self._modality_names = modality_names

    def forward(
        self,
        z_dict: Dict[str, EncoderOut],
        mask_dict: Dict[str, torch.Tensor],
    ) -> FusionOut:

        modality_names = sorted(z_dict.keys())

        # Single modality: just run through transformer
        if len(modality_names) == 1:
            mod = modality_names[0]
            if self._transformers is None or mod not in self._transformers:
                self._init_transformers(modality_names, z_dict[mod]["tokens"].device)
            z = z_dict[mod]
            tokens = self.pos_encoding(z["tokens"])
            mask = z["mask"]
            tokens = self._transformers[mod](tokens, src_key_padding_mask=~mask)
            pooled = pool_tokens(tokens, mask, self.pooling_type)
            return FusionOut(tokens=tokens, pooled=pooled)

        # Lazy init per-modality transformers
        if self._transformers is None or self._modality_names != modality_names:
            device = z_dict[modality_names[0]]["tokens"].device
            self._init_transformers(modality_names, device)

        # Process each modality independently
        mod_tokens = {}
        mod_masks = {}
        for mod in modality_names:
            tokens = self.pos_encoding(z_dict[mod]["tokens"])
            mask = z_dict[mod]["mask"]
            mod_tokens[mod] = self._transformers[mod](tokens, src_key_padding_mask=~mask)
            mod_masks[mod] = mask

        # Align to shortest temporal length, then average
        min_t = min(mod_tokens[mod].shape[1] for mod in modality_names)

        stacked_tokens = torch.stack(
            [mod_tokens[mod][:, :min_t, :] for mod in modality_names], dim=0
        )  # [M, B, T, D]
        averaged_tokens = stacked_tokens.mean(dim=0)  # [B, T, D]

        stacked_masks = torch.stack(
            [mod_masks[mod][:, :min_t] for mod in modality_names], dim=0
        )  # [M, B, T]
        fused_mask = stacked_masks.any(dim=0)  # [B, T]

        pooled = pool_tokens(averaged_tokens, fused_mask, self.pooling_type)

        return FusionOut(tokens=averaged_tokens, pooled=pooled)
