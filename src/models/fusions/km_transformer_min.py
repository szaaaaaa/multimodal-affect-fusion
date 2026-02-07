"""
KM-only Transformer fusion (minimal baseline).

Reproduces the legacy KM-only Transformer regressor behavior while fitting
the new Encoder -> Fusion -> Head pipeline. Uses the encoder's tokens and
mask, applies a Transformer encoder, then masked mean pooling.
"""

from __future__ import annotations

from typing import Dict

import torch
from torch import nn

from src.core.registry import FUSIONS
from src.core.types import BaseFusion, EncoderOut, FusionOut


@FUSIONS.register("km_transformer_min")
class KMTransformerMinFusion(BaseFusion):
    """
    Minimal KM-only Transformer fusion.

    Parameters (via cfg dict)
    -------------------------
    d_model : int (default 64)
    nhead : int (default 4)
    num_layers : int (default 2)
    dropout : float (default 0.1)
    pooling : str (default "mean")  -- "mean", "max", or "cls"
    """

    def __init__(self, cfg=None):
        super().__init__()
        if isinstance(cfg, dict):
            _g = cfg.get
        else:
            _g = lambda k, d=None: getattr(cfg, k, d)

        d_model = _g("d_model", 64)
        nhead = _g("nhead", 4)
        num_layers = _g("num_layers", 2)
        dropout = _g("dropout", 0.1)
        self.pooling_type = _g("pooling", "mean")
        self.d_model = d_model

        enc_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dropout=dropout,
            batch_first=True,
            norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(
            enc_layer,
            num_layers=num_layers,
            enable_nested_tensor=False,
        )

        if self.pooling_type == "cls":
            self.cls_token = nn.Parameter(torch.randn(1, 1, d_model) * 0.02)

    def _select_modality(self, z_dict: Dict[str, EncoderOut]) -> str:
        if "km" in z_dict:
            return "km"
        if len(z_dict) == 1:
            return next(iter(z_dict.keys()))
        raise ValueError("KMTransformerMinFusion expects a single modality (preferably 'km').")

    def forward(
        self,
        z_dict: Dict[str, EncoderOut],
        mask_dict: Dict[str, torch.Tensor],
    ) -> FusionOut:
        mod = self._select_modality(z_dict)
        z = z_dict[mod]
        tokens = z["tokens"]  # [B, T, D]
        masks = mask_dict[mod]  # [B, T]

        if self.pooling_type == "cls":
            B = tokens.size(0)
            cls_tokens = self.cls_token.expand(B, -1, -1)
            tokens = torch.cat([cls_tokens, tokens], dim=1)
            cls_mask = torch.ones(B, 1, dtype=torch.bool, device=tokens.device)
            masks = torch.cat([cls_mask, masks], dim=1)

        padding_mask = ~masks
        fused = self.transformer(tokens, src_key_padding_mask=padding_mask)

        if self.pooling_type == "cls":
            pooled = fused[:, 0, :]
        elif self.pooling_type == "max":
            fused_masked = fused.masked_fill(padding_mask.unsqueeze(-1), float("-inf"))
            pooled = fused_masked.max(dim=1)[0]
        else:
            mask_f = masks.float().unsqueeze(-1)
            pooled = (fused * mask_f).sum(dim=1) / mask_f.sum(dim=1).clamp(min=1.0)

        return FusionOut(tokens=fused, pooled=pooled)
