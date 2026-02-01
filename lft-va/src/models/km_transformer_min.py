"""
Minimal KM-only Transformer regressor.


"""

from __future__ import annotations

import sys
from pathlib import Path

import torch
from torch import nn

# Ensure project root on path to import encoder/*
ROOT = Path(__file__).resolve().parents[4]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from encoder.km.km_encoder_1dCNN import KM1DCNNEncoder
from encoder.km.km_encoder_stat import KMStatTokenEncoder


class KMTransformerRegressor(nn.Module):
    """
    KM-only Transformer baseline with optional CNN encoder.
    """

    def __init__(
        self,
        d_in: int = 15,
        d_model: int = 64,
        nhead: int = 4,
        num_layers: int = 2,
        km_encoder: str = "stat",
    ):
        super().__init__()
        if km_encoder == "cnn":
            self.in_proj = KM1DCNNEncoder(d_in=d_in, d_model=d_model)
        else:
            self.in_proj = KMStatTokenEncoder(d_in=d_in, d_model=d_model)

        enc_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead, batch_first=True)
        self.encoder = nn.TransformerEncoder(enc_layer, num_layers=num_layers)
        self.head = nn.Linear(d_model, 1)

    def forward(self, km: torch.Tensor, km_mask: torch.Tensor) -> torch.Tensor:
        x = self.in_proj(km)

        if km_mask is None:
            pad_mask = None
            mask_f = torch.ones_like(x[:, :, 0], dtype=torch.float32)
        else:
            pad_mask = ~km_mask
            mask_f = km_mask.to(torch.float32)

        x = self.encoder(x, src_key_padding_mask=pad_mask)

        denom = mask_f.sum(dim=1, keepdim=True).clamp(min=1.0)
        pooled = (x * mask_f.unsqueeze(-1)).sum(dim=1) / denom

        return self.head(pooled)
