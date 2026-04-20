"""
Telem 60Hz stream encoder —— 连续遥测流 → 视频 5Hz 同步 token 序列。

Input
-----
x    : FloatTensor[B, T_t, D_t]   60Hz 栅格，T_t = T_v * 12
mask : BoolTensor[B, T_t]         (几乎全 True)

Output
------
tokens : [B, T_v, d_model]
pooled : [B, d_model]
mask   : [B, T_v]

架构
----
Linear(D_t → d) + LN + Dropout
→ 3 层 dilated Conv1d (kernel=5, dilation=[1,2,4]), residual, GELU, LayerNorm
→ 学习式下采样 Conv1d(kernel=12, stride=12) → [B, T_v, d]
"""

from __future__ import annotations

from typing import Optional

import torch
from torch import nn

from src.core.registry import get_encoder_registry
from src.core.types import BaseEncoder, EncoderOut
from src.models.components.fusion_utils import cfg_get, masked_mean_pool


class _DilatedTCNBlock(nn.Module):
    def __init__(self, d_model: int, kernel: int, dilation: int, dropout: float):
        super().__init__()
        pad = ((kernel - 1) // 2) * dilation
        self.conv = nn.Conv1d(d_model, d_model, kernel_size=kernel, padding=pad, dilation=dilation)
        self.act = nn.GELU()
        self.drop = nn.Dropout(dropout)
        self.ln = nn.LayerNorm(d_model)

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        # h: [B, D, T]
        y = self.drop(self.act(self.conv(h)))
        y = h + y  # residual
        # LN on channel dim → transpose
        y = self.ln(y.transpose(1, 2)).transpose(1, 2)
        return y


@get_encoder_registry("telem_60hz").register("stream_60hz")
class TelemStream60HzEncoder(BaseEncoder):
    def __init__(self, cfg):
        super().__init__()
        d_in = cfg_get(cfg, "feature_dim", 23)
        d_model = cfg_get(cfg, "d_model", 512)
        tcn_layers = cfg_get(cfg, "tcn_layers", 3)
        kernel = cfg_get(cfg, "kernel", 5)
        dilations = cfg_get(cfg, "dilations", None) or [2 ** i for i in range(tcn_layers)]
        dropout = cfg_get(cfg, "dropout", 0.1)
        downsample_stride = cfg_get(cfg, "downsample_stride", 12)

        if len(dilations) != tcn_layers:
            raise ValueError(f"dilations ({len(dilations)}) must match tcn_layers ({tcn_layers})")

        self.d_model = d_model
        self.downsample_stride = int(downsample_stride)

        self.proj = nn.Linear(d_in, d_model)
        self.ln_in = nn.LayerNorm(d_model)
        self.drop_in = nn.Dropout(dropout)
        self.blocks = nn.ModuleList(
            [_DilatedTCNBlock(d_model, kernel, d, dropout) for d in dilations]
        )
        self.downsample = nn.Conv1d(
            d_model, d_model, kernel_size=self.downsample_stride, stride=self.downsample_stride
        )

    def forward(
        self, x: torch.Tensor, mask: Optional[torch.Tensor] = None
    ) -> EncoderOut:
        B, T_t, _ = x.shape
        if mask is None:
            mask = torch.ones(B, T_t, dtype=torch.bool, device=x.device)

        # Zero-out invalid 60Hz frames so TCN doesn't see raw padding
        h = self.proj(x) * mask.float().unsqueeze(-1)
        h = self.drop_in(self.ln_in(h))

        h = h.transpose(1, 2)  # [B, D, T_t]
        for blk in self.blocks:
            h = blk(h)
        h = self.downsample(h)  # [B, D, T_v]
        tokens_v = h.transpose(1, 2)  # [B, T_v, D]
        T_v = tokens_v.shape[1]

        # Downsampled mask: any True in a stride-12 window → True
        mask_pool = mask.unsqueeze(1).float()  # [B, 1, T_t]
        mask_pool = torch.nn.functional.avg_pool1d(
            mask_pool, kernel_size=self.downsample_stride, stride=self.downsample_stride
        ).squeeze(1)  # [B, T_v]
        mask_v = mask_pool > 0  # True if the 60Hz window had any valid sample

        pooled = masked_mean_pool(tokens_v, mask_v)
        return EncoderOut(tokens=tokens_v, pooled=pooled, mask=mask_v)
