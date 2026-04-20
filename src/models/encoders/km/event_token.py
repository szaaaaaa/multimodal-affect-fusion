"""
KM event token encoder —— 离散键鼠事件 → 视频 5Hz 同步 token 序列。

Input (encoder.forward x)
-------------------------
x : FloatTensor[B, T_k, 4]
    channel 0 : event_type_id（long cast 回 long）
    channel 1 : t_rel_sec      相对视频起点的秒数
    channel 2 : dt_sec         相邻事件间隔
    channel 3 : bin_id         floor(t_rel_sec / 0.2)，目标 5Hz bucket
mask : BoolTensor[B, T_k]     True = valid event

Output
------
tokens : [B, T_v, d_model]
pooled : [B, d_model]
mask   : [B, T_v]  固定全 True —— 事件→5Hz bucket 聚合后每个视频帧槽位都"存在"
                   （空 bucket 的 token 是零向量，表示该帧内无事件）

架构
----
type_embed (Embedding[vocab, d]) + time_embed (sinusoidal) + dt_embed (Linear)
→ LayerNorm → Dropout
→ TransformerEncoder (num_layers, nhead) with key_padding_mask=~mask
→ scatter-mean pool by bin_id → [B, T_v, d]
"""

from __future__ import annotations

import math
from typing import Optional

import torch
from torch import nn

from src.core.registry import get_encoder_registry
from src.core.types import BaseEncoder, EncoderOut
from src.models.components.fusion_utils import cfg_get, masked_mean_pool


def _sinusoidal_time_embed(t: torch.Tensor, d_model: int) -> torch.Tensor:
    """连续时间正弦位置编码。t: [B, T_k] (seconds) → [B, T_k, d_model]。"""
    device = t.device
    half = d_model // 2
    div_term = torch.exp(
        torch.arange(0, half, device=device, dtype=torch.float32)
        * -(math.log(10000.0) / max(half, 1))
    )  # [half]
    phase = t.unsqueeze(-1).float() * div_term  # [B, T_k, half]
    pe = torch.zeros(*t.shape, d_model, device=device, dtype=torch.float32)
    pe[..., 0::2] = torch.sin(phase)
    pe[..., 1::2] = torch.cos(phase)
    return pe


@get_encoder_registry("km_event").register("event_token")
class KMEventTokenEncoder(BaseEncoder):
    def __init__(self, cfg):
        super().__init__()
        d_model = cfg_get(cfg, "d_model", 512)
        vocab_size = cfg_get(cfg, "vocab_size", None)
        if vocab_size is None:
            raise ValueError("km.event_token encoder requires cfg.vocab_size")
        num_layers = cfg_get(cfg, "num_layers", 2)
        nhead = cfg_get(cfg, "nhead", 4)
        dim_ff = cfg_get(cfg, "dim_feedforward", 2 * d_model)
        dropout = cfg_get(cfg, "dropout", 0.1)
        target_T_v = cfg_get(cfg, "target_T_v", None)
        if target_T_v is None:
            raise ValueError(
                "km.event_token encoder requires cfg.target_T_v (= seq_len_video_frames)"
            )

        self.d_model = d_model
        self.target_T_v = int(target_T_v)
        self.pad_idx = 0  # matches extract_km_event.py PAD_TOKEN id

        self.type_embed = nn.Embedding(vocab_size, d_model, padding_idx=self.pad_idx)
        self.dt_embed = nn.Linear(1, d_model)
        self.ln = nn.LayerNorm(d_model)
        self.drop = nn.Dropout(dropout)

        enc_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_ff,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(enc_layer, num_layers=num_layers)

    def forward(
        self, x: torch.Tensor, mask: Optional[torch.Tensor] = None
    ) -> EncoderOut:
        B, T_k, _ = x.shape
        type_id = x[..., 0].long()
        t_rel = x[..., 1]          # seconds
        dt = x[..., 2]             # seconds
        bin_id = x[..., 3].long()  # [B, T_k] in [0, T_v-1]

        if mask is None:
            mask = torch.ones(B, T_k, dtype=torch.bool, device=x.device)

        # 防御：全 pad 行（窗口内 0 个事件）会让 src_key_padding_mask 全 True，
        # MHA softmax 退化为 NaN。强制把 pos 0 标 valid（type=PAD, embed=0），
        # 让 transformer 至少有一个 token 可以注意，对 pooled 输出几乎无影响。
        all_pad_rows = ~mask.any(dim=1)               # [B]
        if all_pad_rows.any():
            mask = mask.clone()
            mask[all_pad_rows, 0] = True

        # masked 位置 type_id / bin_id 可能是任意值，统一钳制
        type_id = type_id.masked_fill(~mask, self.pad_idx)
        bin_id = bin_id.clamp(min=0, max=self.target_T_v - 1)

        te = self.type_embed(type_id)                                     # [B, T_k, D]
        time = _sinusoidal_time_embed(t_rel, self.d_model)
        de = self.dt_embed(torch.log1p(dt.clamp(min=0.0)).unsqueeze(-1))  # [B, T_k, D]
        h = self.drop(self.ln(te + time + de))

        # TransformerEncoder 的 src_key_padding_mask: True = pad（忽略）
        tokens_k = self.transformer(h, src_key_padding_mask=~mask)        # [B, T_k, D]

        # 把 masked 位置先清零，再按 bin_id scatter_add
        mask_f = mask.float().unsqueeze(-1)                                # [B, T_k, 1]
        tokens_k_m = tokens_k * mask_f

        T_v = self.target_T_v
        idx_exp = bin_id.unsqueeze(-1).expand(-1, -1, self.d_model)        # [B, T_k, D]
        tokens_v_sum = torch.zeros(B, T_v, self.d_model, device=x.device, dtype=tokens_k.dtype)
        tokens_v_sum.scatter_add_(1, idx_exp, tokens_k_m)

        counts = torch.zeros(B, T_v, device=x.device, dtype=tokens_k.dtype)
        counts.scatter_add_(1, bin_id, mask.float())
        tokens_v = tokens_v_sum / counts.clamp(min=1.0).unsqueeze(-1)     # 空 bin → 0

        mask_v = torch.ones(B, T_v, dtype=torch.bool, device=x.device)
        pooled = masked_mean_pool(tokens_v, mask_v)
        return EncoderOut(tokens=tokens_v, pooled=pooled, mask=mask_v)
