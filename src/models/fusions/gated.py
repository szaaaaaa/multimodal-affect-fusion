"""
Gated Fusion — modality-adaptive gating at each timestep.

门控融合 — 每个时间步自适应决定各模态权重。

Architecture:
    For each timestep:
        gate_m = sigmoid(W_m @ concat(all_modality_tokens))
        fused  = sum(gate_m * tokens_m)

Motivation:
    LFT concatenates tokens and relies on self-attention to learn cross-modal
    interactions, but shorter modalities can be "drowned out" by longer ones.
    Gated fusion explicitly computes per-modality importance weights at each
    timestep, allowing the model to upweight combat-relevant modalities
    (e.g., telemetry damage events) during high-arousal moments.

References:
    - MAG (Rahman et al., 2020): Multimodal Adaptation Gate
    - MAG+ extension: Multi-layer gating with modality reinforcement
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
from src.models.components.fusion_utils import (
    cfg_get,
    masked_mean_pool,
    pool_tokens,
)


@FUSIONS.register("gated")
class GatedFusion(BaseFusion):
    """
    Modality-adaptive gated fusion.

    Aligns all modalities to the shortest temporal length, then computes
    per-modality sigmoid gates conditioned on the concatenation of all
    modality tokens. Fused tokens are the gate-weighted sum.

    Optionally followed by a lightweight Transformer refinement layer.

    Parameters (via cfg dict)
    -------------------------
    d_model : int (default 512)
    dropout : float (default 0.1)
    gate_hidden_dim : int (default 256) — hidden dim in gate network
    refine_layers : int (default 1) — Transformer layers after gating (0 = none)
    refine_nhead : int (default 8)
    refine_dim_feedforward : int (default 512)
    pos_encoding_type : str (default "sinusoidal")
    max_seq_len : int (default 1000)
    pooling : str (default "mean") — "mean", "max"
    """

    def __init__(self, cfg=None):
        super().__init__()
        cfg = cfg or {}
        self.d_model = cfg_get(cfg, "d_model", 512)
        dropout = cfg_get(cfg, "dropout", 0.1)
        self.gate_hidden_dim = cfg_get(cfg, "gate_hidden_dim", 256)
        refine_layers = cfg_get(cfg, "refine_layers", 1)
        refine_nhead = cfg_get(cfg, "refine_nhead", 8)
        refine_dim_ff = cfg_get(cfg, "refine_dim_feedforward", 512)
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

        # Gate networks are lazy-initialized on first forward
        # because modality count is unknown at __init__ time.
        self._gate_nets: Optional[nn.ModuleDict] = None
        self._num_modalities: Optional[int] = None
        self._modality_names: Optional[list] = None

        # Layer norm after gated sum
        self.norm = nn.LayerNorm(self.d_model)
        self.dropout = nn.Dropout(dropout)

        # Optional Transformer refinement after gating
        self.refine_layers = refine_layers
        if refine_layers > 0:
            encoder_layer = nn.TransformerEncoderLayer(
                d_model=self.d_model,
                nhead=refine_nhead,
                dim_feedforward=refine_dim_ff,
                dropout=dropout,
                batch_first=True,
                norm_first=True,
            )
            self.transformer = nn.TransformerEncoder(
                encoder_layer,
                num_layers=refine_layers,
                enable_nested_tensor=False,
            )

    def _init_gates(self, modality_names: list, device: torch.device) -> None:
        """Lazy-initialize gate networks for the given modality set."""
        M = len(modality_names)
        gate_input_dim = M * self.d_model

        gate_nets = {}
        for mod in modality_names:
            gate_nets[mod] = nn.Sequential(
                nn.Linear(gate_input_dim, self.gate_hidden_dim),
                nn.GELU(),
                nn.Linear(self.gate_hidden_dim, 1),
                nn.Sigmoid(),
            )

        self._gate_nets = nn.ModuleDict(gate_nets).to(device)
        self._num_modalities = M
        self._modality_names = modality_names

    def forward(
        self,
        z_dict: Dict[str, EncoderOut],
        mask_dict: Dict[str, torch.Tensor],
    ) -> FusionOut:

        # ── Single modality: pass through ──
        if len(z_dict) == 1:
            z = next(iter(z_dict.values()))
            tokens = self.pos_encoding(z["tokens"])
            mask = z["mask"]
            if self.refine_layers > 0:
                tokens = self.transformer(tokens, src_key_padding_mask=~mask)
            pooled = pool_tokens(tokens, mask, self.pooling_type)
            return FusionOut(tokens=tokens, pooled=pooled)

        modality_names = sorted(z_dict.keys())

        # ── Lazy init gates ──
        if self._gate_nets is None or self._modality_names != modality_names:
            device = z_dict[modality_names[0]]["tokens"].device
            self._init_gates(modality_names, device)

        # ── Align to shortest temporal length ──
        min_t = min(z_dict[mod]["tokens"].shape[1] for mod in modality_names)

        trimmed_tokens = {}
        trimmed_masks = []
        for mod in modality_names:
            tok = z_dict[mod]["tokens"][:, :min_t, :]       # [B, T, D]
            tok = self.pos_encoding(tok)
            trimmed_tokens[mod] = tok
            trimmed_masks.append(mask_dict[mod][:, :min_t])  # [B, T]

        # Union mask: valid if any modality is valid
        fused_mask = torch.stack(trimmed_masks, dim=0).any(dim=0)  # [B, T]

        # ── Compute gates ──
        # Concatenate all modality tokens as gate input: [B, T, M*D]
        gate_input = torch.cat(
            [trimmed_tokens[mod] for mod in modality_names], dim=-1
        )

        # Gate-weighted sum: [B, T, D]
        fused = torch.zeros_like(trimmed_tokens[modality_names[0]])
        for mod in modality_names:
            gate = self._gate_nets[mod](gate_input)   # [B, T, 1]
            fused = fused + gate * trimmed_tokens[mod]

        fused = self.norm(self.dropout(fused))

        # ── Optional Transformer refinement ──
        if self.refine_layers > 0:
            padding_mask = ~fused_mask
            fused = self.transformer(fused, src_key_padding_mask=padding_mask)

        # ── Pool ──
        pooled = pool_tokens(fused, fused_mask, self.pooling_type)

        return FusionOut(tokens=fused, pooled=pooled)
