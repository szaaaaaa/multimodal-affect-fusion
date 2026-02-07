"""
Late Fusion Transformer (LFT) — extracted from the original monolithic model.

晚期融合 Transformer，从原始整体模型中拆分出来的融合层。

Architecture:
    For each modality:  tokens → pos_encoding → modality_embedding
    Concatenate all tokens → Transformer Encoder × N → Pooling
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


@FUSIONS.register("lft")
class LFTFusion(BaseFusion):
    """
    Late Fusion Transformer.

    Adds positional encoding + modality embedding to each modality's tokens,
    concatenates them, and passes through a shared Transformer encoder.

    Parameters (via cfg dict)
    -------------------------
    d_model : int (default 256)
    nhead : int (default 8)
    num_layers : int (default 4)
    dim_feedforward : int (default 1024)
    dropout : float (default 0.1)
    max_seq_len : int (default 1000)
    pos_encoding_type : str (default "sinusoidal")
    pooling : str (default "mean")  — "mean", "max", or "cls"
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

        # Positional encoding (shared across modalities)
        if pos_type == "learnable":
            self.pos_encoding = LearnablePositionalEncoding(
                d_model=d_model, max_len=max_seq_len, dropout=dropout,
            )
        else:
            self.pos_encoding = SinusoidalPositionalEncoding(
                d_model=d_model, max_len=max_seq_len, dropout=dropout,
            )

        # Modality embedding — dynamically sized, will be (re-)created as needed
        self._modality_emb: Optional[ModalityEmbedding] = None
        self._modality_names: Optional[list] = None
        self._d_model = d_model

        # Transformer encoder
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

        # CLS token (only used when pooling == "cls")
        if self.pooling_type == "cls":
            self.cls_token = nn.Parameter(torch.randn(1, 1, d_model) * 0.02)

    def _get_modality_emb(self, modality_names: list) -> ModalityEmbedding:
        """Lazy-init modality embedding for the given set of modalities."""
        if self._modality_emb is None or self._modality_names != modality_names:
            self._modality_names = modality_names
            self._modality_emb = ModalityEmbedding(
                d_model=self._d_model,
                num_modalities=len(modality_names),
                modality_names=modality_names,
            ).to(next(self.parameters()).device)
        return self._modality_emb

    def _prepare_tokens_and_masks(
        self,
        z_dict: Dict[str, EncoderOut],
        mask_dict: Dict[str, torch.Tensor],
    ) -> tuple[torch.Tensor, torch.Tensor]:
        modality_names = sorted(z_dict.keys())
        mod_emb = self._get_modality_emb(modality_names)

        all_tokens = []
        all_masks = []
        for mod in modality_names:
            z = z_dict[mod]
            tok = z["tokens"]                            # [B, T_m, D]
            tok = self.pos_encoding(tok)                 # + positional encoding
            tok = mod_emb(tok, modality=mod)              # + modality embedding
            all_tokens.append(tok)
            all_masks.append(mask_dict[mod])

        tokens = torch.cat(all_tokens, dim=1)            # [B, T_total, D]
        masks = torch.cat(all_masks, dim=1)              # [B, T_total]
        return tokens, masks

    def _add_cls_if_needed(
        self,
        tokens: torch.Tensor,
        masks: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if self.pooling_type != "cls":
            return tokens, masks
        B = tokens.size(0)
        cls_tokens = self.cls_token.expand(B, -1, -1)
        tokens = torch.cat([cls_tokens, tokens], dim=1)
        cls_mask = torch.ones(B, 1, dtype=torch.bool, device=tokens.device)
        masks = torch.cat([cls_mask, masks], dim=1)
        return tokens, masks

    def forward(
        self,
        z_dict: Dict[str, EncoderOut],
        mask_dict: Dict[str, torch.Tensor],
    ) -> FusionOut:
        tokens, masks = self._prepare_tokens_and_masks(z_dict, mask_dict)

        # Optional CLS token
        tokens, masks = self._add_cls_if_needed(tokens, masks)

        # Transformer expects True = ignore for src_key_padding_mask
        padding_mask = ~masks

        fused = self.transformer(tokens, src_key_padding_mask=padding_mask)

        # Pooling
        if self.pooling_type == "cls":
            pooled = fused[:, 0, :]
        elif self.pooling_type == "max":
            fused_masked = fused.masked_fill(padding_mask.unsqueeze(-1), float("-inf"))
            pooled = fused_masked.max(dim=1)[0]
        else:
            # mean pooling
            mask_f = masks.float().unsqueeze(-1)
            pooled = (fused * mask_f).sum(dim=1) / mask_f.sum(dim=1).clamp(min=1.0)

        return FusionOut(tokens=fused, pooled=pooled)

    def get_attention_weights(
        self,
        z_dict: Dict[str, EncoderOut],
        mask_dict: Dict[str, torch.Tensor],
        average_attn_weights: bool = True,
    ) -> list[torch.Tensor]:
        """
        Get attention weights from each Transformer encoder layer.

        Returns a list of attention weight tensors, one per layer.
        """
        tokens, masks = self._prepare_tokens_and_masks(z_dict, mask_dict)
        tokens, masks = self._add_cls_if_needed(tokens, masks)
        padding_mask = ~masks

        x = tokens
        attn_weights_all: list[torch.Tensor] = []

        for layer in self.transformer.layers:
            if layer.norm_first:
                x_norm = layer.norm1(x)
                try:
                    attn_out, attn_weights = layer.self_attn(
                        x_norm,
                        x_norm,
                        x_norm,
                        key_padding_mask=padding_mask,
                        need_weights=True,
                        average_attn_weights=average_attn_weights,
                    )
                except TypeError:
                    attn_out, attn_weights = layer.self_attn(
                        x_norm,
                        x_norm,
                        x_norm,
                        key_padding_mask=padding_mask,
                        need_weights=True,
                    )
                x = x + layer.dropout1(attn_out)

                y = layer.norm2(x)
                y = layer.linear2(layer.dropout(layer.activation(layer.linear1(y))))
                y = layer.dropout2(y)
                x = x + y
            else:
                try:
                    attn_out, attn_weights = layer.self_attn(
                        x,
                        x,
                        x,
                        key_padding_mask=padding_mask,
                        need_weights=True,
                        average_attn_weights=average_attn_weights,
                    )
                except TypeError:
                    attn_out, attn_weights = layer.self_attn(
                        x,
                        x,
                        x,
                        key_padding_mask=padding_mask,
                        need_weights=True,
                    )
                x = layer.norm1(x + layer.dropout1(attn_out))

                y = layer.linear2(layer.dropout(layer.activation(layer.linear1(x))))
                y = layer.dropout2(y)
                x = layer.norm2(x + y)

            attn_weights_all.append(attn_weights)

        return attn_weights_all
