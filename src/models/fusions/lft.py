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
        self.temporal_merge = _g("temporal_merge", "none")
        self.d_model = d_model

        if self.temporal_merge not in {"none", "mean", "linear"}:
            raise ValueError(
                f"Unsupported temporal_merge='{self.temporal_merge}'. "
                "Choose from: none, mean, linear."
            )

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
        self._temporal_merge_layers = nn.ModuleDict()

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
    ) -> tuple[torch.Tensor, torch.Tensor, list[str], list[int]]:
        modality_names = sorted(z_dict.keys())
        mod_emb = self._get_modality_emb(modality_names)

        all_tokens = []
        all_masks = []
        modality_lengths = []
        for mod in modality_names:
            z = z_dict[mod]
            tok = z["tokens"]                            # [B, T_m, D]
            tok = self.pos_encoding(tok)                 # + positional encoding
            tok = mod_emb(tok, modality=mod)              # + modality embedding
            all_tokens.append(tok)
            all_masks.append(mask_dict[mod])
            modality_lengths.append(int(tok.shape[1]))

        tokens = torch.cat(all_tokens, dim=1)            # [B, T_total, D]
        masks = torch.cat(all_masks, dim=1)              # [B, T_total]
        return tokens, masks, modality_names, modality_lengths

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

    def _merge_temporal_tokens(
        self,
        fused: torch.Tensor,
        masks: torch.Tensor,
        modality_lengths: list[int],
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if self.temporal_merge == "none" or len(modality_lengths) <= 1:
            return fused, masks

        fused_parts = torch.split(fused, modality_lengths, dim=1)
        mask_parts = torch.split(masks, modality_lengths, dim=1)
        min_t = min(part.shape[1] for part in fused_parts)

        fused_trim = [part[:, :min_t, :] for part in fused_parts]
        masks_trim = [part[:, :min_t] for part in mask_parts]
        merged_mask = torch.stack(masks_trim, dim=0).any(dim=0)

        if self.temporal_merge == "mean":
            merged_tokens = torch.stack(fused_trim, dim=0).mean(dim=0)
            return merged_tokens, merged_mask

        # A2: learnable per-timestep modality fusion, [B, T, M*D] -> [B, T, D]
        num_modalities = len(fused_trim)
        key = str(num_modalities)
        if key not in self._temporal_merge_layers:
            layer = nn.Linear(self.d_model * num_modalities, self.d_model)
            layer = layer.to(device=fused.device, dtype=fused.dtype)
            self._temporal_merge_layers[key] = layer

        merge_layer = self._temporal_merge_layers[key]
        merged_tokens = merge_layer(torch.cat(fused_trim, dim=-1))
        return merged_tokens, merged_mask

    def forward(
        self,
        z_dict: Dict[str, EncoderOut],
        mask_dict: Dict[str, torch.Tensor],
    ) -> FusionOut:
        tokens, masks, _, modality_lengths = self._prepare_tokens_and_masks(z_dict, mask_dict)

        # Optional CLS token
        tokens, masks = self._add_cls_if_needed(tokens, masks)

        # Transformer expects True = ignore for src_key_padding_mask
        padding_mask = ~masks

        fused = self.transformer(tokens, src_key_padding_mask=padding_mask)

        out_tokens = fused
        out_masks = masks
        if self.temporal_merge != "none" and len(modality_lengths) > 1:
            if self.pooling_type == "cls":
                raise ValueError("temporal_merge is not compatible with pooling='cls'.")
            out_tokens, out_masks = self._merge_temporal_tokens(fused, masks, modality_lengths)

        # Pooling
        if self.pooling_type == "cls":
            pooled = fused[:, 0, :]
        elif self.pooling_type == "max":
            padding_mask_out = ~out_masks
            fused_masked = out_tokens.masked_fill(padding_mask_out.unsqueeze(-1), float("-inf"))
            pooled = fused_masked.max(dim=1)[0]
        else:
            # mean pooling
            mask_f = out_masks.float().unsqueeze(-1)
            pooled = (out_tokens * mask_f).sum(dim=1) / mask_f.sum(dim=1).clamp(min=1.0)

        return FusionOut(tokens=out_tokens, pooled=pooled)

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
        tokens, masks, _, _ = self._prepare_tokens_and_masks(z_dict, mask_dict)
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
