"""
Cross-modal attention fusion inspired by MulT-style directional attention.
"""

from __future__ import annotations

import re
from typing import Dict, Optional

import torch
from torch import nn

from src.core.registry import FUSIONS
from src.core.types import BaseFusion, EncoderOut, FusionOut
from src.models.components import (
    LearnablePositionalEncoding,
    ModalityEmbedding,
    SinusoidalPositionalEncoding,
)


class CrossModalAttentionLayer(nn.Module):
    """
    Pre-LN cross-attention block with residual FFN.
    """

    def __init__(
        self,
        d_model: int,
        nhead: int,
        dim_feedforward: int,
        dropout: float,
    ):
        super().__init__()
        self.target_norm = nn.LayerNorm(d_model)
        self.source_norm = nn.LayerNorm(d_model)
        self.cross_attn = nn.MultiheadAttention(
            embed_dim=d_model,
            num_heads=nhead,
            dropout=dropout,
            batch_first=True,
        )
        self.dropout1 = nn.Dropout(dropout)

        self.ffn_norm = nn.LayerNorm(d_model)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, dim_feedforward),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim_feedforward, d_model),
        )
        self.dropout2 = nn.Dropout(dropout)

    def forward(
        self,
        target: torch.Tensor,
        source: torch.Tensor,
        source_key_padding_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        q = self.target_norm(target)
        src = self.source_norm(source)
        attn_out, _ = self.cross_attn(
            q,
            src,
            src,
            key_padding_mask=source_key_padding_mask,
            need_weights=False,
        )
        h = target + self.dropout1(attn_out)
        h = h + self.dropout2(self.ffn(self.ffn_norm(h)))
        return h


class CrossModalTransformer(nn.Module):
    """
    Stack of cross-attention layers.
    """

    def __init__(
        self,
        d_model: int,
        nhead: int,
        num_layers: int,
        dim_feedforward: int,
        dropout: float,
    ):
        super().__init__()
        self.layers = nn.ModuleList(
            [
                CrossModalAttentionLayer(
                    d_model=d_model,
                    nhead=nhead,
                    dim_feedforward=dim_feedforward,
                    dropout=dropout,
                )
                for _ in range(num_layers)
            ]
        )

    def forward(
        self,
        target: torch.Tensor,
        source: torch.Tensor,
        source_key_padding_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        h = target
        for layer in self.layers:
            h = layer(
                target=h,
                source=source,
                source_key_padding_mask=source_key_padding_mask,
            )
        return h


@FUSIONS.register("cma")
class CMAFusion(BaseFusion):
    """
    Anchor-based cross-modal attention fusion.

    Non-anchor modalities are enhanced by attending to the anchor modality,
    then all modality tokens are concatenated and optionally refined by a
    shared self-attention Transformer.
    """

    def __init__(self, cfg):
        super().__init__()
        if isinstance(cfg, dict):
            _g = cfg.get
        else:
            _g = lambda k, d=None: getattr(cfg, k, d)

        d_model = _g("d_model", 256)
        nhead = _g("nhead", 8)
        cm_layers = _g("cm_layers", 4)
        sa_layers = _g("sa_layers", 2)
        dim_feedforward = _g("dim_feedforward", 1024)
        dropout = _g("dropout", 0.1)
        max_seq_len = _g("max_seq_len", 1000)
        pos_type = _g("pos_encoding_type", "sinusoidal")
        self.pooling_type = _g("pooling", "mean")
        self.anchor_modality = _g("anchor_modality", "video")
        self.d_model = d_model

        if self.pooling_type not in {"mean", "max", "cls"}:
            raise ValueError(
                f"Unsupported pooling='{self.pooling_type}'. Choose from: mean, max, cls."
            )

        if pos_type == "learnable":
            self.pos_encoding = LearnablePositionalEncoding(
                d_model=d_model,
                max_len=max_seq_len,
                dropout=dropout,
            )
        else:
            self.pos_encoding = SinusoidalPositionalEncoding(
                d_model=d_model,
                max_len=max_seq_len,
                dropout=dropout,
            )

        self._modality_emb: Optional[ModalityEmbedding] = None
        self._modality_names: Optional[list[str]] = None
        self._cm_transformers = nn.ModuleDict()
        self._cm_cfg = {
            "d_model": d_model,
            "nhead": nhead,
            "num_layers": cm_layers,
            "dim_feedforward": dim_feedforward,
            "dropout": dropout,
        }

        if sa_layers > 0:
            layer = nn.TransformerEncoderLayer(
                d_model=d_model,
                nhead=nhead,
                dim_feedforward=dim_feedforward,
                dropout=dropout,
                batch_first=True,
                norm_first=True,
            )
            self.self_attn_transformer = nn.TransformerEncoder(
                layer,
                num_layers=sa_layers,
                enable_nested_tensor=False,
            )
        else:
            self.self_attn_transformer = None

        if self.pooling_type == "cls":
            self.cls_token = nn.Parameter(torch.randn(1, 1, d_model) * 0.02)

    def _infer_module_device_dtype(self) -> tuple[torch.device, torch.dtype]:
        param = next(self.parameters(), None)
        if param is not None:
            return param.device, param.dtype

        buffer = next(self.buffers(), None)
        if buffer is not None:
            return buffer.device, buffer.dtype

        return torch.device("cpu"), torch.float32

    def _get_or_create_cm(self, modality: str) -> CrossModalTransformer:
        if modality not in self._cm_transformers:
            module = CrossModalTransformer(**self._cm_cfg)
            device, dtype = self._infer_module_device_dtype()
            self._cm_transformers[modality] = module.to(device=device, dtype=dtype)
        return self._cm_transformers[modality]

    def _get_modality_emb(self, modality_names: list[str]) -> ModalityEmbedding:
        if self._modality_emb is None or self._modality_names != modality_names:
            self._modality_names = modality_names
            module = ModalityEmbedding(
                d_model=self.d_model,
                num_modalities=len(modality_names),
                modality_names=modality_names,
            )
            device, _ = self._infer_module_device_dtype()
            self._modality_emb = module.to(device=device)
        return self._modality_emb

    def prepare_lazy_layers_from_state_dict(self, state_dict: Dict[str, torch.Tensor]) -> None:
        for key in state_dict:
            match = re.match(r"(?:^|.*\.)_cm_transformers\.([^.]+)\.", key)
            if match:
                self._get_or_create_cm(match.group(1))

    def _encode_positions(
        self,
        z_dict: Dict[str, EncoderOut],
    ) -> Dict[str, torch.Tensor]:
        tokens_pe: Dict[str, torch.Tensor] = {}
        for mod, z in z_dict.items():
            tokens_pe[mod] = self.pos_encoding(z["tokens"])
        return tokens_pe

    def _add_cls_if_needed(
        self,
        tokens: torch.Tensor,
        masks: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if self.pooling_type != "cls":
            return tokens, masks
        batch_size = tokens.size(0)
        cls_token = self.cls_token.expand(batch_size, -1, -1)
        tokens = torch.cat([cls_token, tokens], dim=1)
        cls_mask = torch.ones(batch_size, 1, dtype=torch.bool, device=tokens.device)
        masks = torch.cat([cls_mask, masks], dim=1)
        return tokens, masks

    def forward(
        self,
        z_dict: Dict[str, EncoderOut],
        mask_dict: Dict[str, torch.Tensor],
    ) -> FusionOut:
        if not z_dict:
            raise ValueError("cma fusion received empty z_dict")

        modality_names = sorted(z_dict.keys())
        tokens_pe = self._encode_positions(z_dict)
        enhanced: Dict[str, torch.Tensor] = {}

        has_anchor = self.anchor_modality in z_dict
        if has_anchor and len(modality_names) > 1:
            anchor_tokens = tokens_pe[self.anchor_modality]
            anchor_mask_ignore = ~mask_dict[self.anchor_modality]
            for mod in modality_names:
                if mod == self.anchor_modality:
                    enhanced[mod] = tokens_pe[mod]
                    continue
                cm = self._get_or_create_cm(mod)
                enhanced[mod] = cm(
                    target=tokens_pe[mod],
                    source=anchor_tokens,
                    source_key_padding_mask=anchor_mask_ignore,
                )
        else:
            enhanced = tokens_pe

        mod_emb = self._get_modality_emb(modality_names)
        all_tokens = []
        all_masks = []
        for mod in modality_names:
            all_tokens.append(mod_emb(enhanced[mod], modality=mod))
            all_masks.append(mask_dict[mod])

        tokens = torch.cat(all_tokens, dim=1)
        masks = torch.cat(all_masks, dim=1)
        tokens, masks = self._add_cls_if_needed(tokens, masks)

        if self.self_attn_transformer is not None:
            fused = self.self_attn_transformer(tokens, src_key_padding_mask=~masks)
        else:
            fused = tokens

        if self.pooling_type == "cls":
            pooled = fused[:, 0, :]
        elif self.pooling_type == "max":
            fused_masked = fused.masked_fill(~masks.unsqueeze(-1), float("-inf"))
            pooled = fused_masked.max(dim=1)[0]
        else:
            mask_f = masks.float().unsqueeze(-1)
            pooled = (fused * mask_f).sum(dim=1) / mask_f.sum(dim=1).clamp(min=1.0)

        return FusionOut(tokens=fused, pooled=pooled)
