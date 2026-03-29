"""
Shared utility functions for fusion and encoder modules.

融合与编码器模块的共享工具函数。
"""

from __future__ import annotations

from typing import Any

import torch
from torch import nn


def cfg_get(cfg, key: str, default: Any = None) -> Any:
    """Unified config accessor supporting both dict and object-style configs."""
    if isinstance(cfg, dict):
        return cfg.get(key, default)
    return getattr(cfg, key, default)


def masked_mean_pool(tokens: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """Masked mean pooling over the time dimension.

    Parameters
    ----------
    tokens : Tensor[B, T, D]
    mask : Tensor[B, T] (bool, True = valid)

    Returns
    -------
    Tensor[B, D]
    """
    mask_f = mask.float().unsqueeze(-1)  # [B, T, 1]
    return (tokens * mask_f).sum(dim=1) / mask_f.sum(dim=1).clamp(min=1.0)


def pool_tokens(
    tokens: torch.Tensor,
    mask: torch.Tensor,
    pooling_type: str,
) -> torch.Tensor:
    """Pool token sequence into a single vector.

    Parameters
    ----------
    tokens : Tensor[B, T, D]
    mask : Tensor[B, T] (bool, True = valid)
    pooling_type : "mean" | "max" | "cls"
    """
    if pooling_type == "cls":
        return tokens[:, 0, :]
    if pooling_type == "max":
        fused_masked = tokens.masked_fill(~mask.unsqueeze(-1), float("-inf"))
        return fused_masked.max(dim=1)[0]
    return masked_mean_pool(tokens, mask)


def add_cls_token(
    tokens: torch.Tensor,
    mask: torch.Tensor,
    cls_param: nn.Parameter,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Prepend a CLS token to the sequence.

    Parameters
    ----------
    tokens : Tensor[B, T, D]
    mask : Tensor[B, T]
    cls_param : Parameter[1, 1, D]

    Returns
    -------
    (tokens_with_cls, mask_with_cls)
    """
    B = tokens.size(0)
    cls_tokens = cls_param.expand(B, -1, -1)
    tokens = torch.cat([cls_tokens, tokens], dim=1)
    cls_mask = torch.ones(B, 1, dtype=torch.bool, device=tokens.device)
    mask = torch.cat([cls_mask, mask], dim=1)
    return tokens, mask
