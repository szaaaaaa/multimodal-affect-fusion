"""
Multitask masked sequence loss for mixed task types.

Supports per-task regression (MSE) and classification (CE).
"""

from __future__ import annotations

from typing import Dict

import torch
import torch.nn.functional as F
from torch import nn

from src.core.registry import LOSSES


def _cfg_get(cfg, key, default):
    if isinstance(cfg, dict):
        return cfg.get(key, default)
    return getattr(cfg, key, default)


@LOSSES.register("multitask_mixed_seq_loss")
class MultiTaskMixedSequenceLoss(nn.Module):
    """
    For each task:
      - regression: masked MSE
      - classification: masked CE
    If a task has no valid timestep in batch, that task loss is zero.

    Total:
      sum(task_weight[task] * task_loss[task])
    """

    def __init__(self, cfg=None):
        super().__init__()
        if cfg is None:
            cfg = {}

        self.task_weights: Dict[str, float] = {
            k: float(v)
            for k, v in dict(_cfg_get(cfg, "task_weights", {})).items()
        }
        self.task_types: Dict[str, str] = {
            str(k): str(v).lower()
            for k, v in dict(_cfg_get(cfg, "task_types", {})).items()
        }

    @staticmethod
    def _masked_mse(
        pred: torch.Tensor,
        target: torch.Tensor,
        mask: torch.Tensor | None,
    ) -> torch.Tensor:
        if pred.ndim == 2:
            pred = pred.unsqueeze(-1)
        if target.ndim == 2:
            target = target.unsqueeze(-1)

        if mask is None:
            return torch.mean((pred - target) ** 2)

        if mask.ndim == 2:
            mask = mask.unsqueeze(-1)
        mask = mask.bool()

        if not torch.any(mask):
            return pred.sum() * 0.0

        loss = (pred - target) ** 2
        mask_f = mask.float()
        denom = mask_f.sum().clamp(min=1.0)
        return (loss * mask_f).sum() / denom

    @staticmethod
    def _masked_ce(
        logits: torch.Tensor,
        target: torch.Tensor,
        mask: torch.Tensor | None,
    ) -> torch.Tensor:
        # logits: [B, T, C], target: [B, T]
        b, t, c = logits.shape

        if mask is None:
            return F.cross_entropy(logits.reshape(-1, c), target.reshape(-1))

        valid = mask.reshape(-1).bool()
        if not torch.any(valid):
            return logits.sum() * 0.0

        logits_flat = logits.reshape(-1, c)[valid]
        target_flat = target.reshape(-1)[valid]
        return F.cross_entropy(logits_flat, target_flat)

    def _resolve_task_type(
        self,
        task: str,
        pred: torch.Tensor,
        target: torch.Tensor,
    ) -> str:
        task_type = self.task_types.get(task, "")
        if task_type in {"regression", "classification"}:
            return task_type

        # Fallback inference for backward compatibility.
        if pred.ndim == 3 and pred.shape[-1] > 1:
            return "classification"
        if target.dtype in (torch.int64, torch.int32, torch.int16, torch.int8):
            if pred.ndim == 3 and pred.shape[-1] > 1:
                return "classification"
        return "regression"

    def forward(
        self,
        pred: Dict[str, torch.Tensor],
        target: Dict[str, torch.Tensor],
        mask: Dict[str, torch.Tensor] | None = None,
    ) -> torch.Tensor:
        total = None

        for task, pred_task in pred.items():
            if task not in target:
                raise KeyError(f"Missing target for task '{task}'")
            target_task = target[task]
            mask_task = None if mask is None else mask.get(task)

            task_type = self._resolve_task_type(task, pred_task, target_task)
            if task_type == "classification":
                task_loss = self._masked_ce(pred_task, target_task, mask_task)
            else:
                task_loss = self._masked_mse(pred_task, target_task, mask_task)

            weighted = self.task_weights.get(task, 1.0) * task_loss
            total = weighted if total is None else total + weighted

        if total is None:
            return torch.tensor(0.0, dtype=torch.float32)
        return total
