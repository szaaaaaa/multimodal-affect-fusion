"""
Weighted multitask masked sequence cross-entropy loss.
"""

from __future__ import annotations

from typing import Dict

import torch
import torch.nn.functional as F
from torch import nn

from src.core.registry import LOSSES


@LOSSES.register("multitask_ce_seq_masked")
class MultiTaskMaskedSequenceCELoss(nn.Module):
    """
    For each task:
      CE is computed only on mask==True timesteps.
      If a task has no valid timestep in the batch, task loss is zero.

    Total loss:
      sum(task_weight[task] * task_loss[task])
    """

    def __init__(self, cfg=None):
        super().__init__()
        if cfg is None:
            cfg = {}
        _g = cfg.get if isinstance(cfg, dict) else (lambda k, d=None: getattr(cfg, k, d))
        self.task_weights: Dict[str, float] = {
            k: float(v) for k, v in dict(_g("task_weights", {"state": 1.0, "trend": 1.0})).items()
        }

    def _task_loss(
        self,
        logits: torch.Tensor,
        target: torch.Tensor,
        mask: torch.Tensor | None,
    ) -> torch.Tensor:
        # logits: [B, T, C], target: [B, T]
        if mask is None:
            b, t, c = logits.shape
            return F.cross_entropy(logits.reshape(-1, c), target.reshape(-1))

        valid = mask.reshape(-1).bool()
        if not torch.any(valid):
            return logits.sum() * 0.0

        b, t, c = logits.shape
        logits_flat = logits.reshape(-1, c)[valid]
        target_flat = target.reshape(-1)[valid]
        return F.cross_entropy(logits_flat, target_flat)

    def forward(
        self,
        pred: Dict[str, torch.Tensor],
        target: Dict[str, torch.Tensor],
        mask: Dict[str, torch.Tensor] | None = None,
    ) -> torch.Tensor:
        total = None
        for task, logits in pred.items():
            if task not in target:
                raise KeyError(f"Missing target for task '{task}'")
            task_target = target[task]
            task_mask = None if mask is None else mask.get(task)
            task_loss = self._task_loss(logits, task_target, task_mask)
            weighted = self.task_weights.get(task, 1.0) * task_loss
            total = weighted if total is None else total + weighted

        if total is None:
            # Defensive fallback for empty pred dict.
            return torch.tensor(0.0, dtype=torch.float32)
        return total
