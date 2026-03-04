"""
Multitask sequence head for mixed task types (regression + classification).
"""

from __future__ import annotations

from typing import Dict

import torch
from torch import nn

from src.core.registry import HEADS
from src.core.types import BaseHead, FusionOut


def _cfg_get(cfg, key, default):
    if isinstance(cfg, dict):
        return cfg.get(key, default)
    return getattr(cfg, key, default)


@HEADS.register("multitask_mixed_seq")
class MultiTaskMixedSeqHead(BaseHead):
    """
    Token-wise multitask head with per-task type.

    - regression task  -> [B, T, out_dim] (default out_dim=1)
    - classification task -> [B, T, num_classes]
    """

    def __init__(self, cfg):
        super().__init__()
        d_model = _cfg_get(cfg, "d_model", 256)
        task_heads = _cfg_get(cfg, "task_heads", None)

        if not isinstance(task_heads, dict) or not task_heads:
            raise ValueError("multitask_mixed_seq requires non-empty `task_heads` config")

        self.task_defs: Dict[str, Dict] = {}
        self.heads = nn.ModuleDict()

        for task_name, task_cfg in task_heads.items():
            if not isinstance(task_cfg, dict):
                raise ValueError(f"task_heads.{task_name} must be a dict")

            task_type = str(task_cfg.get("type", "classification")).lower()
            hidden_dim = int(task_cfg.get("hidden_dim", 128))
            dropout = float(task_cfg.get("dropout", 0.1))

            if task_type == "regression":
                out_dim = int(task_cfg.get("out_dim", 1))
                out_features = out_dim
            elif task_type == "classification":
                num_classes = int(task_cfg.get("num_classes", 3))
                out_features = num_classes
            else:
                raise ValueError(
                    f"Unsupported task type for {task_name}: {task_type}. "
                    "Use 'regression' or 'classification'."
                )

            self.task_defs[task_name] = {"type": task_type}
            self.heads[task_name] = nn.Sequential(
                nn.Linear(d_model, hidden_dim),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim, out_features),
            )

    def forward(self, h: FusionOut) -> Dict[str, torch.Tensor]:
        tokens = h.get("tokens", None)
        if tokens is None:
            raise ValueError("multitask_mixed_seq head requires h['tokens']")
        return {task: head(tokens) for task, head in self.heads.items()}
