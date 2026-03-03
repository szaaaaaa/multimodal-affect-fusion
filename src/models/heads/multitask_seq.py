"""
Multitask sequence classification head.
"""

from __future__ import annotations

from typing import Dict, List

import torch
from torch import nn

from src.core.registry import HEADS
from src.core.types import BaseHead, FusionOut


@HEADS.register("multitask_seq")
class MultiTaskSeqHead(BaseHead):
    """
    Token-wise multitask MLP head for sequence classification.

    Input:
      h["tokens"]: [B, T, D]
    Output:
      {
        "state": [B, T, C_state],
        "trend": [B, T, C_trend],
      }
    """

    def __init__(self, cfg):
        super().__init__()
        if isinstance(cfg, dict):
            _g = cfg.get
        else:
            _g = lambda k, d=None: getattr(cfg, k, d)

        d_model = _g("d_model", 256)
        hidden_dim = _g("hidden_dim", 128)
        dropout = _g("dropout", 0.1)
        task_names: List[str] = list(_g("task_names", ["state", "trend"]))
        default_num_classes = int(_g("num_classes", 3))
        num_classes_by_task = _g("num_classes_by_task", None)

        self.task_names = task_names
        self.heads = nn.ModuleDict()
        for task in task_names:
            if isinstance(num_classes_by_task, dict) and task in num_classes_by_task:
                num_classes = int(num_classes_by_task[task])
            else:
                num_classes = default_num_classes

            self.heads[task] = nn.Sequential(
                nn.Linear(d_model, hidden_dim),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim, num_classes),
            )

    def forward(self, h: FusionOut) -> Dict[str, torch.Tensor]:
        tokens = h.get("tokens", None)
        if tokens is None:
            raise ValueError("multitask_seq head requires h['tokens']")
        return {task: head(tokens) for task, head in self.heads.items()}
