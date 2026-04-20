"""
Task-aware multitask sequence classification head (Direction F).

State branch: direct MLP on tokens (same as multitask_seq).
Trend branch: temporal diff features + MLP — captures arousal change direction
              without depending on absolute arousal level.

差分特征是相对量，天然对 arousal 基线漂移鲁棒（缓解 within-subject F1_State 崩塌）。
"""

from __future__ import annotations

from typing import Dict, List

import torch
import torch.nn.functional as F
from torch import nn

from src.core.registry import HEADS
from src.core.types import BaseHead, FusionOut


@HEADS.register("task_aware_multitask_seq")
class TaskAwareMultiTaskSeqHead(BaseHead):
    """
    Token-wise multitask head with task-specific input features.

    Tasks listed in ``diff_tasks`` (default: ["trend"]) receive
    ``concat(tokens, tokens[t] - tokens[t-k])`` as input — a temporal
    difference that encodes *how* the representation is changing rather
    than *what* the absolute state is.  All other tasks receive raw tokens.

    Input:
      h["tokens"]: [B, T, D]
    Output:
      {"state": [B, T, C], "trend": [B, T, C], ...}

    Extra config keys (vs multitask_seq):
      diff_k      : int   — diff step in frames (default 5 = 1s @ 5Hz)
      diff_tasks  : list  — which tasks use diff features (default ["trend"])
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
        self.diff_k = int(_g("diff_k", 5))
        self.diff_tasks = set(_g("diff_tasks", ["trend"]))

        self.task_names = task_names
        self.heads = nn.ModuleDict()
        self.diff_projs = nn.ModuleDict()

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

            if task in self.diff_tasks:
                self.diff_projs[task] = nn.Sequential(
                    nn.Linear(d_model * 2, d_model),
                    nn.LayerNorm(d_model),
                    nn.GELU(),
                )

    def forward(self, h: FusionOut) -> Dict[str, torch.Tensor]:
        tokens = h.get("tokens", None)
        if tokens is None:
            raise ValueError("task_aware_multitask_seq head requires h['tokens']")

        results: Dict[str, torch.Tensor] = {}

        diff = None  # lazy compute, shared across diff_tasks

        for task in self.task_names:
            if task in self.diff_tasks:
                if diff is None:
                    T = tokens.shape[1]
                    if T > self.diff_k:
                        # diff[t] = tokens[t] - tokens[t-k], zero-padded for t < k
                        shifted = F.pad(tokens[:, :-self.diff_k, :], (0, 0, self.diff_k, 0))
                    else:
                        # T <= diff_k: no valid diff, fall back to zeros
                        shifted = torch.zeros_like(tokens)
                    diff = tokens - shifted  # [B, T, D]

                proj_input = torch.cat([tokens, diff], dim=-1)  # [B, T, 2D]
                task_input = self.diff_projs[task](proj_input)  # [B, T, D]
            else:
                task_input = tokens

            results[task] = self.heads[task](task_input)

        return results
