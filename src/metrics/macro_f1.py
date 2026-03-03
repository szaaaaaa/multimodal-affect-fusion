"""
Macro-F1 metric for classification tasks.
"""

from __future__ import annotations

import torch
from sklearn.metrics import f1_score

from src.core.registry import METRICS


@METRICS.register("macro_f1")
class MacroF1Metric:
    """Compute macro-averaged F1 score."""

    def __init__(self, cfg=None):
        pass

    def __call__(self, pred: torch.Tensor, target: torch.Tensor) -> float:
        if pred.ndim == 2 and pred.shape[-1] > 1:
            pred_labels = pred.argmax(dim=-1).numpy()
        else:
            pred_labels = pred.squeeze().long().numpy()
        target_labels = target.squeeze().long().numpy()
        return float(f1_score(target_labels, pred_labels, average="macro", zero_division=0))
