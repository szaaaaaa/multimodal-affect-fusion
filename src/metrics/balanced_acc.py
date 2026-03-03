"""
Balanced accuracy metric for classification tasks.
"""

from __future__ import annotations

import torch
from sklearn.metrics import balanced_accuracy_score

from src.core.registry import METRICS


@METRICS.register("balanced_acc")
class BalancedAccMetric:
    """Compute balanced accuracy score."""

    def __init__(self, cfg=None):
        pass

    def __call__(self, pred: torch.Tensor, target: torch.Tensor) -> float:
        if pred.ndim == 2 and pred.shape[-1] > 1:
            pred_labels = pred.argmax(dim=-1).numpy()
        else:
            pred_labels = pred.squeeze().long().numpy()
        target_labels = target.squeeze().long().numpy()
        return float(balanced_accuracy_score(target_labels, pred_labels))
