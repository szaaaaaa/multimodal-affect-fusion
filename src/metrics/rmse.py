"""
RMSE metric — root mean squared error.

RMSE 指标。
"""

from __future__ import annotations

import torch

from src.core.registry import METRICS


@METRICS.register("rmse")
class RMSEMetric:
    """Compute RMSE between predictions and targets."""

    def __init__(self, cfg=None):
        pass

    def __call__(self, pred: torch.Tensor, target: torch.Tensor) -> float:
        return torch.sqrt(torch.mean((pred.float() - target.float()) ** 2)).item()
