"""MSE metric — mean squared error."""

from __future__ import annotations

import torch

from src.core.registry import METRICS


@METRICS.register("mse")
class MSEMetric:
    """Compute MSE between predictions and targets."""

    def __init__(self, cfg=None):
        pass

    def __call__(self, pred: torch.Tensor, target: torch.Tensor) -> float:
        return torch.mean((pred.float() - target.float()) ** 2).item()
