"""
MSE and Smooth-L1 loss wrappers (registered).

MSE 和 Smooth-L1 损失包装（已注册）。
"""

from __future__ import annotations

import torch
from torch import nn
import torch.nn.functional as F

from src.core.registry import LOSSES


@LOSSES.register("mse")
class MSELoss(nn.Module):
    def __init__(self, cfg=None):
        super().__init__()

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        return F.mse_loss(pred, target)


@LOSSES.register("smooth_l1")
class SmoothL1Loss(nn.Module):
    def __init__(self, cfg=None):
        super().__init__()

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        return F.smooth_l1_loss(pred, target)
