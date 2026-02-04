"""
Reproducibility utilities.

可复现性工具。
"""

from __future__ import annotations

import random

import torch

try:
    import numpy as np
    NUMPY_AVAILABLE = True
except ImportError:
    NUMPY_AVAILABLE = False


def set_seed(seed: int = 42) -> None:
    """
    Set random seed for reproducibility across all backends.

    设置随机种子以保证可复现性。
    """
    random.seed(seed)
    torch.manual_seed(seed)
    if NUMPY_AVAILABLE:
        np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
