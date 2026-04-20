"""
Frozen abstract interfaces for the extensible multimodal framework.

可扩展多模态框架的冻结抽象接口。

These 4 interfaces (BaseEncoder, BaseFusion, BaseHead, BaseDataModule) plus
the typed dictionaries (Batch, EncoderOut, FusionOut) form the stable contract.
Future extensions implement these interfaces — they never change.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, TypedDict

import torch
from torch import nn
from torch.utils.data import DataLoader


# ──────────────────────────────────────────────
# Typed dictionaries — shape contracts
# ──────────────────────────────────────────────

class EncoderOut(TypedDict):
    """
    Encoder output contract.

    tokens : Tensor[B, T, D]  — token-level representations
    pooled : Tensor[B, D]     — global pooled representation
    mask   : Tensor[B, T]     — bool, True = valid token
    """
    tokens: torch.Tensor
    pooled: torch.Tensor
    mask: torch.Tensor


class FusionOut(TypedDict):
    """
    Fusion output contract.

    tokens : Tensor[B, T, D] | None — fused token sequence (optional)
    pooled : Tensor[B, D]           — fused global representation (required)
    """
    tokens: Optional[torch.Tensor]
    pooled: torch.Tensor


class Batch(TypedDict, total=False):
    """
    Unified batch schema.

    x    : Dict[str, Tensor]  — modality name -> feature tensor
    mask : Dict[str, Tensor]  — modality name -> bool mask
    y    : Tensor             — labels
    meta : Dict               — optional metadata (id, player, session, …)
    """
    x: Dict[str, torch.Tensor]
    mask: Dict[str, torch.Tensor]
    y: torch.Tensor
    meta: Dict[str, Any]


# ──────────────────────────────────────────────
# Abstract base classes — frozen interfaces
# ──────────────────────────────────────────────

class BaseEncoder(ABC, nn.Module):
    """
    Abstract encoder interface (frozen).

    Every modality encoder must subclass this and return EncoderOut.
    """

    @abstractmethod
    def forward(
        self,
        x: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> EncoderOut:
        """
        Encode modality features.

        Parameters
        ----------
        x : Tensor
            Input features, [B, T, D_in] (sequence) or [B, D_in] (non-sequence).
        mask : Tensor, optional
            Bool mask [B, T], True = valid. None means all valid.

        Returns
        -------
        EncoderOut
            {"tokens": [B, T, D], "pooled": [B, D], "mask": [B, T]}
        """
        ...


class BaseFusion(ABC, nn.Module):
    """
    Abstract fusion interface (frozen).

    Must handle arbitrary subsets of modalities in z_dict.
    """

    def init_for_modalities(
        self,
        modality_names: list,
        device: torch.device,
    ) -> None:
        """Pre-allocate per-modality modules so they are captured by the optimizer.

        Called by the runner before optimizer construction. Subclasses with
        lazy per-modality modules should override this.
        """

    @abstractmethod
    def forward(
        self,
        z_dict: Dict[str, EncoderOut],
        mask_dict: Dict[str, torch.Tensor],
    ) -> FusionOut:
        """
        Fuse encoded modality representations.

        Parameters
        ----------
        z_dict : dict
            {modality_name: EncoderOut} — may contain 1..N modalities.
        mask_dict : dict
            {modality_name: Tensor[B, T]} — bool masks.

        Returns
        -------
        FusionOut
            {"tokens": [B, T, D] or None, "pooled": [B, D]}
        """
        ...


class BaseHead(ABC, nn.Module):
    """
    Abstract prediction head interface (frozen).
    """

    @abstractmethod
    def forward(self, h: FusionOut) -> torch.Tensor:
        """
        Predict from fused representation.

        Parameters
        ----------
        h : FusionOut
            {"tokens": ... , "pooled": [B, D]}

        Returns
        -------
        Tensor [B, out_dim]
        """
        ...


class BaseDataModule(ABC):
    """
    Abstract data module interface (frozen).

    Subclasses produce DataLoaders whose batches conform to the Batch schema.
    """

    @abstractmethod
    def train_dataloader(self) -> DataLoader:
        ...

    @abstractmethod
    def val_dataloader(self) -> DataLoader:
        ...

    def test_dataloader(self) -> Optional[DataLoader]:
        return None
