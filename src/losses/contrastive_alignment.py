"""
Cross-modal contrastive alignment loss (Direction C).

Adds an InfoNCE auxiliary loss between encoder outputs *before* fusion.
Same-sample representations from different modalities are positive pairs;
different-sample representations are negative pairs.

Uses pooled (sequence-level) representations for efficiency — avoids
O((B*T)²) memory cost of token-level InfoNCE.

跨模态对比对齐辅助损失。强制编码器将同一情感状态的不同模态映射到邻近区域。
"""

from __future__ import annotations

from typing import Dict, List, Optional

import torch
import torch.nn.functional as F
from torch import nn


class ContrastiveAlignmentLoss(nn.Module):
    """
    Cross-modal InfoNCE loss on pooled encoder outputs.

    Projection heads are eagerly initialized via ``init_projs()`` so that
    their parameters are visible to the optimizer.

    Parameters (via cfg dict)
    -------------------------
    temperature : float (default 0.07)
    lambda_align : float (default 0.1) — weight of alignment loss
    proj_dim : int (default 128) — projection head output dimension
    d_model : int (default 512) — encoder output dimension
    """

    def __init__(self, cfg: Optional[Dict] = None):
        super().__init__()
        cfg = cfg or {}
        self.temperature = float(cfg.get("temperature", 0.07))
        self.lambda_align = float(cfg.get("lambda_align", 0.1))
        self.proj_dim = int(cfg.get("proj_dim", 128))
        self._d_model = int(cfg.get("d_model", 512))
        self.projs = nn.ModuleDict()

    def init_projs(self, modality_names: List[str]) -> None:
        """Eagerly create projection heads. Call before optimizer construction."""
        for mod in modality_names:
            self.projs[mod] = nn.Sequential(
                nn.Linear(self._d_model, self._d_model),
                nn.ReLU(),
                nn.Linear(self._d_model, self.proj_dim),
            )

    def forward(self, z_dict: Dict[str, Dict[str, torch.Tensor]]) -> torch.Tensor:
        """
        Parameters
        ----------
        z_dict : {mod: EncoderOut} where EncoderOut has "tokens" [B,T,D] and "mask" [B,T]

        Returns
        -------
        Scalar loss (already scaled by lambda_align).
        """
        modalities = sorted(z_dict.keys())
        if len(modalities) < 2:
            return torch.tensor(0.0, device=next(iter(z_dict.values()))["tokens"].device)

        device = z_dict[modalities[0]]["tokens"].device

        # Pool each modality to [B, D] using masked mean
        pooled = {}
        for mod in modalities:
            tokens = z_dict[mod]["tokens"]  # [B, T, D]
            mask = z_dict[mod]["mask"]      # [B, T]
            mask_f = mask.unsqueeze(-1).float()  # [B, T, 1]
            denom = mask_f.sum(dim=1).clamp(min=1.0)  # [B, 1]
            p = (tokens * mask_f).sum(dim=1) / denom   # [B, D]
            pooled[mod] = F.normalize(self.projs[mod](p), dim=-1)  # [B, proj_dim]

        # InfoNCE for each modality pair
        loss = torch.tensor(0.0, device=device)
        count = 0
        for i, m1 in enumerate(modalities):
            for m2 in modalities[i + 1:]:
                z1 = pooled[m1]  # [B, proj_dim]
                z2 = pooled[m2]  # [B, proj_dim]
                sim = z1 @ z2.T / self.temperature  # [B, B]
                labels = torch.arange(sim.size(0), device=device)
                loss = loss + (F.cross_entropy(sim, labels)
                               + F.cross_entropy(sim.T, labels)) / 2
                count += 1

        return self.lambda_align * loss / max(count, 1)
