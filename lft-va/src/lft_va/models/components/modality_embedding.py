"""
Modality embedding module for multimodal fusion.
 
多模态融合的模态嵌入模块。
"""
 
from __future__ import annotations
 
from typing import Dict, Optional, Union
 
import torch
from torch import nn
 
 
class ModalityEmbedding(nn.Module):
    """
    Learnable modality embedding to distinguish different modalities.
 
    可学习的模态嵌入，用于区分不同模态。
 
    Each modality gets a unique learnable embedding vector that is added
    to all tokens from that modality. This helps the Transformer distinguish
    between tokens from different modalities.
 
    Parameters
    ----------
    d_model : int
        Model dimension (embedding size).
        / 模型维度。
    num_modalities : int
        Number of modalities.
        / 模态数量。
    modality_names : list of str, optional
        Names for each modality. If None, uses indices.
        / 模态名称列表。
 
    Examples
    --------
    >>> # Two modalities: video and keyboard/mouse
    >>> mod_emb = ModalityEmbedding(
    ...     d_model=256,
    ...     num_modalities=2,
    ...     modality_names=["video", "km"]
    ... )
    >>> video_tokens = torch.randn(8, 100, 256)  # [B, T_v, D]
    >>> km_tokens = torch.randn(8, 50, 256)      # [B, T_km, D]
    >>> video_out = mod_emb(video_tokens, modality="video")
    >>> km_out = mod_emb(km_tokens, modality="km")
    """
 
    def __init__(
        self,
        d_model: int,
        num_modalities: int = 2,
        modality_names: Optional[list] = None,
    ):
        super().__init__()
        self.d_model = d_model
        self.num_modalities = num_modalities
 
        # Set up modality name to index mapping
        if modality_names is None:
            modality_names = [str(i) for i in range(num_modalities)]
 
        if len(modality_names) != num_modalities:
            raise ValueError(
                f"modality_names length ({len(modality_names)}) "
                f"must match num_modalities ({num_modalities})"
            )
 
        self.modality_names = modality_names
        self.name_to_idx = {name: idx for idx, name in enumerate(modality_names)}
 
        # Learnable modality embeddings
        # Initialize with small random values (similar to position embeddings)
        self.embeddings = nn.Embedding(num_modalities, d_model)
        nn.init.normal_(self.embeddings.weight, mean=0.0, std=0.02)
 
    def forward(
        self,
        x: torch.Tensor,
        modality: Union[int, str],
    ) -> torch.Tensor:
        """
        Add modality embedding to input tokens.
 
        将模态嵌入添加到输入 token。
 
        Parameters
        ----------
        x : torch.Tensor
            Input tokens of shape (B, T, D).
            / 输入 token。
        modality : int or str
            Modality index or name.
            / 模态索引或名称。
 
        Returns
        -------
        torch.Tensor
            Tokens with modality embedding added, shape (B, T, D).
            / 添加模态嵌入后的 token。
        """
        # Get modality index
        if isinstance(modality, str):
            if modality not in self.name_to_idx:
                raise ValueError(
                    f"Unknown modality name: {modality}. "
                    f"Available: {self.modality_names}"
                )
            mod_idx = self.name_to_idx[modality]
        else:
            mod_idx = int(modality)
 
        # Get embedding for this modality
        # Shape: [1, 1, d_model] for broadcasting
        mod_emb = self.embeddings(
            torch.tensor([mod_idx], device=x.device)
        ).unsqueeze(0)  # [1, 1, D]
 
        # Add to all tokens
        return x + mod_emb
 
    def get_embedding(self, modality: Union[int, str]) -> torch.Tensor:
        """
        Get the embedding vector for a specific modality.
 
        获取特定模态的嵌入向量。
 
        Parameters
        ----------
        modality : int or str
            Modality index or name.
            / 模态索引或名称。
 
        Returns
        -------
        torch.Tensor
            Embedding vector of shape (d_model,).
            / 嵌入向量。
        """
        if isinstance(modality, str):
            mod_idx = self.name_to_idx[modality]
        else:
            mod_idx = int(modality)
 
        return self.embeddings.weight[mod_idx]
 
    def get_all_embeddings(self) -> Dict[str, torch.Tensor]:
        """
        Get all modality embeddings as a dictionary.
 
        获取所有模态嵌入的字典。
 
        Returns
        -------
        dict
            Dictionary mapping modality names to embedding vectors.
            / 模态名称到嵌入向量的字典。
        """
        return {
            name: self.embeddings.weight[idx]
            for name, idx in self.name_to_idx.items()
        }
 
    def extra_repr(self) -> str:
        return (
            f"d_model={self.d_model}, "
            f"num_modalities={self.num_modalities}, "
            f"modalities={self.modality_names}"
        )
 
 
class MultiModalityEmbedding(nn.Module):
    """
    Multi-modality embedding with separate projections per modality.
 
    带有独立投影的多模态嵌入。
 
    This module provides:
    1. Per-modality linear projection (if input dimensions differ)
    2. Learnable modality embedding
 
    Parameters
    ----------
    d_model : int
        Output model dimension.
        / 输出模型维度。
    modality_dims : dict
        Dictionary mapping modality name to input dimension.
        / 模态名称到输入维度的字典。
 
    Examples
    --------
    >>> mm_emb = MultiModalityEmbedding(
    ...     d_model=256,
    ...     modality_dims={"video": 1280, "km": 25}
    ... )
    >>> video_feat = torch.randn(8, 100, 1280)
    >>> km_feat = torch.randn(8, 50, 25)
    >>> video_out = mm_emb(video_feat, modality="video")  # [8, 100, 256]
    >>> km_out = mm_emb(km_feat, modality="km")  # [8, 50, 256]
    """
 
    def __init__(
        self,
        d_model: int,
        modality_dims: Dict[str, int],
    ):
        super().__init__()
        self.d_model = d_model
        self.modality_dims = modality_dims
        self.modality_names = list(modality_dims.keys())
 
        # Create per-modality projections
        self.projections = nn.ModuleDict({
            name: nn.Sequential(
                nn.Linear(dim, d_model),
                nn.LayerNorm(d_model),
            )
            for name, dim in modality_dims.items()
        })
 
        # Modality embedding
        self.modality_embedding = ModalityEmbedding(
            d_model=d_model,
            num_modalities=len(modality_dims),
            modality_names=self.modality_names,
        )
 
    def forward(
        self,
        x: torch.Tensor,
        modality: str,
    ) -> torch.Tensor:
        """
        Project and add modality embedding.
 
        投影并添加模态嵌入。
 
        Parameters
        ----------
        x : torch.Tensor
            Input features of shape (B, T, D_in).
            / 输入特征。
        modality : str
            Modality name.
            / 模态名称。
 
        Returns
        -------
        torch.Tensor
            Output tokens of shape (B, T, d_model).
            / 输出 token。
        """
        if modality not in self.projections:
            raise ValueError(
                f"Unknown modality: {modality}. "
                f"Available: {self.modality_names}"
            )
 
        # Project to d_model
        x = self.projections[modality](x)
 
        # Add modality embedding
        x = self.modality_embedding(x, modality=modality)
 
        return x
 
 
if __name__ == "__main__":
    # Demo
    print("Testing Modality Embedding modules...")
 
    # Test basic modality embedding
    mod_emb = ModalityEmbedding(
        d_model=256,
        num_modalities=2,
        modality_names=["video", "km"]
    )
 
    video_tokens = torch.randn(4, 100, 256)
    km_tokens = torch.randn(4, 50, 256)
 
    video_out = mod_emb(video_tokens, modality="video")
    km_out = mod_emb(km_tokens, modality="km")
 
    print(f"Video tokens: {video_tokens.shape} -> {video_out.shape}")
    print(f"KM tokens: {km_tokens.shape} -> {km_out.shape}")
    print(f"Modality embedding parameters: {sum(p.numel() for p in mod_emb.parameters()):,}")
 
    # Verify embeddings are different
    video_emb = mod_emb.get_embedding("video")
    km_emb = mod_emb.get_embedding("km")
    print(f"Embedding distance: {(video_emb - km_emb).norm().item():.4f}")
 
    # Test multi-modality embedding
    print("\nTesting MultiModalityEmbedding...")
    mm_emb = MultiModalityEmbedding(
        d_model=256,
        modality_dims={"video": 1280, "km": 25}
    )
 
    video_feat = torch.randn(4, 100, 1280)
    km_feat = torch.randn(4, 50, 25)
 
    video_out = mm_emb(video_feat, modality="video")
    km_out = mm_emb(km_feat, modality="km")
 
    print(f"Video features: {video_feat.shape} -> {video_out.shape}")
    print(f"KM features: {km_feat.shape} -> {km_out.shape}")
    print(f"Multi-modality embedding parameters: {sum(p.numel() for p in mm_emb.parameters()):,}")
