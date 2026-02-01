"""
Late Fusion Transformer (LFT) for Valence-Arousal prediction.
 
晚期融合 Transformer，用于情绪效价/唤醒度预测。
 
Architecture:
    Video Modality          KM Modality
         |                       |
    [EmotiEffLib]          [KM Encoder]
         |                       |
    [Linear + LN]          [Linear + LN]
         |                       |
    [+ Pos Enc]            [+ Pos Enc]
         |                       |
    [+ Mod Emb]            [+ Mod Emb]
         |                       |
    Video Tokens           KM Tokens
         |                       |
         +-------+-------+-------+
                 |
         [Concatenate]
                 |
         [Transformer Encoder x N]
                 |
         [Fused Tokens]
                 |
         [Pooling + MLP Head]
                 |
         [Valence / Arousal]
"""
 
from __future__ import annotations
 
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union
 
import torch
from torch import nn
 
# Ensure project root on path
ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from encoder.face.emotieff_encoder import EmotiEffTokenEncoder, EMOTIEFF_MODELS
from encoder.km.km_encoder_stat import KMStatTokenEncoder
from encoder.km.km_encoder_1dCNN import KM1DCNNEncoder
from models.components import (
    SinusoidalPositionalEncoding,
    LearnablePositionalEncoding,
    ModalityEmbedding,
)
 
 
@dataclass
class LFTConfig:
    """
    Configuration for Late Fusion Transformer.
 
    Late Fusion Transformer 配置。
    """
 
    # Model dimensions
    d_model: int = 256
    nhead: int = 8
    num_encoder_layers: int = 4
    dim_feedforward: int = 1024
    dropout: float = 0.1
 
    # Video encoder
    video_feature_dim: int = 1280  # EmotiEffLib EfficientNet-B0 output
    video_model_name: str = "enet_b0_8_best_afew"
 
    # KM encoder
    km_feature_dim: int = 25  # KM statistical features
    km_encoder_type: str = "stat"  # "stat" or "cnn"
 
    # Positional encoding
    max_seq_len: int = 1000
    pos_encoding_type: str = "sinusoidal"  # "sinusoidal" or "learnable"
 
    # Output head
    output_dim: int = 2  # Valence and Arousal
    head_hidden_dim: int = 128
    pooling: str = "mean"  # "mean", "max", or "cls"
 
    # Modality names
    modality_names: List[str] = field(default_factory=lambda: ["video", "km"])
 
 
class LateFusionTransformer(nn.Module):
    """
    Late Fusion Transformer for multimodal Valence-Arousal prediction.
 
    晚期融合 Transformer，用于多模态情绪预测。
 
    This model fuses visual features (from EmotiEffLib) and keyboard/mouse
    behavioral features to predict valence and arousal values.
 
    Parameters
    ----------
    config : LFTConfig
        Model configuration.
        / 模型配置。
 
    Examples
    --------
    >>> config = LFTConfig(d_model=256, num_encoder_layers=4)
    >>> model = LateFusionTransformer(config)
    >>> video_feat = torch.randn(8, 100, 1280)  # [B, T_v, 1280]
    >>> km_feat = torch.randn(8, 200, 25)       # [B, T_km, 25]
    >>> output = model(video_feat, km_feat)     # [B, 2]
    """
 
    def __init__(self, config: LFTConfig):
        super().__init__()
        self.config = config
 
        # ========== Video Branch ==========
        # Linear projection + LayerNorm for video features
        self.video_encoder = EmotiEffTokenEncoder(
            feature_dim=config.video_feature_dim,
            d_model=config.d_model,
            dropout=config.dropout,
        )
 
        # ========== KM Branch ==========
        # KM encoder (statistical or CNN)
        if config.km_encoder_type == "cnn":
            self.km_encoder = KM1DCNNEncoder(
                d_in=config.km_feature_dim,
                d_model=config.d_model,
            )
        else:
            self.km_encoder = KMStatTokenEncoder(
                d_in=config.km_feature_dim,
                d_model=config.d_model,
            )
 
        # ========== Positional Encoding ==========
        # Shared or separate positional encoding for each modality
        if config.pos_encoding_type == "learnable":
            self.video_pos_encoding = LearnablePositionalEncoding(
                d_model=config.d_model,
                max_len=config.max_seq_len,
                dropout=config.dropout,
            )
            self.km_pos_encoding = LearnablePositionalEncoding(
                d_model=config.d_model,
                max_len=config.max_seq_len,
                dropout=config.dropout,
            )
        else:
            self.video_pos_encoding = SinusoidalPositionalEncoding(
                d_model=config.d_model,
                max_len=config.max_seq_len,
                dropout=config.dropout,
            )
            self.km_pos_encoding = SinusoidalPositionalEncoding(
                d_model=config.d_model,
                max_len=config.max_seq_len,
                dropout=config.dropout,
            )
 
        # ========== Modality Embedding ==========
        self.modality_embedding = ModalityEmbedding(
            d_model=config.d_model,
            num_modalities=len(config.modality_names),
            modality_names=config.modality_names,
        )
 
        # ========== Transformer Encoder ==========
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=config.d_model,
            nhead=config.nhead,
            dim_feedforward=config.dim_feedforward,
            dropout=config.dropout,
            batch_first=True,
            norm_first=True,  # Pre-LN for better training stability
        )
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer,
            num_layers=config.num_encoder_layers,
        )
 
        # ========== Output Head ==========
        self.pooling = config.pooling
 
        # Optional CLS token for pooling
        if config.pooling == "cls":
            self.cls_token = nn.Parameter(torch.randn(1, 1, config.d_model) * 0.02)
 
        # MLP prediction head
        self.head = nn.Sequential(
            nn.Linear(config.d_model, config.head_hidden_dim),
            nn.GELU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.head_hidden_dim, config.output_dim),
        )
 
    def forward(
        self,
        video_feat: torch.Tensor,
        km_feat: torch.Tensor,
        video_mask: Optional[torch.Tensor] = None,
        km_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Forward pass.
 
        Parameters
        ----------
        video_feat : torch.Tensor
            Video features of shape (B, T_v, video_feature_dim).
            Pre-extracted EmotiEffLib features.
            / 视频特征（预提取的 EmotiEffLib 特征）。
        km_feat : torch.Tensor
            Keyboard/mouse features of shape (B, T_km, km_feature_dim).
            / 键盘/鼠标特征。
        video_mask : torch.Tensor, optional
            Boolean mask of shape (B, T_v), True for valid positions.
            / 视频掩码。
        km_mask : torch.Tensor, optional
            Boolean mask of shape (B, T_km), True for valid positions.
            / 键鼠掩码。
 
        Returns
        -------
        torch.Tensor
            Predictions of shape (B, output_dim).
            For VA prediction: (B, 2) with [valence, arousal].
            / 预测输出。
        """
        B = video_feat.size(0)
        T_v = video_feat.size(1)
        T_km = km_feat.size(1)
 
        # ========== Video Branch ==========
        # Project video features
        z_v = self.video_encoder(video_feat)  # [B, T_v, d_model]
 
        # Add positional encoding
        z_v = self.video_pos_encoding(z_v)  # [B, T_v, d_model]
 
        # Add modality embedding
        z_v = self.modality_embedding(z_v, modality="video")  # [B, T_v, d_model]
 
        # ========== KM Branch ==========
        # Encode KM features
        z_km = self.km_encoder(km_feat)  # [B, T_km, d_model]
 
        # Add positional encoding
        z_km = self.km_pos_encoding(z_km)  # [B, T_km, d_model]
 
        # Add modality embedding
        z_km = self.modality_embedding(z_km, modality="km")  # [B, T_km, d_model]
 
        # ========== Fusion ==========
        # Concatenate tokens along sequence dimension
        if self.pooling == "cls":
            # Prepend CLS token
            cls_tokens = self.cls_token.expand(B, -1, -1)  # [B, 1, d_model]
            z_combined = torch.cat([cls_tokens, z_v, z_km], dim=1)  # [B, 1+T_v+T_km, d_model]
        else:
            z_combined = torch.cat([z_v, z_km], dim=1)  # [B, T_v+T_km, d_model]
 
        # Build attention mask (True = masked/ignored)
        if video_mask is not None or km_mask is not None:
            if video_mask is None:
                video_mask = torch.ones(B, T_v, dtype=torch.bool, device=video_feat.device)
            if km_mask is None:
                km_mask = torch.ones(B, T_km, dtype=torch.bool, device=km_feat.device)
 
            if self.pooling == "cls":
                # CLS token is always valid
                cls_mask = torch.ones(B, 1, dtype=torch.bool, device=video_feat.device)
                combined_mask = torch.cat([cls_mask, video_mask, km_mask], dim=1)
            else:
                combined_mask = torch.cat([video_mask, km_mask], dim=1)
 
            # Transformer expects True = ignore, so invert
            padding_mask = ~combined_mask
        else:
            padding_mask = None
 
        # ========== Transformer Encoder ==========
        z_fused = self.transformer_encoder(
            z_combined,
            src_key_padding_mask=padding_mask,
        )  # [B, T_total, d_model]
 
        # ========== Pooling ==========
        if self.pooling == "cls":
            # Use CLS token representation
            pooled = z_fused[:, 0, :]  # [B, d_model]
        elif self.pooling == "max":
            # Max pooling over sequence
            if padding_mask is not None:
                # Mask out padded positions
                z_fused = z_fused.masked_fill(padding_mask.unsqueeze(-1), float("-inf"))
            pooled = z_fused.max(dim=1)[0]  # [B, d_model]
        else:  # mean pooling
            if padding_mask is not None:
                # Masked mean pooling
                mask_f = (~padding_mask).float().unsqueeze(-1)  # [B, T, 1]
                pooled = (z_fused * mask_f).sum(dim=1) / mask_f.sum(dim=1).clamp(min=1.0)
            else:
                pooled = z_fused.mean(dim=1)  # [B, d_model]
 
        # ========== Prediction ==========
        output = self.head(pooled)  # [B, output_dim]
 
        return output
 
    def get_attention_weights(
        self,
        video_feat: torch.Tensor,
        km_feat: torch.Tensor,
    ) -> List[torch.Tensor]:
        """
        Get attention weights from each transformer layer.
 
        获取每个 Transformer 层的注意力权重。
 
        Note: This requires hooking into the transformer layers.
        For visualization and interpretability.
 
        Parameters
        ----------
        video_feat : torch.Tensor
            Video features.
        km_feat : torch.Tensor
            KM features.
 
        Returns
        -------
        list of torch.Tensor
            Attention weights from each layer.
        """
        attention_weights = []
 
        def hook_fn(module, input, output):
            # MultiheadAttention returns (attn_output, attn_weights)
            if isinstance(output, tuple) and len(output) > 1:
                attention_weights.append(output[1])
 
        # Register hooks
        hooks = []
        for layer in self.transformer_encoder.layers:
            hook = layer.self_attn.register_forward_hook(hook_fn)
            hooks.append(hook)
 
        # Forward pass
        with torch.no_grad():
            _ = self.forward(video_feat, km_feat)
 
        # Remove hooks
        for hook in hooks:
            hook.remove()
 
        return attention_weights
 
 
class LateFusionTransformerVA(LateFusionTransformer):
    """
    Late Fusion Transformer specifically for Valence-Arousal prediction.
 
    专用于效价-唤醒度预测的晚期融合 Transformer。
 
    Convenience class with default output_dim=2 and provides
    separate valence/arousal outputs.
    """
 
    def __init__(self, config: Optional[LFTConfig] = None, **kwargs):
        if config is None:
            config = LFTConfig(**kwargs)
        config.output_dim = 2  # Valence and Arousal
        super().__init__(config)
 
    def forward(
        self,
        video_feat: torch.Tensor,
        km_feat: torch.Tensor,
        video_mask: Optional[torch.Tensor] = None,
        km_mask: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass with named outputs.
 
        Returns
        -------
        dict
            {"valence": Tensor[B], "arousal": Tensor[B], "va": Tensor[B, 2]}
        """
        output = super().forward(video_feat, km_feat, video_mask, km_mask)
        return {
            "valence": output[:, 0],
            "arousal": output[:, 1],
            "va": output,
        }
 
 
def create_lft_model(
    d_model: int = 256,
    num_layers: int = 4,
    nhead: int = 8,
    video_feature_dim: int = 1280,
    km_feature_dim: int = 25,
    dropout: float = 0.1,
    **kwargs,
) -> LateFusionTransformer:
    """
    Factory function to create LFT model with common configurations.
 
    创建 LFT 模型的工厂函数。
 
    Parameters
    ----------
    d_model : int
        Model dimension.
    num_layers : int
        Number of Transformer encoder layers.
    nhead : int
        Number of attention heads.
    video_feature_dim : int
        Video feature dimension (1280 for B0, 1408 for B2).
    km_feature_dim : int
        KM feature dimension.
    dropout : float
        Dropout probability.
 
    Returns
    -------
    LateFusionTransformer
        Configured model instance.
    """
    config = LFTConfig(
        d_model=d_model,
        num_encoder_layers=num_layers,
        nhead=nhead,
        video_feature_dim=video_feature_dim,
        km_feature_dim=km_feature_dim,
        dropout=dropout,
        **kwargs,
    )
    return LateFusionTransformer(config)
 
 
if __name__ == "__main__":
    # Demo
    print("=" * 60)
    print("Late Fusion Transformer Demo")
    print("=" * 60)
 
    # Create model
    config = LFTConfig(
        d_model=256,
        nhead=8,
        num_encoder_layers=4,
        video_feature_dim=1280,
        km_feature_dim=25,
    )
    model = LateFusionTransformer(config)
 
    # Print model summary
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"\nModel Configuration:")
    print(f"  d_model: {config.d_model}")
    print(f"  nhead: {config.nhead}")
    print(f"  num_layers: {config.num_encoder_layers}")
    print(f"  video_feature_dim: {config.video_feature_dim}")
    print(f"  km_feature_dim: {config.km_feature_dim}")
    print(f"\nTotal parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
 
    # Test forward pass
    print("\nTesting forward pass...")
    batch_size = 4
    T_v = 100  # Video sequence length
    T_km = 200  # KM sequence length
 
    video_feat = torch.randn(batch_size, T_v, config.video_feature_dim)
    km_feat = torch.randn(batch_size, T_km, config.km_feature_dim)
 
    # Without masks
    output = model(video_feat, km_feat)
    print(f"  Video features: {video_feat.shape}")
    print(f"  KM features: {km_feat.shape}")
    print(f"  Output: {output.shape}")
 
    # With masks
    video_mask = torch.ones(batch_size, T_v, dtype=torch.bool)
    km_mask = torch.ones(batch_size, T_km, dtype=torch.bool)
    km_mask[:, -50:] = False  # Mask last 50 positions
 
    output_masked = model(video_feat, km_feat, video_mask, km_mask)
    print(f"  Output (with masks): {output_masked.shape}")
 
    # Test VA model
    print("\nTesting LateFusionTransformerVA...")
    model_va = LateFusionTransformerVA(config)
    output_va = model_va(video_feat, km_feat)
    print(f"  Valence: {output_va['valence'].shape}")
    print(f"  Arousal: {output_va['arousal'].shape}")
    print(f"  VA combined: {output_va['va'].shape}")
 
    print("\nDemo completed successfully!")
