"""
ResNet-50-based visual feature encoder for gameplay frames.

基于 ResNet-50 的视觉特征编码器，用于游戏画面帧。
"""

from __future__ import annotations

from typing import List, Optional

import cv2
import torch
from torch import nn

try:
    from torchvision.models import resnet50, ResNet50_Weights
    TORCHVISION_AVAILABLE = True
except ImportError:
    TORCHVISION_AVAILABLE = False


RESNET_FEATURE_DIM = 2048
IMAGENET_MEAN = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
IMAGENET_STD = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)


class ResNetFrameEncoder:
    """
    ResNet-50 feature extractor for raw frames.

    ResNet-50 逐帧特征提取器。

    Parameters
    ----------
    device : str
        Device to run the model on ('cpu' or 'cuda').
    pretrained : bool
        Whether to load ImageNet pretrained weights.
    freeze : bool
        Whether to freeze backbone parameters.
    """

    def __init__(self, device: str = "cpu", pretrained: bool = True, freeze: bool = True):
        if not TORCHVISION_AVAILABLE:
            raise ImportError("torchvision is not installed. Install with: pip install torchvision")

        weights = ResNet50_Weights.DEFAULT if pretrained else None
        self.model = resnet50(weights=weights)
        self.model.fc = nn.Identity()
        self.model.to(device)
        self.model.eval()

        if freeze:
            for p in self.model.parameters():
                p.requires_grad = False

        self.device = device
        self.feature_dim = RESNET_FEATURE_DIM

    def _preprocess(self, frame_bgr, frame_size: int = 224) -> torch.Tensor:
        frame = cv2.resize(frame_bgr, (frame_size, frame_size))
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        tensor = torch.from_numpy(frame).permute(2, 0, 1).float() / 255.0
        tensor = tensor.unsqueeze(0).to(self.device)
        tensor = (tensor - IMAGENET_MEAN.to(self.device)) / IMAGENET_STD.to(self.device)
        return tensor

    def extract_features(self, frame_bgr) -> torch.Tensor:
        """
        Extract features from a single frame.

        Returns
        -------
        torch.Tensor
            Feature vector of shape (feature_dim,).
        """
        with torch.no_grad():
            x = self._preprocess(frame_bgr)
            feat = self.model(x).squeeze(0)
        return feat

    def extract_batch_features(self, frames_bgr: List, frame_size: int = 224) -> torch.Tensor:
        """
        Extract features from a batch of frames.

        Returns
        -------
        torch.Tensor
            Feature matrix of shape (B, feature_dim).
        """
        with torch.no_grad():
            batch = torch.cat([self._preprocess(f, frame_size) for f in frames_bgr], dim=0)
            feats = self.model(batch)
        return feats


class ResNetTokenEncoder(nn.Module):
    """
    Project ResNet features to model dimension with LayerNorm.

    将 ResNet 特征投影到模型维度。
    """

    def __init__(self, feature_dim: int = RESNET_FEATURE_DIM, d_model: int = 512, dropout: float = 0.1):
        super().__init__()
        self.proj = nn.Sequential(
            nn.Linear(feature_dim, d_model),
            nn.LayerNorm(d_model),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.proj(x)


class VideoFeatureExtractor(nn.Module):
    """
    Video feature extractor using ResNet-50 + projection.

    Supports pre-extracted features or on-the-fly frame extraction.
    """

    def __init__(
        self,
        d_model: int = 512,
        dropout: float = 0.1,
        device: str = "cpu",
        use_pretrained: bool = True,
        freeze_backbone: bool = True,
    ):
        super().__init__()
        self.device = device
        self.feature_dim = RESNET_FEATURE_DIM

        self._extractor: Optional[ResNetFrameEncoder] = None
        if use_pretrained and TORCHVISION_AVAILABLE:
            self._extractor = ResNetFrameEncoder(
                device=device,
                pretrained=use_pretrained,
                freeze=freeze_backbone,
            )

        self.token_encoder = ResNetTokenEncoder(
            feature_dim=self.feature_dim,
            d_model=d_model,
            dropout=dropout,
        )

    def extract_features_from_frames(self, frames: List) -> torch.Tensor:
        if self._extractor is None:
            raise RuntimeError(
                "Feature extractor not initialized. "
                "Set use_pretrained=True and ensure torchvision is installed."
            )
        return self._extractor.extract_batch_features(frames)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.token_encoder(x)


if __name__ == "__main__":
    print("Testing ResNetTokenEncoder...")
    encoder = ResNetTokenEncoder(feature_dim=RESNET_FEATURE_DIM, d_model=512)
    x = torch.randn(4, 50, RESNET_FEATURE_DIM)
    out = encoder(x)
    print(f"Input shape: {x.shape}")
    print(f"Output shape: {out.shape}")
    print(f"Parameters: {sum(p.numel() for p in encoder.parameters()):,}")
