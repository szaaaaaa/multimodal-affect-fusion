"""
EmotiEffLib-based visual feature encoder for facial emotion recognition.

基于 EmotiEffLib 的视觉特征编码器，用于面部情感识别。
"""

from __future__ import annotations

from typing import List, Optional, Union

import torch
from torch import nn

try:
    from emotiefflib.facial_emotions import HSEmotionRecognizer
    EMOTIEFFLIB_AVAILABLE = True
except ImportError:
    EMOTIEFFLIB_AVAILABLE = False
    HSEmotionRecognizer = None


# Model configurations: model_name -> (feature_dim, description)
EMOTIEFF_MODELS = {
    "enet_b0_8_best_afew": (1280, "EfficientNet-B0, 8 emotions, best on AFEW"),
    "enet_b0_8_best_vgaf": (1280, "EfficientNet-B0, 8 emotions, best on VGAF"),
    "enet_b0_8_va_mtl": (1280, "EfficientNet-B0, 8 emotions, VA multi-task"),
    "enet_b2_8": (1408, "EfficientNet-B2, 8 emotions"),
    "enet_b2_7": (1408, "EfficientNet-B2, 7 emotions"),
}


class EmotiEffEncoder:
    """
    Feature extractor wrapper for EmotiEffLib.

    EmotiEffLib 特征提取器包装类。

    Parameters
    ----------
    model_name : str
        Name of the pre-trained model.
        / 预训练模型名称。
    device : str
        Device to run the model on ('cpu' or 'cuda').
        / 模型运行设备。

    Attributes
    ----------
    feature_dim : int
        Dimension of extracted features.
        / 提取特征的维度。

    Examples
    --------
    >>> encoder = EmotiEffEncoder(model_name="enet_b0_8_best_afew", device="cuda")
    >>> features = encoder.extract_features(face_image)  # [1280]
    >>> batch_features = encoder.extract_batch_features(face_images)  # [B, 1280]
    """

    def __init__(
        self,
        model_name: str = "enet_b0_8_best_afew",
        device: str = "cpu",
    ):
        if not EMOTIEFFLIB_AVAILABLE:
            raise ImportError(
                "emotiefflib is not installed. "
                "Install it with: pip install emotiefflib[torch]"
            )

        if model_name not in EMOTIEFF_MODELS:
            raise ValueError(
                f"Unknown model_name: {model_name}. "
                f"Available models: {list(EMOTIEFF_MODELS.keys())}"
            )

        self.model_name = model_name
        self.device = device
        self.feature_dim = EMOTIEFF_MODELS[model_name][0]

        # Initialize the recognizer
        self.recognizer = HSEmotionRecognizer(
            model_name=model_name,
            device=device,
        )

    def extract_features(self, face_image) -> torch.Tensor:
        """
        Extract features from a single face image.

        从单张人脸图像提取特征。

        Parameters
        ----------
        face_image : np.ndarray
            Face image in BGR format (H, W, 3).
            / BGR 格式的人脸图像。

        Returns
        -------
        torch.Tensor
            Feature vector of shape (feature_dim,).
            / 特征向量。
        """
        features = self.recognizer.extract_features(face_image)
        return torch.tensor(features, dtype=torch.float32, device=self.device)

    def extract_batch_features(self, face_images: List) -> torch.Tensor:
        """
        Extract features from a batch of face images.

        从一批人脸图像提取特征。

        Parameters
        ----------
        face_images : List[np.ndarray]
            List of face images in BGR format.
            / BGR 格式的人脸图像列表。

        Returns
        -------
        torch.Tensor
            Feature matrix of shape (B, feature_dim).
            / 特征矩阵。
        """
        features = self.recognizer.extract_multi_features(face_images)
        return torch.tensor(features, dtype=torch.float32, device=self.device)


class EmotiEffTokenEncoder(nn.Module):
    """
    Neural network wrapper that projects EmotiEffLib features to model dimension.

    将 EmotiEffLib 特征投影到模型维度的神经网络包装器。

    This module takes pre-extracted EmotiEffLib features and projects them
    to a specified model dimension with layer normalization.

    Parameters
    ----------
    feature_dim : int
        Input feature dimension from EmotiEffLib (1280 for B0, 1408 for B2).
        / EmotiEffLib 输入特征维度。
    d_model : int
        Output model dimension.
        / 输出模型维度。
    dropout : float
        Dropout probability.
        / Dropout 概率。

    Examples
    --------
    >>> encoder = EmotiEffTokenEncoder(feature_dim=1280, d_model=256)
    >>> x = torch.randn(8, 100, 1280)  # [B, T, feature_dim]
    >>> out = encoder(x)  # [B, T, d_model]
    """

    def __init__(
        self,
        feature_dim: int = 1280,
        d_model: int = 256,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.feature_dim = feature_dim
        self.d_model = d_model

        self.proj = nn.Sequential(
            nn.Linear(feature_dim, d_model),
            nn.LayerNorm(d_model),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Parameters
        ----------
        x : torch.Tensor
            Input features of shape (B, T, feature_dim).
            / 输入特征。

        Returns
        -------
        torch.Tensor
            Projected features of shape (B, T, d_model).
            / 投影后的特征。
        """
        return self.proj(x)


class VideoFeatureExtractor(nn.Module):
    """
    End-to-end video feature extractor using EmotiEffLib.

    使用 EmotiEffLib 的端到端视频特征提取器。

    This module handles both feature extraction (via EmotiEffLib) and
    projection to model dimension. It can work with pre-extracted features
    or extract features on-the-fly.

    Parameters
    ----------
    model_name : str
        EmotiEffLib model name.
        / EmotiEffLib 模型名称。
    d_model : int
        Output model dimension.
        / 输出模型维度。
    dropout : float
        Dropout probability.
        / Dropout 概率。
    device : str
        Device for computation.
        / 计算设备。
    use_pretrained : bool
        Whether to load pretrained EmotiEffLib model.
        / 是否加载预训练模型。

    Examples
    --------
    >>> extractor = VideoFeatureExtractor(
    ...     model_name="enet_b0_8_best_afew",
    ...     d_model=256,
    ...     device="cuda"
    ... )
    >>> # With pre-extracted features
    >>> features = torch.randn(8, 100, 1280)  # [B, T, 1280]
    >>> out = extractor(features)  # [B, T, 256]
    """

    def __init__(
        self,
        model_name: str = "enet_b0_8_best_afew",
        d_model: int = 256,
        dropout: float = 0.1,
        device: str = "cpu",
        use_pretrained: bool = True,
    ):
        super().__init__()

        self.model_name = model_name
        self.d_model = d_model
        self.device = device

        # Get feature dimension for the model
        if model_name in EMOTIEFF_MODELS:
            self.feature_dim = EMOTIEFF_MODELS[model_name][0]
        else:
            # Default to EfficientNet-B0 dimension
            self.feature_dim = 1280

        # Feature extractor (optional, for on-the-fly extraction)
        self._extractor: Optional[EmotiEffEncoder] = None
        if use_pretrained and EMOTIEFFLIB_AVAILABLE:
            self._extractor = EmotiEffEncoder(
                model_name=model_name,
                device=device,
            )

        # Token encoder for projection
        self.token_encoder = EmotiEffTokenEncoder(
            feature_dim=self.feature_dim,
            d_model=d_model,
            dropout=dropout,
        )

    def extract_features_from_frames(self, frames: List) -> torch.Tensor:
        """
        Extract features from raw video frames.

        从原始视频帧提取特征。

        Parameters
        ----------
        frames : List[np.ndarray]
            List of face images (T frames), each in BGR format.
            / 人脸图像列表。

        Returns
        -------
        torch.Tensor
            Features of shape (T, feature_dim).
            / 特征张量。
        """
        if self._extractor is None:
            raise RuntimeError(
                "Feature extractor not initialized. "
                "Set use_pretrained=True and ensure emotiefflib is installed."
            )
        return self._extractor.extract_batch_features(frames)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass with pre-extracted features.

        Parameters
        ----------
        x : torch.Tensor
            Pre-extracted EmotiEffLib features of shape (B, T, feature_dim).
            / 预提取的特征。

        Returns
        -------
        torch.Tensor
            Projected features of shape (B, T, d_model).
            / 投影后的特征。
        """
        return self.token_encoder(x)


if __name__ == "__main__":
    # Demo: test the token encoder (doesn't require emotiefflib)
    print("Testing EmotiEffTokenEncoder...")
    encoder = EmotiEffTokenEncoder(feature_dim=1280, d_model=256)
    x = torch.randn(4, 50, 1280)  # [B, T, feature_dim]
    out = encoder(x)
    print(f"Input shape: {x.shape}")
    print(f"Output shape: {out.shape}")
    print(f"Parameters: {sum(p.numel() for p in encoder.parameters()):,}")
