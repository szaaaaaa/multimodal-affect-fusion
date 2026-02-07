"""
Visual feature encoders.

视觉特征编码器。
"""
from .ResNet50 import (
    ResNetFrameEncoder,
    ResNetTokenEncoder,
    VideoFeatureExtractor,
    RESNET_FEATURE_DIM,
)

__all__ = [
    "ResNetFrameEncoder",
    "ResNetTokenEncoder",
    "VideoFeatureExtractor",
    "RESNET_FEATURE_DIM",
]
