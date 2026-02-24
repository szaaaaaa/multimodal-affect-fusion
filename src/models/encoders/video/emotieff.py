"""
Deprecated alias for video encoder.

The legacy name "emotieff" now maps to the per-frame ResNet projection encoder
encoder to avoid hard dependencies on the legacy EmotiEff pipeline.
"""

from __future__ import annotations

from src.core.registry import get_encoder_registry
from src.models.encoders.video.resnet2d import VideoResNet2dEncoder


@get_encoder_registry("video").register("emotieff")
class VideoEmotiEffEncoder(VideoResNet2dEncoder):
    """Backward-compatible alias for resnet2d encoder."""

    pass

