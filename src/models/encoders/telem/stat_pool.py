"""
Telemetry statistical-pool encoder — BaseEncoder wrapper.

Pre-computed telemetry stat features -> linear projection -> EncoderOut.

Identical to KMStatEncoder; only the default feature_dim differs.
"""

from __future__ import annotations

from src.core.registry import get_encoder_registry
from src.models.encoders.km.stat import KMStatEncoder


@get_encoder_registry("telem").register("stat_pool")
class TelemStatPoolEncoder(KMStatEncoder):
    """
    Linear projection for pre-computed telemetry statistical features.

    Input: [B, T, D_in] -> Output: EncoderOut
    """

    def __init__(self, cfg):
        if isinstance(cfg, dict):
            cfg = {**cfg}
            cfg.setdefault("feature_dim", 109)
        super().__init__(cfg)
