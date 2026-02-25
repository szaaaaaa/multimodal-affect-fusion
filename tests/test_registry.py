"""
Registry integrity tests.

注册表完整性测试 — 确保所有模块正确注册。
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

# Ensure project root on path
ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

# Import registries (triggers auto-registration)
from src.core.registry import (
    ENCODERS,
    FUSIONS,
    DATAMODULES,
    HEADS,
    LOSSES,
    METRICS,
    get_encoder_registry,
)

# Force imports to trigger registration
import src.models.encoders.km       # noqa: F401
import src.models.encoders.video    # noqa: F401
import src.models.fusions           # noqa: F401
import src.models.heads             # noqa: F401
import src.losses                   # noqa: F401
import src.metrics                  # noqa: F401
import src.data.datamodules         # noqa: F401


class TestEncoderRegistries:
    def test_km_encoders_registered(self):
        km_reg = get_encoder_registry("km")
        assert "stat" in km_reg, f"KM 'stat' not registered. Available: {list(km_reg.keys())}"
        assert "cnn1d" in km_reg, f"KM 'cnn1d' not registered. Available: {list(km_reg.keys())}"

    def test_video_encoders_registered(self):
        video_reg = get_encoder_registry("video")
        assert "resnet2d" in video_reg, f"Video 'resnet2d' not registered. Available: {list(video_reg.keys())}"

    def test_encoder_registry_auto_creates(self):
        """Requesting a new modality creates an empty registry."""
        reg = get_encoder_registry("test_modality_xyz")
        assert reg is not None
        assert len(list(reg.keys())) == 0


class TestFusionRegistry:
    def test_fusions_registered(self):
        assert "single" in FUSIONS, f"'single' not registered. Available: {list(FUSIONS.keys())}"
        assert "lft" in FUSIONS, f"'lft' not registered. Available: {list(FUSIONS.keys())}"
        assert "aligned_mean" in FUSIONS, f"'aligned_mean' not registered. Available: {list(FUSIONS.keys())}"


class TestHeadRegistry:
    def test_heads_registered(self):
        assert "regression" in HEADS, f"'regression' not registered. Available: {list(HEADS.keys())}"
        assert "regression_seq" in HEADS, f"'regression_seq' not registered. Available: {list(HEADS.keys())}"


class TestLossRegistry:
    def test_losses_registered(self):
        assert "ccc" in LOSSES, f"'ccc' not registered. Available: {list(LOSSES.keys())}"
        assert "mse" in LOSSES, f"'mse' not registered. Available: {list(LOSSES.keys())}"
        assert "smooth_l1" in LOSSES, f"'smooth_l1' not registered. Available: {list(LOSSES.keys())}"
        assert "mse_seq_masked" in LOSSES, f"'mse_seq_masked' not registered. Available: {list(LOSSES.keys())}"


class TestMetricRegistry:
    def test_metrics_registered(self):
        assert "ccc" in METRICS, f"'ccc' not registered. Available: {list(METRICS.keys())}"
        assert "rmse" in METRICS, f"'rmse' not registered. Available: {list(METRICS.keys())}"


class TestDataModuleRegistry:
    def test_datamodules_registered(self):
        assert "amucs" in DATAMODULES, f"'amucs' not registered. Available: {list(DATAMODULES.keys())}"
        assert "amucs_seq" in DATAMODULES, f"'amucs_seq' not registered. Available: {list(DATAMODULES.keys())}"


class TestRegistryBuild:
    def test_build_encoder(self):
        km_reg = get_encoder_registry("km")
        encoder = km_reg.build("stat", {"d_in": 25, "d_model": 64})
        assert encoder is not None

    def test_build_fusion(self):
        fusion = FUSIONS.build("single")
        assert fusion is not None

    def test_build_head(self):
        head = HEADS.build("regression", {"d_model": 64, "hidden_dim": 32, "out_dim": 1})
        assert head is not None
        head_seq = HEADS.build("regression_seq", {"d_model": 64, "hidden_dim": 32, "out_dim": 1})
        assert head_seq is not None

    def test_build_loss(self):
        loss = LOSSES.build("ccc")
        assert loss is not None
        seq_loss = LOSSES.build("mse_seq_masked")
        assert seq_loss is not None

    def test_build_metric(self):
        metric = METRICS.build("ccc")
        assert metric is not None

    def test_build_unknown_raises(self):
        with pytest.raises(KeyError):
            FUSIONS.build("nonexistent_fusion")

    def test_duplicate_register_raises(self):
        from src.core.registry import Registry
        reg = Registry("test")
        reg.register("foo")(type("Foo", (), {}))
        with pytest.raises(KeyError):
            reg.register("foo")(type("Bar", (), {}))

