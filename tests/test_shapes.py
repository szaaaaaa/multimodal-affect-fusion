"""
Shape contract tests — the most effective guard against future breakage.

形状契约测试 — 防止未来扩展破坏既有流程的最有效手段。

Tests that every module produces outputs conforming to the frozen interfaces.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest
import torch

# Ensure project root on path
ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

# Force imports to trigger registration
import src.models.encoders.km       # noqa: F401
import src.models.encoders.video    # noqa: F401
import src.models.fusions           # noqa: F401
import src.models.heads             # noqa: F401
import src.losses                   # noqa: F401
import src.metrics                  # noqa: F401

from src.core.registry import FUSIONS, HEADS, LOSSES, METRICS, get_encoder_registry


B = 4       # batch size for tests
D = 64      # d_model for tests
T = 20      # sequence length for tests


# ──────────────────────────────────────────────
# Encoder contract tests
# ──────────────────────────────────────────────

@pytest.mark.parametrize("modality,name,d_in", [
    ("km", "stat", 25),
    ("km", "cnn1d", 25),
    ("video", "resnet2d", 2048),
])
def test_encoder_output_contract(modality, name, d_in):
    """Every encoder must return EncoderOut with correct shapes."""
    x = torch.randn(B, T, d_in)
    encoder = get_encoder_registry(modality).build(name, {"d_in": d_in, "feature_dim": d_in, "d_model": D})
    out = encoder(x)

    assert "tokens" in out, "EncoderOut missing 'tokens'"
    assert "pooled" in out, "EncoderOut missing 'pooled'"
    assert "mask" in out, "EncoderOut missing 'mask'"

    expected_t = T
    assert out["tokens"].shape == (B, expected_t, D), f"tokens shape: {out['tokens'].shape} != ({B}, {expected_t}, {D})"
    assert out["pooled"].shape == (B, D), f"pooled shape: {out['pooled'].shape} != ({B}, {D})"
    assert out["mask"].shape == (B, expected_t), f"mask shape: {out['mask'].shape} != ({B}, {expected_t})"
    assert out["mask"].dtype == torch.bool, f"mask dtype: {out['mask'].dtype} != torch.bool"


@pytest.mark.parametrize("modality,name,d_in", [
    ("km", "stat", 25),
    ("km", "cnn1d", 25),
    ("video", "resnet2d", 2048),
])
def test_encoder_with_mask(modality, name, d_in):
    """Encoder must accept an explicit mask."""
    x = torch.randn(B, T, d_in)
    mask = torch.ones(B, T, dtype=torch.bool)
    mask[:, -5:] = False

    encoder = get_encoder_registry(modality).build(name, {"d_in": d_in, "feature_dim": d_in, "d_model": D})
    out = encoder(x, mask=mask)

    expected_t = T
    assert out["tokens"].shape == (B, expected_t, D)
    assert out["mask"].shape == (B, expected_t)
    # Mask should be the one we passed in
    assert torch.equal(out["mask"], mask)


# ──────────────────────────────────────────────
# Fusion contract tests
# ──────────────────────────────────────────────

def _make_z_dict(modalities):
    """Helper to create mock z_dict for given modality names."""
    z_dict = {}
    mask_dict = {}
    for mod in modalities:
        z_dict[mod] = {
            "tokens": torch.randn(B, T, D),
            "pooled": torch.randn(B, D),
            "mask": torch.ones(B, T, dtype=torch.bool),
        }
        mask_dict[mod] = z_dict[mod]["mask"]
    return z_dict, mask_dict


@pytest.mark.parametrize("modality_subset", [
    ["km"],
    ["video"],
    ["video", "km"],
])
def test_single_fusion_arbitrary_subset(modality_subset):
    """SingleFusion must handle any modality subset."""
    z_dict, mask_dict = _make_z_dict(modality_subset)
    fusion = FUSIONS.build("single")
    out = fusion(z_dict, mask_dict)

    assert "pooled" in out, "FusionOut missing 'pooled'"
    assert out["pooled"].shape == (B, D), f"pooled shape: {out['pooled'].shape}"


@pytest.mark.parametrize("modality_subset", [
    ["km"],
    ["video"],
    ["video", "km"],
])
def test_lft_fusion_arbitrary_subset(modality_subset):
    """LFTFusion must handle any modality subset."""
    z_dict, mask_dict = _make_z_dict(modality_subset)
    fusion = FUSIONS.build("lft", {"d_model": D, "nhead": 4, "num_layers": 1, "dim_feedforward": 128})
    out = fusion(z_dict, mask_dict)

    assert "pooled" in out, "FusionOut missing 'pooled'"
    assert out["pooled"].shape == (B, D), f"pooled shape: {out['pooled'].shape}"
    assert "tokens" in out


def test_aligned_mean_fusion_shape():
    z_dict, mask_dict = _make_z_dict(["video", "km"])
    fusion = FUSIONS.build("aligned_mean")
    out = fusion(z_dict, mask_dict)
    assert out["tokens"].shape == (B, T, D)
    assert out["pooled"].shape == (B, D)


# ──────────────────────────────────────────────
# Head contract tests
# ──────────────────────────────────────────────

@pytest.mark.parametrize("out_dim", [1, 2])
def test_head_output_shape(out_dim):
    """Head must produce [B, out_dim]."""
    h = {"tokens": None, "pooled": torch.randn(B, D)}
    head = HEADS.build("regression", {"d_model": D, "hidden_dim": 32, "out_dim": out_dim})
    y_hat = head(h)
    assert y_hat.shape == (B, out_dim), f"head output: {y_hat.shape} != ({B}, {out_dim})"


def test_seq_head_output_shape():
    h = {"tokens": torch.randn(B, T, D), "pooled": torch.randn(B, D)}
    head = HEADS.build("regression_seq", {"d_model": D, "hidden_dim": 32, "out_dim": 1})
    y_hat = head(h)
    assert y_hat.shape == (B, T, 1)


# ──────────────────────────────────────────────
# End-to-end single forward pass
# ──────────────────────────────────────────────

def test_end_to_end_forward():
    """DataModule batch → Encoders → Fusion → Head → Loss: one forward, no errors."""
    # Mock batch
    batch = {
        "x": {
            "video": torch.randn(B, T, 2048),
            "km": torch.randn(B, T, 25),
        },
        "mask": {
            "video": torch.ones(B, T, dtype=torch.bool),
            "km": torch.ones(B, T, dtype=torch.bool),
        },
        "y": torch.randn(B, 1),
    }

    # Build
    video_enc = get_encoder_registry("video").build("resnet2d", {"feature_dim": 2048, "d_model": D})
    km_enc = get_encoder_registry("km").build("stat", {"d_in": 25, "d_model": D})
    fusion = FUSIONS.build("lft", {"d_model": D, "nhead": 4, "num_layers": 1, "dim_feedforward": 128})
    head = HEADS.build("regression", {"d_model": D, "hidden_dim": 32, "out_dim": 1})
    loss_fn = LOSSES.build("ccc")

    # Forward
    z_video = video_enc(batch["x"]["video"], batch["mask"]["video"])
    z_km = km_enc(batch["x"]["km"], batch["mask"]["km"])

    z_dict = {"video": z_video, "km": z_km}
    mask_dict = {"video": z_video["mask"], "km": z_km["mask"]}

    h = fusion(z_dict, mask_dict)
    y_hat = head(h)
    loss = loss_fn(y_hat, batch["y"])

    assert y_hat.shape == (B, 1)
    assert loss.ndim == 0  # scalar


def test_end_to_end_single_modality():
    """Single-modality forward pass works end-to-end."""
    batch = {
        "x": {"km": torch.randn(B, T, 25)},
        "mask": {"km": torch.ones(B, T, dtype=torch.bool)},
        "y": torch.randn(B, 1),
    }

    km_enc = get_encoder_registry("km").build("cnn1d", {"d_in": 25, "d_model": D})
    fusion = FUSIONS.build("single")
    head = HEADS.build("regression", {"d_model": D, "hidden_dim": 32, "out_dim": 1})
    loss_fn = LOSSES.build("smooth_l1")

    z_km = km_enc(batch["x"]["km"], batch["mask"]["km"])
    z_dict = {"km": z_km}
    mask_dict = {"km": z_km["mask"]}

    h = fusion(z_dict, mask_dict)
    y_hat = head(h)
    loss = loss_fn(y_hat, batch["y"])

    assert y_hat.shape == (B, 1)
    assert loss.ndim == 0


# ──────────────────────────────────────────────
# Loss and metric contract tests
# ──────────────────────────────────────────────

@pytest.mark.parametrize("loss_name", ["ccc", "mse", "smooth_l1"])
def test_loss_returns_scalar(loss_name):
    pred = torch.randn(B, 1)
    target = torch.randn(B, 1)
    loss_fn = LOSSES.build(loss_name)
    loss = loss_fn(pred, target)
    assert loss.ndim == 0, f"Loss {loss_name} is not scalar: ndim={loss.ndim}"


def test_masked_seq_loss_returns_scalar():
    pred = torch.randn(B, T, 1)
    target = torch.randn(B, T, 1)
    mask = torch.ones(B, T, dtype=torch.bool)
    mask[:, -5:] = False
    loss_fn = LOSSES.build("mse_seq_masked")
    loss = loss_fn(pred, target, mask)
    assert loss.ndim == 0


@pytest.mark.parametrize("metric_name", ["ccc", "rmse"])
def test_metric_returns_float(metric_name):
    pred = torch.randn(B, 1)
    target = torch.randn(B, 1)
    metric_fn = METRICS.build(metric_name)
    val = metric_fn(pred, target)
    assert isinstance(val, float), f"Metric {metric_name} returned {type(val)}, expected float"

