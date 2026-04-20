"""
Shape tests for Path B encoders: km.event_token and telem.stream_60hz.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest
import torch

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import src.models.encoders.km     # noqa: F401
import src.models.encoders.telem  # noqa: F401

from src.core.registry import get_encoder_registry


B = 2
D = 64
T_v = 30   # video frame count in a window
T_k = 50   # km event count
T_t = T_v * 12  # 60Hz stream length aligned to T_v
VOCAB = 80


def _build_km_event_encoder():
    return get_encoder_registry("km_event").build(
        "event_token",
        {
            "d_model": D,
            "vocab_size": VOCAB,
            "num_layers": 2,
            "nhead": 4,
            "dropout": 0.0,
            "target_T_v": T_v,
        },
    )


def _build_telem_stream_encoder(D_t: int = 23):
    return get_encoder_registry("telem_60hz").build(
        "stream_60hz",
        {
            "d_model": D,
            "feature_dim": D_t,
            "tcn_layers": 3,
            "kernel": 5,
            "dropout": 0.0,
            "downsample_stride": 12,
        },
    )


def test_km_event_token_shape_no_mask():
    enc = _build_km_event_encoder()
    type_ids = torch.randint(0, VOCAB, (B, T_k))
    t_rel = torch.linspace(0.0, T_v * 0.2, T_k).unsqueeze(0).expand(B, -1).contiguous()
    dt = torch.zeros(B, T_k)
    dt[:, 1:] = t_rel[:, 1:] - t_rel[:, :-1]
    bin_id = torch.clamp((t_rel / 0.2).floor().long(), max=T_v - 1)
    x = torch.stack([type_ids.float(), t_rel, dt, bin_id.float()], dim=-1)

    out = enc(x)
    assert out["tokens"].shape == (B, T_v, D)
    assert out["pooled"].shape == (B, D)
    assert out["mask"].shape == (B, T_v)
    assert out["mask"].dtype == torch.bool


def test_km_event_token_shape_with_mask():
    enc = _build_km_event_encoder()
    type_ids = torch.randint(0, VOCAB, (B, T_k))
    t_rel = torch.rand(B, T_k) * (T_v * 0.2)
    t_rel, _ = t_rel.sort(dim=1)
    dt = torch.zeros(B, T_k)
    dt[:, 1:] = t_rel[:, 1:] - t_rel[:, :-1]
    bin_id = torch.clamp((t_rel / 0.2).floor().long(), max=T_v - 1)
    x = torch.stack([type_ids.float(), t_rel, dt, bin_id.float()], dim=-1)

    mask = torch.ones(B, T_k, dtype=torch.bool)
    mask[:, -10:] = False  # last 10 events are padding

    out = enc(x, mask=mask)
    assert out["tokens"].shape == (B, T_v, D)
    assert out["pooled"].shape == (B, D)
    assert out["mask"].shape == (B, T_v)


def test_km_event_token_all_pad_row_no_nan():
    """3.7% of real training windows have 0 km events; ensure encoder handles all-pad rows."""
    enc = _build_km_event_encoder()
    type_ids = torch.zeros(B, T_k, dtype=torch.long)
    t_rel = torch.zeros(B, T_k)
    dt = torch.zeros(B, T_k)
    bin_id = torch.zeros(B, T_k)
    x = torch.stack([type_ids.float(), t_rel, dt, bin_id], dim=-1)
    mask = torch.zeros(B, T_k, dtype=torch.bool)  # all pad

    out = enc(x, mask=mask)
    assert out["tokens"].shape == (B, T_v, D)
    assert torch.isfinite(out["tokens"]).all(), "all-pad row produced NaN/Inf"
    assert torch.isfinite(out["pooled"]).all()


def test_km_event_token_backward():
    enc = _build_km_event_encoder()
    type_ids = torch.randint(0, VOCAB, (B, T_k))
    t_rel = torch.rand(B, T_k) * (T_v * 0.2)
    t_rel, _ = t_rel.sort(dim=1)
    dt = torch.zeros(B, T_k)
    dt[:, 1:] = t_rel[:, 1:] - t_rel[:, :-1]
    bin_id = torch.clamp((t_rel / 0.2).floor().long(), max=T_v - 1)
    x = torch.stack([type_ids.float(), t_rel, dt, bin_id.float()], dim=-1)

    out = enc(x)
    loss = out["tokens"].sum() + out["pooled"].sum()
    loss.backward()


def test_telem_stream_60hz_shape():
    enc = _build_telem_stream_encoder(D_t=23)
    x = torch.randn(B, T_t, 23)
    out = enc(x)
    assert out["tokens"].shape == (B, T_v, D)
    assert out["pooled"].shape == (B, D)
    assert out["mask"].shape == (B, T_v)
    assert out["mask"].dtype == torch.bool


def test_telem_stream_60hz_with_mask():
    enc = _build_telem_stream_encoder(D_t=23)
    x = torch.randn(B, T_t, 23)
    mask = torch.ones(B, T_t, dtype=torch.bool)
    mask[:, -24:] = False  # last 2 video-frame-worths are invalid
    out = enc(x, mask=mask)
    assert out["tokens"].shape == (B, T_v, D)
    # The last 2 pooled frames should have mask=False
    assert out["mask"].shape == (B, T_v)
    assert (out["mask"][:, -2:] == False).all()


def test_telem_stream_60hz_backward():
    enc = _build_telem_stream_encoder(D_t=23)
    x = torch.randn(B, T_t, 23, requires_grad=True)
    out = enc(x)
    loss = out["tokens"].sum() + out["pooled"].sum()
    loss.backward()
    assert x.grad is not None


def test_pathB_encoders_produce_same_T_v():
    """Critical: km_event + telem_60hz + video must share T after encoding for CMA."""
    km_enc = _build_km_event_encoder()
    telem_enc = _build_telem_stream_encoder()

    type_ids = torch.randint(0, VOCAB, (B, T_k))
    t_rel = torch.rand(B, T_k) * (T_v * 0.2)
    t_rel, _ = t_rel.sort(dim=1)
    dt = torch.zeros(B, T_k)
    bin_id = torch.clamp((t_rel / 0.2).floor().long(), max=T_v - 1)
    km_x = torch.stack([type_ids.float(), t_rel, dt, bin_id.float()], dim=-1)
    telem_x = torch.randn(B, T_t, 23)

    km_out = km_enc(km_x)
    telem_out = telem_enc(telem_x)
    assert km_out["tokens"].shape[1] == T_v
    assert telem_out["tokens"].shape[1] == T_v
