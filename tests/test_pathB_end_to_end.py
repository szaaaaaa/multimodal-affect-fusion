"""
End-to-end smoke test for Path B:
  datamodule → per-modality encoders → CMA fusion → multitask_seq head → loss → backward.

Uses real extracted features under features/aligned/{video_clip, km_event, telem_60hz}
and the cross-subject split, but with a small batch and short seq_len for speed.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest
import torch

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

# Trigger registrations
import src.data.datamodules  # noqa: F401
import src.models.encoders.video  # noqa: F401
import src.models.encoders.km  # noqa: F401
import src.models.encoders.telem  # noqa: F401
import src.models.fusions  # noqa: F401
import src.models.heads  # noqa: F401
import src.losses  # noqa: F401

from src.core.registry import (
    DATAMODULES, FUSIONS, HEADS, LOSSES, get_encoder_registry,
)


EXP_ROOT = Path("G:/我的云端硬盘/AmuCS_experiment")
DATA_ROOT = EXP_ROOT / "features/aligned"
SPLIT_PATH = EXP_ROOT / "splits/session_tvt.json"
LABELS_PATH = EXP_ROOT / "labels/arousal_state_trend_seq.json"


@pytest.mark.skipif(
    not (DATA_ROOT / "km_event").is_dir() or not (DATA_ROOT / "telem_60hz").is_dir(),
    reason="Path B extracted features not yet materialized",
)
def test_pathB_end_to_end_forward_backward():
    D = 128  # smaller d_model for smoke
    T_v = 60  # short window for smoke

    dm_cfg = {
        "modalities": ["video", "km_event", "telem_60hz"],
        "data_root": str(DATA_ROOT),
        "labels_seq_path": str(LABELS_PATH),
        "split_path": str(SPLIT_PATH),
        "seq_len_video_frames": T_v,
        "train_stride_video_frames": T_v,
        "val_stride_video_frames": T_v,
        "test_stride_video_frames": T_v,
        "normalize": True,
        "task_names": ["state", "trend"],
        "label_dtype": "long",
        "batch_size": 2,
        "num_workers": 0,
        "max_events_per_window": 512,
        "modality_dir_map": {
            "video": "video_clip",
            "km_event": "km_event",
            "telem_60hz": "telem_60hz",
        },
    }

    dm = DATAMODULES.build("amucs_seq_mixed_rate", dm_cfg)
    loader = dm.train_dataloader()
    batch = next(iter(loader))

    assert "x" in batch and "video" in batch["x"]
    B = batch["x"]["video"].shape[0]
    assert batch["x"]["video"].shape == (B, T_v, 768), batch["x"]["video"].shape
    assert batch["x"]["telem_60hz"].shape == (B, T_v * 12, 23), batch["x"]["telem_60hz"].shape
    assert batch["x"]["km_event"].shape[0] == B and batch["x"]["km_event"].shape[2] == 4

    # Build encoders
    video_enc = get_encoder_registry("video").build(
        "resnet2d",
        {"d_model": D, "feature_dim": 768, "dropout": 0.1, "temporal_pool": "none"},
    )
    km_enc = get_encoder_registry("km_event").build(
        "event_token",
        {
            "d_model": D, "vocab_size": 256, "num_layers": 2, "nhead": 4,
            "dropout": 0.1, "target_T_v": T_v,
        },
    )
    telem_enc = get_encoder_registry("telem_60hz").build(
        "stream_60hz",
        {
            "d_model": D, "feature_dim": 23, "tcn_layers": 2, "kernel": 5,
            "dropout": 0.1, "downsample_stride": 12,
        },
    )

    video_out = video_enc(batch["x"]["video"], mask=batch["mod_mask"]["video"])
    km_out = km_enc(batch["x"]["km_event"], mask=batch["mod_mask"]["km_event"])
    telem_out = telem_enc(batch["x"]["telem_60hz"], mask=batch["mod_mask"]["telem_60hz"])

    assert video_out["tokens"].shape == (B, T_v, D)
    assert km_out["tokens"].shape == (B, T_v, D)
    assert telem_out["tokens"].shape == (B, T_v, D)

    # CMA fusion
    fusion = FUSIONS.build("cma", {
        "d_model": D, "nhead": 4, "cm_layers": 1, "sa_layers": 1,
        "dim_feedforward": 256, "dropout": 0.1,
        "anchor_modality": "video", "max_seq_len": 2000,
        "pos_encoding_type": "sinusoidal", "pooling": "mean",
        "temporal_merge": "mean",
    })
    z_dict = {"video": video_out, "km_event": km_out, "telem_60hz": telem_out}
    mask_dict = {m: o["mask"] for m, o in z_dict.items()}
    fused = fusion(z_dict, mask_dict)
    assert fused["tokens"].shape == (B, T_v, D)

    # Multitask seq head
    head = HEADS.build("multitask_seq", {
        "d_model": D, "task_names": ["state", "trend"],
        "hidden_dim": 64, "num_classes": 3, "dropout": 0.1,
    })
    pred = head(fused)
    assert pred["state"].shape == (B, T_v, 3)
    assert pred["trend"].shape == (B, T_v, 3)

    # Loss
    loss_fn = LOSSES.build("multitask_ce_seq_masked", {
        "task_weights": {"state": 1.0, "trend": 1.0},
        "label_smoothing": 0.0,
    })
    loss = loss_fn(pred, batch["y"], batch["mask"])
    assert torch.isfinite(loss)
    loss.backward()
    print(f"[smoke] B={B} T_v={T_v} loss={loss.item():.4f}")
