#!/bin/bash
# ============================================================
# Colab: Extract CLIP ViT-L/14 video features + align to 5Hz
# ============================================================
# Copy each section into a separate Colab cell.
#
# Output:
#   features/video_clip/           — raw CLIP features at 8fps
#   features/aligned/video_clip/   — 5Hz aligned CLIP features
#
# Training: use data_root=.../features/aligned
# Config: model.encoders.video.feature_dim: 768
# ============================================================

# ── Cell 1: Clone repo + install deps ─────────────────────
# from getpass import getpass
# import os
# token = getpass('GitHub PAT: ')
# !git clone https://{token}@github.com/szaaaaaa/ProjectExperiment.git
# %cd ProjectExperiment
# !pip install open_clip_torch

# ── Cell 2: Mount Drive ───────────────────────────────────
# from google.colab import drive
# drive.mount('/content/drive')

# ── Cell 3: Extract CLIP features (A100/H100, ~2-3h) ─────
# %%bash
# cd /content/ProjectExperiment
# python scripts/extract_video_features.py \
#   --video_dir "/content/drive/MyDrive/AmuCS/Affective Multimodal Counter-Strike video game dataset (AMuCS) - Public/researchdata/gameplay_videos_nospeech" \
#   --output_dir "/content/drive/MyDrive/AmuCS_experiment/features/video_clip" \
#   --backbone clip_vit_l14 \
#   --target_fps 8 \
#   --batch_size 64 \
#   --device cuda \
#   --amp \
#   --session_mode subdirs \
#   --name_mode amucs

# ── Cell 4: Verify extraction ────────────────────────────
# import glob, torch
# pts = sorted(glob.glob("/content/drive/MyDrive/AmuCS_experiment/features/video_clip/*.pt"))
# print(f"Extracted: {len(pts)} files")
# if pts:
#     obj = torch.load(pts[0], map_location="cpu", weights_only=False)
#     print(f"Feature shape: {obj['features'].shape}")
#     print(f"Backbone: {obj['meta']['backbone']}")

# ── Cell 5: Align video_clip to 5Hz ──────────────────────
# Only aligns video_clip. km and telem are already in aligned/.
# Uses same parameters (uniform 5Hz, time_origin=zero) as original alignment,
# so timestamps are consistent with aligned/km/ and aligned/telem/.
#
# %%bash
# FEAT="/content/drive/MyDrive/AmuCS_experiment/features"
# STAGE="$FEAT/_stage_video_clip"
# mkdir -p "$STAGE"
# ln -sfn "$FEAT/video_clip" "$STAGE/video_clip"
#
# cd /content/ProjectExperiment
# python scripts/sync_data.py \
#   --input_root "$STAGE" \
#   --output_root "$FEAT/aligned" \
#   --modalities video_clip \
#   --grid_mode uniform \
#   --target_hz 5 \
#   --resample nearest \
#   --time_origin zero
#
# rm -rf "$STAGE"

# ── Cell 6: Verify alignment ─────────────────────────────
# import glob, torch
# pts = sorted(glob.glob("/content/drive/MyDrive/AmuCS_experiment/features/aligned/video_clip/*.pt"))
# print(f"Aligned video_clip: {len(pts)} files")
# if pts:
#     obj = torch.load(pts[0], map_location="cpu", weights_only=False)
#     print(f"Aligned shape: {obj['features'].shape}")
#     # Compare with existing aligned video to confirm same length
#     vpt = pts[0].replace("video_clip", "video")
#     if os.path.exists(vpt):
#         vobj = torch.load(vpt, map_location="cpu", weights_only=False)
#         print(f"Original video shape: {vobj['features'].shape}")
#         print(f"Length match: {obj['features'].shape[0] == vobj['features'].shape[0]}")
