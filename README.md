# ProjectExperiment

An extensible multimodal sequence-learning framework centered on **Late Fusion Transformer (LFT)** for AMuCS-style gameplay affect modeling.

## What Is Supported

- Single-task sequence regression (continuous arousal)
- Single-task sequence classification (3-class tasks)
- Multitask sequence classification (`state` + `trend`)
- **Mixed multitask sequence learning**:
  - `arousal` regression (continuous)
  - `trend` classification (3 classes)

All major experiments use the same LFT backbone and the same modality families:

- `video` -> `resnet2d`
- `km` -> `stat`
- `telem` -> `stat_pool`

## Core Architecture

Pipeline:

1. Modality encoders
2. LFT fusion
3. Task head
4. Mask-aware loss and metrics

Key registries:

- `DATAMODULES`
- `FUSIONS`
- `HEADS`
- `LOSSES`
- `METRICS`

Main training entrypoint:

- `scripts/train.py`

## New Mixed Multitask (Arousal Regression + Trend Classification)

### Label Merge Script

Use:

```bash
python scripts/merge_arousal_reg_trend_labels.py \
  --arousal /path/to/arousal_seq_z_perparticipant.json \
  --trend /path/to/arousal_3trend_seq.json \
  --output /path/to/arousal_reg_trend_seq.json
```

Output schema:

```json
{
  "<stem>": {
    "arousal": {"values": [...], "mask": [...]},
    "trend": {"values": [...], "mask": [...]}
  }
}
```

### Configs (7 Modality Combinations)

- `configs/amucs_seq_lft_video_multitask_arousal_trend.yaml`
- `configs/amucs_seq_lft_km_multitask_arousal_trend.yaml`
- `configs/amucs_seq_lft_telem_multitask_arousal_trend.yaml`
- `configs/amucs_seq_lft_video_km_multitask_arousal_trend.yaml`
- `configs/amucs_seq_lft_video_telem_multitask_arousal_trend.yaml`
- `configs/amucs_seq_lft_km_telem_multitask_arousal_trend.yaml`
- `configs/amucs_seq_lft_video_km_telem_multitask_arousal_trend.yaml`

### New Components

- Mixed multitask head:
  - `src/models/heads/multitask_mixed_seq.py`
- Mixed multitask loss:
  - `src/losses/multitask_mixed_seq_loss.py`
- Datamodule extension (task-level dtype support):
  - `src/data/datamodules/amucs_seq_multitask.py`
- Runner extension (task-level metrics routing + optional mixed score):
  - `src/core/runner.py`

## Existing State+Trend Multitask Classification

State+Trend multitask classification configs remain fully supported:

- `configs/amucs_seq_lft_*_multitask_state_trend.yaml`

This path is backward compatible and unchanged in behavior.

## Notebook Workflow

Main notebook:

- `train.ipynb`

Relevant cells:

- `Cell 26`: multitask classification (`state + trend`)
- `Cell 27`: mixed multitask (`arousal regression + trend classification`) with full **7 combos x 3 seeds**
- `Cell 28`: lag sweep analysis for regression runs

## Typical Train Command

Example (mixed multitask, video+km):

```bash
python -u scripts/train.py \
  --config configs/amucs_seq_lft_video_km_multitask_arousal_trend.yaml \
  --override \
    data.data_root=/path/to/features/aligned \
    data.labels_seq_path=/path/to/arousal_reg_trend_seq.json \
    data.split_path=/path/to/session_tvt.json \
    train.seed=0
```

## Environment

Recommended:

- Python 3.10+
- PyTorch + torchvision
- pyyaml, numpy, pandas, scikit-learn, tqdm, pytest

## Output Structure

Each run writes to:

```text
runs/{timestamp}__{dataset}__{fusion}__{modalities}__seed{seed}/
  config.yaml
  seed.txt
  git_commit.txt
  ckpt_best.pt
  ckpt_last.pt
  metrics.json
```

## Notes

- Masked timesteps (`mask=False`) are excluded from loss/metrics.
- Mixed multitask supports per-task metrics and optional weighted composite metric (`val_score_mixed`) for early stopping.
- Existing single-task and previous multitask experiments are not broken by these additions.
