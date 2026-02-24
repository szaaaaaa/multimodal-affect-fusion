# ProjectExperiment

An extensible multimodal framework for game emotion prediction (Valence/Arousal), focused on:
- `video` modality (pre-extracted ResNet-50 frame features)
- `km` modality (keyboard/mouse interaction features)

The current training stack is already refactored to an interface + registry + config architecture.

## 1. What Is Implemented Now

### 1.1 Core experiment methods
- Multimodal regression on AMuCS-style session stems (for example `S001_P1`)
- Supported tasks:
  - Arousal regression (`out_dim: 1`)
  - Joint Valence/Arousal regression (`out_dim: 2`)
- Supported training setups:
  - `km` single modality + `single` fusion
  - `video + km` multimodal + `lft` fusion

### 1.2 Main model architecture
- Encoders:
  - `video/resnet2d`: per-frame projection to `d_model` with mask-aware temporal mean pooling for `pooled`
  - `km/stat`: linear projection encoder for KM statistics
  - `km/cnn1d`: 1D CNN encoder for KM sequences
- Fusion:
  - `single`: pass-through for one modality; concat + masked mean for multiple modalities
  - `lft`: Late Fusion Transformer with positional encoding + modality embedding
  - `lft_video_valence`: optional special fusion for split VA pathway
- Heads:
  - `regression`: MLP head
  - `va_split`: valence/arousal split head for `lft_video_valence`
- Losses:
  - `smooth_l1`, `mse`, `ccc`
- Metrics:
  - `ccc`, `rmse`

### 1.3 Extensible design
- Frozen interfaces in `src/core/types.py`:
  - `BaseEncoder`, `BaseFusion`, `BaseHead`, `BaseDataModule`
- Registry-based composition in `src/core/registry.py`:
  - `DATAMODULES`, `FUSIONS`, `HEADS`, `LOSSES`, `METRICS`
  - per-modality encoder registry via `get_encoder_registry(modality)`
- Thin training entry:
  - `scripts/train.py` only does `config -> Runner.fit()`

## 2. Repository Structure

```text
ProjectExperiment/
  configs/
    base.yaml
    experiments/
  scripts/
    train.py
    extract_video_features.py
    extract_km_features.py
    build_multimodal_split.py
    build_km_arousal_split.py
    summarize.py
  src/
    core/
    data/
    models/
    losses/
    metrics/
  tests/
  docs/
```

## 3. Data and Feature Format

### 3.1 Training expects this feature layout

```text
data/features/amucs/
  video/
    S001_P1.pt
    S001_P2.pt
    ...
  km/
    S001_P1.pt
    S001_P2.pt
    ...
```

And:
- labels file: `data/labels_arousal.json`
- split file:
  - multimodal: `data/splits/multimodal_split.json`
  - km single: `data/splits/km_arousal_split.json`

### 3.2 Video feature `.pt` content

Each file is a dict with at least:
- `features`: tensor `[T, D]` (usually `D=2048`)
- `timestamps`: per-feature timestamps (seconds)
- `fps`, `sample_fps`, `stride`
- `meta`: source path and extraction metadata

### 3.3 KM feature `.pt` content

Extracted from session CSV logs:
- `keyboard.csv`
- `mousebuttons.csv`

Output is a dict containing encoded `features` and metadata.

## 4. Environment Setup

Use Python 3.10+.

Install core dependencies:

```bash
pip install torch torchvision pyyaml pandas opencv-python tqdm pytest
```

## 5. End-to-End Usage

## 5.1 Extract video features

Example for session-subdirectory style input (for example `s001`, `s002`, ...):

```bash
python scripts/extract_video_features.py \
  --video_dir "/path/to/gameplay_videos_nospeech" \
  --output_dir "data/features/amucs/video" \
  --session_mode subdirs \
  --name_mode amucs \
  --device cuda
```

Notes:
- `--name_mode amucs` maps session/video names to stems like `S001_P1`.
- Session-level resume is built in:
  - writes done markers under `output_dir/.session_done/`
  - rerun will skip completed sessions/files.
- Use `--overwrite` to force recomputation.

## 5.2 Extract KM features

Input root should follow `S*/P*`:

```text
/path/to/km_raw/
  S001/
    P1/
      keyboard.csv
      mousebuttons.csv
```

Run:

```bash
python scripts/extract_km_features.py \
  --root_dir "/path/to/km_raw" \
  --output_dir "data/features/amucs/km" \
  --encoder stat
```

Notes:
- Default behavior skips existing output files (resume-friendly).
- Use `--overwrite` to recompute.

## 5.3 Build train/val splits

Multimodal split:

```bash
python scripts/build_multimodal_split.py \
  --video_dir data/features/amucs/video \
  --km_dir data/features/amucs/km \
  --labels_path data/labels_arousal.json \
  --output_path data/splits/multimodal_split.json
```

KM-only split:

```bash
python scripts/build_km_arousal_split.py --seed 42 --train_ratio 0.8
```

## 5.4 Train experiments

Video + KM LFT:

```bash
python scripts/train.py --config configs/experiments/video_km_lft.yaml
```

KM single:

```bash
python scripts/train.py --config configs/experiments/km_single.yaml
```

KM single CNN:

```bash
python scripts/train.py --config configs/experiments/km_single_cnn.yaml
```

Video + KM VA output:

```bash
python scripts/train.py --config configs/experiments/video_km_lft_va.yaml
```

CLI override example:

```bash
python scripts/train.py \
  --config configs/experiments/video_km_lft.yaml \
  --override train.epochs=100 train.seed=0 model.fusion.num_layers=2
```

## 5.5 Resume training

Resume from run directory:

```bash
python scripts/train.py \
  --config configs/experiments/video_km_lft.yaml \
  --resume runs/<run_dir_name>
```

Or directly from checkpoint:

```bash
python scripts/train.py \
  --config configs/experiments/video_km_lft.yaml \
  --resume runs/<run_dir_name>/ckpt_last.pt
```

Important:
- Resume is epoch-level.
- If checkpoint epoch already reached configured `train.epochs`, increase it via override.

## 5.6 Summarize results

```bash
python scripts/summarize.py --runs_dir runs --output leaderboard.csv
```

## 6. Training Outputs

Each run creates:

```text
runs/{timestamp}__{dataset}__{fusion}__{modalities}__seed{seed}/
  config.yaml (or config.json)
  seed.txt
  git_commit.txt
  ckpt_best.pt
  ckpt_last.pt
  metrics.json
```

## 7. Current Configs

- `configs/experiments/video_km_lft.yaml`
  - multimodal (`video`, `km`) with `lft` fusion
- `configs/experiments/km_single.yaml`
  - KM-only with `single` fusion
- `configs/experiments/km_single_cnn.yaml`
  - KM-only with `cnn1d` encoder + `single` fusion
- `configs/experiments/video_km_lft_va.yaml`
  - multimodal with 2-D target output (`out_dim: 2`)

Base defaults are in `configs/base.yaml`.

## 8. Quality Checks

Run tests:

```bash
pytest -q
```

Includes:
- shape-contract tests (`tests/test_shapes.py`)
- registry integrity tests (`tests/test_registry.py`)

## 9. Colab/Drive Practical Notes

To avoid losing progress after Colab interruption:
- keep `runs_dir` on Google Drive
- keep feature output directories on Drive
- rerun the same extraction/training commands with resume flags

For video extraction on session folders in Drive, prefer:

```bash
--session_mode subdirs --name_mode amucs
```

This avoids cross-session filename collisions and supports session-level resume.
