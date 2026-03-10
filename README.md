# ProjectExperiment

An extensible multimodal sequence-learning framework for **gameplay affect modeling**, supporting both **Late Fusion Transformer (LFT)** and **Cross-Modal Attention (CMA)** fusion architectures. Designed around a plugin-based architecture with frozen interfaces and a registry system so new modalities, encoders, and tasks can be added without touching core training code.

---

## Table of Contents

- [Background](#background)
- [Supported Tasks](#supported-tasks)
- [Model Architecture](#model-architecture)
  - [Overall Pipeline](#overall-pipeline)
  - [Modality Encoders](#modality-encoders)
  - [Late Fusion Transformer (LFT)](#late-fusion-transformer-lft)
  - [Cross-Modal Attention (CMA)](#cross-modal-attention-cma)
  - [Task Heads](#task-heads)
- [Experiment Design](#experiment-design)
- [Framework Design](#framework-design)
- [Repository Structure](#repository-structure)
- [Installation](#installation)
- [Data Preparation](#data-preparation)
- [Usage](#usage)
  - [Training](#training)
  - [Configuration System](#configuration-system)
  - [Modality Combinations](#modality-combinations)
  - [Mixed Multitask (Arousal + Trend)](#mixed-multitask-arousal--trend)
- [Adding New Components](#adding-new-components)
- [Testing](#testing)
- [Output Structure](#output-structure)
- [Results Aggregation](#results-aggregation)

---

## Background

This framework is built for the **AMuCS** (Adaptive Multimodal Computer Systems) gameplay dataset, where the goal is to predict participant affect (arousal/valence) from multimodal behavioral signals recorded during gameplay. The three modality families used throughout experiments are:

| Modality | Source signal | Feature dim |
|---|---|---|
| `video` | Per-frame ResNet-50 features | 2048 |
| `km` | Keyboard + mouse statistical features | 25 |
| `telem` | Game telemetry statistics | varies |

Affect annotation is provided as continuous per-second scores (arousal/valence) collected via self-report, aligned to session timestamps.

---

## Supported Tasks

| Task type | Head | Loss | Metrics | Config suffix |
|---|---|---|---|---|
| Single-task arousal regression | `regression` | Smooth L1 | CCC, RMSE | `*_arousal.yaml` |
| Single-task 3-class classification | `classification` | Masked CE | Macro-F1, Balanced Acc | `*_state.yaml` |
| Multitask classification (state + trend) | `multitask_seq` | Masked multi-CE | per-task F1 | `*_multitask_state_trend.yaml` |
| **Mixed multitask (arousal reg + trend cls)** | `multitask_mixed_seq` | MSE + CE | CCC/RMSE + F1/Acc | `*_multitask_arousal_trend.yaml` |

All tasks share the same fusion backbone (LFT or CMA). Only the head, loss function, and label format differ.

---

## Model Architecture

### Overall Pipeline

```
┌──────────────────────────────────────────────────────────────────────┐
│                         Batch from DataModule                        │
│  x: {video: [B,T_v,2048], km: [B,T_k,25], telem: [B,T_t,D_t]}      │
│  mask: {video: [B,T_v], km: [B,T_k], telem: [B,T_t]}                │
│  y: [B,T] (regression) or {task: [B,T]} (multitask)                 │
└──────────┬───────────────────────┬───────────────────────┬───────────┘
           │                       │                       │
           ▼                       ▼                       ▼
   ┌───────────────┐     ┌─────────────────┐     ┌────────────────────┐
   │ VideoResNet2d │     │  KMStatEncoder  │     │  TelemStatPool     │
   │  Encoder      │     │  Encoder        │     │  Encoder           │
   │ [B,T_v,2048]  │     │ [B,T_k,25]      │     │ [B,T_t,D_t]        │
   │      ↓        │     │      ↓          │     │       ↓            │
   │ proj+LN+Drop  │     │  Linear+LN      │     │ Linear+LN          │
   │      ↓        │     │      ↓          │     │       ↓            │
   │ tokens[B,T,D] │     │ tokens[B,T,D]  │     │ tokens[B,T,D]      │
   │ pooled[B,D]   │     │ pooled[B,D]    │     │ pooled[B,D]        │
   └───────┬───────┘     └────────┬────────┘     └──────────┬─────────┘
           │                      │                         │
           └──────────────────────┴─────────────────────────┘
                                  │  z_dict: {mod: EncoderOut}
                                  ▼
               ┌─────────────────────────────────────────┐
               │         Late Fusion Transformer          │
               │                                          │
               │  For each modality:                      │
               │    tokens + pos_encoding + mod_embedding │
               │              ↓                           │
               │  Concatenate along time dim              │
               │    [B, T_v+T_k+T_t, D]                  │
               │              ↓                           │
               │  TransformerEncoder (N layers, Pre-LN)   │
               │    (self-attention + FFN)                 │
               │              ↓                           │
               │  Mask-aware pooling (mean/max/cls)        │
               │    → pooled [B, D]                       │
               │    → tokens [B, T_total, D]              │
               └─────────────────────┬───────────────────┘
                                     │  FusionOut
                                     ▼
                        ┌────────────────────────┐
                        │       Task Head         │
                        │  regression:  [B, 1]   │
                        │  classif:     [B, T, C]│
                        │  mixed:       dict[task]│
                        └────────────┬────────────┘
                                     │
                                     ▼
                          ┌──────────────────┐
                          │  Mask-aware Loss  │
                          │  + Metrics        │
                          └──────────────────┘
```

### Modality Encoders

All encoders implement the frozen `BaseEncoder` interface and return an `EncoderOut` TypedDict:

```python
EncoderOut = {
    "tokens": Tensor[B, T, D],   # per-timestep representations
    "pooled": Tensor[B, D],      # mask-aware mean of tokens
    "mask":   Tensor[B, T],      # bool, True = valid timestep
}
```

#### `video` / `resnet2d` — `VideoResNet2dEncoder`

Operates on **pre-extracted** per-frame ResNet-50 features (2048-dim). No CNN is run during training; feature extraction is done offline.

```
Input:  [B, T_v, 2048]
  → Linear(2048 → D) + LayerNorm + Dropout
  → tokens: [B, T_v, D]
  → pooled: mask-aware temporal mean  [B, D]
```

This keeps the training loop light while preserving temporal resolution for the fusion stage.

#### `km` / `stat` — `KMStatEncoder`

Lightweight projection for pre-computed keyboard/mouse statistical features (key-press rates, mouse velocity statistics, etc., computed in sliding windows).

```
Input:  [B, T_k, 25]
  → Linear(25 → D) + LayerNorm
  → tokens: [B, T_k, D]
  → pooled: mask-aware temporal mean  [B, D]
```

#### `telem` / `stat_pool` — `TelemStatPoolEncoder`

Game telemetry statistics (player health, score, position derivatives, etc.) encoded similarly to KM features with an additional pooling stage.

### Late Fusion Transformer (LFT)

The LFT is the core fusion module. It operates on the token sequences from all active modalities and produces a unified multimodal representation.

**Architecture detail:**

1. **Positional encoding** — each modality's token sequence independently gets a sinusoidal (or learnable) positional encoding to preserve temporal ordering.
2. **Modality embedding** — a learned per-modality embedding vector is added to distinguish modality identity after concatenation.
3. **Token concatenation** — all modality token sequences are concatenated along the time axis:
   ```
   tokens_concat = [tokens_video | tokens_km | tokens_telem]  # [B, T_total, D]
   ```
4. **Transformer Encoder** — `N` layers of Pre-LayerNorm Transformer encoder (self-attention + FFN), with a padding mask that marks invalid timesteps as padding positions.
5. **Pooling** — the fused sequence is aggregated into a global vector:
   - `mean` (default): mask-aware mean pooling over valid tokens
   - `max`: mask-aware max pooling
   - `cls`: a prepended learnable CLS token is used

**Key parameters (configurable via YAML):**

| Parameter | Default | Description |
|---|---|---|
| `d_model` | 512 | Hidden dimension |
| `nhead` | 8 | Number of attention heads |
| `num_layers` | 4 | Transformer encoder layers |
| `dim_feedforward` | 1024 | FFN hidden size |
| `dropout` | 0.1 | Dropout rate |
| `pos_encoding_type` | `sinusoidal` | `sinusoidal` or `learnable` |
| `pooling` | `mean` | `mean`, `max`, or `cls` |
| `input_token_merge` | `none` | Pre-fusion merge: `none`, `mean`, `linear` |
| `temporal_merge` | `none` | Post-fusion merge: `none`, `mean`, `linear` |

**Fusion variants:**

| Fusion name | Description |
|---|---|
| `lft` | Standard Late Fusion Transformer (tokens concat → shared Transformer) |
| `cma` | Cross-Modal Attention (directional cross-attention + self-attention) |
| `single` | Single-modality pass-through (no cross-modal attention) |
| `aligned_mean` | Temporally aligned mean pooling across modalities |

### Cross-Modal Attention (CMA)

CMA is an alternative fusion module that addresses a limitation of LFT: when modalities have very different sequence lengths (e.g., video has far more tokens than km), shorter-sequence modalities can be "drowned out" in LFT's shared self-attention.

**Key difference from LFT**: Instead of concatenating all tokens and relying on a single shared self-attention, CMA first uses **directional cross-modal attention** to let non-anchor modalities attend to an anchor modality (default: `video`), and then applies self-attention for final fusion.

**Architecture detail:**

```
Encoders → EncoderOut per modality
                │
                ▼
┌───────────────────────────────────────────────────┐
│           Cross-Modal Attention Phase              │
│                                                    │
│  anchor (video) ──────────────────────────┐        │
│                                           │        │
│  km tokens ─── CrossModalTransformer ◄────┘        │
│                  Q=km, K/V=video                   │
│                  (D layers)            ┌───┘        │
│  telem tokens ─ CrossModalTransformer ◄┘           │
│                  Q=telem, K/V=video                │
│                                                    │
│  (anchor tokens pass through unchanged)            │
└────────────────────┬──────────────────────────────┘
                     │
                     ▼
┌───────────────────────────────────────────────────┐
│          Self-Attention Refinement Phase            │
│                                                    │
│  Modality embeddings + concatenation               │
│    [B, T_v + T_k + T_t, D]                        │
│              ↓                                     │
│  TransformerEncoder (N_sa layers, Pre-LN)          │
│              ↓                                     │
│  Mask-aware pooling → pooled [B, D]                │
└───────────────────────────────────────────────────┘
```

**LFT vs CMA comparison:**

| Aspect | LFT | CMA |
|---|---|---|
| Cross-modal interaction | Implicit (shared self-attention) | Explicit directional cross-attention |
| Token handling | All tokens concatenated equally | Anchor modality serves as K/V source |
| Attention layers | `num_layers` SA layers | `cm_layers` CM layers + `sa_layers` SA layers |
| Temporal alignment | Via position encoding | Naturally handled (Q and K/V can differ in length) |
| Modality asymmetry | None | Anchor modality privileged as information source |

**Fallback behavior** — CMA degrades gracefully when modalities are absent:
- If the anchor modality is missing, cross-modal attention is skipped (equivalent to LFT)
- Single-modality input degrades to pure self-attention

**Key CMA parameters (configurable via YAML):**

| Parameter | Default | Description |
|---|---|---|
| `cm_layers` | 4 | Number of cross-modal attention layers |
| `sa_layers` | 2 | Number of self-attention refinement layers |
| `anchor_modality` | `video` | Modality used as K/V source for cross-attention |
| `nhead` | 8 | Number of attention heads |
| `dim_feedforward` | 1024 | FFN hidden size |
| `dropout` | 0.1 | Dropout rate |
| `pooling` | `mean` | `mean`, `max`, or `cls` |

### Task Heads

All heads implement `BaseHead` and accept a `FusionOut` dict as input.

| Head name | Output | Notes |
|---|---|---|
| `regression` | `[B, out_dim]` | Dense layer from pooled representation; `out_dim=1` for arousal |
| `classification` (seq) | `[B, T, num_classes]` | Per-timestep logits from token sequence |
| `multitask_seq` | `{task: [B, T, C]}` | Multiple classification branches |
| `multitask_mixed_seq` | `{arousal: [B,T,1], trend: [B,T,3]}` | Mixed regression + classification branches |

---

## Experiment Design

### Modality Combinations

All experiments are run across **7 modality combinations** to assess individual and complementary modality contributions:

| Combination | Config file |
|---|---|
| video only | `*_video_*.yaml` |
| km only | `*_km_*.yaml` |
| telem only | `*_telem_*.yaml` |
| video + km | `*_video_km_*.yaml` |
| video + telem | `*_video_telem_*.yaml` |
| km + telem | `*_km_telem_*.yaml` |
| video + km + telem | `*_video_km_telem_*.yaml` |

### Reproducibility

Each run uses a fixed random seed (default 42) controlling PyTorch, NumPy, and Python random state. Results are typically reported as the mean ± std over 3 seeds (0, 1, 42).

### Loss Functions

| Task | Loss | Notes |
|---|---|---|
| Arousal regression | Smooth L1 (configurable: MSE, CCC) | Applied only to valid (unmasked) timesteps |
| State/Trend classification | Masked Cross-Entropy | Ignores padded timesteps |
| Mixed multitask | `w_reg * MSE + w_cls * CE` | Per-task weights configurable |

### Evaluation Metrics

| Task | Primary | Secondary |
|---|---|---|
| Regression | CCC (Concordance Correlation Coefficient) | RMSE |
| Classification | Macro-F1 | Balanced Accuracy |
| Mixed multitask | `val_score_mixed = 0.5*CCC + 0.5*F1` | per-task metrics |

### Early Stopping

Patience-based early stopping monitors `val_ccc` (regression) or `val_score_mixed` (mixed multitask), restoring the best checkpoint automatically.

---

## Framework Design

The codebase uses an **"Interface + Registry + Config"** pattern to enable zero-modification extension.

### Frozen Interfaces (`src/core/types.py`)

Four abstract base classes define the immutable contracts:

```
BaseEncoder  → EncoderOut {tokens, pooled, mask}
BaseFusion   → FusionOut  {tokens, pooled}
BaseHead     → Tensor[B, out_dim]
BaseDataModule → produces Batch dicts
```

These interfaces **never change**. All new implementations must conform to them.

### Registry System (`src/core/registry.py`)

Modules self-register via decorators:

```python
@get_encoder_registry("km").register("stat")
class KMStatEncoder(BaseEncoder): ...

@FUSIONS.register("lft")
class LFTFusion(BaseFusion): ...
```

The runner looks up components by string key at runtime — no `if/else` chains, no hardcoded imports. Adding a new component requires only:

1. A new file implementing the relevant interface
2. A `@registry.register("name")` decorator
3. A config update: `model.fusion.name: my_new_fusion`

### Batch Format

All DataModules produce the standard `Batch` schema:

```python
{
    "x":    {modality: Tensor[B, T, D], ...},  # arbitrary modality subset
    "mask": {modality: Tensor[B, T],    ...},  # True = valid timestep
    "y":    Tensor[B, T] | {task: Tensor},     # labels
    "meta": {...},                             # session metadata (optional)
}
```

---

## Repository Structure

```
ProjectExperiment/
├── src/
│   ├── core/
│   │   ├── types.py          # Frozen interfaces & TypedDicts
│   │   ├── registry.py       # Plugin registration system
│   │   ├── runner.py         # Training orchestration
│   │   ├── config.py         # YAML config with inheritance
│   │   ├── logging.py        # Run directory management
│   │   └── seed.py           # Reproducibility
│   ├── data/
│   │   └── datamodules/
│   │       ├── amucs_seq.py              # Base sequence DataModule
│   │       └── amucs_seq_multitask.py    # Multitask extension
│   ├── models/
│   │   ├── encoders/
│   │   │   ├── km/           # stat, cnn1d
│   │   │   ├── video/        # resnet2d, emotieff
│   │   │   └── telem/        # stat_pool
│   │   ├── fusions/
│   │   │   ├── lft.py          # Late Fusion Transformer
│   │   │   ├── cma.py          # Cross-Modal Attention fusion
│   │   │   ├── single.py       # Single-modality pass-through
│   │   │   └── aligned_mean.py
│   │   ├── heads/
│   │   │   ├── regression.py
│   │   │   ├── multitask_seq.py
│   │   │   └── multitask_mixed_seq.py
│   │   └── components/       # Shared building blocks (pos encoding, etc.)
│   ├── losses/               # ccc, mse, multitask_mixed_seq_loss
│   └── metrics/              # ccc, rmse, macro_f1, balanced_acc
├── configs/
│   ├── base.yaml             # Global defaults
│   ├── amucs_seq_lft_*_multitask_arousal_trend.yaml   # LFT, 7 combos
│   ├── amucs_seq_cma_*_multitask_arousal_trend.yaml   # CMA, 7 combos
│   └── amucs_seq_lft_*_multitask_state_trend.yaml     # 7 combos
├── scripts/
│   ├── train.py              # Main entry point
│   ├── merge_arousal_reg_trend_labels.py
│   └── summarize.py          # Results aggregation
├── tests/
│   └── test_shapes.py        # Shape contract tests
├── docs/                     # Technical design documents
├── runs/                     # Training outputs (gitignored)
└── legacy/                   # Archived legacy code
```

---

## Installation

```bash
# Clone the repository
git clone <repo_url>
cd ProjectExperiment

# Install dependencies
pip install torch torchvision
pip install pyyaml numpy pandas scikit-learn tqdm pytest
```

**Requirements:**
- Python 3.10+
- PyTorch ≥ 1.9 (CUDA recommended)
- torchvision

---

## Data Preparation

### Feature Pre-extraction

All modality features must be pre-extracted and stored as `.npy` (or `.pt`) files organized by session:

```
data/features/aligned/
└── {session_stem}/
    ├── video_features.npy    # [T_v, 2048]   ResNet-50 frame features
    ├── km_features.npy       # [T_k, 25]     KM statistical features
    └── telem_features.npy    # [T_t, D_t]    Telemetry features
```

### Label Files

**Single-task regression** (arousal):
```json
{
  "<session_stem>": {
    "values": [0.12, 0.35, ...],
    "mask":   [true, true, ...]
  }
}
```

**Mixed multitask** (arousal regression + trend classification):

Use the provided merge script to generate the combined label file:

```bash
python scripts/merge_arousal_reg_trend_labels.py \
  --arousal /path/to/arousal_seq_z_perparticipant.json \
  --trend   /path/to/arousal_3trend_seq.json \
  --output  /path/to/arousal_reg_trend_seq.json
```

Output schema:
```json
{
  "<session_stem>": {
    "arousal": {"values": [0.12, 0.35, ...], "mask": [true, true, ...]},
    "trend":   {"values": [1, 2, 0, ...],    "mask": [true, true, ...]}
  }
}
```

### Split File

Train/val/test split in JSON:
```json
{
  "train": ["session_001", "session_002", ...],
  "val":   ["session_010", ...],
  "test":  ["session_020", ...]
}
```

---

## Usage

### Training

```bash
# Single-task arousal regression, video + km
python scripts/train.py \
  --config configs/base.yaml \
  --override \
    data.data_root=/path/to/features/aligned \
    data.labels_path=/path/to/arousal_labels.json \
    data.split_path=/path/to/session_tvt.json \
    train.seed=0

# Mixed multitask (arousal regression + trend classification), all 3 modalities
python -u scripts/train.py \
  --config configs/amucs_seq_lft_video_km_telem_multitask_arousal_trend.yaml \
  --override \
    data.data_root=/path/to/features/aligned \
    data.labels_seq_path=/path/to/arousal_reg_trend_seq.json \
    data.split_path=/path/to/session_tvt.json \
    train.seed=42
```

### Configuration System

Configs use hierarchical YAML with `_base_` inheritance. CLI overrides use dot-notation:

```bash
python scripts/train.py \
  --config configs/base.yaml \
  --override model.fusion.name=single model.fusion.num_layers=2 train.seed=1
```

**Key config sections (`configs/base.yaml`):**

```yaml
data:
  name: amucs                    # DataModule registry key
  modalities: [video, km]        # Active modality list
  normalize: true                # Per-participant z-score normalization

model:
  d_model: 512                   # Shared model dimension (all components)
  encoders:
    video:
      name: resnet2d             # Encoder registry key
      feature_dim: 2048
      dropout: 0.1
    km:
      name: stat
      feature_dim: 25
  fusion:
    name: lft                    # Fusion registry key
    nhead: 8
    num_layers: 4
    dim_feedforward: 1024
    dropout: 0.1
    pooling: mean
  head:
    name: regression             # Head registry key
    hidden_dim: 128
    out_dim: 1

train:
  loss: smooth_l1                # Loss registry key
  optimizer:
    name: adamw
    lr: 1.0e-4
    weight_decay: 0.01
  batch_size: 8
  epochs: 50
  early_stopping:
    patience: 10
    metric: val_ccc
    mode: max

eval:
  metrics: [ccc, rmse]

device: auto                     # auto / cuda / cpu
```

### Modality Combinations

To run a single-modality experiment, override `data.modalities` and set `model.fusion.name=single`:

```bash
python scripts/train.py \
  --config configs/base.yaml \
  --override data.modalities=[km] model.fusion.name=single
```

### Mixed Multitask (Arousal + Trend)

Seven pre-built configs cover all modality combinations:

| Modalities | Config |
|---|---|
| video | `configs/amucs_seq_lft_video_multitask_arousal_trend.yaml` |
| km | `configs/amucs_seq_lft_km_multitask_arousal_trend.yaml` |
| telem | `configs/amucs_seq_lft_telem_multitask_arousal_trend.yaml` |
| video + km | `configs/amucs_seq_lft_video_km_multitask_arousal_trend.yaml` |
| video + telem | `configs/amucs_seq_lft_video_telem_multitask_arousal_trend.yaml` |
| km + telem | `configs/amucs_seq_lft_km_telem_multitask_arousal_trend.yaml` |
| video + km + telem | `configs/amucs_seq_lft_video_km_telem_multitask_arousal_trend.yaml` |

The mixed multitask head outputs:
- `arousal` branch: `[B, T, 1]` — continuous regression
- `trend` branch: `[B, T, 3]` — 3-class logits (decreasing / stable / increasing)

Early stopping monitors `val_score_mixed = 0.5 * val_ccc_arousal + 0.5 * val_macro_f1_trend`.

### Notebook Workflow

The main notebook `train.ipynb` contains pre-configured cells for batch experiments:

| Cell | Content |
|---|---|
| Cell 26 | State + trend multitask classification (7 combos × 3 seeds) |
| Cell 27 | Mixed multitask: arousal regression + trend classification (7 combos × 3 seeds) |
| Cell 28 | Lag sweep analysis for regression experiments |

---

## Adding New Components

### New Encoder

```python
# src/models/encoders/km/transformer.py
from src.core.registry import get_encoder_registry
from src.core.types import BaseEncoder, EncoderOut

@get_encoder_registry("km").register("transformer")
class KMTransformerEncoder(BaseEncoder):
    def __init__(self, cfg):
        super().__init__()
        ...

    def forward(self, x, mask=None) -> EncoderOut:
        ...
        return EncoderOut(tokens=tokens, pooled=pooled, mask=mask)
```

Then in config: `model.encoders.km.name: transformer`. No other files need changing.

### New Fusion Method

```python
# src/models/fusions/mult.py
from src.core.registry import FUSIONS
from src.core.types import BaseFusion, FusionOut

@FUSIONS.register("mult")
class MulTFusion(BaseFusion):
    def forward(self, z_dict, mask_dict) -> FusionOut:
        # must handle arbitrary modality subsets
        ...
```

### New Modality

1. DataModule outputs `x["new_mod"]` and `mask["new_mod"]`
2. Create `src/models/encoders/new_mod/name.py` and register
3. Add to config: `data.modalities: [..., new_mod]` and `model.encoders.new_mod: {...}`
4. Fusion handles it automatically (dynamic key iteration)

### Extension Summary

| What to add | Files to create | Files to change |
|---|---|---|
| New encoder | `src/models/encoders/{mod}/{name}.py` | Config only |
| New modality | Encoder file + DataModule extension | Config only |
| New fusion | `src/models/fusions/{name}.py` | Config only |
| New head | `src/models/heads/{name}.py` | Config only |
| New loss | `src/losses/{name}.py` | Config only |
| New dataset | `src/data/datamodules/{name}.py` | Config only |

---

## Testing

Tests focus on **shape contracts** to ensure all components conform to the frozen interfaces and that new additions don't break existing functionality.

```bash
# Run all tests
pytest tests/

# Verbose output
pytest tests/ -v

# Specific test
pytest tests/test_shapes.py -v
```

Key test categories in `tests/test_shapes.py`:
- Encoder output shape verification (tokens/pooled/mask dimensions)
- Fusion handling of 1..N arbitrary modality subsets
- Head output shape validation
- End-to-end forward pass (DataModule → Encoder → Fusion → Head → Loss)
- Loss returns scalar; metrics return float

---

## Output Structure

Each training run creates a self-contained directory:

```
runs/{timestamp}__{dataset}__{fusion}__{modalities}__seed{seed}/
├── config.yaml        # Complete merged configuration (for exact reproducibility)
├── seed.txt           # Random seed used
├── git_commit.txt     # Git commit hash at training time
├── ckpt_best.pt       # Best model checkpoint (by val metric)
├── ckpt_last.pt       # Checkpoint at final epoch
└── metrics.json       # All tracked metrics
```

**Example run directory name:**
```
2026-02-04_14-30-22__amucs__lft__video_km__seed42/
```

**`metrics.json` example:**
```json
{
  "best_val_ccc": 0.72,
  "best_val_epoch": 35,
  "test_ccc": 0.68,
  "test_rmse": 0.21,
  "total_epochs": 50,
  "early_stopped": true
}
```

---

## Results Aggregation

Aggregate all run results into a leaderboard CSV:

```bash
python scripts/summarize.py
```

Produces `leaderboard.csv` with one row per run, including dataset, fusion, modalities, seed, and all metrics. Useful for comparing modality ablations and hyperparameter effects across experiments.

---

## Notes

- Masked timesteps (`mask=False`) are fully excluded from loss computation and metric calculation.
- Mixed multitask supports per-task metrics and a configurable composite metric (`val_score_mixed`) for early stopping to prevent single-task dominance.
- All existing single-task and state+trend multitask experiments are unaffected by the mixed multitask additions.
- `modality_dropout` (set in config) randomly zeroes a modality's mask during training to improve robustness to missing modalities at inference time.
- For detailed CMA design rationale, see `docs/crossmodal_attention_fusion_design.md`.
