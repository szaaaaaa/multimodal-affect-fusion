# AGENTS.md - AI Assistant Guide for ProjectExperiment

This document provides guidance for AI assistants working with this codebase.

## Project Overview

This is an **extensible multimodal deep learning framework** for **Valence/Arousal (emotion) prediction** supporting multiple fusion architectures (EFT, MFT, LFT, CMA, etc.). The project implements a plugin-based architecture that supports arbitrary modalities through dynamic composition while maintaining frozen stable interfaces.

**Current Modalities:**
- **Video**: CLIP ViT-based frame features (768-dim, feature file: `video_clip`). Note: the consumer encoder class is named `VideoResNet2dEncoder` / registered as `resnet2d` for legacy reasons, but it does NOT extract ResNet features — it consumes pre-extracted CLIP features.
- **Keyboard/Mouse (KM)**: Behavioral interaction statistical features (25-dim)
- **Telemetry**: In-game event statistical features (109-dim)

## Repository Structure

```
ProjectExperiment/
├── src/                       # Main source code (extensible architecture)
│   ├── core/                  # Stable infrastructure (rarely changes)
│   │   ├── types.py           # Frozen abstract interfaces & TypedDicts
│   │   ├── registry.py        # Plugin registration system
│   │   ├── runner.py          # Training orchestration
│   │   ├── config.py          # YAML configuration with inheritance
│   │   ├── logging.py         # Experiment management
│   │   └── seed.py            # Reproducibility control
│   ├── data/                  # Data modules and transforms
│   │   └── datamodules/       # DataModule implementations
│   ├── models/                # Model components
│   │   ├── encoders/          # Per-modality encoders (km/, video/)
│   │   ├── fusions/           # Multimodal fusion methods
│   │   ├── heads/             # Prediction heads
│   │   └── components/        # Reusable building blocks
│   ├── losses/                # Loss functions (ccc, mse)
│   └── metrics/               # Evaluation metrics (ccc, rmse)
├── configs/                   # YAML configuration files
│   ├── base.yaml              # Global defaults
│   ├── experiments/           # Experiment-specific overrides
│   └── sweeps/                # Hyperparameter grid search configs
├── scripts/                   # Training and utility scripts
│   ├── train.py               # Main training entry point
│   └── summarize.py           # Results aggregation
├── tests/                     # Shape contract tests
├── docs/                      # Technical documentation
├── runs/                      # Training outputs (gitignored)
├── encoder/                   # Legacy encoders (backward compat)
└── legacy/                    # Archived legacy code and configs
```

## Architecture Philosophy

The codebase uses an **"Interface + Registry + Config"** pattern:

### Frozen Interfaces (`src/core/types.py`)

Four abstract base classes that **NEVER change**:
- `BaseEncoder` → returns `EncoderOut`
- `BaseFusion` → returns `FusionOut`
- `BaseHead` → returns `Tensor[B, out_dim]`
- `BaseDataModule` → produces `Batch` dicts

Three TypedDicts for shape contracts:
- `EncoderOut`: `{tokens: [B,T,D], pooled: [B,D], mask: [B,T]}`
- `FusionOut`: `{tokens: [B,T,D]|None, pooled: [B,D]}`
- `Batch`: `{x: {mod: Tensor}, mask: {mod: Tensor}, y: Tensor, meta: {}}`

### Registry System (`src/core/registry.py`)

All modules self-register via decorators:

```python
# Per-modality encoder registration
@get_encoder_registry("km").register("stat")
class KMStatEncoder(BaseEncoder): ...

# Global registry registration
@FUSIONS.register("eft")
class EFTFusion(BaseFusion): ...
```

Global registries: `FUSIONS`, `HEADS`, `LOSSES`, `METRICS`, `DATAMODULES`
Per-modality registries: `get_encoder_registry("modality_name")`

## Common Commands

### Training

```bash
# Train with base config
python scripts/train.py --config configs/base.yaml

# Train with experiment config
python scripts/train.py --config configs/experiments/video_km_eft.yaml

# Train with CLI overrides
python scripts/train.py --config configs/base.yaml --override model.fusion.name=single train.seed=0
```

### Testing

```bash
# Run all tests
pytest tests/

# Run specific test file
pytest tests/test_shapes.py

# Run with verbose output
pytest tests/ -v
```

### Results Aggregation

```bash
python scripts/summarize.py
```

## Configuration System

Configs use hierarchical YAML with `_base_` inheritance:

```yaml
# configs/experiments/km_single.yaml
_base_: ../base.yaml

data:
  modalities: [km]

model:
  fusion:
    name: single
```

### Key Config Sections

```yaml
data:
  name: amucs                    # DataModule name (registered)
  modalities: [video, km]        # Dynamic modality list
  normalize: true                # Enable z-score normalization

model:
  d_model: 512                   # Shared model dimension
  encoders:
    video:
      name: resnet2d             # Encoder name
      feature_dim: 2048          # Input feature dimension
    km:
      name: stat
      feature_dim: 25
  fusion:
    name: eft                    # Fusion method (eft/mft/lft/cma/...)
    nhead: 8
    num_layers: 4
  head:
    name: regression
    out_dim: 1                   # 1=single, 2=valence+arousal

train:
  loss: ccc
  epochs: 50
  early_stopping:
    patience: 10
    metric: val_ccc
    mode: max

device: auto                     # auto/cuda/cpu
```

## Adding New Components

### Adding a New Encoder

1. Create file: `src/models/encoders/{modality}/{name}.py`
2. Implement `BaseEncoder` interface
3. Register with decorator:
   ```python
   @get_encoder_registry("modality").register("name")
   class MyEncoder(BaseEncoder):
       def forward(self, x, mask=None) -> EncoderOut:
           ...
   ```
4. Add test case to `tests/test_shapes.py`
5. Update config to use: `model.encoders.{modality}.name: myname`

### Adding a New Fusion Method

1. Create file: `src/models/fusions/{name}.py`
2. Implement `BaseFusion` interface (must handle arbitrary modality subsets)
3. Register: `@FUSIONS.register("name")`
4. Add test case for arbitrary modality subsets
5. Update config: `model.fusion.name: myname`

### Adding a New Modality

1. DataModule must output `x["modality"]` and `mask["modality"]`
2. Create encoder in `src/models/encoders/{modality}/`
3. Register encoder with `@get_encoder_registry("modality").register("name")`
4. Add to config: `data.modalities: [..., modality]`
5. Fusion automatically handles new modality (tested for arbitrary subsets)

## Code Conventions

### Naming Patterns

| Component | Pattern | Example |
|-----------|---------|---------|
| Encoder | `{Modality}{Method}Encoder` | `KMStatEncoder`, `VideoResNet2dEncoder` |
| Fusion | `{Method}Fusion` | `EFTFusion`, `LFTFusion`, `MFTFusion` |
| Head | `{Task}Head` | `RegressionHead` |
| DataModule | `{Dataset}DataModule` | `AMuCSDataModule` |
| Loss | `{Method}Loss` | `CCCLoss` |
| Metric | `{Method}Metric` | `RMSEMetric` |

### Import Pattern

Modules auto-register when imported. `runner.py` explicitly imports all:

```python
import src.models.encoders.km      # Triggers @register decorators
import src.models.encoders.video
import src.models.fusions
import src.models.heads
import src.losses
import src.metrics
import src.data.datamodules
```

### Type Hints

- Use `from __future__ import annotations`
- Use TypedDict for shape contracts
- Use Optional, Dict, List from typing

## Testing Philosophy

Tests focus on **shape contracts**, not unit tests:
- Verify all modules conform to frozen interfaces
- Test arbitrary modality subsets to ensure generalization
- Prevent future extensions from breaking existing functionality

Key test categories in `tests/test_shapes.py`:
- Encoder output contract verification
- Fusion handling of 1..N modalities
- Head output shape validation
- End-to-end forward passes
- Loss returns scalar, metrics return float

## Output Directory Structure

Training creates standardized run directories:

```
runs/
└── 2026-02-04_14-30-22__amucs__lft__video_km__seed42/
    ├── config.yaml              # Complete merged config
    ├── metrics.json             # Final metrics
    ├── seed.txt                 # Random seed used
    ├── git_commit.txt           # Git commit hash
    ├── ckpt_best.pt             # Best model weights
    └── ckpt_last.pt             # Final model weights
```

## Data Flow

```
Batch from DataModule
  {x: {mod: [B,T,D]}, mask: {mod: [B,T]}, y: [B,1]}
          │
          ▼
    Per-modality Encoders → EncoderOut
          │
          ▼
    Fusion (handles arbitrary subsets) → FusionOut
          │
          ▼
    Head → Predictions [B, out_dim]
          │
          ▼
    Loss → Scalar
```

## Key Design Principles

1. **Frozen Interfaces**: 4 ABCs + 3 TypedDicts form the immutable contract. Future extensions NEVER change these.

2. **Plugin Registration**: All implementations self-register via decorators. No hardcoded component selection.

3. **Configuration-Driven**: YAML specifies component names as strings. Runner looks them up in registries.

4. **Dynamic Modality Handling**: Config specifies modalities, runner builds encoders dynamically, fusion handles arbitrary subsets.

5. **Batch Format Unification**: All data modules produce the standard Batch schema.

6. **Zero-Modification Extension**: New modules require no changes to core files (train.py, runner.py, etc.)

## Dependencies

- PyTorch >= 1.9
- torchvision (for video feature extraction)
- PyYAML
- pytest (for testing)

## Documentation

- `docs/extensible_multimodal_framework.md` - Technical design of plugin architecture
- `docs/late_fusion_transformer_architecture.md` - Fusion architecture specification

## Troubleshooting

### Module Not Found in Registry
- Ensure the module file is imported (check `runner.py` imports)
- Verify the `@register` decorator is applied correctly
- Check the registry name matches config

### Shape Mismatch Errors
- Run `pytest tests/test_shapes.py` to verify interface compliance
- Check `d_model` is consistent across encoder, fusion, and head
- Verify input `feature_dim` matches actual data dimensions

### Config Inheritance Issues
- Ensure `_base_` path is relative to current config file
- CLI overrides use dot notation: `--override model.fusion.name=single`

