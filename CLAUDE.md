# CLAUDE.md - AI Assistant Guide for ProjectExperiment

## Project Overview

This is a **machine learning research project** for **arousal prediction using keyboard and mouse (KM) behavioral patterns**. The project is part of the AMuCS (Arousal Measurement using Computational Signal Processing) research initiative, analyzing user engagement/stress levels through keyboard and mouse event sequences.

**Key Capabilities:**
- Encoding raw keyboard/mouse events into 25-dimensional statistical feature sequences
- Training transformer-based models for arousal regression
- Session-level and window-level dataset management
- Train/validation splitting with reproducible seeds

## Repository Structure

```
ProjectExperiment/
├── encoder/                              # Event encoding and feature extraction
│   ├── face/                             # Face features (placeholder, not implemented)
│   │   └── extract_face_features.py
│   └── km/                               # Keyboard-mouse encoding
│       ├── km_encoder_stat.py            # Statistical encoder (events → [T, 25] features)
│       ├── km_encoder_1dCNN.py           # 1D-CNN encoder variant
│       ├── extract_km_features.py        # Main feature extraction pipeline
│       ├── build_arousal_labels.py       # Generate labels from ranktrace CSVs
│       ├── filter_arousal_ranktrace.py   # Session filtering utility
│       └── check_km_features.py          # Feature inspection/debugging tool
│
└── lft-va/                               # Training framework
    ├── src/lft_va/                       # Main package
    │   ├── datasets/
    │   │   └── km_window_dataset.py      # Windowed dataset loader
    │   ├── models/
    │   │   └── km_transformer_min.py     # Transformer regressor model
    │   └── utils/
    │       └── config.py                 # Configuration (placeholder)
    ├── scripts/
    │   ├── train_km_arousal_first.py     # Main training script
    │   └── build_km_arousal_split.py     # Train/val split builder
    ├── configs/
    │   └── default.yaml                  # Configuration (placeholder)
    └── data/                             # Data directory (not in repo)
        ├── features/amucs/km/            # Extracted .pt feature files
        ├── labels_arousal.json           # Session arousal labels
        ├── splits/km_arousal_split.json  # Train/val session lists
        └── km_input_stats.json           # Input normalization stats
```

## Quick Start Commands

### Data Preparation Pipeline
```bash
# 1. Find sessions with arousal labels
python encoder/km/filter_arousal_ranktrace.py --amucs_root /path/to/amucs --output arousal_sessions.json

# 2. Build aggregated labels
python encoder/km/build_arousal_labels.py --amucs_root /path/to/amucs --output labels_arousal.json

# 3. Extract KM features
python encoder/km/extract_km_features.py --amucs_root /path/to/amucs --out_dir lft-va/data/features/amucs/km --encoder stat --dt 0.2

# 4. Create train/val split
python lft-va/scripts/build_km_arousal_split.py --seed 42 --train_ratio 0.8

# 5. Inspect extracted features
python encoder/km/check_km_features.py --km_dir lft-va/data/features/amucs/km
```

### Training
```bash
# Train with statistical encoder (default)
python lft-va/scripts/train_km_arousal_first.py --km_encoder stat

# Train with CNN encoder
python lft-va/scripts/train_km_arousal_first.py --km_encoder cnn
```

**Training Outputs** (in `lft-va/outputs/{km_arousal_first|km_arousal_cnn}/{timestamp}/`):
- `best.pt` - Best validation checkpoint
- `last.pt` - Final epoch checkpoint
- `metrics.json` - Train/val loss history
- `loss_curve.png` - Loss visualization
- `train.log` - Training log

## Architecture Overview

### Data Flow
```
Raw CSV Files (keyboard, mousebuttons, mouseposition)
        │
        ▼
KMStatEncoder (dt=0.2s bins → 25 statistical features)
        │
        ▼
Feature Tensors [T, 25] saved as .pt files
        │
        ▼
KMWindDataset (L=300 windows, stride=150, normalized)
        │
        ▼
DataLoader (batch_size=8)
        │
        ▼
KMTransformerRegressor
    ├── Input projection (Linear or 1D-CNN) → [B, L, 64]
    ├── TransformerEncoder (2 layers, 4 heads)
    ├── Masked mean pooling → [B, 64]
    └── Linear head → [B, 1] (arousal prediction)
        │
        ▼
Smooth L1 Loss → Adam optimizer (lr=1e-3)
```

### KM Feature Dimensions (25 features)
| Index | Feature Name | Description |
|-------|--------------|-------------|
| 0 | key_down_count | Keyboard key press count per bin |
| 1 | key_up_count | Keyboard key release count per bin |
| 2 | mouse_move_event_count | Mouse movement event count |
| 3 | mouse_move_distance_sum | Total mouse travel distance (pixels) |
| 4 | mouse_speed_mean | Average mouse speed |
| 5 | mouse_speed_max | Maximum mouse speed |
| 6 | mouse_button_down_count | Total mouse button presses |
| 7 | left_button_down_count | Left button presses |
| 8 | right_button_down_count | Right button presses |
| 9 | mouse_button_up_count | Total mouse button releases |
| 10 | left_button_up_count | Left button releases |
| 11 | right_button_up_count | Right button releases |
| 12 | scroll_event_count | Scroll wheel event count |
| 13 | scroll_delta_sum | Total scroll delta |
| 14 | inter_key_interval_mean | Mean time between keystrokes |
| 15 | mouse_dx_sum | Cumulative horizontal movement |
| 16 | mouse_dy_sum | Cumulative vertical movement |
| 17 | mouse_accel_mean | Mean mouse acceleration |
| 18 | key_down_delta | Change in key_down_count |
| 19 | key_up_delta | Change in key_up_count |
| 20 | key_down_rate | Key presses per second |
| 21 | mouse_click_rate | Clicks per second |
| 22 | scroll_rate | Scrolls per second |
| 23 | mouse_move_rate | Mouse moves per second |
| 24 | unique_key_count | Distinct keys pressed per bin |

## Code Conventions

### Python Style
- **Type hints**: Modern Python annotations with `from __future__ import annotations`
- **Docstrings**: NumPy-style with bilingual (English/Chinese) documentation
- **Path handling**: Always use `pathlib.Path`, not `os.path`
- **Private functions**: Prefix with underscore (e.g., `_set_seed()`, `_add_src_to_path()`)

### Naming Conventions
- **Files**: snake_case (e.g., `km_encoder_stat.py`)
- **Classes**: PascalCase (e.g., `KMTransformerRegressor`, `KMWindDataset`)
- **Functions/Methods**: snake_case (e.g., `encode()`, `_build_index()`)
- **Constants**: UPPER_SNAKE_CASE (if any)

### Import Organization
```python
from __future__ import annotations  # Always first

import json                         # Standard library
import random
from pathlib import Path

import torch                        # Third-party
import numpy as np
from torch import nn

from lft_va.datasets import ...     # Local imports
```

### Device Handling
```python
# Use configurable device strings
device = "cpu"  # or "cuda"
tensor = torch.tensor(data, device=device)
model = torch.load(path, map_location="cpu")  # Always load to CPU first
```

## Dependencies

**Required packages** (inferred from imports):
- `torch` (PyTorch) - Neural networks and tensor operations
- `numpy` - Numerical computing
- `pandas` - CSV reading (for raw data processing)
- `matplotlib` - Training visualization

**Python version**: 3.10+ (for type hint syntax)

## Key Implementation Details

### KMStatEncoder (`encoder/km/km_encoder_stat.py`)
- Bins events into fixed time windows (default `dt=0.2` seconds)
- Supports two input formats: direct event lists or table-like CSV data
- Returns `{"features": [T, 25], "mask": [T], "meta": {...}}`

### KMWindDataset (`lft-va/src/lft_va/datasets/km_window_dataset.py`)
- Creates sliding windows from session features (L=300, stride=150)
- Computes input normalization stats on training set, applies to val
- Returns `{"km": [L, D], "km_mask": [L], "y": [1], "stem": str}`

### KMTransformerRegressor (`lft-va/src/lft_va/models/km_transformer_min.py`)
- Input projection: Linear (`stat`) or 1D-CNN (`cnn`) to d_model=64
- Transformer: 2 encoder layers, 4 attention heads, batch_first=True
- Output: Masked mean pooling → linear head → scalar prediction

## Testing and Validation

**No formal test suite exists.** Validate changes by:
1. Running `check_km_features.py` to inspect feature statistics
2. Checking training converges (`loss_curve.png`)
3. Comparing `metrics.json` across runs

## Common Tasks for AI Assistants

### Adding a New Feature to the Encoder
1. Modify `KMStatEncoder.encode()` in `encoder/km/km_encoder_stat.py`
2. Update the feature count (currently 25) and `feature_names` list
3. Update model's `d_in` parameter or let it auto-detect

### Adding a New Model Architecture
1. Create new file in `lft-va/src/lft_va/models/`
2. Follow the pattern: `__init__` accepts d_in, forward takes (km, km_mask)
3. Import and use in training script

### Modifying Training Hyperparameters
Key locations in `train_km_arousal_first.py`:
- Epochs: line 117 (`epochs = 3`)
- Batch size: lines 101-102 (`batch_size=8`)
- Learning rate: line 107 (`lr=1e-3`)
- Optimizer: line 107 (Adam)
- Loss function: line 56 (`smooth_l1_loss`)

### Adding New Data Sources
1. Extend `KMStatEncoder._events_from_tables()` for new CSV formats
2. Update `extract_km_features.py` to handle new directory structures
3. Update `build_arousal_labels.py` for new label sources

## Project Status

**Stage**: Early-stage research prototype

**Known Limitations**:
- Placeholder files: `face/extract_face_features.py`, `configs/default.yaml`, `requirements.txt`
- No formal testing framework
- Hardcoded hyperparameters (should move to config)
- Limited error handling in data loading

## File Locations Reference

| Purpose | Path |
|---------|------|
| Statistical encoder | `encoder/km/km_encoder_stat.py` |
| CNN encoder | `encoder/km/km_encoder_1dCNN.py` |
| Feature extraction | `encoder/km/extract_km_features.py` |
| Dataset loader | `lft-va/src/lft_va/datasets/km_window_dataset.py` |
| Transformer model | `lft-va/src/lft_va/models/km_transformer_min.py` |
| Training script | `lft-va/scripts/train_km_arousal_first.py` |
| Split builder | `lft-va/scripts/build_km_arousal_split.py` |
