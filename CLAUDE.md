# CLAUDE.md - AI Assistant Guidelines

This document provides context and conventions for AI assistants working on the ProjectExperiment codebase.

## Project Overview

**Domain**: Affective Computing / Emotion Recognition
**Goal**: Predict emotional arousal levels from keyboard and mouse behavioral data
**Dataset**: AMuCS (multimodal affective dataset)
**Task Type**: Regression (continuous arousal value prediction)

### Core Hypothesis

User behavior patterns (key press frequency, mouse movement speed) correlate with emotional states:
- High arousal (tension/excitement) → faster keyboard input, more frequent mouse movement
- Low arousal (relaxation/boredom) → slower, less frequent interactions

## Repository Structure

```
ProjectExperiment/
├── CLAUDE.md                # This file - AI assistant guidelines
├── ARCHITECTURE.md          # Detailed architecture docs (Chinese)
│
├── encoder/                 # Feature extraction module
│   └── km/                  # Keyboard & Mouse encoder (core)
│       ├── km_encoder_stat.py      # Statistical feature encoder (25 dims)
│       ├── km_encoder_1dCNN.py     # Alternative CNN encoder
│       ├── extract_km_features.py  # Batch feature extraction
│       ├── build_arousal_labels.py # Label extraction from CSV
│       ├── filter_arousal_ranktrace.py # Dataset filtering
│       └── check_km_features.py    # Feature validation tool
│
└── lft-va/                  # Late Fusion Transformer training framework
    ├── data/                # Runtime-generated data (git-ignored)
    │   ├── features/amucs/km/   # .pt feature files
    │   ├── splits/              # Train/val split JSON
    │   ├── labels_arousal.json  # Arousal labels
    │   └── km_input_stats.json  # Normalization parameters
    ├── outputs/             # Training outputs (git-ignored)
    ├── src/lft_va/
    │   ├── datasets/
    │   │   └── km_window_dataset.py  # Windowed dataset class
    │   └── models/
    │       └── km_transformer_min.py # Transformer model
    └── scripts/
        ├── build_km_arousal_split.py # Create train/val split
        └── train_km_arousal_first.py # Training script
```

## Key Files for Common Tasks

| Task | Primary File(s) |
|------|-----------------|
| Modify model architecture | `lft-va/src/lft_va/models/km_transformer_min.py` |
| Change dataset loading | `lft-va/src/lft_va/datasets/km_window_dataset.py` |
| Adjust training params | `lft-va/scripts/train_km_arousal_first.py` |
| Add/modify features | `encoder/km/km_encoder_stat.py` |
| Debug data issues | Use `encoder/km/check_km_features.py` |

## Development Workflow

### Data Pipeline (if starting from raw AMuCS)

```bash
# 1. Filter sessions with arousal annotations
python encoder/km/filter_arousal_ranktrace.py --root /path/to/AMuCS

# 2. Build arousal labels
python encoder/km/build_arousal_labels.py

# 3. Extract keyboard/mouse features
python encoder/km/extract_km_features.py

# 4. Create train/val split
python lft-va/scripts/build_km_arousal_split.py
```

### Training

```bash
# Using default statistical encoder
python lft-va/scripts/train_km_arousal_first.py

# Using CNN encoder
python lft-va/scripts/train_km_arousal_first.py --km_encoder cnn
```

### Loading Trained Models

```python
import torch
from lft_va.models.km_transformer_min import KMTransformerRegressor

checkpoint = torch.load("path/to/best.pt")
model = KMTransformerRegressor(d_in=25)
model.load_state_dict(checkpoint["model"])
model.eval()
```

## Technical Details

### Training Hyperparameters

| Parameter | Value |
|-----------|-------|
| batch_size | 8 |
| learning_rate | 1e-3 |
| epochs | 3 |
| optimizer | Adam |
| loss | smooth_l1_loss (Huber) |
| seed | 123 |

### Data Parameters

| Parameter | Value | Description |
|-----------|-------|-------------|
| dt | 0.2s | Time bin size |
| win_len | 300 | Window length (60 seconds) |
| stride | 150 | Stride (30 seconds, 50% overlap) |
| feature_dim | 25 | Features per time bin |

### Model Architecture

```
Input [B, 300, 25] → Input Projection [B, 300, 64] → Transformer Encoder (2 layers, 4 heads)
→ Masked Average Pooling [B, 64] → Linear Head → Output [B, 1]
```

Total parameters: ~100K

## Code Conventions

### Python Style

- Python 3.9+ (uses `|` union type syntax)
- Use `pathlib.Path` for file operations (not string paths)
- Type hints on all function signatures
- NumPy-style docstrings (Parameters/Returns format)

### Module Pattern

Scripts use a path setup pattern for imports:
```python
def _add_src_to_path():
    import sys
    src = Path(__file__).resolve().parents[1] / "src"
    if str(src) not in sys.path:
        sys.path.insert(0, str(src))
```

### Data Structures

Event data uses frozen dataclasses:
```python
@dataclass(frozen=True)
class KMEvent:
    t: float              # timestamp
    kind: str             # event type
    x, y: Optional[float] # mouse position
    button: Optional[str] # mouse button
    scroll: Optional[float]
    key: Optional[str]
```

Supported event kinds: `key_down`, `key_up`, `mouse_move`, `mouse_click`, `mouse_button_up`, `mouse_scroll`

### File Formats

- **Features**: PyTorch tensors (`.pt`) containing `{features, mask, meta}`
- **Labels/Splits**: JSON files
- **Configs**: YAML (mostly placeholder for now)

## Critical Implementation Details

### Mask Semantics

This is a common source of bugs. Be careful:
- **Dataset masks**: `True` = valid position, `False` = padding
- **Transformer masks**: `True` = position to ignore (inverted)
- Code inverts for Transformer: `src_key_padding_mask=~km_mask`

### Normalization Flow

1. Training set computes mean/std and saves to `km_input_stats.json`
2. Validation set loads and reuses training statistics (standard ML practice)
3. Never compute stats from validation data

### Time Binning Logic

- Bin index formula: `i = int((t - t0) / dt)`
- Boundary handling: clamp to `[0, T-1]`
- dt = 0.2 seconds (200ms per bin)

### Feature Encoder Input Flexibility

`KMStatEncoder.encode()` accepts two formats:
1. Direct events: `{"events": [...], "t0": ..., "t1": ...}`
2. Table format: `{"keyboard": [...], "mousebuttons": [...], "mouseposition": [...]}`

## 25-Dimensional Feature Set

| Index | Feature | Type |
|-------|---------|------|
| 0-1 | key_down/up_count | Count |
| 2 | mouse_move_event_count | Count |
| 3-5 | mouse_move_distance_sum, speed_mean, speed_max | Distance/Speed |
| 6-11 | mouse_button_*_count (all, left, right) | Count |
| 12-13 | scroll_event_count, scroll_delta_sum | Count |
| 14 | inter_key_interval_mean | Time |
| 15-16 | mouse_dx/dy_sum | Distance |
| 17 | mouse_accel_mean | Acceleration |
| 18-19 | key_down/up_delta | Delta |
| 20-23 | key_down_rate, mouse_click_rate, scroll_rate, mouse_move_rate | Rate |
| 24 | unique_key_count | Count |

## Dependencies

Core stack (requirements.txt files are empty but code requires):
- PyTorch (torch, nn, DataLoader)
- NumPy
- Pandas (CSV handling)
- Matplotlib (loss curve visualization)

## Git Workflow

- Data directories (`lft-va/data/`, `lft-va/outputs/`) are git-ignored
- Model weights (`.pt` files) are git-ignored
- Main documentation: `ARCHITECTURE.md` (Chinese, very detailed)

## Testing

- No formal test suite currently
- Validation happens during training loop on held-out split
- Use `check_km_features.py` for feature file validation
- Run training with small subset for quick sanity checks

## Output Structure

Training creates timestamped output directories:
```
outputs/km_arousal_first/YYYYMMDD_HHMMSS/
├── best.pt           # Best model (lowest val loss)
├── last.pt           # Final model
├── metrics.json      # {"train_loss": [...], "val_loss": [...]}
├── train.log         # Per-epoch logs
└── loss_curve.png    # Matplotlib visualization
```

## Future Extension Points

The architecture supports:
1. **Multi-modal fusion**: Add face, speech, other modalities
2. **Valence prediction**: Extend from Arousal-only to VA pair
3. **Alternative encoders**: LSTM, GRU, attention-only variants
4. **Hyperparameter configs**: Extended training via YAML configs

## Common Issues and Solutions

| Issue | Solution |
|-------|----------|
| Missing features | Run full data pipeline from raw AMuCS |
| Import errors | Ensure `_add_src_to_path()` is called |
| Mask dimension mismatch | Check mask inversion for Transformer |
| Empty validation set | Verify split JSON and feature file paths match |
| NaN in training | Check for division by zero in normalization |

## References

- **Primary Documentation**: See `ARCHITECTURE.md` for detailed diagrams and Chinese explanations
- **Model Definition**: `lft-va/src/lft_va/models/km_transformer_min.py`
- **Dataset Implementation**: `lft-va/src/lft_va/datasets/km_window_dataset.py`
- **Feature Encoder**: `encoder/km/km_encoder_stat.py`
