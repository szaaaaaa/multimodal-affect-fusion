# Late Fusion Transformer (LFT) Architecture

## Overview

The Late Fusion Transformer is a multimodal architecture designed for **Valence/Arousal Prediction**. It processes two modalities (Video and Keyboard/Mouse) through separate encoding branches before fusing them in a shared Transformer encoder.

---

## Architecture Diagram

```
┌─────────────────────────────────────────────────────────────────────────────────┐
│                          Late Fusion Transformer (LFT)                          │
├────────────────────────────────┬────────────────────────────────────────────────┤
│        Video Modality          │           Keyboard / Mouse Modality            │
├────────────────────────────────┼────────────────────────────────────────────────┤
│                                │                                                │
│    ┌──────────────────┐        │        ┌──────────────────┐                    │
│    │ Input Sequence   │        │        │ Input Sequence   │                    │
│    │      X_v         │        │        │      X_km        │                    │
│    └────────┬─────────┘        │        └────────┬─────────┘                    │
│             │                  │                 │                              │
│             ▼                  │                 ▼                              │
│    ┌──────────────────┐        │        ┌──────────────────┐                    │
│    │ Video Encoder /  │        │        │  KM Encoder /    │                    │
│    │    Features      │        │        │    Features      │                    │
│    └────────┬─────────┘        │        └────────┬─────────┘                    │
│             │                  │                 │                              │
│             ▼                  │                 ▼                              │
│    ┌────────────────────────────────────────────────────────┐                   │
│    │  Linear Projection + LN  │  Positional/Time  │ Modality│                   │
│    │                          │     Encoding      │Embedding│                   │
│    └────────────────────────────────────────────────────────┘                   │
│             │                  │                 │                              │
│             ▼                  │                 ▼                              │
│         [ Add ]                │             [ Add ]                            │
│             │                  │                 │                              │
│             ▼                  │                 ▼                              │
│    ┌──────────────────┐        │        ┌──────────────────┐                    │
│    │ Video Tokens Z_v │        │        │  KM Tokens Z_km  │                    │
│    └────────┬─────────┘        │        └────────┬─────────┘                    │
│             │                  │                 │                              │
└─────────────┼──────────────────┴─────────────────┼──────────────────────────────┘
              │                                    │
              └──────────────┬─────────────────────┘
                             ▼
              ┌──────────────────────────┐
              │ Concatenate Tokens       │
              │      (dim = 1)           │
              └────────────┬─────────────┘
                           ▼
              ┌──────────────────────────────────┐
              │     Transformer Encoder × N      │
              │  ┌────────────────────────────┐  │
              │  │ Multi-Head Self-Attention  │  │
              │  └─────────────┬──────────────┘  │
              │                ▼                 │
              │  ┌────────────────────────────┐  │
              │  │       Add & Norm           │  │
              │  └─────────────┬──────────────┘  │
              │                ▼                 │
              │  ┌────────────────────────────┐  │
              │  │       Feed Forward         │  │
              │  └─────────────┬──────────────┘  │
              │                ▼                 │
              │  ┌────────────────────────────┐  │
              │  │       Add & Norm           │  │
              │  └─────────────┬──────────────┘  │
              └────────────────┼─────────────────┘
                               ▼
              ┌──────────────────────────┐
              │  Fused Token Sequence Z  │
              └────────────┬─────────────┘
                           ▼
              ┌──────────────────────────┐
              │    Linear / MLP Head     │
              └────────────┬─────────────┘
                           ▼
              ┌──────────────────────────┐
              │  Valence / Arousal       │
              │      Prediction          │
              └──────────────────────────┘
```

---

## Module Specifications

### 1. Input Modalities

#### 1.1 Video Modality Input (`X_v`)
```python
# Input shape: (batch_size, seq_len_v, video_feature_dim)
# Example: (B, T_v, D_v)
```

#### 1.2 Keyboard/Mouse Modality Input (`X_km`)
```python
# Input shape: (batch_size, seq_len_km, km_feature_dim)
# Example: (B, T_km, D_km)
```

---

### 2. Feature Encoders

#### 2.1 Video Encoder / Features
- **Purpose**: Extract features from raw video input
- **Options**:
  - Pre-trained CNN (ResNet, VGG, EfficientNet)
  - Vision Transformer (ViT)
  - 3D CNN for temporal features (C3D, I3D)
- **Output**: `(B, T_v, D_encoded)`

```python
class VideoEncoder(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        # Encoder layers
        pass

    def forward(self, x_v):
        # x_v: (B, T_v, D_v)
        # return: (B, T_v, D_encoded)
        pass
```

#### 2.2 KM (Keyboard/Mouse) Encoder / Features
- **Purpose**: Extract features from keyboard/mouse behavioral data
- **Options**:
  - 1D CNN
  - LSTM/GRU
  - Temporal CNN
- **Output**: `(B, T_km, D_encoded)`

```python
class KMEncoder(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        # Encoder layers
        pass

    def forward(self, x_km):
        # x_km: (B, T_km, D_km)
        # return: (B, T_km, D_encoded)
        pass
```

---

### 3. Token Embedding Components

#### 3.1 Linear Projection + Layer Normalization
- **Purpose**: Project encoded features to transformer dimension
- **Components**:
  - Linear layer: `D_encoded → D_model`
  - Layer Normalization

```python
class LinearProjection(nn.Module):
    def __init__(self, input_dim, model_dim):
        self.linear = nn.Linear(input_dim, model_dim)
        self.layer_norm = nn.LayerNorm(model_dim)

    def forward(self, x):
        return self.layer_norm(self.linear(x))
```

#### 3.2 Positional / Time Encoding
- **Purpose**: Encode temporal position information
- **Options**:
  - Sinusoidal positional encoding
  - Learnable positional embedding
  - Temporal encoding for time-series data

```python
class PositionalEncoding(nn.Module):
    def __init__(self, model_dim, max_seq_len=5000):
        # Sinusoidal or learnable positional encoding
        pass

    def forward(self, seq_len):
        # return: (1, seq_len, D_model) or (seq_len, D_model)
        pass
```

#### 3.3 Modality Embedding
- **Purpose**: Distinguish between different modalities
- **Implementation**: Learnable embedding per modality

```python
class ModalityEmbedding(nn.Module):
    def __init__(self, num_modalities, model_dim):
        # e_v for video, e_km for keyboard/mouse
        self.embeddings = nn.Embedding(num_modalities, model_dim)

    def forward(self, modality_id):
        # modality_id: 0 for video, 1 for km
        # return: (1, 1, D_model) - broadcastable
        pass
```

---

### 4. Token Generation

#### 4.1 Video Tokens (`Z_v`)
```python
# Z_v = LinearProjection(VideoEncoder(X_v)) + PositionalEncoding + e_v
# Shape: (B, T_v, D_model)
```

#### 4.2 KM Tokens (`Z_km`)
```python
# Z_km = LinearProjection(KMEncoder(X_km)) + PositionalEncoding + e_km
# Shape: (B, T_km, D_model)
```

---

### 5. Token Concatenation

- **Operation**: Concatenate along sequence dimension (dim=1)
- **Output**: Combined token sequence

```python
# Z_combined = torch.cat([Z_v, Z_km], dim=1)
# Shape: (B, T_v + T_km, D_model)
```

---

### 6. Transformer Encoder (× N layers)

#### 6.1 Architecture per Layer
Each Transformer encoder layer contains:

1. **Multi-Head Self-Attention**
2. **Add & Norm** (Residual connection + Layer Normalization)
3. **Feed Forward Network**
4. **Add & Norm** (Residual connection + Layer Normalization)

```python
class TransformerEncoderLayer(nn.Module):
    def __init__(self, model_dim, num_heads, ff_dim, dropout=0.1):
        self.self_attn = nn.MultiheadAttention(model_dim, num_heads, dropout=dropout)
        self.feed_forward = nn.Sequential(
            nn.Linear(model_dim, ff_dim),
            nn.ReLU(),  # or GELU
            nn.Dropout(dropout),
            nn.Linear(ff_dim, model_dim)
        )
        self.norm1 = nn.LayerNorm(model_dim)
        self.norm2 = nn.LayerNorm(model_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        # Multi-Head Self-Attention + Add & Norm
        attn_out, _ = self.self_attn(x, x, x)
        x = self.norm1(x + self.dropout(attn_out))

        # Feed Forward + Add & Norm
        ff_out = self.feed_forward(x)
        x = self.norm2(x + self.dropout(ff_out))

        return x
```

#### 6.2 Stacked Encoder
```python
class TransformerEncoder(nn.Module):
    def __init__(self, num_layers, model_dim, num_heads, ff_dim, dropout=0.1):
        self.layers = nn.ModuleList([
            TransformerEncoderLayer(model_dim, num_heads, ff_dim, dropout)
            for _ in range(num_layers)
        ])

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x  # Fused Token Sequence Z
```

---

### 7. Output Head

#### 7.1 Fused Token Sequence (`Z`)
- Output from Transformer Encoder
- Shape: `(B, T_v + T_km, D_model)`

#### 7.2 Linear / MLP Head
- **Purpose**: Map fused representations to predictions
- **Options**:
  - Global pooling (mean/max) + Linear
  - CLS token + Linear
  - MLP with multiple layers

```python
class PredictionHead(nn.Module):
    def __init__(self, model_dim, hidden_dim, output_dim=2):
        # output_dim=2 for Valence and Arousal
        self.pooling = 'mean'  # or 'cls', 'max'
        self.mlp = nn.Sequential(
            nn.Linear(model_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, output_dim)
        )

    def forward(self, z):
        # z: (B, T, D_model)
        if self.pooling == 'mean':
            pooled = z.mean(dim=1)  # (B, D_model)
        elif self.pooling == 'cls':
            pooled = z[:, 0, :]  # (B, D_model)

        return self.mlp(pooled)  # (B, 2) for valence & arousal
```

#### 7.3 Output: Valence / Arousal Prediction
- **Valence**: Measures positive/negative emotion (-1 to 1)
- **Arousal**: Measures activation/deactivation level (-1 to 1)
- **Output Shape**: `(B, 2)` - [valence, arousal]

---

## Complete Model Implementation

```python
import torch
import torch.nn as nn

class LateFusionTransformer(nn.Module):
    def __init__(self, config):
        super().__init__()

        # Config parameters
        self.model_dim = config.model_dim

        # Video branch
        self.video_encoder = VideoEncoder(
            input_dim=config.video_input_dim,
            hidden_dim=config.video_hidden_dim,
            output_dim=config.video_encoded_dim
        )
        self.video_projection = LinearProjection(
            input_dim=config.video_encoded_dim,
            model_dim=config.model_dim
        )

        # KM branch
        self.km_encoder = KMEncoder(
            input_dim=config.km_input_dim,
            hidden_dim=config.km_hidden_dim,
            output_dim=config.km_encoded_dim
        )
        self.km_projection = LinearProjection(
            input_dim=config.km_encoded_dim,
            model_dim=config.model_dim
        )

        # Positional encoding (shared or separate)
        self.positional_encoding = PositionalEncoding(
            model_dim=config.model_dim,
            max_seq_len=config.max_seq_len
        )

        # Modality embeddings
        self.modality_embedding = ModalityEmbedding(
            num_modalities=2,
            model_dim=config.model_dim
        )

        # Transformer encoder
        self.transformer_encoder = TransformerEncoder(
            num_layers=config.num_encoder_layers,
            model_dim=config.model_dim,
            num_heads=config.num_heads,
            ff_dim=config.ff_dim,
            dropout=config.dropout
        )

        # Prediction head
        self.prediction_head = PredictionHead(
            model_dim=config.model_dim,
            hidden_dim=config.head_hidden_dim,
            output_dim=2  # Valence & Arousal
        )

    def forward(self, x_v, x_km):
        """
        Args:
            x_v: Video input (B, T_v, D_v)
            x_km: Keyboard/Mouse input (B, T_km, D_km)

        Returns:
            predictions: (B, 2) - [valence, arousal]
        """
        # === Video Branch ===
        # Encode video features
        video_features = self.video_encoder(x_v)  # (B, T_v, D_encoded)

        # Project to model dimension + Layer Norm
        video_proj = self.video_projection(video_features)  # (B, T_v, D_model)

        # Add positional encoding
        video_proj = video_proj + self.positional_encoding(video_proj.size(1))

        # Add modality embedding
        z_v = video_proj + self.modality_embedding(0)  # (B, T_v, D_model)

        # === KM Branch ===
        # Encode KM features
        km_features = self.km_encoder(x_km)  # (B, T_km, D_encoded)

        # Project to model dimension + Layer Norm
        km_proj = self.km_projection(km_features)  # (B, T_km, D_model)

        # Add positional encoding
        km_proj = km_proj + self.positional_encoding(km_proj.size(1))

        # Add modality embedding
        z_km = km_proj + self.modality_embedding(1)  # (B, T_km, D_model)

        # === Fusion ===
        # Concatenate tokens along sequence dimension
        z_combined = torch.cat([z_v, z_km], dim=1)  # (B, T_v + T_km, D_model)

        # Transformer encoder
        z_fused = self.transformer_encoder(z_combined)  # (B, T_v + T_km, D_model)

        # === Prediction ===
        predictions = self.prediction_head(z_fused)  # (B, 2)

        return predictions
```

---

## Configuration Template

```python
from dataclasses import dataclass

@dataclass
class LFTConfig:
    # Model dimensions
    model_dim: int = 512
    ff_dim: int = 2048
    num_heads: int = 8
    num_encoder_layers: int = 6
    dropout: float = 0.1
    max_seq_len: int = 1000

    # Video encoder
    video_input_dim: int = 2048  # e.g., ResNet feature dim
    video_hidden_dim: int = 1024
    video_encoded_dim: int = 512

    # KM encoder
    km_input_dim: int = 64  # keyboard/mouse feature dim
    km_hidden_dim: int = 256
    km_encoded_dim: int = 512

    # Prediction head
    head_hidden_dim: int = 256
```

---

## Training Considerations

### Loss Function
```python
# For continuous Valence/Arousal prediction
criterion = nn.MSELoss()  # or nn.L1Loss(), nn.SmoothL1Loss()

# Or CCC Loss (Concordance Correlation Coefficient) - common in affective computing
class CCCLoss(nn.Module):
    def forward(self, pred, target):
        # Implementation of CCC loss
        pass
```

### Data Format
```python
# Batch structure
batch = {
    'video': torch.Tensor,      # (B, T_v, D_v)
    'keyboard_mouse': torch.Tensor,  # (B, T_km, D_km)
    'valence': torch.Tensor,    # (B,) ground truth
    'arousal': torch.Tensor     # (B,) ground truth
}
```

---

## File Structure for Implementation

```
project/
├── models/
│   ├── __init__.py
│   ├── late_fusion_transformer.py  # Main model
│   ├── encoders/
│   │   ├── __init__.py
│   │   ├── video_encoder.py
│   │   └── km_encoder.py
│   ├── components/
│   │   ├── __init__.py
│   │   ├── positional_encoding.py
│   │   ├── modality_embedding.py
│   │   └── prediction_head.py
│   └── transformer/
│       ├── __init__.py
│       └── encoder.py
├── configs/
│   └── lft_config.py
├── data/
│   └── dataloader.py
├── training/
│   ├── train.py
│   └── loss.py
└── utils/
    └── metrics.py
```

---

## Key Implementation Notes

1. **Sequence Length Handling**: Video and KM sequences may have different lengths. Handle with:
   - Padding + attention masks
   - Fixed-length sampling
   - Dynamic batching

2. **Modality Dropout**: Consider dropping one modality during training for robustness

3. **Pre-training**: Video encoder can use pre-trained weights (ImageNet, Kinetics)

4. **Attention Visualization**: Multi-head attention weights can reveal cross-modal interactions

5. **Gradient Checkpointing**: Use for memory efficiency with long sequences
