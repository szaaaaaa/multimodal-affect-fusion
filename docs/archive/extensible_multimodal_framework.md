# 可扩展多模态实验框架 — 技术设计文档

> **项目**：多模态游戏情绪识别（Valence / Arousal）
> **当前模态**：Video (ResNet-50 frame features) + Keyboard/Mouse (KM)
> **目标融合方法**：Single Transformer / LFT / MulT / MoE

---

## 0. 为什么需要预留接口？

**核心问题**：当前代码中 encoder 的选择通过 `if/else` 硬编码在模型类内部（`late_fusion_transformer.py:136-145`），模态列表固定为 `["video", "km"]`，`train_lft.py` 中训练逻辑与具体数据集、模型类深度耦合。这意味着：

- **新增模态**（如 telemetry、pose）→ 必须改模型类的 `__init__` 和 `forward`
- **新增 encoder**（如 km 的 transformer encoder）→ 必须在模型类里加 `if` 分支
- **新增融合方式**（如 MulT）→ 必须写一个全新的模型类 + 全新的训练脚本
- **新增数据集** → 必须在 `train.py` 里加 `if/else`

**解决方案**：是的，你需要预留接口。具体来说需要两样东西：

1. **抽象接口（Abstract Interface）**：用 Python ABC 定义 4 个稳定的契约（DataModule / Encoder / Fusion / Head），规定输入输出的形状和类型。一旦定死，未来扩展不动它们。
2. **注册表（Registry）**：一个字典式的查找机制，让新模块通过装饰器自注册，训练入口通过配置文件中的字符串 key 查找并实例化模块，无需 `import` 或 `if/else`。

两者缺一不可：接口保证新模块与既有流程兼容（shape 和方法签名一致），注册表保证新模块被发现和构建（无需改调用方代码）。

### 当前代码 vs 目标架构的差距

| 维度 | 当前状态 | 目标状态 |
|------|---------|---------|
| Encoder 选择 | `if config.km_encoder_type == "cnn"` 硬编码 | `ENCODERS["km"][cfg.name]` 注册表查找 |
| 模态列表 | 固定 video + km | `cfg.modalities` 任意子集，动态遍历 |
| 融合方式 | 仅 EFT（concat + Transformer） | `FUSIONS[cfg.name]` 可选 single/eft/lft/mft/cma |
| Batch 格式 | `{"video", "video_mask", "km", "km_mask", "y"}` 平铺 | `{"x": {...}, "mask": {...}, "y", "meta"}` 嵌套字典 |
| 训练脚本 | 249 行，含数据集构建 + 模型构建 + 训练循环 | 极薄入口：`cfg → build → runner.fit()` |
| 配置 | argparse CLI 参数 | 分层 YAML (base + override) |
| 输出管理 | 时间戳目录 + metrics.json | 标准化目录 + config.yaml + git hash + seed |

---

## 1. 设计目标

| 编号 | 目标 | 约束 |
|------|------|------|
| G1 | 训练主流程永不改 | 新增模型/模态/encoder 只"加文件 + 注册"，不改 `train.py` |
| G2 | 统一 batch / 特征张量形状 | 多模态只是 `batch["x"]` 里 key 变多 |
| G3 | 可插拔 | 所有实现通过 Registry/Plugin 发现与构建 |
| G4 | 缺失模态从第一天内建 | `mask` 永远存在，fusion 接受任意子集 |
| G5 | 可复现 | 每次 run 保存完整配置、commit hash、seed |

---

## 2. 稳定接口（冻结，未来扩展不动它们）

只定 **4 个抽象接口**。所有可变实现都放在具体子类中。

### 2.1 接口 A — DataModule 输出 batch 规范

```python
# DataModule.__getitem__ 或 collate_fn 产出的 batch 必须满足：
batch = {
    "x": {
        "video": Tensor,       # [B, T_v, D_v] 或 [B, D_v]
        "km": Tensor,          # [B, T_km, D_km]
        "telemetry": Tensor,   # [B, T_t, D_t]
        # ... 任意子集，由 cfg.modalities 决定
    },
    "mask": {
        "video": Tensor,       # [B, T_v] bool, True=有效
        "km": Tensor,          # [B, T_km] bool
        # ... 与 x 的 key 一一对应
    },
    "y": Tensor,               # [B, ...] — 标签
    "meta": {                  # 可选，不参与训练
        "id": ...,
        "player": ...,
        "session": ...,
        "timestamps": ...,
    },
}
```

**约束**：

- `x[modality]` 要么是 `[B, T, ...]`（序列），要么 `[B, ...]`（非序列），由该模态 encoder 负责适配
- `mask` **永远存在**，即使单模态也给 `mask={"km": ones}`，方便缺失模态统一处理
- DataModule 的抽象基类只要求实现 `train_dataloader()` / `val_dataloader()` / `test_dataloader()`

```python
# src/core/types.py
from typing import TypedDict, Dict, Optional
import torch

class Batch(TypedDict):
    x: Dict[str, torch.Tensor]
    mask: Dict[str, torch.Tensor]
    y: torch.Tensor
    meta: Optional[Dict]
```

### 2.2 接口 B — Encoder（冻结）

每个模态的 encoder 必须返回统一的 `EncoderOut` 字典：

```python
# 调用方式
z = encoder(x, mask=None)

# 返回值（强制 dict，避免不同 encoder 返回不同形状导致下游爆炸）
z: EncoderOut = {
    "tokens": Tensor,   # [B, T, D] — token 级表示
    "pooled": Tensor,   # [B, D]   — 池化后的全局表示
    "mask":   Tensor,   # [B, T]   — 有效 token 的 mask
}
```

**抽象基类**：

```python
# src/core/types.py
from abc import ABC, abstractmethod

class EncoderOut(TypedDict):
    tokens: torch.Tensor   # [B, T, D]
    pooled: torch.Tensor   # [B, D]
    mask: torch.Tensor     # [B, T]

class BaseEncoder(ABC, nn.Module):
    @abstractmethod
    def forward(self, x: torch.Tensor, mask: torch.Tensor | None = None) -> EncoderOut:
        ...
```

### 2.3 接口 C — Fusion（冻结）

融合层接收**任意数量**模态的 encoder 输出，返回统一的 `FusionOut`：

```python
# 调用方式
h = fusion(z_dict, mask_dict)
# z_dict:   {"video": EncoderOut, "km": EncoderOut, ...}  任意子集
# mask_dict: {"video": Tensor[B,T], "km": Tensor[B,T], ...}

# 返回值
h: FusionOut = {
    "tokens": Tensor | None,  # [B, T, D] — 融合后的 token 序列（可选）
    "pooled": Tensor,          # [B, D]   — 融合后的全局表示（必须）
}
```

**抽象基类**：

```python
class FusionOut(TypedDict):
    tokens: torch.Tensor | None
    pooled: torch.Tensor

class BaseFusion(ABC, nn.Module):
    @abstractmethod
    def forward(
        self,
        z_dict: Dict[str, EncoderOut],
        mask_dict: Dict[str, torch.Tensor],
    ) -> FusionOut:
        ...
```

**关键约束**：fusion 必须能处理 `z_dict` 中任意数量的 key（1 个、2 个、N 个），不能 hardcode 模态名。

### 2.4 接口 D — Head（冻结）

```python
# 调用方式
y_hat = head(h)  # h: FusionOut

# 返回值
y_hat: Tensor  # [B, out_dim]
```

**抽象基类**：

```python
class BaseHead(ABC, nn.Module):
    @abstractmethod
    def forward(self, h: FusionOut) -> torch.Tensor:
        ...
```

### 接口总结：数据流图

```
DataModule                Encoders              Fusion              Head
─────────               ────────              ──────              ────
batch["x"]["video"] ──→ video_encoder ──→ ┐
batch["x"]["km"]    ──→ km_encoder    ──→ ├──→ fusion(z_dict) ──→ head(h) ──→ y_hat
batch["x"]["tel"]   ──→ tel_encoder   ──→ ┘
                                          │
batch["mask"]       ─────────────────────→┘
```

---

## 3. Registry 机制（扩展的关键）

### 3.1 实现方式

```python
# src/core/registry.py
from typing import Dict, Type, Callable

class Registry:
    """通用注册表。key 只允许字符串（稳定可序列化）。"""

    def __init__(self, name: str):
        self.name = name
        self._registry: Dict[str, Type] = {}

    def register(self, key: str) -> Callable:
        """装饰器注册。"""
        def decorator(cls):
            if key in self._registry:
                raise KeyError(f"{self.name}: '{key}' already registered")
            self._registry[key] = cls
            return cls
        return decorator

    def build(self, key: str, cfg):
        """根据 key 查找并实例化，构造函数只吃 cfg。"""
        if key not in self._registry:
            raise KeyError(
                f"{self.name}: '{key}' not found. "
                f"Available: {list(self._registry.keys())}"
            )
        return self._registry[key](cfg)

    def __contains__(self, key: str) -> bool:
        return key in self._registry

    def keys(self):
        return self._registry.keys()


# ──── 需要的 registries ────
ENCODERS: Dict[str, Registry] = {}  # ENCODERS["km"] = Registry("km_encoders")
FUSIONS = Registry("fusions")       # single / lft / mult / moe ...
DATAMODULES = Registry("datamodules")
HEADS = Registry("heads")
LOSSES = Registry("losses")         # 可选
METRICS = Registry("metrics")       # 可选

def get_encoder_registry(modality: str) -> Registry:
    """按模态获取 encoder 注册表，不存在则自动创建。"""
    if modality not in ENCODERS:
        ENCODERS[modality] = Registry(f"{modality}_encoders")
    return ENCODERS[modality]
```

### 3.2 注册示例

```python
# src/models/encoders/km/stat.py
from src.core.registry import get_encoder_registry

@get_encoder_registry("km").register("stat")
class KMStatEncoder(BaseEncoder):
    def __init__(self, cfg):
        super().__init__()
        self.proj = nn.Linear(cfg.d_in, cfg.d_model)
        ...

    def forward(self, x, mask=None):
        tokens = self.proj(x)
        pooled = tokens.mean(dim=1)
        if mask is None:
            mask = torch.ones(x.shape[0], x.shape[1], dtype=torch.bool, device=x.device)
        return {"tokens": tokens, "pooled": pooled, "mask": mask}
```

```python
# src/models/fusions/lft.py
from src.core.registry import FUSIONS

@FUSIONS.register("lft")
class LFTFusion(BaseFusion):
    def __init__(self, cfg):
        super().__init__()
        ...

    def forward(self, z_dict, mask_dict):
        ...
```

### 3.3 规则

| 规则 | 说明 |
|------|------|
| Registry key 只允许字符串 | 稳定可序列化，可在 YAML 中引用 |
| 构造函数只吃 `cfg` | 避免参数地狱，cfg 是 dataclass 或 dict |
| 注册时检查重复 | 防止静默覆盖 |
| Encoder registry 按模态分离 | `ENCODERS["km"]["stat"]`、`ENCODERS["video"]["vit"]` |

---

## 4. 推荐仓库结构（可长期生长）

```
ProjectExperiment/
  configs/
    base.yaml                         # 全局默认值
    experiments/
      km_single.yaml                  # 单模态 KM 实验
      video_km_lft.yaml               # 双模态 LFT
      all_lft.yaml                    # 全模态 LFT
    sweeps/
      lft_grid.yaml                   # 超参网格搜索

  src/
    core/
      __init__.py
      types.py                        # Batch / EncoderOut / FusionOut / 抽象基类
      registry.py                     # 注册表实现
      config.py                       # YAML 读取 / 合并 / 校验
      runner.py                       # 训练循环入口（稳定，不改）
      logging.py                      # metrics.json + ckpt + wandb/tb
      seed.py                         # 随机种子设置

    data/
      __init__.py
      datamodules/
        __init__.py
        amucs.py                      # AMuCS 数据集的 DataModule
        again.py                      # AGAIN 数据集（未来）
      transforms/
        windowing.py                  # 窗口化
        alignment.py                  # 时间对齐
        normalization.py              # z-score / min-max

    models/
      __init__.py
      encoders/
        km/
          __init__.py
          stat.py                     # KM 统计特征 encoder
          cnn1d.py                    # KM 1D CNN encoder
          # 未来新增: transformer.py
        video/
          __init__.py
          resnet2d.py                 # Per-frame projection + mask-aware temporal mean pooling
          # 未来新增: vit.py
        telemetry/
          __init__.py
          mlp.py                      # 遥测 MLP encoder（未来）
        # 未来新增: pose/ ...
      fusions/
        __init__.py
        single.py                     # 单模态直通
        lft.py                        # Late Fusion Transformer
        mult.py                       # MulT（未来）
        moe.py                        # MoE（未来）
      heads/
        __init__.py
        regression.py                 # 回归头 (VA 预测)
        classification.py            # 分类头（未来）

    losses/
      __init__.py
      ccc.py                          # CCC loss
      mse.py                          # MSE loss

    metrics/
      __init__.py
      ccc.py                          # CCC metric
      rmse.py                         # RMSE metric

  scripts/
    train.py                          # 极薄入口：cfg → build → runner.fit()
    sweep.py                          # 实验组合生成 + 调度（可选）
    summarize.py                      # 汇总 runs → leaderboard.csv

  tests/
    test_shapes.py                    # Shape contract tests（核心）
    test_registry.py                  # 注册表完整性测试

  runs/                               # 输出目录（.gitignore）
  pyproject.toml
  requirements.txt
```

**关键设计**：

- `scripts/train.py` **极薄**：只做 `cfg → build → runner.fit()`，不含任何业务逻辑
- 所有可变逻辑在 `src/models/*`、`src/data/*`，且都通过 registry 构建
- `src/core/` 中的文件一旦稳定后**极少改动**

---

## 5. 配置设计：让扩展不破坏旧实验

### 5.1 分层配置（base + override）

```yaml
# configs/base.yaml — 全局默认值
data:
  name: amucs
  modalities: [video, km]
  window:
    length_s: 5.0
    stride_s: 0.2
  alignment: late
  normalization:
    x: zscore
    y: zscore

model:
  d_model: 512
  encoders:
    video:
      name: resnet2d
      params:
        feature_dim: 2048
    km:
      name: stat
      params:
        feature_dim: 25
  fusion:
    name: lft
    params:
      nhead: 8
      num_layers: 4
      dim_feedforward: 1024
      dropout: 0.1
      pooling: mean
  head:
    name: regression
    params:
      hidden_dim: 128
      out_dim: 1              # 1=arousal, 2=VA

train:
  loss: ccc
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
  seed: 42

eval:
  metrics: [ccc, rmse]
  missing_modality_tests:     # 缺失模态鲁棒性测试（可选开关）
    - drop: [video]
    - drop: [km]
```

### 5.2 实验 override 示例

```yaml
# configs/experiments/km_single.yaml
_base_: ../base.yaml

data:
  modalities: [km]            # 只用 KM

model:
  fusion:
    name: single              # 单模态直通，不做跨模态融合
  head:
    params:
      out_dim: 1
```

```yaml
# configs/experiments/video_km_lft.yaml
_base_: ../base.yaml

data:
  modalities: [video, km]

model:
  fusion:
    name: lft
    params:
      num_layers: 4
```

```yaml
# configs/experiments/all_mult.yaml — 未来新增 MulT
_base_: ../base.yaml

data:
  modalities: [video, km, telemetry]

model:
  encoders:
    telemetry:
      name: mlp
      params:
        feature_dim: 32
  fusion:
    name: mult                # MulT 只需改这一行 + 注册
    params:
      num_layers: 3
      attn_dropout: 0.1
```

### 5.3 扩展时的约束

| 扩展类型 | 需要做的 | 不需要改的 |
|----------|---------|-----------|
| 新增 encoder | 新文件 + `@register` + config 里加 `encoders.<mod>.name` 可选值 | `train.py`、`runner.py`、任何其他 encoder |
| 新增模态 | DataModule 产出 `x[new_mod]` + 新 encoder + config `modalities` 列表加一项 | fusion、head、训练主流程 |
| 新增融合方式 | `models/fusions/xxx.py` + `@register` + config `fusion.name` | 所有 encoder、data、head |
| 新增数据集 | `data/datamodules/xxx.py` + `@register` + config `data.name` | 模型层、训练循环 |
| 新增 loss | `losses/xxx.py` + `@register` + config `train.loss` | 一切 |

---

## 6. 训练入口与 Runner（稳定层）

### 6.1 scripts/train.py（极薄）

```python
# scripts/train.py — 这个文件写完后几乎不改
"""
Usage:
    python scripts/train.py --config configs/experiments/video_km_lft.yaml
    python scripts/train.py --config configs/base.yaml --override model.fusion.name=mult
"""
import argparse
from src.core.config import load_config
from src.core.runner import Runner

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--override", nargs="*", default=[])
    args = parser.parse_args()

    cfg = load_config(args.config, overrides=args.override)
    runner = Runner(cfg)
    runner.fit()

if __name__ == "__main__":
    main()
```

### 6.2 Runner.fit()（伪代码，说明构建流程）

```python
# src/core/runner.py
class Runner:
    def __init__(self, cfg):
        self.cfg = cfg
        self._build()

    def _build(self):
        # 1. 数据
        self.dm = DATAMODULES.build(cfg.data.name, cfg.data)

        # 2. Encoders — 按 cfg.data.modalities 动态构建
        self.encoders = {}
        for mod in cfg.data.modalities:
            enc_cfg = cfg.model.encoders[mod]
            registry = get_encoder_registry(mod)
            self.encoders[mod] = registry.build(enc_cfg.name, enc_cfg.params)

        # 3. Fusion
        self.fusion = FUSIONS.build(cfg.model.fusion.name, cfg.model.fusion.params)

        # 4. Head
        self.head = HEADS.build(cfg.model.head.name, cfg.model.head.params)

        # 5. Loss
        self.loss_fn = LOSSES.build(cfg.train.loss, cfg.train)

        # 组装成 nn.Module（或直接分开调用）
        ...

    def fit(self):
        # 标准训练循环：
        # for epoch:
        #   for batch in train_loader:
        #     z_dict = {mod: self.encoders[mod](batch["x"][mod], batch["mask"][mod])
        #               for mod in self.encoders}
        #     mask_dict = {mod: z_dict[mod]["mask"] for mod in z_dict}
        #     h = self.fusion(z_dict, mask_dict)
        #     y_hat = self.head(h)
        #     loss = self.loss_fn(y_hat, batch["y"])
        #     ...
        ...
```

**注意**：这里的 `for mod in self.encoders` 循环是关键 — 它让模态数量变成运行时动态的，不需要任何 `if video ... if km ...` 判断。

---

## 7. 缺失模态与可选模态：从第一天内建

即使暂时不做缺失模态实验，也要把接口留好：

### 7.1 mask 永远存在

DataModule 必须对每个模态产出 mask。单模态场景下：

```python
batch["mask"] = {"km": torch.ones(B, T_km, dtype=torch.bool)}
```

### 7.2 Fusion 必须接受任意子集

```python
class LFTFusion(BaseFusion):
    def forward(self, z_dict, mask_dict):
        # z_dict 可能只有 1 个 key，也可能有 5 个
        all_tokens = []
        all_masks = []
        for mod, z in z_dict.items():
            all_tokens.append(z["tokens"])
            all_masks.append(z["mask"])
        tokens = torch.cat(all_tokens, dim=1)
        masks = torch.cat(all_masks, dim=1)
        # ... transformer encoder ...
```

### 7.3 Modality Dropout（配置开关）

训练时随机丢弃模态，提升鲁棒性：

```yaml
train:
  modality_dropout: 0.1  # 10% 概率丢弃某个模态（设 mask 全 0）
```

```python
# runner.py 训练循环中
if self.cfg.train.modality_dropout > 0 and phase == "train":
    for mod in list(batch["mask"].keys()):
        if torch.rand(1).item() < self.cfg.train.modality_dropout:
            batch["mask"][mod] = torch.zeros_like(batch["mask"][mod])
```

不需要改任何模型结构，因为 mask 本来就存在。

---

## 8. 新增模块的最小改动清单

### 8.1 新增 MulT 融合

| 步骤 | 文件 | 操作 |
|------|------|------|
| 1 | `src/models/fusions/mult.py` | 新建，实现 `forward(z_dict, mask_dict) → FusionOut` |
| 2 | 同上 | 加 `@FUSIONS.register("mult")` |
| 3 | `configs/experiments/xxx_mult.yaml` | `model.fusion.name: mult` |
| **不改** | `train.py` / `runner.py` / 任何 datamodule / 任何 encoder | — |

### 8.2 新增模态（例如 pose）

| 步骤 | 文件 | 操作 |
|------|------|------|
| 1 | `src/data/datamodules/amucs.py` | 让 DataModule 产出 `x["pose"]` + `mask["pose"]` |
| 2 | `src/models/encoders/pose/` | 新建目录和 encoder 文件 + `@register` |
| 3 | config | `data.modalities` 加 `pose`，`model.encoders.pose` 加配置 |
| 可选 | `src/data/transforms/` | 新增 pose 的预处理 transform |
| **不改** | fusion / head / 训练主流程 | 只要 fusion 能处理新增 key（它本来就能） |

### 8.3 新增 encoder（例如 KM 的 Transformer encoder）

| 步骤 | 文件 | 操作 |
|------|------|------|
| 1 | `src/models/encoders/km/transformer.py` | 新建，实现 `BaseEncoder` |
| 2 | 同上 | `@get_encoder_registry("km").register("transformer")` |
| 3 | config | `model.encoders.km.name: transformer` |
| **不改** | 其他任何代码 | — |

---

## 9. Shape Contract Tests（防止未来扩展炸掉）

**最有效的可扩展性保障手段是契约测试**。

### 9.1 必测项

```python
# tests/test_shapes.py
import pytest
import torch

# ──── Encoder 契约 ────
@pytest.mark.parametrize("modality,name", [
    ("km", "stat"),
    ("km", "cnn1d"),
    ("video", "resnet2d"),
])
def test_encoder_output_shape(modality, name):
    """每个 encoder：输入 mock → 输出必须包含 tokens/pooled/mask 且维度合法。"""
    B, T, D_in = 4, 50, 25 if modality == "km" else 2048
    x = torch.randn(B, T, D_in)

    encoder = get_encoder_registry(modality).build(name, mock_cfg(d_model=256, d_in=D_in))
    out = encoder(x)

    assert "tokens" in out and out["tokens"].shape == (B, T, 256)
    assert "pooled" in out and out["pooled"].shape == (B, 256)
    assert "mask" in out and out["mask"].shape == (B, T)
    assert out["mask"].dtype == torch.bool

# ──── Fusion 契约 ────
@pytest.mark.parametrize("modality_subset", [
    ["km"],
    ["video"],
    ["video", "km"],
    ["video", "km", "telemetry"],
])
def test_fusion_arbitrary_subset(modality_subset):
    """给定任意模态子集，fusion 都能跑通。"""
    B, T, D = 4, 50, 256
    z_dict = {mod: {"tokens": torch.randn(B, T, D),
                     "pooled": torch.randn(B, D),
                     "mask": torch.ones(B, T, dtype=torch.bool)}
              for mod in modality_subset}
    mask_dict = {mod: z_dict[mod]["mask"] for mod in modality_subset}

    fusion = FUSIONS.build("lft", mock_fusion_cfg(d_model=256))
    out = fusion(z_dict, mask_dict)

    assert "pooled" in out and out["pooled"].shape == (B, D)

# ──── 端到端单步前向 ────
def test_end_to_end_forward():
    """DataModule → Encoders → Fusion → Head → Loss 一次 forward 不报错。"""
    B, D = 4, 256
    batch = make_mock_batch(B=B, modalities=["video", "km"])

    # build all
    encoders = {mod: get_encoder_registry(mod).build(...) for mod in ["video", "km"]}
    fusion = FUSIONS.build("lft", ...)
    head = HEADS.build("regression", ...)

    # forward
    z_dict = {mod: encoders[mod](batch["x"][mod], batch["mask"][mod])
              for mod in encoders}
    mask_dict = {mod: z_dict[mod]["mask"] for mod in z_dict}
    h = fusion(z_dict, mask_dict)
    y_hat = head(h)

    assert y_hat.shape == (B, 1)  # or (B, 2) for VA
```

### 9.2 为什么这比单元测试更有价值

- 单元测试验证"函数返回对不对" — 但无法防止 shape 不兼容
- 契约测试验证"模块能不能嵌入流水线" — 直接保证 **新增模块不会破坏既有流程**
- 每次新增模块后只需把参数加到 `@pytest.mark.parametrize` 列表里

---

## 10. 实验管理：统一输出与可复现

### 10.1 每次 run 必须保存

```
runs/
  2026-02-04_14-30-22__amucs__lft__video_km__seed42/
    config.yaml           # 最终展开后的完整配置（不是 override，是 merge 后的）
    metrics.json          # {"best_val_ccc": 0.72, "test_ccc": 0.68, ...}
    git_commit.txt        # 当前 commit hash
    seed.txt              # 42
    ckpt_best.pt          # 最优模型权重
    ckpt_last.pt          # 最后一个 epoch
    events.tfevents.*     # TensorBoard 日志（可选）
    loss_curve.png        # 训练曲线（可选）
```

### 10.2 目录命名规范

```
{timestamp}__{dataset}__{fusion}__{modalities_joined}__{seed}
```

示例：
- `2026-02-04_14-30-22__amucs__lft__video_km__seed42`
- `2026-02-04_15-00-00__amucs__mult__video_km_telemetry__seed0`
- `2026-02-04_16-00-00__amucs__single__km__seed42`

### 10.3 metrics.json 标准格式

```json
{
  "best_val_ccc": 0.72,
  "best_val_epoch": 35,
  "test_ccc": 0.68,
  "test_rmse": 0.21,
  "train_ccc_final": 0.85,
  "total_epochs": 50,
  "early_stopped": true,
  "total_params": 2345678,
  "gpu_mem_peak_mb": 4096,
  "train_time_s": 1234.5
}
```

### 10.4 summarize.py 汇总

```bash
python scripts/summarize.py --runs_dir runs/ --output leaderboard.csv
```

输出 `leaderboard.csv`：

| dataset | fusion | modalities | encoder_km | seed | best_val_ccc | test_ccc | params |
|---------|--------|------------|------------|------|-------------|----------|--------|
| amucs | lft | video+km | stat | 42 | 0.72 | 0.68 | 2.3M |
| amucs | single | km | cnn1d | 42 | 0.65 | 0.61 | 0.5M |

---

## 11. 自动实验编排（Sweep / Orchestration）

### 11.1 Sweep 配置

```yaml
# configs/sweeps/lft_grid.yaml
_base_: ../base.yaml

sweep:
  method: grid        # grid / random / bayesian（未来）
  parameters:
    data.modalities:
      - [km]
      - [video, km]
    model.encoders.km.name:
      - stat
      - cnn1d
    model.fusion.name:
      - single
      - lft
    model.fusion.params.num_layers:
      - 2
      - 4
    train.seed:
      - 0
      - 42
  # 条件规则（可选）：单模态时 fusion 只能是 single
  constraints:
    - if:
        data.modalities: [km]
      then:
        model.fusion.name: single
```

### 11.2 sweep.py

```python
# scripts/sweep.py
"""
生成所有实验组合 → 逐个/并行调度 train.py
"""
# 从 sweep yaml 读取参数空间
# 生成笛卡尔积（应用 constraints 过滤）
# 每个组合生成一个临时 config yaml
# 调度执行：
#   - 本机串行：subprocess.run(["python", "scripts/train.py", "--config", tmp_cfg])
#   - 本机并行：multiprocessing / joblib
#   - 集群：生成 Slurm/ARC job 脚本
```

---

## 12. MVP 落地优先级

按照依赖关系和收益，建议以下实施顺序：

### Phase 1：基础设施（必须先做）

1. `src/core/types.py` — 定义 `Batch`, `EncoderOut`, `FusionOut`, 抽象基类
2. `src/core/registry.py` — 实现 Registry + 各注册表实例
3. `src/core/config.py` — YAML 加载 + merge + 校验

### Phase 2：迁移现有代码到新接口

4. 迁移 `KMStatTokenEncoder` → `src/models/encoders/km/stat.py`，实现 `BaseEncoder`，注册
5. 迁移 `KM1DCNNEncoder` → `src/models/encoders/km/cnn1d.py`，实现 `BaseEncoder`，注册
6. 迁移 `ResNetTokenEncoder` → `src/models/encoders/video/resnet2d.py`，实现 `BaseEncoder`，注册
7. 实现 `SingleFusion`（单模态直通）+ `LFTFusion`（从 `LateFusionTransformer` 中拆出），注册
8. 实现 `RegressionHead`，注册
9. 迁移 `MultimodalDataset` → `src/data/datamodules/amucs.py`，输出新 batch 格式

### Phase 3：组装训练流程

10. `src/core/runner.py` — 实现 Runner（用 registry 构建所有模块）
11. `scripts/train.py` — 极薄入口
12. `configs/base.yaml` + 几个实验 yaml

### Phase 4：测试与管理

13. `tests/test_shapes.py` — 契约测试
14. `tests/test_registry.py` — 注册表完整性
15. `src/core/logging.py` — 标准化输出目录
16. `scripts/summarize.py` — leaderboard 汇总

### Phase 5：扩展（无需改既有代码）

17. 新增 MulT：`src/models/fusions/mult.py` + `@register`
18. 新增 MoE：`src/models/fusions/moe.py` + `@register`
19. 新增 telemetry 模态：encoder + datamodule 扩展
20. `scripts/sweep.py` — 实验编排

---

## 13. 与当前代码的映射关系

| 当前文件 | 目标位置 | 改动说明 |
|----------|---------|---------|
| `encoder/km/km_encoder_stat.py` | `src/models/encoders/km/stat.py` | 包装为 `BaseEncoder`，输出 `EncoderOut` |
| `encoder/km/km_encoder_1dCNN.py` | `src/models/encoders/km/cnn1d.py` | 同上 |
| `encoder/video/ResNet50.py` | `src/models/encoders/video/resnet2d.py` | 同上 |
| `legacy/lft_va_src/models/late_fusion_transformer.py` | 拆分为 `fusions/lft.py` + `heads/regression.py` | 融合逻辑与 head 分离 |
| `legacy/lft_va_src/models/components/` | `src/models/components/` | 保持，被 encoder 和 fusion 引用 |
| `legacy/lft_va_src/datasets/multimodal_dataset.py` | `src/data/datamodules/amucs.py` | 输出改为新 batch 格式 |
| `legacy/lft_va_train_lft.py` | `scripts/train.py` | 重写为极薄入口 |
| `legacy/lft_va_configs/default.yaml` | `configs/base.yaml` | 填充完整默认配置 |

---

## 14. 总结

```
                   ┌──────────────────────────┐
                   │     configs/*.yaml       │  ← 变化在这里
                   └────────────┬─────────────┘
                                │
                   ┌────────────▼─────────────┐
                   │   scripts/train.py       │  ← 极薄，不改
                   │   cfg → Runner.fit()     │
                   └────────────┬─────────────┘
                                │
           ┌────────────────────▼────────────────────┐
           │           src/core/runner.py             │  ← 稳定，很少改
           │                                         │
           │  dm = DATAMODULES.build(cfg)             │
           │  enc = {m: ENCODERS[m].build(cfg)}       │
           │  fus = FUSIONS.build(cfg)                │
           │  head = HEADS.build(cfg)                 │
           │  loss = LOSSES.build(cfg)                │
           └──────────┬──────────────────────┬───────┘
                      │                      │
         ┌────────────▼──────┐    ┌──────────▼──────────┐
         │  src/data/*       │    │  src/models/*        │  ← 扩展在这里
         │  (DataModules)    │    │  (Encoders/Fusions/  │
         │                   │    │   Heads)             │
         └───────────────────┘    └─────────────────────┘
                                           │
                                  通过 @register 自注册
                                  新增模块零改动主流程
```

**核心结论**：是的，你需要预留接口。4 个抽象接口 + Registry 机制是实现"新增模块不改训练流程"的最小且充分的方案。接口保证兼容性，Registry 保证可发现性。两者结合让你的框架具备真正的可扩展性。

