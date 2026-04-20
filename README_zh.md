# ProjectExperiment（中文版）

这是一个面向**游戏情绪建模**的可扩展多模态时序学习框架，支持 **EFT（Early Fusion Transformer）**、**MFT（Mid Fusion Transformer）**、**LFT（Late Fusion Transformer）** 和 **CMA（Cross-Modal Attention）** 等多种融合架构。框架基于插件化架构设计，通过冻结接口和注册表机制，支持在不改动核心训练代码的前提下扩展新模态、新编码器和新任务。

---

## 目录

- [背景与数据集](#背景与数据集)
- [支持的任务](#支持的任务)
- [模型架构](#模型架构)
  - [整体流程](#整体流程)
  - [模态编码器](#模态编码器)
  - [融合方法](#融合方法)
  - [CMA 融合层（Cross-Modal Attention）](#cma-融合层cross-modal-attention)
  - [任务头](#任务头)
- [实验设计](#实验设计)
- [框架设计原则](#框架设计原则)
- [代码结构](#代码结构)
- [安装环境](#安装环境)
- [数据准备](#数据准备)
- [使用方法](#使用方法)
  - [启动训练](#启动训练)
  - [配置系统](#配置系统)
  - [模态组合](#模态组合)
  - [混合多任务（Arousal 回归 + Trend 分类）](#混合多任务arousal-回归--trend-分类)
- [扩展新组件](#扩展新组件)
- [测试](#测试)
- [输出目录结构](#输出目录结构)
- [结果汇总](#结果汇总)

---

## 背景与数据集

本框架专为 **AMuCS**（Adaptive Multimodal Computer Systems）游戏数据集而构建，目标是从游戏过程中采集的多模态行为信号预测参与者的情绪状态（唤醒度/效价）。实验中使用的三类模态如下：

| 模态 | 信号来源 | 特征维度 |
|---|---|---|
| `video` | 逐帧 ResNet-50 特征 | 2048 |
| `km` | 键鼠统计特征（按键频率、鼠标速度等） | 25 |
| `telem` | 游戏遥测统计数据（血量、得分等） | 可变 |

情绪标注为连续时序分值（唤醒度/效价），通过自我报告方式采集，与会话时间戳对齐。

---

## 支持的任务

| 任务类型 | 任务头 | 损失函数 | 评估指标 | 配置文件后缀 |
|---|---|---|---|---|
| 单任务 arousal 回归 | `regression` | Smooth L1 | CCC, RMSE | `*_arousal.yaml` |
| 单任务三分类 | `classification` | Masked CE | Macro-F1, Balanced Acc | `*_state.yaml` |
| 多任务分类（state + trend） | `multitask_seq` | 多任务 Masked CE | 各任务 F1 | `*_multitask_state_trend.yaml` |
| **混合多任务（arousal 回归 + trend 分类）** | `multitask_mixed_seq` | MSE + CE | CCC/RMSE + F1/Acc | `*_multitask_arousal_trend.yaml` |

所有任务共用同一融合 backbone（EFT、MFT、LFT、CMA 等），仅任务头、损失函数和标签格式不同。

---

## 模型架构

### 整体流程

```
┌──────────────────────────────────────────────────────────────────────┐
│                    来自 DataModule 的 Batch                          │
│  x:    {video:[B,T_v,2048], km:[B,T_k,25], telem:[B,T_t,D_t]}       │
│  mask: {video:[B,T_v],      km:[B,T_k],    telem:[B,T_t]}            │
│  y:    [B,T]（回归）或 {task: [B,T]}（多任务）                       │
└──────────┬────────────────────────┬───────────────────────┬──────────┘
           │                        │                       │
           ▼                        ▼                       ▼
   ┌───────────────┐     ┌──────────────────┐     ┌────────────────────┐
   │ VideoResNet2d │     │  KMStatEncoder   │     │  TelemStatPool     │
   │     Encoder   │     │    Encoder       │     │    Encoder         │
   │ [B,T_v,2048]  │     │ [B,T_k,25]       │     │ [B,T_t,D_t]        │
   │       ↓       │     │       ↓          │     │        ↓           │
   │ 线性投影+LN   │     │   线性投影+LN    │     │  线性投影+LN       │
   │       ↓       │     │       ↓          │     │        ↓           │
   │ tokens[B,T,D] │     │ tokens[B,T,D]    │     │ tokens[B,T,D]      │
   │ pooled[B,D]   │     │ pooled[B,D]      │     │ pooled[B,D]        │
   └───────┬───────┘     └────────┬─────────┘     └──────────┬─────────┘
           │                      │                          │
           └──────────────────────┴──────────────────────────┘
                                  │  z_dict: {mod: EncoderOut}
                                  ▼
               ┌──────────────────────────────────────────┐
               │     融合层（EFT / MFT / LFT / CMA）       │
               │                                           │
               │  详见下方"融合方法"章节                    │
               │              ↓                            │
               │  Mask 感知池化（mean/max/cls）             │
               │    → pooled [B, D]                        │
               │    → tokens [B, T_total, D]               │
               └──────────────────┬────────────────────────┘
                                  │  FusionOut
                                  ▼
                        ┌──────────────────────┐
                        │        任务头         │
                        │  回归:    [B, 1]      │
                        │  分类:    [B, T, C]   │
                        │  混合:    dict[task]  │
                        └──────────┬────────────┘
                                   │
                                   ▼
                         ┌─────────────────────┐
                         │  Mask 感知 Loss       │
                         │  + 评估指标          │
                         └─────────────────────┘
```

### 模态编码器

所有编码器均实现冻结的 `BaseEncoder` 接口，返回统一的 `EncoderOut` TypedDict：

```python
EncoderOut = {
    "tokens": Tensor[B, T, D],   # 每个时间步的表示
    "pooled": Tensor[B, D],      # tokens 的 mask 感知均值
    "mask":   Tensor[B, T],      # bool，True 表示有效时间步
}
```

#### `video` / `resnet2d` — `VideoResNet2dEncoder`

使用**离线预提取**的逐帧 ResNet-50 特征（2048 维），训练时不运行 CNN，保持训练循环轻量同时保留时序分辨率。

```
输入：[B, T_v, 2048]
  → Linear(2048 → D) + LayerNorm + Dropout
  → tokens: [B, T_v, D]
  → pooled: mask 感知时序均值  [B, D]
```

#### `km` / `stat` — `KMStatEncoder`

对预计算的键鼠统计特征（滑动窗口内的按键频率、鼠标速度统计等）进行轻量线性投影。

```
输入：[B, T_k, 25]
  → Linear(25 → D) + LayerNorm
  → tokens: [B, T_k, D]
  → pooled: mask 感知时序均值  [B, D]
```

#### `telem` / `stat_pool` — `TelemStatPoolEncoder`

游戏遥测统计数据（玩家血量、得分、位置导数等），编码方式类似 KM，带额外池化步骤。

### 融合方法

框架提供三种基于 Transformer 的融合架构（EFT、MFT、LFT），核心区别在于**跨模态交互首次发生的阶段**：

```
EFT（早期融合 Transformer）：
  video → encoder ─┐
  km    → encoder ─┼─ concat(time_dim) + mod_emb → 共享 Transformer → head
  telem → encoder ─┘
  跨模态交互：从 Transformer 第 1 层开始

MFT（中期融合 Transformer）：
  video → encoder → 独立 Transformer ─┐
  km    → encoder → 独立 Transformer ─┼─ cross-attention 层 → concat → head
  telem → encoder → 独立 Transformer ─┘
  跨模态交互：在私有层之后，通过 cross-attention

LFT（晚期融合 Transformer）：
  video → encoder → 独立 Transformer → pool ─┐
  km    → encoder → 独立 Transformer → pool ─┼─ 注意力加权融合 → head
  telem → encoder → 独立 Transformer → pool ─┘
  跨模态交互：仅在最终融合阶段
```

#### Early Fusion Transformer (EFT) — `eft`

所有模态 token 沿时间轴拼接，通过共享 Transformer 处理。跨模态交互从第一层 self-attention 即开始。

| 参数 | 默认值 | 说明 |
|---|---|---|
| `d_model` | 512 | 隐层维度 |
| `nhead` | 8 | 注意力头数 |
| `num_layers` | 4 | Transformer 编码器层数 |
| `dim_feedforward` | 1024 | FFN 隐层大小 |
| `pooling` | `mean` | 池化方式：`mean`、`max`、`cls` |

#### Mid Fusion Transformer (MFT) — `mft`

各模态先经过独立 Transformer 私有层提取模态特有特征，再通过 cross-attention 层进行跨模态信息交换。

| 参数 | 默认值 | 说明 |
|---|---|---|
| `num_private_layers` | 2 | 各模态独立 Transformer 层数 |
| `num_cross_layers` | 2 | 跨模态 cross-attention 层数 |
| `d_model` | 512 | 隐层维度 |
| `nhead` | 8 | 注意力头数 |
| `dim_feedforward` | 1024 | FFN 隐层大小 |

#### Late Fusion Transformer (LFT) — `lft`

各模态由独立 Transformer 编码器完整处理，编码过程无跨模态信息流。最终通过注意力加权融合模态表示。

| 参数 | 默认值 | 说明 |
|---|---|---|
| `d_model` | 512 | 隐层维度 |
| `nhead` | 8 | 注意力头数 |
| `num_layers` | 4 | 各模态独立 Transformer 层数 |
| `dim_feedforward` | 1024 | FFN 隐层大小 |
| `pooling` | `mean` | 各模态融合前的池化方式 |

#### 所有融合变体

| 融合名称 | 注册名 | 跨模态交互 | 实现文件 |
|---|---|---|---|
| `eft` | Early Fusion Transformer | 从第 1 层开始（共享 self-attention） | `src/models/fusions/eft.py` |
| `mft` | Mid Fusion Transformer | 私有层之后（cross-attention） | `src/models/fusions/mft.py` |
| `lft` | Late Fusion Transformer | 仅最终融合阶段 | `src/models/fusions/lft.py` |
| `late` | Late Fusion（平均） | 无（独立 Transformer + 平均） | `src/models/fusions/late.py` |
| `cma` | Cross-Modal Attention | 定向跨模态注意力 + 自注意力 | `src/models/fusions/cma.py` |
| `gated` | Gated Fusion | Sigmoid 门控 | `src/models/fusions/gated.py` |
| `single` | 单模态直通 | 不适用 | `src/models/fusions/single.py` |
| `aligned_mean` | 时间对齐均值 | 简单平均 | `src/models/fusions/aligned_mean.py` |

### CMA 融合层（Cross-Modal Attention）

CMA 是 EFT 的替代融合模块，旨在解决 EFT 的一个局限：当各模态序列长度差异较大时（如 video token 远多于 km），短序列模态在 EFT 的共享自注意力中容易被"淹没"。

**与 EFT 的核心区别**：CMA 不是将所有 token 拼接后统一做自注意力，而是先通过**定向跨模态注意力**让非锚点模态关注锚点模态（默认为 `video`），再通过自注意力进行最终融合。

**架构细节：**

```
各模态 Encoder → EncoderOut
                │
                ▼
┌───────────────────────────────────────────────────┐
│            跨模态注意力阶段                         │
│                                                    │
│  anchor (video) ──────────────────────────┐        │
│                                           │        │
│  km tokens ─── CrossModalTransformer ◄────┘        │
│                  Q=km, K/V=video                   │
│                  (D 层)                ┌───┘        │
│  telem tokens ─ CrossModalTransformer ◄┘           │
│                  Q=telem, K/V=video                │
│                                                    │
│  （anchor tokens 不经跨模态注意力，直接传递）       │
└────────────────────┬──────────────────────────────┘
                     │
                     ▼
┌───────────────────────────────────────────────────┐
│             自注意力精炼阶段                        │
│                                                    │
│  模态嵌入 + 拼接                                   │
│    [B, T_v + T_k + T_t, D]                        │
│              ↓                                     │
│  TransformerEncoder（N_sa 层，Pre-LN）              │
│              ↓                                     │
│  Mask 感知池化 → pooled [B, D]                     │
└───────────────────────────────────────────────────┘
```

**EFT 与 CMA 对比：**

| 方面 | EFT | CMA |
|---|---|---|
| 跨模态交互 | 隐式（共享自注意力） | 显式定向跨模态注意力 |
| Token 处理 | 所有模态 token 平等拼接 | 锚点模态作为 K/V 源 |
| 注意力层数 | `num_layers` 层自注意力 | `cm_layers` 层跨模态 + `sa_layers` 层自注意力 |
| 时序对齐 | 通过位置编码 | 天然支持（Q 与 K/V 长度可不同） |
| 模态非对称性 | 无 | 锚点模态作为信息源享有特权地位 |

**降级行为** — CMA 在模态缺失时优雅降级：
- 锚点模态缺失时，跳过跨模态注意力（等价于 EFT）
- 单模态输入时，退化为纯自注意力

**CMA 关键参数（通过 YAML 配置）：**

| 参数 | 默认值 | 说明 |
|---|---|---|
| `cm_layers` | 4 | 跨模态注意力层数 |
| `sa_layers` | 2 | 自注意力精炼层数 |
| `anchor_modality` | `video` | 作为跨模态注意力 K/V 源的模态 |
| `nhead` | 8 | 注意力头数 |
| `dim_feedforward` | 1024 | FFN 隐层大小 |
| `dropout` | 0.1 | Dropout 率 |
| `pooling` | `mean` | 池化方式：`mean`、`max`、`cls` |

### 任务头

所有任务头均实现 `BaseHead` 接口，接受 `FusionOut` 字典作为输入。

| 任务头名称 | 输出形状 | 说明 |
|---|---|---|
| `regression` | `[B, out_dim]` | 从 pooled 表示输出密集预测；`out_dim=1` 对应 arousal |
| `classification`（seq） | `[B, T, num_classes]` | 基于 token 序列的逐时间步 logits |
| `multitask_seq` | `{task: [B, T, C]}` | 多分类分支 |
| `multitask_mixed_seq` | `{arousal: [B,T,1], trend: [B,T,3]}` | 混合回归 + 分类分支 |

---

## 实验设计

### 模态组合

所有实验均覆盖 **7 种模态组合**，以评估各模态的独立贡献及互补性：

| 组合 | 配置文件 |
|---|---|
| 仅 video | `*_video_*.yaml` |
| 仅 km | `*_km_*.yaml` |
| 仅 telem | `*_telem_*.yaml` |
| video + km | `*_video_km_*.yaml` |
| video + telem | `*_video_telem_*.yaml` |
| km + telem | `*_km_telem_*.yaml` |
| video + km + telem | `*_video_km_telem_*.yaml` |

### 可复现性

每次运行使用固定的随机种子（默认 42），控制 PyTorch、NumPy 和 Python 随机状态。结果通常报告为 3 个种子（0、1、42）的均值 ± 标准差。

### 损失函数

| 任务 | 损失函数 | 说明 |
|---|---|---|
| Arousal 回归 | Smooth L1（可配置为 MSE、CCC） | 仅对有效（非 mask）时间步计算 |
| State/Trend 分类 | Masked 交叉熵 | 忽略填充时间步 |
| 混合多任务 | `w_reg * MSE + w_cls * CE` | 各任务权重可在配置中设置 |

### 评估指标

| 任务 | 主指标 | 辅助指标 |
|---|---|---|
| 回归 | CCC（一致相关系数） | RMSE |
| 分类 | Macro-F1 | Balanced Accuracy |
| 混合多任务 | `val_score_mixed = 0.5*CCC + 0.5*F1` | 各任务独立指标 |

### 早停

基于 patience 的早停策略，监控 `val_ccc`（回归）或 `val_score_mixed`（混合多任务），自动恢复最优 checkpoint。

---

## 框架设计原则

代码库采用**"接口 + 注册表 + 配置"**模式，实现零改动式扩展。

### 冻结接口（`src/core/types.py`）

四个抽象基类定义了不可变的设计契约：

```
BaseEncoder    → EncoderOut {tokens, pooled, mask}
BaseFusion     → FusionOut  {tokens, pooled}
BaseHead       → Tensor[B, out_dim]
BaseDataModule → 产出标准 Batch 字典
```

这些接口**永远不会改变**，所有新实现必须遵守这些契约。

### 注册表系统（`src/core/registry.py`）

模块通过装饰器自注册：

```python
@get_encoder_registry("km").register("stat")
class KMStatEncoder(BaseEncoder): ...

@FUSIONS.register("lft")
class LFTFusion(BaseFusion): ...
```

Runner 在运行时通过字符串 key 查找组件，无需 `if/else` 链或硬编码 import。新增组件只需：

1. 新建文件，实现相应接口
2. 添加 `@registry.register("name")` 装饰器
3. 修改配置：`model.fusion.name: my_new_fusion`

### Batch 格式

所有 DataModule 产出统一的 `Batch` 结构：

```python
{
    "x":    {modality: Tensor[B, T, D], ...},  # 任意模态子集
    "mask": {modality: Tensor[B, T],    ...},  # True 表示有效时间步
    "y":    Tensor[B, T] | {task: Tensor},     # 标签
    "meta": {...},                             # 会话元数据（可选）
}
```

---

## 代码结构

```
ProjectExperiment/
├── src/
│   ├── core/
│   │   ├── types.py          # 冻结接口与 TypedDict
│   │   ├── registry.py       # 插件注册系统
│   │   ├── runner.py         # 训练编排
│   │   ├── config.py         # YAML 配置（支持继承）
│   │   ├── logging.py        # 运行目录管理
│   │   └── seed.py           # 可复现性控制
│   ├── data/
│   │   └── datamodules/
│   │       ├── amucs_seq.py              # 基础时序 DataModule
│   │       └── amucs_seq_multitask.py    # 多任务扩展
│   ├── models/
│   │   ├── encoders/
│   │   │   ├── km/           # stat, cnn1d
│   │   │   ├── video/        # resnet2d, emotieff
│   │   │   └── telem/        # stat_pool
│   │   ├── fusions/
│   │   │   ├── lft.py          # Late Fusion Transformer
│   │   │   ├── cma.py          # Cross-Modal Attention 融合
│   │   │   ├── single.py       # 单模态直通
│   │   │   └── aligned_mean.py
│   │   ├── heads/
│   │   │   ├── regression.py
│   │   │   ├── multitask_seq.py
│   │   │   └── multitask_mixed_seq.py
│   │   └── components/       # 共享组件（位置编码等）
│   ├── losses/               # ccc, mse, multitask_mixed_seq_loss 等
│   └── metrics/              # ccc, rmse, macro_f1, balanced_acc 等
├── configs/
│   ├── base.yaml             # 全局默认配置
│   ├── amucs_seq_lft_*_multitask_arousal_trend.yaml   # LFT，7 种组合
│   ├── amucs_seq_cma_*_multitask_arousal_trend.yaml   # CMA，7 种组合
│   └── amucs_seq_lft_*_multitask_state_trend.yaml     # 7 种组合
├── scripts/
│   ├── train.py              # 训练主入口
│   ├── merge_arousal_reg_trend_labels.py
│   └── summarize.py          # 结果汇总
├── tests/
│   └── test_shapes.py        # 形状契约测试
├── docs/                     # 技术设计文档
├── runs/                     # 训练输出（已 gitignore）
└── legacy/                   # 历史遗留代码
```

---

## 安装环境

```bash
# 克隆仓库
git clone <repo_url>
cd ProjectExperiment

# 安装依赖
pip install torch torchvision
pip install pyyaml numpy pandas scikit-learn tqdm pytest
```

**环境要求：**
- Python 3.10+
- PyTorch ≥ 1.9（推荐 CUDA）
- torchvision

---

## 数据准备

### 特征预提取

所有模态特征须离线预提取并按会话组织存储：

```
data/features/aligned/
└── {session_stem}/
    ├── video_features.npy    # [T_v, 2048]   ResNet-50 逐帧特征
    ├── km_features.npy       # [T_k, 25]     键鼠统计特征
    └── telem_features.npy    # [T_t, D_t]    遥测特征
```

### 标签文件

**单任务回归**（arousal）：
```json
{
  "<session_stem>": {
    "values": [0.12, 0.35, ...],
    "mask":   [true, true, ...]
  }
}
```

**混合多任务**（arousal 回归 + trend 分类）：

使用提供的合并脚本生成联合标签文件：

```bash
python scripts/merge_arousal_reg_trend_labels.py \
  --arousal /path/to/arousal_seq_z_perparticipant.json \
  --trend   /path/to/arousal_3trend_seq.json \
  --output  /path/to/arousal_reg_trend_seq.json
```

输出格式：
```json
{
  "<session_stem>": {
    "arousal": {"values": [0.12, 0.35, ...], "mask": [true, true, ...]},
    "trend":   {"values": [1, 2, 0, ...],    "mask": [true, true, ...]}
  }
}
```

### 数据划分文件

训练/验证/测试划分：
```json
{
  "train": ["session_001", "session_002", ...],
  "val":   ["session_010", ...],
  "test":  ["session_020", ...]
}
```

---

## 使用方法

### 启动训练

```bash
# 单任务 arousal 回归，video + km
python scripts/train.py \
  --config configs/base.yaml \
  --override \
    data.data_root=/path/to/features/aligned \
    data.labels_path=/path/to/arousal_labels.json \
    data.split_path=/path/to/session_tvt.json \
    train.seed=0

# 混合多任务（arousal 回归 + trend 分类），全部三模态
python -u scripts/train.py \
  --config configs/amucs_seq_lft_video_km_telem_multitask_arousal_trend.yaml \
  --override \
    data.data_root=/path/to/features/aligned \
    data.labels_seq_path=/path/to/arousal_reg_trend_seq.json \
    data.split_path=/path/to/session_tvt.json \
    train.seed=42
```

### 配置系统

配置文件使用带 `_base_` 继承的分层 YAML，CLI 覆盖项使用点号语法：

```bash
python scripts/train.py \
  --config configs/base.yaml \
  --override model.fusion.name=single model.fusion.num_layers=2 train.seed=1
```

**核心配置项（`configs/base.yaml`）：**

```yaml
data:
  name: amucs                    # DataModule 注册表 key
  modalities: [video, km]        # 激活的模态列表
  normalize: true                # 按参与者 z-score 归一化

model:
  d_model: 512                   # 共享模型维度（所有组件统一）
  encoders:
    video:
      name: resnet2d             # Encoder 注册表 key
      feature_dim: 2048
      dropout: 0.1
    km:
      name: stat
      feature_dim: 25
  fusion:
    name: lft                    # Fusion 注册表 key
    nhead: 8
    num_layers: 4
    dim_feedforward: 1024
    dropout: 0.1
    pooling: mean
  head:
    name: regression             # Head 注册表 key
    hidden_dim: 128
    out_dim: 1

train:
  loss: smooth_l1                # Loss 注册表 key
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

### 模态组合

单模态实验示例（以 KM 为例）：

```bash
python scripts/train.py \
  --config configs/base.yaml \
  --override data.modalities=[km] model.fusion.name=single
```

### 混合多任务（Arousal 回归 + Trend 分类）

七种模态组合均有预构建配置：

| 模态 | 配置文件 |
|---|---|
| 仅 video | `configs/amucs_seq_lft_video_multitask_arousal_trend.yaml` |
| 仅 km | `configs/amucs_seq_lft_km_multitask_arousal_trend.yaml` |
| 仅 telem | `configs/amucs_seq_lft_telem_multitask_arousal_trend.yaml` |
| video + km | `configs/amucs_seq_lft_video_km_multitask_arousal_trend.yaml` |
| video + telem | `configs/amucs_seq_lft_video_telem_multitask_arousal_trend.yaml` |
| km + telem | `configs/amucs_seq_lft_km_telem_multitask_arousal_trend.yaml` |
| video + km + telem | `configs/amucs_seq_lft_video_km_telem_multitask_arousal_trend.yaml` |

混合多任务头输出：
- `arousal` 分支：`[B, T, 1]` — 连续值回归
- `trend` 分支：`[B, T, 3]` — 三分类 logits（下降 / 平稳 / 上升）

早停监控指标：`val_score_mixed = 0.5 * val_ccc_arousal + 0.5 * val_macro_f1_trend`。

### Notebook 工作流

主 notebook `train.ipynb` 包含批量实验的预配置单元：

| 单元 | 内容 |
|---|---|
| Cell 26 | State + Trend 多任务分类（7 组合 × 3 种子） |
| Cell 27 | 混合多任务：arousal 回归 + trend 分类（7 组合 × 3 种子） |
| Cell 28 | 回归实验的 Lag Sweep 分析 |

---

## 扩展新组件

### 新增 Encoder

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

然后在配置中设置：`model.encoders.km.name: transformer`。**无需修改其他任何文件。**

### 新增 Fusion 方法

```python
# src/models/fusions/mult.py
from src.core.registry import FUSIONS
from src.core.types import BaseFusion, FusionOut

@FUSIONS.register("mult")
class MulTFusion(BaseFusion):
    def forward(self, z_dict, mask_dict) -> FusionOut:
        # 必须能处理任意数量的模态子集
        ...
```

### 新增模态

1. DataModule 输出 `x["new_mod"]` 和 `mask["new_mod"]`
2. 创建 `src/models/encoders/new_mod/name.py` 并注册
3. 配置中添加：`data.modalities: [..., new_mod]` 和 `model.encoders.new_mod: {...}`
4. Fusion 层自动处理（动态遍历 key），**无需修改 fusion 代码**

### 扩展清单

| 扩展类型 | 需要创建的文件 | 需要修改的文件 |
|---|---|---|
| 新增 encoder | `src/models/encoders/{mod}/{name}.py` | 仅配置文件 |
| 新增模态 | Encoder 文件 + DataModule 扩展 | 仅配置文件 |
| 新增 fusion | `src/models/fusions/{name}.py` | 仅配置文件 |
| 新增任务头 | `src/models/heads/{name}.py` | 仅配置文件 |
| 新增 loss | `src/losses/{name}.py` | 仅配置文件 |
| 新增数据集 | `src/data/datamodules/{name}.py` | 仅配置文件 |

---

## 测试

测试重点关注**形状契约**，确保所有组件符合冻结接口规范，新增组件不会破坏现有功能。

```bash
# 运行所有测试
pytest tests/

# 详细输出
pytest tests/ -v

# 指定测试文件
pytest tests/test_shapes.py -v
```

`tests/test_shapes.py` 中的主要测试类别：
- Encoder 输出形状验证（tokens/pooled/mask 维度）
- Fusion 对 1~N 任意模态子集的处理能力
- 任务头输出形状验证
- 端到端前向传播（DataModule → Encoder → Fusion → Head → Loss）
- Loss 返回标量，指标返回浮点数

---

## 输出目录结构

每次训练创建独立的自包含目录：

```
runs/{timestamp}__{dataset}__{fusion}__{modalities}__seed{seed}/
├── config.yaml        # 完整合并后的配置（精确复现用）
├── seed.txt           # 使用的随机种子
├── git_commit.txt     # 训练时的 Git commit hash
├── ckpt_best.pt       # 最优模型 checkpoint（按验证指标）
├── ckpt_last.pt       # 最后一个 epoch 的 checkpoint
└── metrics.json       # 所有记录的指标
```

**目录名示例：**
```
2026-02-04_14-30-22__amucs__lft__video_km__seed42/
```

**`metrics.json` 示例：**
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

## 结果汇总

汇总所有运行结果为排行榜 CSV：

```bash
python scripts/summarize.py
```

生成 `leaderboard.csv`，每行对应一次运行，包含数据集、融合方式、模态组合、种子和所有指标。便于跨实验比较模态消融结果和超参数影响。

---

## 注意事项

- `mask=False` 的时间步完全不参与 loss 计算和指标统计。
- 混合多任务支持各任务独立指标以及可配置的综合指标 `val_score_mixed`（用于早停），避免单一任务主导训练。
- 所有已有的单任务和 state+trend 多任务实验不受混合多任务新增内容的影响。
- `modality_dropout`（配置项）在训练时随机将某模态的 mask 置零，提升推理时对缺失模态的鲁棒性。
- CMA 的详细设计文档参见 `docs/crossmodal_attention_fusion_design.md`。
