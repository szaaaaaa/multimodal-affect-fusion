# ProjectExperiment 代码仓库架构文档

> 面向新人的详细讲解文档，包含实验设计、训练流程、模型结构及完整架构图

---

## 目录

1. [项目概述](#1-项目概述)
2. [目录结构](#2-目录结构)
3. [实验设计](#3-实验设计)
4. [数据处理流程](#4-数据处理流程)
5. [模型架构详解](#5-模型架构详解)
6. [训练流程](#6-训练流程)
7. [完整架构图](#7-完整架构图)
8. [快速上手指南](#8-快速上手指南)

---

## 1. 项目概述

### 1.1 研究目标

这是一个**情感计算 (Affective Computing)** 实验项目，核心目标是：

**通过用户的键盘和鼠标行为数据，预测其情感唤醒度 (Arousal)**

### 1.2 核心假设

1. 用户的键盘敲击频率、鼠标移动速度等行为模式与其情感状态相关
2. 高唤醒度 (紧张/兴奋) 可能表现为更快的键盘输入、更频繁的鼠标移动
3. 低唤醒度 (放松/无聊) 可能表现为较慢、较少的交互行为

### 1.3 技术方案

- **数据源**: AMuCS 数据集 (多模态情感数据集)
- **输入模态**: 键盘事件 + 鼠标事件
- **模型架构**: Transformer Encoder + 回归头
- **任务类型**: 回归任务 (预测连续 Arousal 值)

---

## 2. 目录结构

```
ProjectExperiment/
├── ARCHITECTURE.md              # 本文档
├── .gitignore                   # Git 忽略配置
│
├── encoder/                     # 特征编码器模块
│   ├── README.md
│   ├── requirements.txt
│   ├── face/                    # 面部特征 (预留，未实现)
│   │   └── extract_face_features.py
│   └── km/                      # 键盘/鼠标特征编码器 ⭐核心
│       ├── km_encoder_stat.py       # 统计特征编码器
│       ├── km_encoder_1dCNN.py      # 1D-CNN 编码器
│       ├── extract_km_features.py   # 特征提取脚本
│       ├── build_arousal_labels.py  # 构建 Arousal 标签
│       ├── filter_arousal_ranktrace.py  # 筛选数据
│       └── check_km_features.py     # 特征检查工具
│
└── lft-va/                      # 训练框架 (Late Fusion Transformer for VA)
    ├── README.md
    ├── requirements.txt
    ├── configs/
    │   └── default.yaml
    ├── data/                    # 数据目录 (运行时生成)
    │   ├── features/amucs/km/       # .pt 特征文件
    │   ├── splits/km_arousal_split.json  # 训练/验证划分
    │   ├── labels_arousal.json      # 标签文件
    │   └── km_input_stats.json      # 输入标准化参数
    ├── outputs/                 # 训练输出目录
    │   └── km_arousal_first/YYYYMMDD_HHMMSS/
    │       ├── best.pt              # 最佳模型
    │       ├── last.pt              # 最终模型
    │       ├── metrics.json         # 损失记录
    │       ├── train.log            # 训练日志
    │       └── loss_curve.png       # 损失曲线
    ├── src/lft_va/
    │   ├── __init__.py
    │   ├── datasets/
    │   │   ├── __init__.py
    │   │   ├── dataloader.py
    │   │   └── km_window_dataset.py # 窗口数据集 ⭐核心
    │   ├── models/
    │   │   ├── __init__.py
    │   │   └── km_transformer_min.py # Transformer 模型 ⭐核心
    │   └── utils/
    │       ├── __init__.py
    │       └── config.py
    └── scripts/
        ├── build_km_arousal_split.py    # 数据划分脚本
        └── train_km_arousal_first.py    # 训练脚本 ⭐核心
```

---

## 3. 实验设计

### 3.1 数据预处理管线

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                         数据预处理流程 (Data Pipeline)                        │
└─────────────────────────────────────────────────────────────────────────────┘

  AMuCS 原始数据
  (多个 Session)
       │
       ▼
┌──────────────────┐     ┌──────────────────────────────────────────────┐
│ filter_arousal_  │────▶│ arousal_sessions.json                        │
│ ranktrace.py     │     │ 格式: {"S001_P1": "/path/to/ranktrace.csv"}  │
└──────────────────┘     └──────────────────────────────────────────────┘
       │
       ▼
┌──────────────────┐     ┌──────────────────────────────────────────────┐
│ build_arousal_   │────▶│ labels_arousal.json                          │
│ labels.py        │     │ 格式: {"S001_P1": 0.523, "S001_P2": 0.687}   │
└──────────────────┘     └──────────────────────────────────────────────┘
       │
       ▼
┌──────────────────┐     ┌──────────────────────────────────────────────┐
│ extract_km_      │────▶│ features/*.pt                                │
│ features.py      │     │ 每个 session 一个 .pt 文件                   │
└──────────────────┘     │ 包含: features[T,25], mask[T], meta          │
                         └──────────────────────────────────────────────┘
       │
       ▼
┌──────────────────┐     ┌──────────────────────────────────────────────┐
│ build_km_arousal_│────▶│ km_arousal_split.json                        │
│ split.py         │     │ 格式: {"train": [...], "val": [...]}         │
└──────────────────┘     │ 默认 80/20 划分                              │
                         └──────────────────────────────────────────────┘
```

### 3.2 时间分箱策略 (Time Binning)

原始键鼠事件是连续的时间序列，需要离散化为固定长度的特征向量：

```
时间轴 (秒)
0.0    0.2    0.4    0.6    0.8    1.0    ...
│      │      │      │      │      │
├──────┼──────┼──────┼──────┼──────┼──────►
│ Bin0 │ Bin1 │ Bin2 │ Bin3 │ Bin4 │ ...
│      │      │      │      │      │
│ 25维 │ 25维 │ 25维 │ 25维 │ 25维 │
└──────┴──────┴──────┴──────┴──────┘

dt = 0.2 秒 (200ms) 为一个时间分箱
每个分箱提取 25 维统计特征
```

### 3.3 滑动窗口采样

长序列被切分为固定长度的窗口用于训练：

```
原始序列 (假设 T=600 个时间步，即 120 秒)
├────────────────────────────────────────────────────────────────┤

窗口参数:
  win_len = 300 (60 秒)
  stride  = 150 (30 秒，50% 重叠)

采样结果:
Window 0: [0, 300)     ├────────────────────────────────┤
Window 1: [150, 450)              ├────────────────────────────────┤
Window 2: [300, 600)                          ├────────────────────────────────┤
```

---

## 4. 数据处理流程

### 4.1 原始事件结构 (KMEvent)

```python
@dataclass(frozen=True)
class KMEvent:
    t: float              # 时间戳 (秒)
    kind: str             # 事件类型
    x: Optional[float]    # 鼠标 X 坐标
    y: Optional[float]    # 鼠标 Y 坐标
    button: Optional[str] # 鼠标按钮
    scroll: Optional[float] # 滚轮增量
    key: Optional[str]    # 键名
```

**支持的事件类型 (kind)**:
| kind | 描述 |
|------|------|
| `key_down` | 键盘按键按下 |
| `key_up` | 键盘按键释放 |
| `mouse_move` | 鼠标移动 |
| `mouse_click` | 鼠标按钮按下 |
| `mouse_button_up` | 鼠标按钮释放 |
| `mouse_scroll` | 鼠标滚轮滚动 |

### 4.2 统计特征编码 (25 维)

`KMStatEncoder` 将每个时间分箱内的事件编码为 25 维特征向量：

| 索引 | 特征名称 | 描述 | 类型 |
|------|----------|------|------|
| 0 | `key_down_count` | 按键按下次数 | 计数 |
| 1 | `key_up_count` | 按键释放次数 | 计数 |
| 2 | `mouse_move_event_count` | 鼠标移动事件数 | 计数 |
| 3 | `mouse_move_distance_sum` | 鼠标移动距离总和 | 累计 |
| 4 | `mouse_speed_mean` | 鼠标平均速度 | 统计 |
| 5 | `mouse_speed_max` | 鼠标最大速度 | 统计 |
| 6 | `mouse_button_down_count` | 鼠标按下次数 | 计数 |
| 7 | `left_button_down_count` | 左键按下次数 | 计数 |
| 8 | `right_button_down_count` | 右键按下次数 | 计数 |
| 9 | `mouse_button_up_count` | 鼠标释放次数 | 计数 |
| 10 | `left_button_up_count` | 左键释放次数 | 计数 |
| 11 | `right_button_up_count` | 右键释放次数 | 计数 |
| 12 | `scroll_event_count` | 滚轮事件数 | 计数 |
| 13 | `scroll_delta_sum` | 滚轮增量总和 | 累计 |
| 14 | `inter_key_interval_mean` | 平均按键间隔 | 统计 |
| 15 | `mouse_dx_sum` | 鼠标 X 方向位移 | 累计 |
| 16 | `mouse_dy_sum` | 鼠标 Y 方向位移 | 累计 |
| 17 | `mouse_accel_mean` | 鼠标平均加速度 | 统计 |
| 18 | `key_down_delta` | 按键按下变化量 | 差分 |
| 19 | `key_up_delta` | 按键释放变化量 | 差分 |
| 20 | `key_down_rate` | 按键按下速率 (次/秒) | 速率 |
| 21 | `mouse_click_rate` | 鼠标点击速率 | 速率 |
| 22 | `scroll_rate` | 滚轮速率 | 速率 |
| 23 | `mouse_move_rate` | 鼠标移动速率 | 速率 |
| 24 | `unique_key_count` | 独特按键数量 | 计数 |

### 4.3 特征文件格式 (.pt)

```python
{
    "features": Tensor[T, 25],   # T 个时间步，25 维特征
    "mask": Tensor[T],           # 有效位掩码 (全 True)
    "meta": {
        "modality": "keyboard_mouse",
        "dt": 0.2,               # 时间分箱大小
        "t0": float,             # 起始时间
        "t1": float,             # 结束时间
        "feature_dim": 25,
        "feature_names": [...]   # 特征名列表
    }
}
```

---

## 5. 模型架构详解

### 5.1 整体架构

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                    KMTransformerRegressor 模型架构                           │
└─────────────────────────────────────────────────────────────────────────────┘

输入:
  km      : Tensor[B, L, D]     # B=批次, L=序列长度(300), D=特征维度(25)
  km_mask : Tensor[B, L]        # 有效位掩码

                    ┌─────────────────────────────────────┐
                    │         Input Projection            │
                    │  (KMStatTokenEncoder / KM1DCNNEncoder)  │
                    │     [B, L, 25] → [B, L, 64]         │
                    └─────────────────┬───────────────────┘
                                      │
                                      ▼
                    ┌─────────────────────────────────────┐
                    │       Transformer Encoder           │
                    │         2 layers, 4 heads           │
                    │         d_model = 64                │
                    │     [B, L, 64] → [B, L, 64]         │
                    └─────────────────┬───────────────────┘
                                      │
                                      ▼
                    ┌─────────────────────────────────────┐
                    │      Masked Average Pooling         │
                    │     [B, L, 64] → [B, 64]            │
                    └─────────────────┬───────────────────┘
                                      │
                                      ▼
                    ┌─────────────────────────────────────┐
                    │          Linear Head                │
                    │       [B, 64] → [B, 1]              │
                    └─────────────────┬───────────────────┘
                                      │
                                      ▼
输出:
  arousal : Tensor[B, 1]        # 预测的 Arousal 值
```

### 5.2 输入投影层 (两种选择)

#### 方案 A: KMStatTokenEncoder (默认)

简单的线性投影 + 归一化：

```python
class KMStatTokenEncoder(nn.Module):
    def __init__(self, d_in: int, d_model: int):
        self.proj = nn.Linear(d_in, d_model)    # 25 → 64
        self.ln = nn.LayerNorm(d_model)

    def forward(self, x):
        return self.ln(self.proj(x))
```

```
输入 [B, L, 25]
      │
      ▼
┌──────────────┐
│ Linear(25→64)│
└──────┬───────┘
       │
       ▼
┌──────────────┐
│  LayerNorm   │
└──────┬───────┘
       │
       ▼
输出 [B, L, 64]
```

#### 方案 B: KM1DCNNEncoder

双层 1D 卷积 + 残差连接：

```python
class KM1DCNNEncoder(nn.Module):
    def __init__(self, d_in, d_model, kernel_size=5, dropout=0.1):
        self.conv1 = nn.Conv1d(d_in, d_model, kernel_size, padding=kernel_size//2)
        self.conv2 = nn.Conv1d(d_model, d_model, kernel_size, padding=kernel_size//2)
        self.act = nn.GELU()
        self.drop = nn.Dropout(dropout)
        self.ln = nn.LayerNorm(d_model)
```

```
输入 [B, L, 25]
      │
      ▼ transpose
[B, 25, L]
      │
      ▼
┌──────────────────┐
│ Conv1d(25→64)    │──────────────┐
│ kernel_size=5    │              │
└──────┬───────────┘              │
       │                          │
       ▼                          │
┌──────────────────┐              │
│      GELU        │              │
└──────┬───────────┘              │
       │                          │ 残差连接
       ▼                          │
┌──────────────────┐              │
│   Dropout(0.1)   │              │
└──────┬───────────┘              │
       │                          │
       ▼                          │
┌──────────────────┐              │
│ Conv1d(64→64)    │              │
│ kernel_size=5    │              │
└──────┬───────────┘              │
       │                          │
       ▼                          │
┌──────────────────┐              │
│      GELU        │              │
└──────┬───────────┘              │
       │                          │
       ▼                          │
       +◄─────────────────────────┘
       │
       ▼ transpose
[B, L, 64]
       │
       ▼
┌──────────────────┐
│   LayerNorm      │
└──────┬───────────┘
       │
       ▼
输出 [B, L, 64]
```

### 5.3 Transformer Encoder

```
┌─────────────────────────────────────────────────────────────────┐
│                 TransformerEncoderLayer × 2                      │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  ┌─────────────────────────────────────────────────────────┐    │
│  │              Multi-Head Self-Attention                   │    │
│  │                   nhead = 4                              │    │
│  │                   d_model = 64                           │    │
│  │                   d_k = d_v = 64/4 = 16                  │    │
│  └─────────────────────────┬───────────────────────────────┘    │
│                            │                                     │
│                     Add & LayerNorm                              │
│                            │                                     │
│  ┌─────────────────────────▼───────────────────────────────┐    │
│  │                Feed-Forward Network                      │    │
│  │         Linear(64 → 256) → ReLU → Linear(256 → 64)      │    │
│  └─────────────────────────┬───────────────────────────────┘    │
│                            │                                     │
│                     Add & LayerNorm                              │
│                            │                                     │
└────────────────────────────┼────────────────────────────────────┘
                             │
                             ▼
                        [B, L, 64]
```

**注意力机制中的 Mask 处理**:
- `km_mask` 为 `True` 表示有效位置
- 传入 Transformer 时取反 (`~km_mask`)，因为 `src_key_padding_mask` 中 `True` 表示需要被忽略的位置

### 5.4 Masked Average Pooling

```python
# mask_f: [B, L] float32, 1.0 表示有效, 0.0 表示 padding
denom = mask_f.sum(dim=1, keepdim=True).clamp(min=1.0)  # [B, 1]
pooled = (x * mask_f.unsqueeze(-1)).sum(dim=1) / denom   # [B, 64]
```

```
x       : [B, L, 64]
mask_f  : [B, L]
                    │
                    ▼
        对每个序列只聚合有效位置
        sum(x[mask]) / count(mask)
                    │
                    ▼
pooled  : [B, 64]
```

### 5.5 模型参数量估算

| 组件 | 参数量 |
|------|--------|
| KMStatTokenEncoder | 25×64 + 64 + 64×2 = 1,792 |
| TransformerEncoder (2层) | ≈ 2×(4×64² + 64×256×2) ≈ 98,816 |
| Linear Head | 64×1 + 1 = 65 |
| **总计** | **≈ 100K** |

---

## 6. 训练流程

### 6.1 训练配置

```python
# 超参数
batch_size = 8
learning_rate = 1e-3
epochs = 3
optimizer = Adam
loss_function = smooth_l1_loss (Huber Loss)

# 数据
win_len = 300     # 窗口长度 (60 秒)
stride = 150      # 步长 (30 秒)
normalize = True  # 输入标准化
```

### 6.2 训练循环

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                              训练流程                                        │
└─────────────────────────────────────────────────────────────────────────────┘

初始化:
  ├── 设置随机种子 (seed=123)
  ├── 加载训练集 KMWindDataset(split="train")
  ├── 加载验证集 KMWindDataset(split="val")
  ├── 初始化 KMTransformerRegressor 模型
  └── 初始化 Adam 优化器 (lr=1e-3)

For epoch in 1..3:
  │
  ├── 训练阶段 (Train)
  │   │
  │   └── For batch in train_loader:
  │         ├── 前向传播: y_hat = model(km, km_mask)
  │         ├── 计算损失: loss = smooth_l1_loss(y_hat, y)
  │         ├── 反向传播: loss.backward()
  │         └── 参数更新: optimizer.step()
  │
  ├── 验证阶段 (Validation)
  │   │
  │   └── For batch in val_loader:
  │         ├── 前向传播: y_hat = model(km, km_mask)
  │         └── 计算损失: loss = smooth_l1_loss(y_hat, y)
  │
  ├── 记录 metrics
  │
  └── 保存最佳模型 (if val_loss < best_val)

最终输出:
  ├── best.pt        # 验证损失最低的模型
  ├── last.pt        # 最终模型
  ├── metrics.json   # 训练/验证损失记录
  ├── train.log      # 训练日志
  └── loss_curve.png # 损失曲线图
```

### 6.3 损失函数: Smooth L1 Loss (Huber Loss)

```
            ┌ 0.5 × (y - ŷ)²           if |y - ŷ| < 1
L(y, ŷ) = ─┤
            └ |y - ŷ| - 0.5            otherwise
```

特点:
- 对小误差 (< 1) 使用 L2 损失，梯度平滑
- 对大误差 (≥ 1) 使用 L1 损失，对异常值更鲁棒

### 6.4 输入标准化

```python
# 训练集计算统计量
mean = Σx / N
std = sqrt(Σ(x - mean)² / N)

# 标准化
x_norm = (x - mean) / std
```

- 训练集：计算并保存统计量到 `km_input_stats.json`
- 验证集：加载并复用训练集的统计量

---

## 7. 完整架构图

### 7.1 系统全景图

```
╔═══════════════════════════════════════════════════════════════════════════════╗
║                         ProjectExperiment 系统架构                             ║
╠═══════════════════════════════════════════════════════════════════════════════╣
║                                                                               ║
║  ┌─────────────────────────────────────────────────────────────────────────┐  ║
║  │                         数据预处理阶段                                    │  ║
║  │                                                                         │  ║
║  │   AMuCS 原始数据                                                        │  ║
║  │   ┌──────────┐ ┌──────────┐ ┌──────────┐                               │  ║
║  │   │keyboard. │ │mousebut- │ │mousepos- │                               │  ║
║  │   │csv       │ │tons.csv  │ │ition.csv │                               │  ║
║  │   └────┬─────┘ └────┬─────┘ └────┬─────┘                               │  ║
║  │        │            │            │                                      │  ║
║  │        └────────────┼────────────┘                                      │  ║
║  │                     ▼                                                   │  ║
║  │   ┌─────────────────────────────────────────────────────────────────┐  │  ║
║  │   │                    KMStatEncoder                                 │  │  ║
║  │   │                                                                  │  │  ║
║  │   │  ┌─────────┐    ┌─────────────┐    ┌─────────────────────────┐  │  │  ║
║  │   │  │ 原始事件 │ ─▶ │ 时间分箱     │ ─▶ │ 25 维统计特征向量        │  │  │  ║
║  │   │  │ KMEvent │    │ dt=0.2s    │    │ [T, 25] Tensor          │  │  │  ║
║  │   │  └─────────┘    └─────────────┘    └─────────────────────────┘  │  │  ║
║  │   │                                                                  │  │  ║
║  │   └─────────────────────────────────────────────────────────────────┘  │  ║
║  │                     │                                                   │  ║
║  │                     ▼                                                   │  ║
║  │   ┌─────────────────────────────────────────────────────────────────┐  │  ║
║  │   │  features/S001_P1.pt, S001_P2.pt, ...                           │  │  ║
║  │   └─────────────────────────────────────────────────────────────────┘  │  ║
║  │                                                                         │  ║
║  └─────────────────────────────────────────────────────────────────────────┘  ║
║                                    │                                          ║
║                                    ▼                                          ║
║  ┌─────────────────────────────────────────────────────────────────────────┐  ║
║  │                           数据加载阶段                                    │  ║
║  │                                                                         │  ║
║  │   ┌───────────────────────────────────────────────────────────────┐    │  ║
║  │   │                     KMWindDataset                             │    │  ║
║  │   │                                                               │    │  ║
║  │   │   ┌──────────────┐     ┌──────────────┐     ┌─────────────┐  │    │  ║
║  │   │   │ 加载 .pt 文件 │ ──▶ │ 滑动窗口切分  │ ──▶ │ 输入标准化   │  │    │  ║
║  │   │   │              │     │ win=300     │     │ z-score     │  │    │  ║
║  │   │   │              │     │ stride=150  │     │             │  │    │  ║
║  │   │   └──────────────┘     └──────────────┘     └─────────────┘  │    │  ║
║  │   │                                                               │    │  ║
║  │   │   输出: {"km": [300,25], "km_mask": [300], "y": [1]}         │    │  ║
║  │   │                                                               │    │  ║
║  │   └───────────────────────────────────────────────────────────────┘    │  ║
║  │                                                                         │  ║
║  └─────────────────────────────────────────────────────────────────────────┘  ║
║                                    │                                          ║
║                                    ▼                                          ║
║  ┌─────────────────────────────────────────────────────────────────────────┐  ║
║  │                            模型训练阶段                                   │  ║
║  │                                                                         │  ║
║  │   ┌───────────────────────────────────────────────────────────────┐    │  ║
║  │   │                 KMTransformerRegressor                        │    │  ║
║  │   │                                                               │    │  ║
║  │   │   ┌──────────────┐                                            │    │  ║
║  │   │   │ km [B,300,25]│                                            │    │  ║
║  │   │   └──────┬───────┘                                            │    │  ║
║  │   │          │                                                    │    │  ║
║  │   │          ▼                                                    │    │  ║
║  │   │   ┌──────────────────────────────────────────────────┐       │    │  ║
║  │   │   │          Input Projection Layer                  │       │    │  ║
║  │   │   │   ┌────────────────┐  OR  ┌────────────────┐     │       │    │  ║
║  │   │   │   │ KMStatToken-   │      │ KM1DCNNEncoder │     │       │    │  ║
║  │   │   │   │ Encoder        │      │ (双层Conv1d+   │     │       │    │  ║
║  │   │   │   │ (Linear+LN)    │      │  残差)         │     │       │    │  ║
║  │   │   │   └────────────────┘      └────────────────┘     │       │    │  ║
║  │   │   │                  [B, 300, 64]                    │       │    │  ║
║  │   │   └──────────────────────┬───────────────────────────┘       │    │  ║
║  │   │                          │                                    │    │  ║
║  │   │                          ▼                                    │    │  ║
║  │   │   ┌──────────────────────────────────────────────────┐       │    │  ║
║  │   │   │           Transformer Encoder                    │       │    │  ║
║  │   │   │                                                  │       │    │  ║
║  │   │   │   ┌────────────────────────────────────────┐    │       │    │  ║
║  │   │   │   │  TransformerEncoderLayer × 2            │    │       │    │  ║
║  │   │   │   │  • Multi-Head Self-Attention (4 heads)  │    │       │    │  ║
║  │   │   │   │  • Feed-Forward Network                 │    │       │    │  ║
║  │   │   │   │  • d_model = 64                         │    │       │    │  ║
║  │   │   │   └────────────────────────────────────────┘    │       │    │  ║
║  │   │   │                  [B, 300, 64]                    │       │    │  ║
║  │   │   └──────────────────────┬───────────────────────────┘       │    │  ║
║  │   │                          │                                    │    │  ║
║  │   │                          ▼                                    │    │  ║
║  │   │   ┌──────────────────────────────────────────────────┐       │    │  ║
║  │   │   │         Masked Average Pooling                   │       │    │  ║
║  │   │   │         [B, 300, 64] → [B, 64]                   │       │    │  ║
║  │   │   └──────────────────────┬───────────────────────────┘       │    │  ║
║  │   │                          │                                    │    │  ║
║  │   │                          ▼                                    │    │  ║
║  │   │   ┌──────────────────────────────────────────────────┐       │    │  ║
║  │   │   │              Linear Head                         │       │    │  ║
║  │   │   │              [B, 64] → [B, 1]                    │       │    │  ║
║  │   │   └──────────────────────┬───────────────────────────┘       │    │  ║
║  │   │                          │                                    │    │  ║
║  │   │                          ▼                                    │    │  ║
║  │   │   ┌──────────────┐                                            │    │  ║
║  │   │   │ arousal [B,1]│ ◄── 预测值                                 │    │  ║
║  │   │   └──────────────┘                                            │    │  ║
║  │   │                                                               │    │  ║
║  │   └───────────────────────────────────────────────────────────────┘    │  ║
║  │                          │                                              │  ║
║  │                          ▼                                              │  ║
║  │   ┌───────────────────────────────────────────────────────────────┐    │  ║
║  │   │                   Loss & Optimization                         │    │  ║
║  │   │                                                               │    │  ║
║  │   │   Loss = smooth_l1_loss(arousal_pred, arousal_true)          │    │  ║
║  │   │   Optimizer = Adam(lr=1e-3)                                   │    │  ║
║  │   │                                                               │    │  ║
║  │   └───────────────────────────────────────────────────────────────┘    │  ║
║  │                                                                         │  ║
║  └─────────────────────────────────────────────────────────────────────────┘  ║
║                                    │                                          ║
║                                    ▼                                          ║
║  ┌─────────────────────────────────────────────────────────────────────────┐  ║
║  │                              输出产物                                     │  ║
║  │                                                                         │  ║
║  │   outputs/km_arousal_first/YYYYMMDD_HHMMSS/                            │  ║
║  │   ├── best.pt          # 最佳模型权重                                    │  ║
║  │   ├── last.pt          # 最终模型权重                                    │  ║
║  │   ├── metrics.json     # {"train_loss": [...], "val_loss": [...]}       │  ║
║  │   ├── train.log        # 训练日志                                        │  ║
║  │   └── loss_curve.png   # 损失曲线可视化                                  │  ║
║  │                                                                         │  ║
║  └─────────────────────────────────────────────────────────────────────────┘  ║
║                                                                               ║
╚═══════════════════════════════════════════════════════════════════════════════╝
```

### 7.2 数据流图

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                            数据流转详图                                       │
└─────────────────────────────────────────────────────────────────────────────┘

原始 CSV 文件
keyboard.csv    mousebuttons.csv    mouseposition.csv
     │                 │                   │
     │    ┌────────────┼───────────────────┘
     │    │            │
     ▼    ▼            ▼
┌──────────────────────────────────────┐
│           统一事件格式                 │
│  List[KMEvent]                       │
│  每个事件: (t, kind, x, y, ...)      │
└──────────────────┬───────────────────┘
                   │
                   ▼  时间分箱 (dt=0.2s)
┌──────────────────────────────────────┐
│          分箱统计特征                  │
│  Tensor[T, 25]                       │
│  T = ceil((t1-t0) / 0.2)            │
└──────────────────┬───────────────────┘
                   │
                   ▼  保存为 .pt 文件
┌──────────────────────────────────────┐
│          特征文件                      │
│  {features, mask, meta}              │
└──────────────────┬───────────────────┘
                   │
                   ▼  滑动窗口 + 标准化
┌──────────────────────────────────────┐
│          训练样本                      │
│  km: [300, 25]                       │
│  km_mask: [300]                      │
│  y: [1]                              │
└──────────────────┬───────────────────┘
                   │
                   ▼  DataLoader (batch_size=8)
┌──────────────────────────────────────┐
│          训练批次                      │
│  km: [8, 300, 25]                    │
│  km_mask: [8, 300]                   │
│  y: [8, 1]                           │
└──────────────────┬───────────────────┘
                   │
                   ▼  模型前向传播
┌──────────────────────────────────────┐
│          模型输出                      │
│  arousal_pred: [8, 1]                │
└──────────────────────────────────────┘
```

---

## 8. 快速上手指南

### 8.1 环境准备

```bash
# 1. 安装依赖 (PyTorch, NumPy, Matplotlib)
pip install torch numpy matplotlib

# 2. 进入项目目录
cd /home/user/ProjectExperiment
```

### 8.2 数据预处理 (假设已有 AMuCS 数据)

```bash
# 1. 筛选含 arousal 标注的 session
cd encoder/km
python filter_arousal_ranktrace.py --root_dir /path/to/AMuCS --output arousal_sessions.json

# 2. 构建 arousal 标签
python build_arousal_labels.py --mapping arousal_sessions.json --output labels_arousal.json

# 3. 提取键鼠特征
python extract_km_features.py --root_dir /path/to/AMuCS --output_dir ../../lft-va/data/features/amucs/km

# 4. 划分训练/验证集
cd ../../lft-va/scripts
python build_km_arousal_split.py
```

### 8.3 训练模型

```bash
cd /home/user/ProjectExperiment/lft-va/scripts

# 使用默认 stat 编码器
python train_km_arousal_first.py

# 或使用 CNN 编码器
python train_km_arousal_first.py --km_encoder cnn
```

### 8.4 查看训练结果

```bash
# 输出目录
ls lft-va/outputs/km_arousal_first/YYYYMMDD_HHMMSS/

# 查看损失曲线
open loss_curve.png

# 查看训练日志
cat train.log
```

### 8.5 加载训练好的模型

```python
import torch
from lft_va.models.km_transformer_min import KMTransformerRegressor

# 加载最佳模型
checkpoint = torch.load("outputs/km_arousal_first/YYYYMMDD_HHMMSS/best.pt")
model = KMTransformerRegressor(d_in=25)
model.load_state_dict(checkpoint["model"])
model.eval()

# 推理
with torch.no_grad():
    arousal = model(km_input, km_mask)
```

---

## 附录

### A. 关键代码文件速查

| 文件 | 功能 | 行数 |
|------|------|------|
| `encoder/km/km_encoder_stat.py` | 统计特征编码器 | ~400 |
| `encoder/km/km_encoder_1dCNN.py` | CNN 编码器 | ~40 |
| `lft-va/src/lft_va/models/km_transformer_min.py` | Transformer 模型 | ~60 |
| `lft-va/src/lft_va/datasets/km_window_dataset.py` | 数据集类 | ~250 |
| `lft-va/scripts/train_km_arousal_first.py` | 训练脚本 | ~170 |

### B. 项目未来扩展方向

1. **多模态融合**: 项目名 "lft-va" (Late Fusion Transformer for VA) 暗示可扩展为融合面部表情、键鼠行为等多模态数据
2. **Valence 预测**: 当前只预测 Arousal，可扩展为同时预测 Valence (效价)
3. **更多编码器**: 可添加 LSTM、GRU 等序列编码器
4. **超参数调优**: 当前训练仅 3 epochs，可扩展更多训练配置

---

*文档生成时间: 2026-01-29*
*基于项目 commit: 319853e (initial commit)*
