# CrossModal Attention Fusion (CMA) 设计文档

> 基于 MulT (Tsai et al., ACL 2019) 的跨模态注意力机制，
> 针对 AMuCS 游戏情绪数据集定制的 **video→{km, telem} 单向跨模态融合**方案。

---

## 目录

- [1. 动机与背景](#1-动机与背景)
- [2. 原始 MulT 架构回顾](#2-原始-mult-架构回顾)
- [3. 我们的方案：CMA Fusion](#3-我们的方案cma-fusion)
  - [3.1 核心思路](#31-核心思路)
  - [3.2 完整数据流图](#32-完整数据流图)
  - [3.3 各阶段详细说明](#33-各阶段详细说明)
  - [3.4 输出与接口](#34-输出与接口)
- [4. 与 LFT 的对比](#4-与-lft-的对比)
- [5. 在现有框架上的最小扩展方案](#5-在现有框架上的最小扩展方案)
  - [5.1 需要新建的文件](#51-需要新建的文件)
  - [5.2 需要修改的文件](#52-需要修改的文件)
  - [5.3 代码骨架](#53-代码骨架)
- [6. 配置文件设计](#6-配置文件设计)
- [7. 对各模态组合的处理策略](#7-对各模态组合的处理策略)
- [8. 训练流程](#8-训练流程)
- [9. 超参数推荐](#9-超参数推荐)
- [10. 测试计划](#10-测试计划)
- [11. 实验计划](#11-实验计划)

---

## 1. 动机与背景

### 当前 LFT 的问题

从已有的 105 次实验（5 个实验组 × 7 模态组合 × 3 种子）中观察到：

| 问题 | 证据 |
|------|------|
| 回归性能低 | 最佳 test_CCC = 0.21（triple），视频单模 = 0.19 |
| 分类性能有限 | 最佳 test_F1 = 0.44（3 分类，随机基线 ~0.33） |
| 多模态融合收益微弱 | 回归：triple vs video 仅 +0.023 CCC；分类：+3% F1 |
| km/telem 被稀释 | 在 LFT 的拼接 self-attention 中，km/telem token 数量远少于 video token，注意力被 video 主导 |

### LFT 架构缺陷分析

```
LFT: [video_tokens | km_tokens | telem_tokens] → Self-Attention → Pooling
```

- 所有 token 共享同一个 self-attention，不区分跨模态和模态内的交互
- video token 数量多（T_v），km/telem token 数量少（T_k, T_t），attention 资源分配不均
- 没有显式的跨模态信息传递机制

### 为什么选 MulT 的跨模态注意力

MulT 论文在 CMU-MOSI/MOSEI 情感数据集上的消融实验表明：
- 跨模态注意力（Cross-Modal Attention）比 Late Fusion (+2.6 Acc7) 和 Early Fusion (+2.7) 显著更好
- 仅 V,A→L 单向跨模态（即辅助模态增强主模态）就能达到完整双向 MulT 的 99.6% 性能
- 从低层特征（low-level）进行跨模态注意力比从中间层特征更好

这与我们的场景完美吻合：video 是强模态，km/telem 是辅助模态，用 km/telem 来增强 video 的表示。

---

## 2. 原始 MulT 架构回顾

### 2.1 跨模态注意力公式

**核心公式（论文 Eq. 1）：**

给定 target 模态 α 的序列 X_α ∈ ℝ^{T_α × d} 和 source 模态 β 的序列 X_β ∈ ℝ^{T_β × d}：

```
Y_α = CM_{β→α}(X_α, X_β) = softmax(Q_α K_β^T / √d_k) V_β
```

其中：
- Q_α = X_α W_Q ∈ ℝ^{T_α × d_k}（query 来自 target 模态 α）
- K_β = X_β W_K ∈ ℝ^{T_β × d_k}（key 来自 source 模态 β）
- V_β = X_β W_V ∈ ℝ^{T_β × d_v}（value 来自 source 模态 β）

**关键语义**：CM_{β→α} 表示"source β 的信息流向 target α"。输出 Y_α 保持 target 模态的时间长度 T_α，但融入了 source 模态 β 的信息。

注意力权重矩阵形状为 ℝ^{T_α × T_β}，第 (i, j) 个元素表示 α 的第 i 个时间步对 β 的第 j 个时间步的注意力，**不要求 T_α = T_β**，天然支持不对齐的多模态序列。

### 2.2 跨模态 Transformer Block（D 层堆叠）

**论文 Eq. 4**，以 V→L 为例（Vision 增强 Language）：

```
Layer 0 初始化:
    Z^{V→L}[0] = Z_L[0]    （target 模态的低层特征）

Layer i = 1, ..., D:
    Ẑ^{V→L}[i] = CM^{[i],multi}_{V→L}(LN(Z^{V→L}[i-1]), LN(Z_V[0])) + LN(Z^{V→L}[i-1])
    Z^{V→L}[i] = FFN^{[i]}_{V→L}(LN(Ẑ^{V→L}[i])) + LN(Ẑ^{V→L}[i])
```

每一层的结构是：
1. **LayerNorm** 对 target 当前表示
2. **Multi-head Cross-Attention**：query 来自 target 上一层的输出，key/value 来自 source 的**低层特征**（始终是 Z_V[0]，不是中间层输出）
3. **残差连接**：加上 LN(target 上一层输出)
4. **LayerNorm → FFN → 残差连接**

**关键设计**：每层都从 source 的低层特征 Z_β[0] 取 K/V，而非用上一层的中间表示。论文消融实验证明这比使用中间层特征更好（"low-level features works best"）。

### 2.3 原始 MulT 的完整流程

```
输入 → 1D Conv 投影（统一维度 d） → 位置编码 →

6 个跨模态 Transformer:
  V→L, A→L, L→V, A→V, L→A, V→A

同 target 模态的输出拼接:
  Z_L = [Z^{V→L}[D] ; Z^{A→L}[D]]   ∈ ℝ^{T_L × 2d}
  Z_V = [Z^{L→V}[D] ; Z^{A→V}[D]]   ∈ ℝ^{T_V × 2d}
  Z_A = [Z^{L→A}[D] ; Z^{V→A}[D]]   ∈ ℝ^{T_A × 2d}

→ 各自经过 Self-Attention Transformer
→ 取最后时间步 → FC → 预测
```

### 2.4 原始 MulT 的参考超参数

| 参数 | CMU-MOSI | CMU-MOSEI |
|------|----------|-----------|
| d_model | 40 | 40 |
| CM 层数 D | 4 | 4 |
| attention heads | 10 | 8 |
| optimizer | Adam | Adam |
| lr | 1e-3 | 1e-3 |
| batch_size | 128 | 16 |
| dropout (attn) | 0.1 | 0.1 |
| dropout (res) | 0.1 | 0.1 |

---

## 3. 我们的方案：CMA Fusion

### 3.1 核心思路

**仅做 video→km 和 video→telem 的单向跨模态注意力**，即让 km/telem 的 query attend to video 的 key/value，用 video 信息增强 km/telem 的表示。

具体做法：

```
CM_{video→km}:   Q 来自 km，   K/V 来自 video  → "video 增强的 km 表示"
CM_{video→telem}: Q 来自 telem， K/V 来自 video  → "video 增强的 telem 表示"
```

**为什么这样设计而非反过来（km→video）：**

1. km/telem 单模态性能极弱（CCC < 0.1），说明这些模态的原始特征缺乏情感预测所需的判别性信号
2. Video 是强模态（CCC = 0.19），其逐帧视觉特征蕴含了丰富的行为/表情信息
3. 让 km/telem 的 query 去检索 video 中的相关帧 = 让行为操作数据（键鼠频率、遥测指标）与对应的视觉上下文对齐
4. 这类似于"video 监督下的 km/telem 特征重标定"

**增强后的各模态表示最终通过一个轻量 Self-Attention Transformer 融合，再送入任务头。**

### 3.2 完整数据流图

```
┌──────────────────────────────────────────────────────────────────────┐
│                     来自 DataModule 的 Batch                         │
│  x:    {video:[B,T_v,2048], km:[B,T_k,25], telem:[B,T_t,D_t]}      │
│  mask: {video:[B,T_v],      km:[B,T_k],    telem:[B,T_t]}           │
└──────────┬────────────────────────┬───────────────────────┬──────────┘
           │                        │                       │
           ▼                        ▼                       ▼
   ┌───────────────┐     ┌──────────────────┐     ┌────────────────────┐
   │ VideoResNet2d │     │  KMStatEncoder   │     │  TelemStatPool     │
   │   Encoder     │     │                  │     │    Encoder         │
   └───────┬───────┘     └────────┬─────────┘     └──────────┬─────────┘
           │                      │                           │
     z_video:                z_km:                      z_telem:
     tokens [B,T_v,D]       tokens [B,T_k,D]           tokens [B,T_t,D]
     mask   [B,T_v]         mask   [B,T_k]              mask   [B,T_t]
           │                      │                           │
           │                      ▼                           ▼
           │         ┌─────────────────────────┐  ┌─────────────────────────┐
           │         │  CMA: CrossModal Attn   │  │  CMA: CrossModal Attn   │
           │         │  CM_{video→km}          │  │  CM_{video→telem}       │
           ├────────►│                         │  │                         │◄──┤
           │         │  Q = LN(km_tokens)      │  │  Q = LN(telem_tokens)   │   │
           │         │  K,V = LN(video_tokens) │  │  K,V = LN(video_tokens) │   │
           │         │  × D 层堆叠             │  │  × D 层堆叠              │   │
           │         │  (每层 K/V 来自低层特征)  │  │  (每层 K/V 来自低层特征) │   │
           │         └──────────┬──────────────┘  └──────────┬──────────────┘
           │                   │                              │
           │              km_enhanced                   telem_enhanced
           │              [B, T_k, D]                   [B, T_t, D]
           │                   │                              │
           ▼                   ▼                              ▼
   ┌───────────────────────────────────────────────────────────────────┐
   │                  位置编码 + 模态嵌入                               │
   │                                                                   │
   │  video_tokens + PE + ModEmb("video")     → [B, T_v, D]           │
   │  km_enhanced   + PE + ModEmb("km")       → [B, T_k, D]           │
   │  telem_enhanced + PE + ModEmb("telem")   → [B, T_t, D]           │
   │                                                                   │
   │  沿时间轴拼接: [B, T_v+T_k+T_t, D]                               │
   └──────────────────────────┬────────────────────────────────────────┘
                              │
                              ▼
               ┌──────────────────────────────┐
               │  Self-Attention Transformer   │
               │  (N_self 层, Pre-LN)          │
               │  标准 TransformerEncoder       │
               └──────────────┬────────────────┘
                              │
                              ▼
               ┌──────────────────────────────┐
               │  Mask-Aware Pooling           │
               │  mean / max / cls             │
               │  → pooled [B, D]              │
               │  → tokens [B, T_total, D]     │
               └──────────────┬────────────────┘
                              │
                              ▼
                      FusionOut {tokens, pooled}
```

### 3.3 各阶段详细说明

#### 阶段 1：Encoder 输出（不变）

各模态 Encoder 保持不变，输出 `EncoderOut`。CMA Fusion 接收的输入与 LFT 完全相同：

```python
z_dict: Dict[str, EncoderOut]   # {"video": EncoderOut, "km": EncoderOut, "telem": EncoderOut}
mask_dict: Dict[str, Tensor]    # {"video": [B,T_v], "km": [B,T_k], "telem": [B,T_t]}
```

所有 Encoder 已经将各自的原始特征投影到 d_model 维度（例如 video: 2048→512, km: 25→512, telem: 109→512），**CMA Fusion 不需要额外的维度投影**（这一点与原始 MulT 不同——MulT 用 1D Conv 做投影，但我们的框架中 Encoder 已经完成了这步）。

#### 阶段 2：位置编码注入（low-level 特征准备）

为 video 的 token 序列添加位置编码，作为跨模态注意力中固定使用的 low-level 特征：

```python
video_tokens_pe = pos_encoding(z_dict["video"]["tokens"])  # [B, T_v, D]
```

同样为 km 和 telem 添加位置编码，作为跨模态 Transformer 的初始 target 输入：

```python
km_tokens_pe = pos_encoding(z_dict["km"]["tokens"])        # [B, T_k, D]
telem_tokens_pe = pos_encoding(z_dict["telem"]["tokens"])   # [B, T_t, D]
```

**注意**：这里使用**各模态独立的**位置编码（同一个 SinusoidalPE 模块，但各序列从 position 0 开始编号），因为各模态的时间步数和采样率不同。

#### 阶段 3：CrossModal Transformer Block（核心）

以 CM_{video→km} 为例，D 层堆叠：

```
初始化:
    h_km[0] = km_tokens_pe              # target: km 的低层特征
    src_video = video_tokens_pe          # source: video 的低层特征（每层固定复用）

Layer i = 1, ..., D:
    # Step 1: Multi-Head Cross-Attention
    q = LN(h_km[i-1])                   # Query [B, T_k, D]
    k = LN(src_video)                   # Key   [B, T_v, D]
    v = LN(src_video)                   # Value [B, T_v, D]

    attn_out = MultiHeadAttention(q, k, v, key_padding_mask=~video_mask)
    # attn_out: [B, T_k, D]
    # 注意力矩阵: [B, nhead, T_k, T_v]

    # Step 2: Residual + Dropout
    h_km_mid = LN(h_km[i-1]) + Dropout(attn_out)

    # Step 3: Feed-Forward Network
    ffn_out = FFN(LN(h_km_mid))         # Linear(D, 4D) → GELU → Dropout → Linear(4D, D)

    # Step 4: Residual
    h_km[i] = h_km_mid + Dropout(ffn_out)

输出:
    km_enhanced = h_km[D]               # [B, T_k, D]  — video 增强后的 km 表示
```

**注意事项**：
- **key_padding_mask**：使用 video_mask 来屏蔽 video 中无效的时间步，防止 km 的 query attend to padding token
- **每层的 K/V 始终来自 source 的低层特征** `src_video = video_tokens_pe`，不随层数更新。这是论文消融实验验证的最优设计
- **残差连接**：确保 km 的原始信息不会在多层跨模态注意力中丢失

CM_{video→telem} 的实现完全对称，只是把 km 替换为 telem。

#### 阶段 4：Self-Attention Transformer（融合阶段）

跨模态注意力完成后，将三个模态的表示拼接：

```python
# 各模态添加模态嵌入（区分不同模态的 token）
video_out = video_tokens_pe + modality_emb("video")   # [B, T_v, D]  — video 保持原始
km_out    = km_enhanced     + modality_emb("km")       # [B, T_k, D]  — 已被 video 增强
telem_out = telem_enhanced  + modality_emb("telem")    # [B, T_t, D]  — 已被 video 增强

# 沿时间轴拼接
concat = torch.cat([video_out, km_out, telem_out], dim=1)  # [B, T_v+T_k+T_t, D]
mask_concat = torch.cat([video_mask, km_mask, telem_mask], dim=1)

# Self-Attention Transformer
fused = self_attn_transformer(concat, src_key_padding_mask=~mask_concat)
# fused: [B, T_total, D]
```

这个 Self-Attention Transformer 的作用：
1. 在已增强的跨模态表示之间进行信息整合
2. km_enhanced 中已经嵌入了 video 的上下文信息，self-attn 让增强后的 km 反哺 video
3. 层数 N_self 建议设为较小值（1-2 层），因为主要的跨模态建模已在 CMA 阶段完成

#### 阶段 5：池化与输出

与 LFT 完全相同的 mask-aware pooling：

```python
if pooling == "mean":
    mask_f = mask_concat.float().unsqueeze(-1)
    pooled = (fused * mask_f).sum(dim=1) / mask_f.sum(dim=1).clamp(min=1.0)
elif pooling == "max":
    ...
elif pooling == "cls":
    pooled = fused[:, 0, :]
```

输出 `FusionOut(tokens=fused, pooled=pooled)`，后续的任务头（regression / classification / multitask）**完全不变**。

### 3.4 输出与接口

CMA Fusion 实现 `BaseFusion` 接口，输入输出与 LFT 完全相同：

```python
def forward(
    self,
    z_dict: Dict[str, EncoderOut],    # 任意模态子集
    mask_dict: Dict[str, torch.Tensor],
) -> FusionOut:                        # {tokens: [B,T,D] or None, pooled: [B,D]}
```

---

## 4. 与 LFT 的对比

| 维度 | LFT | CMA (本方案) |
|------|-----|-------------|
| 跨模态建模 | 隐式（self-attn 中所有 token 共享） | 显式（km/telem 的 query attend to video 的 K/V） |
| video 对 km/telem 的信息传递 | 间接，经过多层 self-attn 才可能传递 | 直接，每层 CMA 都从 video 低层特征取信息 |
| 时间对齐要求 | 不要求（拼接后统一处理） | 不要求（cross-attn 的 T_α ≠ T_β） |
| 注意力资源分配 | video token 多 → 被 video 主导 | CMA 阶段 km/telem 单独获得 video 注意力 |
| 计算复杂度 | O((T_v+T_k+T_t)² × D) | CMA: O(T_k × T_v × D) + O(T_t × T_v × D) + SA: O((T_v+T_k+T_t)² × D)，但 SA 层数更少 |
| 参数量增长 | — | +2 个 CMA Block（D_cm 层 × 每层 cross-attn + FFN），SA 层数可以减少来平衡 |

**计算量估算**（以 AMuCS 典型序列长度为例）：

假设 T_v=120, T_k=30, T_t=30, D=512：
- LFT self-attn（4层）：4 × (120+30+30)² × 512 ≈ 4 × 32,400 × 512
- CMA cross-attn（4层）：4 × (30×120 + 30×120) × 512 ≈ 4 × 7,200 × 512
- CMA self-attn（2层）：2 × (120+30+30)² × 512 ≈ 2 × 32,400 × 512

总计相当但 CMA 阶段的跨模态建模更有针对性。

---

## 5. 在现有框架上的最小扩展方案

### 5.1 需要新建的文件

| 文件 | 说明 |
|------|------|
| `src/models/fusions/cma.py` | CMA Fusion 实现（~200 行） |
| `configs/amucs_seq_cma_*_multitask_arousal_trend.yaml` × 7 | 7 种模态组合的配置文件 |

### 5.2 需要修改的文件

| 文件 | 修改内容 |
|------|----------|
| `src/models/fusions/__init__.py` | 添加 `from . import cma  # noqa: F401`（1行） |

**不需要修改的文件**（也不应该修改）：

- `src/core/types.py` — 接口不变
- `src/core/runner.py` — 训练循环不变
- `src/core/registry.py` — 注册系统不变
- 所有 Encoder — 输出格式不变
- 所有 Head — 输入格式不变
- 所有 Loss / Metric — 不变

### 5.3 代码骨架

```python
# src/models/fusions/cma.py

"""
CrossModal Attention Fusion (CMA)

跨模态注意力融合：以 video 为锚模态，通过方向性跨模态注意力
增强 km/telem 的表示，再经 self-attention 融合。

基于 MulT (Tsai et al., ACL 2019) 的跨模态 Transformer 设计。
"""

from __future__ import annotations
from typing import Dict, Optional
import torch
from torch import nn
from src.core.registry import FUSIONS
from src.core.types import BaseFusion, EncoderOut, FusionOut
from src.models.components import (
    SinusoidalPositionalEncoding,
    LearnablePositionalEncoding,
    ModalityEmbedding,
)


class CrossModalTransformerBlock(nn.Module):
    """
    单层跨模态 Transformer Block。

    实现 MulT Eq. 4 中的一层：
        Ẑ[i] = CM(LN(Z[i-1]), LN(Z_src)) + LN(Z[i-1])
        Z[i] = FFN(LN(Ẑ[i])) + LN(Ẑ[i])

    Parameters
    ----------
    d_model : int
        模型维度。
    nhead : int
        注意力头数。
    dim_feedforward : int
        FFN 隐层维度。
    dropout : float
        Dropout 概率。
    """

    def __init__(self, d_model: int, nhead: int, dim_feedforward: int, dropout: float):
        super().__init__()
        self.ln_target = nn.LayerNorm(d_model)
        self.ln_source = nn.LayerNorm(d_model)
        self.cross_attn = nn.MultiheadAttention(
            embed_dim=d_model,
            num_heads=nhead,
            dropout=dropout,
            batch_first=True,
        )
        self.ln_ffn = nn.LayerNorm(d_model)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, dim_feedforward),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim_feedforward, d_model),
            nn.Dropout(dropout),
        )

    def forward(
        self,
        target: torch.Tensor,       # [B, T_tgt, D]  — 当前层 target 表示
        source: torch.Tensor,       # [B, T_src, D]  — source 低层特征（每层固定）
        source_key_padding_mask: Optional[torch.Tensor] = None,  # [B, T_src], True=ignore
    ) -> torch.Tensor:
        """返回 [B, T_tgt, D] — 增强后的 target 表示。"""
        # Cross-Attention + Residual
        tgt_norm = self.ln_target(target)
        src_norm = self.ln_source(source)
        attn_out, _ = self.cross_attn(
            query=tgt_norm,
            key=src_norm,
            value=src_norm,
            key_padding_mask=source_key_padding_mask,
        )
        h = tgt_norm + attn_out  # 残差：LN(target) + cross_attn_output

        # FFN + Residual
        h = h + self.ffn(self.ln_ffn(h))
        return h


class CrossModalTransformer(nn.Module):
    """
    D 层 CrossModalTransformerBlock 堆叠。

    每层的 source K/V 始终来自输入时的低层特征（不更新）。

    Parameters
    ----------
    d_model : int
        模型维度。
    nhead : int
        注意力头数。
    num_layers : int
        堆叠层数 D。
    dim_feedforward : int
        FFN 隐层维度。
    dropout : float
        Dropout 概率。
    """

    def __init__(self, d_model: int, nhead: int, num_layers: int,
                 dim_feedforward: int, dropout: float):
        super().__init__()
        self.layers = nn.ModuleList([
            CrossModalTransformerBlock(d_model, nhead, dim_feedforward, dropout)
            for _ in range(num_layers)
        ])

    def forward(
        self,
        target: torch.Tensor,
        source: torch.Tensor,
        source_key_padding_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Parameters
        ----------
        target : [B, T_tgt, D] — target 模态的低层特征（初始化 h[0]）
        source : [B, T_src, D] — source 模态的低层特征（每层复用）

        Returns
        -------
        [B, T_tgt, D] — 经 D 层 source 增强后的 target 表示
        """
        h = target
        for layer in self.layers:
            h = layer(h, source, source_key_padding_mask)
        return h


@FUSIONS.register("cma")
class CMAFusion(BaseFusion):
    """
    CrossModal Attention Fusion.

    以 anchor_modality（默认 video）为锚，对其余模态执行
    CM_{anchor→target} 跨模态注意力增强，再通过 Self-Attention
    Transformer 融合所有模态的表示。

    当只有单模态或只有锚模态时，退化为普通 Self-Attention。

    Parameters (via cfg dict)
    -------------------------
    d_model : int (default 512)
    nhead : int (default 8)
    cm_layers : int (default 4)
        跨模态 Transformer 的层数 D。
    sa_layers : int (default 2)
        后续 Self-Attention Transformer 的层数。
    dim_feedforward : int (default 1024)
    dropout : float (default 0.1)
    anchor_modality : str (default "video")
        锚模态名称。跨模态注意力方向为 anchor → 其余模态。
    max_seq_len : int (default 1000)
    pos_encoding_type : str (default "sinusoidal")
    pooling : str (default "mean")
    """

    def __init__(self, cfg):
        super().__init__()
        if isinstance(cfg, dict):
            _g = cfg.get
        else:
            _g = lambda k, d=None: getattr(cfg, k, d)

        d_model         = _g("d_model", 512)
        nhead           = _g("nhead", 8)
        cm_layers       = _g("cm_layers", 4)
        sa_layers       = _g("sa_layers", 2)
        dim_feedforward = _g("dim_feedforward", 1024)
        dropout         = _g("dropout", 0.1)
        max_seq_len     = _g("max_seq_len", 1000)
        pos_type        = _g("pos_encoding_type", "sinusoidal")
        self.pooling_type    = _g("pooling", "mean")
        self.anchor_modality = _g("anchor_modality", "video")
        self.d_model = d_model

        # 位置编码（各模态共用同一 PE 模块，但各自从 pos 0 开始）
        if pos_type == "learnable":
            self.pos_encoding = LearnablePositionalEncoding(d_model, max_seq_len, dropout)
        else:
            self.pos_encoding = SinusoidalPositionalEncoding(d_model, max_seq_len, dropout)

        # 模态嵌入（lazy-init）
        self._modality_emb: Optional[ModalityEmbedding] = None
        self._modality_names: Optional[list] = None

        # 跨模态 Transformer（每个 non-anchor 模态共享同一组参数，或各自独立）
        # 这里选择各模态独立的 CM Transformer（参数不共享），
        # 因为 km 和 telem 的特征分布差异大。
        # 用 ModuleDict 按需创建，支持任意模态子集。
        self._cm_transformers = nn.ModuleDict()
        self._cm_cfg = {
            "d_model": d_model,
            "nhead": nhead,
            "num_layers": cm_layers,
            "dim_feedforward": dim_feedforward,
            "dropout": dropout,
        }

        # Self-Attention Transformer
        if sa_layers > 0:
            sa_layer = nn.TransformerEncoderLayer(
                d_model=d_model,
                nhead=nhead,
                dim_feedforward=dim_feedforward,
                dropout=dropout,
                batch_first=True,
                norm_first=True,
            )
            self.self_attn_transformer = nn.TransformerEncoder(
                sa_layer,
                num_layers=sa_layers,
                enable_nested_tensor=False,
            )
        else:
            self.self_attn_transformer = None

        # CLS token（仅 pooling == "cls" 时使用）
        if self.pooling_type == "cls":
            self.cls_token = nn.Parameter(torch.randn(1, 1, d_model) * 0.02)

    def _get_or_create_cm(self, modality: str) -> CrossModalTransformer:
        """懒初始化指定模态的 CrossModal Transformer。"""
        if modality not in self._cm_transformers:
            cm = CrossModalTransformer(**self._cm_cfg)
            ref = next(self.parameters())
            cm = cm.to(device=ref.device, dtype=ref.dtype)
            self._cm_transformers[modality] = cm
        return self._cm_transformers[modality]

    def _get_modality_emb(self, modality_names: list) -> ModalityEmbedding:
        """懒初始化模态嵌入。"""
        if self._modality_emb is None or self._modality_names != modality_names:
            self._modality_names = modality_names
            self._modality_emb = ModalityEmbedding(
                d_model=self.d_model,
                num_modalities=len(modality_names),
                modality_names=modality_names,
            ).to(next(self.parameters()).device)
        return self._modality_emb

    def forward(
        self,
        z_dict: Dict[str, EncoderOut],
        mask_dict: Dict[str, torch.Tensor],
    ) -> FusionOut:
        modality_names = sorted(z_dict.keys())
        anchor = self.anchor_modality
        has_anchor = anchor in z_dict

        # --- 阶段 1: 位置编码 ---
        tokens_pe = {}   # 各模态 tokens + PE
        for mod in modality_names:
            tokens_pe[mod] = self.pos_encoding(z_dict[mod]["tokens"])

        # --- 阶段 2: 跨模态注意力增强 ---
        enhanced = {}
        if has_anchor and len(modality_names) > 1:
            anchor_tokens = tokens_pe[anchor]           # [B, T_v, D]
            anchor_mask_ignore = ~mask_dict[anchor]     # True = ignore

            for mod in modality_names:
                if mod == anchor:
                    enhanced[mod] = tokens_pe[mod]      # anchor 保持原始
                else:
                    cm = self._get_or_create_cm(mod)
                    enhanced[mod] = cm(
                        target=tokens_pe[mod],
                        source=anchor_tokens,
                        source_key_padding_mask=anchor_mask_ignore,
                    )
        else:
            # 无 anchor 或单模态 → 跳过 CMA，直接用原始 PE tokens
            for mod in modality_names:
                enhanced[mod] = tokens_pe[mod]

        # --- 阶段 3: 模态嵌入 + 拼接 ---
        mod_emb = self._get_modality_emb(modality_names)
        all_tokens = []
        all_masks = []
        for mod in modality_names:
            tok = mod_emb(enhanced[mod], modality=mod)
            all_tokens.append(tok)
            all_masks.append(mask_dict[mod])

        tokens_concat = torch.cat(all_tokens, dim=1)   # [B, T_total, D]
        masks_concat = torch.cat(all_masks, dim=1)      # [B, T_total]

        # CLS token
        if self.pooling_type == "cls":
            B = tokens_concat.size(0)
            cls = self.cls_token.expand(B, -1, -1)
            tokens_concat = torch.cat([cls, tokens_concat], dim=1)
            cls_mask = torch.ones(B, 1, dtype=torch.bool, device=tokens_concat.device)
            masks_concat = torch.cat([cls_mask, masks_concat], dim=1)

        # --- 阶段 4: Self-Attention Transformer ---
        if self.self_attn_transformer is not None:
            padding_mask = ~masks_concat
            fused = self.self_attn_transformer(tokens_concat, src_key_padding_mask=padding_mask)
        else:
            fused = tokens_concat

        # --- 阶段 5: Pooling ---
        if self.pooling_type == "cls":
            pooled = fused[:, 0, :]
        elif self.pooling_type == "max":
            fused_masked = fused.masked_fill(~masks_concat.unsqueeze(-1), float("-inf"))
            pooled = fused_masked.max(dim=1)[0]
        else:  # mean
            mask_f = masks_concat.float().unsqueeze(-1)
            pooled = (fused * mask_f).sum(dim=1) / mask_f.sum(dim=1).clamp(min=1.0)

        return FusionOut(tokens=fused, pooled=pooled)
```

---

## 6. 配置文件设计

### 基础配置结构

只需要更改 `model.fusion` 部分。以 `video_km_telem` 三模态的混合多任务为例：

```yaml
# configs/amucs_seq_cma_video_km_telem_multitask_arousal_trend.yaml
_base_: ./experiments/video_km_telem_aligned_seq.yaml

task_type: multitask

data:
  name: amucs_seq_multitask
  modalities: [video, km, telem]
  labels_seq_path: labels/arousal_reg_trend_seq.json
  task_names: [arousal, trend]
  task_label_dtypes:
    arousal: float
    trend: long

model:
  fusion:
    name: cma                       # 改为 cma
    nhead: 8
    cm_layers: 4                    # 跨模态 Transformer 层数
    sa_layers: 2                    # 自注意力 Transformer 层数
    dim_feedforward: 1024
    dropout: 0.1
    anchor_modality: video          # 锚模态
    max_seq_len: 1000
    pos_encoding_type: sinusoidal
    pooling: mean
  head:
    name: multitask_mixed_seq
    task_heads:
      arousal:
        type: regression
        hidden_dim: 128
        out_dim: 1
        dropout: 0.1
      trend:
        type: classification
        hidden_dim: 128
        num_classes: 3
        dropout: 0.1

train:
  loss: multitask_mixed_seq_loss
  loss_cfg:
    task_weights:
      arousal: 1.0
      trend: 1.0
    task_types:
      arousal: regression
      trend: classification
  early_stopping:
    metric: val_score_mixed
    mode: max
    patience: 5

eval:
  task_metrics:
    arousal: [ccc, rmse]
    trend: [macro_f1, balanced_acc]
  multitask_metric_weights:
    ccc_arousal: 0.5
    macro_f1_trend: 0.5
```

### CMA 特有配置参数

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `name` | str | — | 必须为 `"cma"` |
| `cm_layers` | int | 4 | 跨模态 Transformer 的层数 D |
| `sa_layers` | int | 2 | 后续 Self-Attention Transformer 层数（0 = 禁用） |
| `anchor_modality` | str | `"video"` | 锚模态名称 |
| `nhead` | int | 8 | 注意力头数（CM 和 SA 共用） |
| `dim_feedforward` | int | 1024 | FFN 隐层维度 |
| `dropout` | float | 0.1 | Dropout 概率 |
| `pooling` | str | `"mean"` | 池化方式：mean / max / cls |

---

## 7. 对各模态组合的处理策略

CMA Fusion 必须像 LFT 一样处理任意模态子集。下表列出各组合的行为：

| 模态组合 | 行为 | 说明 |
|----------|------|------|
| **video + km + telem** | CM_{video→km} + CM_{video→telem} + SA | 完整流程 |
| **video + km** | CM_{video→km} + SA | 仅 km 被 video 增强 |
| **video + telem** | CM_{video→telem} + SA | 仅 telem 被 video 增强 |
| **km + telem** | 跳过 CMA → 直接 SA（等价于 LFT） | 无 anchor，退化为 self-attention |
| **仅 video** | 跳过 CMA → 直接 SA | 单模态，self-attention only |
| **仅 km** | 跳过 CMA → 直接 SA | 单模态 |
| **仅 telem** | 跳过 CMA → 直接 SA | 单模态 |

代码中的关键判断逻辑：

```python
has_anchor = anchor in z_dict
if has_anchor and len(modality_names) > 1:
    # 执行跨模态注意力
    ...
else:
    # 退化为普通拼接 + self-attention
    ...
```

这确保了 CMA 在所有 7 种模态组合下都能正常工作，无需为每种组合写特殊逻辑。

---

## 8. 训练流程

### 训练步骤（完全复用现有 Runner）

```
1. 配置文件指定 model.fusion.name: cma
2. Runner._build() 中:
   - FUSIONS.build("cma", fusion_cfg) → 构建 CMAFusion
   - 其余所有组件（encoders, head, loss, metrics）照常构建
3. 训练循环完全不变:
   - y_hat = model(x_dict, mask_dict)
   - loss = loss_fn(y_hat, y, y_mask)
   - optimizer.zero_grad() → loss.backward() → optimizer.step()
4. 验证/测试/早停/checkpoint 全部复用
```

### CMA 内部的梯度流

```
反向传播路径 1（loss → SA → CMA → km encoder）：
    loss → pooled → sa_transformer → km_enhanced → cm_transformer → km_tokens_pe → km_encoder

反向传播路径 2（loss → SA → video_tokens_pe → video encoder）：
    loss → pooled → sa_transformer → video_out → video_tokens_pe → video_encoder

反向传播路径 3（loss → CMA → video low-level features → video encoder）：
    loss → km_enhanced → cm_cross_attn(K,V) → video_tokens_pe → video_encoder
```

video encoder 通过两条路径接收梯度：
1. 直接通过 SA 的 self-attention
2. 间接通过 CMA 的 K/V（因为 video_tokens_pe 作为 source 参与了 cross-attention）

这意味着 **video encoder 会被 km/telem 的任务信号间接优化**，促使 video 提取对 km/telem 也有用的特征。

### 训练开销对比（与 LFT）

| 项目 | LFT (4层SA) | CMA (4层CM + 2层SA) |
|------|-------------|---------------------|
| 参数量（d=512, ff=1024） | SA: ~12.6M | CM: ~12.6M × 2 + SA: ~6.3M ≈ 31.5M |
| 训练速度（估计） | 1× | ~1.5×（CM 阶段额外开销） |
| 显存（估计） | 1× | ~1.3×（CM 阶段的中间激活） |

参数量显著增加，但每个 CM Transformer 只在 T_k × T_v 或 T_t × T_v 尺度上计算注意力（远小于 T_total²），实际计算量可控。

如需控制参数量，可以：
- 减小 cm_layers 到 2
- 减小 dim_feedforward 到 512
- 让两个 CM Transformer 共享参数（`shared_cm: true` 配置）

---

## 9. 超参数推荐

基于 MulT 论文超参数和 AMuCS 数据集特点的初始建议：

| 参数 | 推荐值 | 理由 |
|------|--------|------|
| `d_model` | 512 | 与现有 LFT 保持一致，便于对比 |
| `cm_layers` | 4 | 论文默认值，AMuCS 序列不长，4 层足够 |
| `sa_layers` | 2 | CMA 已做跨模态建模，SA 只需少量层做整合 |
| `nhead` | 8 | 与 LFT 一致 |
| `dim_feedforward` | 1024 | 与 LFT 一致 |
| `dropout` | 0.1 | LFT 和 MulT 论文一致 |
| `anchor_modality` | video | 最强模态 |
| `lr` | 1e-4 | 与 LFT 一致，先保持不变 |
| `batch_size` | 8 | 与 LFT 一致 |
| `pooling` | mean | 与 LFT 一致 |

### 需要 sweep 的参数（优先级排序）

1. **cm_layers**: [2, 4, 6] — 跨模态建模深度
2. **sa_layers**: [0, 1, 2] — sa_layers=0 可以验证纯 CMA 是否足够
3. **lr**: [5e-5, 1e-4, 3e-4] — CMA 参数更多，可能需要调 lr

---

## 10. 测试计划

### 形状契约测试

在 `tests/test_shapes.py` 中添加以下测试：

```python
class TestCMAFusion:
    """CMA Fusion 的形状契约测试。"""

    @pytest.fixture
    def cma_cfg(self):
        return {
            "d_model": 64,
            "nhead": 4,
            "cm_layers": 2,
            "sa_layers": 1,
            "dim_feedforward": 128,
            "dropout": 0.0,
            "anchor_modality": "video",
            "pooling": "mean",
        }

    def test_triple_modality(self, cma_cfg):
        """video + km + telem → FusionOut with correct shapes."""
        fusion = CMAFusion(cma_cfg)
        z_dict = {
            "video": EncoderOut(tokens=torch.randn(2,10,64), pooled=torch.randn(2,64), mask=torch.ones(2,10,dtype=torch.bool)),
            "km":    EncoderOut(tokens=torch.randn(2,5,64),  pooled=torch.randn(2,64), mask=torch.ones(2,5, dtype=torch.bool)),
            "telem": EncoderOut(tokens=torch.randn(2,5,64),  pooled=torch.randn(2,64), mask=torch.ones(2,5, dtype=torch.bool)),
        }
        mask_dict = {mod: z_dict[mod]["mask"] for mod in z_dict}
        out = fusion(z_dict, mask_dict)
        assert out["pooled"].shape == (2, 64)
        assert out["tokens"].shape == (2, 20, 64)  # T_v + T_k + T_t

    def test_single_modality(self, cma_cfg):
        """仅 video → 退化为 self-attention。"""

    def test_dual_without_anchor(self, cma_cfg):
        """km + telem（无 anchor）→ 退化为 self-attention。"""

    def test_dual_with_anchor(self, cma_cfg):
        """video + km → CM_{video→km} + SA。"""

    def test_variable_seq_lengths(self, cma_cfg):
        """不同模态不同序列长度 → 正常工作。"""

    def test_with_padding_mask(self, cma_cfg):
        """部分 mask=False → pooling 正确排除 padding。"""

    def test_cls_pooling(self, cma_cfg):
        """pooling='cls' → 额外 CLS token。"""

    def test_backward(self, cma_cfg):
        """前向 + 反向传播无错误。"""
```

---

## 11. 实验计划

### 第一轮：CMA vs LFT 对比（7 组合 × 3 种子）

使用与现有 LFT 实验相同的设置（mixed multitask: arousal 回归 + trend 分类），只将 `model.fusion.name` 从 `lft` 改为 `cma`。

| 实验 | 模态 | Fusion | 配置文件 |
|------|------|--------|----------|
| 1 | single_video | cma | `amucs_seq_cma_video_multitask_arousal_trend.yaml` |
| 2 | single_km | cma | `amucs_seq_cma_km_multitask_arousal_trend.yaml` |
| 3 | single_telem | cma | `amucs_seq_cma_telem_multitask_arousal_trend.yaml` |
| 4 | video + km | cma | `amucs_seq_cma_video_km_multitask_arousal_trend.yaml` |
| 5 | video + telem | cma | `amucs_seq_cma_video_telem_multitask_arousal_trend.yaml` |
| 6 | km + telem | cma | `amucs_seq_cma_km_telem_multitask_arousal_trend.yaml` |
| 7 | video + km + telem | cma | `amucs_seq_cma_video_km_telem_multitask_arousal_trend.yaml` |

**预期结果**：
- 单模态和 km+telem 组合应与 LFT 基本持平（退化为 self-attn）
- video+km、video+telem、triple 组合应有提升（CMA 发挥作用）
- 提升主要体现在 CCC（回归任务）上，因为 CMA 更有效地利用了 km/telem 的时序信息

### 第二轮：超参数搜索

在 triple 模态上对 `cm_layers` 和 `sa_layers` 进行网格搜索：

```
cm_layers: [2, 4]
sa_layers: [0, 1, 2]
→ 6 组 × 3 种子 = 18 个 run
```

### 评估指标（与 LFT 基线对比）

| 指标 | LFT 基线 (triple) | CMA 目标 |
|------|-------------------|----------|
| test_ccc_arousal | 0.2085 ± 0.007 | > 0.23 |
| test_macro_f1_trend | 0.4256 ± 0.011 | > 0.44 |
| test_score_mixed | 0.3170 ± 0.005 | > 0.34 |

---

## 参考文献

- Tsai, Y. H. H., et al. (2019). "Multimodal Transformer for Unaligned Multimodal Language Sequences." ACL 2019. [arXiv:1906.00295](https://arxiv.org/abs/1906.00295)
- [Official MulT Implementation](https://github.com/yaohungt/Multimodal-Transformer)
