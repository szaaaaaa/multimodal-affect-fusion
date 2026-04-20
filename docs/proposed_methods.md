# 改进方法汇总

> 6 个独立改进方向 + 3 个组合方案。每个方向标注基于哪个 baseline、改什么、怎么改。
> 全部实现后通过实验选出最优方案作为论文主方法。

---

## 当前 Pipeline 概览

```
Input:
  video [B, T_v, 768]   (CLIP ViT-L/14 特征, win_len=24)
  km    [B, T_k, 25]    (键鼠统计特征,     win_len=300)
  telem [B, T_t, 109]   (遥测统计特征,     无 win_len)

Encoders (线性投影, 各自独立):
  video: Linear(768→512)  → EncoderOut{tokens:[B,T_v,512], mask}
  km:    Linear(25→512)   → EncoderOut{tokens:[B,T_k,512], mask}
  telem: Linear(109→512)  → EncoderOut{tokens:[B,T_t,512], mask}

Fusion (6 个 baseline, 唯一差异点):
  → FusionOut{tokens:[B,T,512]|None, pooled:[B,512]}

Head (multitask_seq, token-wise):
  tokens[B,T,512] → MLP_state → state[B,T,3]
  tokens[B,T,512] → MLP_trend → trend[B,T,3]

Loss: CE(state) + CE(trend), label_smoothing=0.1
```

### 已有 6 个 Baseline

所有方法统一 Transformer 深度 = 4 层, dim_feedforward = 1024。

| 名称 | 注册名 | 跨模态交互起点 | 核心机制 | 输出 tokens | 实现文件 |
|------|--------|--------------|---------|-------------|---------|
| EFT | `eft` | 第 1 层 (最早) | token concat(time_dim) + modality_emb → 共享 Transformer (4层) | [B,T_total,D] | `fusions/eft.py` |
| MFT | `mft` | 第 3 层 (中间) | 私有 Transformer (2层) → cross-attention (2层) | [B,T_total,D] | `fusions/mft.py` |
| CMA | `cma` | 第 1 层 (定向) | non-anchor cross-attn to anchor(video) → 共享 Transformer | [B,T_total,D] | `fusions/cma.py` |
| Gated | `gated` | 门控加权后 | sigmoid gate → weighted sum → Transformer 精炼 (4层) | [B,T,D] | `fusions/gated.py` |
| LFT | `lft` | 仅最终融合 | 独立 Transformer (4层) → pool → attention-weighted query fusion | **tokens=None**, pooled=[B,D] | `fusions/lft.py` |
| Late | `late` | 仅 average | 独立 Transformer (4层) → average tokens | [B,T,D] | `fusions/late.py` |

> **注意**: LFT 多模态时返回 `tokens=None`（仅 pooled），因此依赖逐帧 tokens 的改进（如方向 F）不兼容 LFT。

### Pipeline 瓶颈

| 编号 | 瓶颈 | 位置 | 影响 |
|------|------|------|------|
| B1 | 编码器仅线性投影，无时序建模 | Encoder | 模态 token 缺乏局部时序上下文 |
| B2 | 三个模态投射到同维度但无语义对齐 | Encoder→Fusion 之间 | 融合层需同时学对齐+整合 |
| B3 | EFT 拼接异质 token 做全局 attention (T_v+T_k+T_t 长序列)，短模态信号被稀释 | Fusion (EFT) | 多模态增益微弱 |
| B4 | state 和 trend 共用同一组 token，无任务特异处理 | Head | trend 需要时间差分信息 |
| B5 | 所有模态视为同等可靠 | Fusion | video 信噪比远低于 telem |

---

## 方向 A: Bottleneck Token Fusion（瓶颈 token 融合）

### 基本信息

| 项目 | 内容 |
|------|------|
| **改哪层** | Fusion |
| **基于谁** | EFT (替换其 Transformer 融合机制) |
| **解决瓶颈** | B3 (异质 token 直接 attention 效率低) |
| **注册名** | `bottleneck` |
| **实现文件** | `src/models/fusions/bottleneck.py` |

### 核心思想

引入 K 个可学习的 bottleneck latent tokens 作为模态间信息传递的"中介"。模态 token 永远不直接互相 attend，所有跨模态信息必须经过 bottleneck。

### 架构图

```
EFT (现有):
  [video_tok | km_tok | telem_tok] → full self-attention → pooling
  问题: O((T_v+T_k+T_t)²), 异质 token 混合

Bottleneck (改进):
  ┌─ bottleneck_tokens [K, D]  (K=4~16, 可学习)
  │
  │  每个 Bottleneck Layer:
  │    Step 1 — Read: bottleneck attend to 各模态 (提取信息)
  │      Q=bottleneck, KV=video  → cross-attn
  │      Q=bottleneck, KV=km     → cross-attn
  │      Q=bottleneck, KV=telem  → cross-attn
  │      → 更新 bottleneck
  │
  │    Step 2 — Self-refine: bottleneck self-attention (整合跨模态信息)
  │
  │    Step 3 — Write (可选): 各模态 attend to bottleneck (分发信息)
  │      Q=video, KV=bottleneck → cross-attn → 增强的 video
  │      Q=km,    KV=bottleneck → cross-attn → 增强的 km
  │      Q=telem, KV=bottleneck → cross-attn → 增强的 telem
  │
  └─ 重复 N 层
     ↓
  pooling(bottleneck_tokens) → [B, D]
```

### 关键实现细节

```python
class BottleneckFusionLayer(nn.Module):
    def __init__(self, d_model, nhead, dim_feedforward, dropout):
        # 每个模态一个 cross-attn (bottleneck → modality)
        self.read_attns = nn.ModuleDict()   # 按需 lazy-init
        self.self_attn = nn.TransformerEncoderLayer(...)  # bottleneck self-attention
        # 可选 write-back
        self.write_attns = nn.ModuleDict()  # 按需 lazy-init

    def forward(self, bottleneck, mod_tokens_dict, mod_masks_dict):
        # Read: bottleneck 从各模态提取信息
        for mod in mod_tokens_dict:
            bottleneck = bottleneck + cross_attn(Q=bottleneck, KV=mod_tokens)
        # Self-refine
        bottleneck = self.self_attn(bottleneck)
        return bottleneck

class BottleneckFusion(BaseFusion):
    def __init__(self, cfg):
        self.num_bottleneck = cfg_get(cfg, "num_bottleneck", 8)
        self.bottleneck = nn.Parameter(torch.randn(1, K, d_model) * 0.02)
        self.layers = nn.ModuleList([BottleneckFusionLayer(...) for _ in range(num_layers)])

    def forward(self, z_dict, mask_dict):
        B = ...
        bt = self.bottleneck.expand(B, -1, -1)
        for layer in self.layers:
            bt = layer(bt, z_dict_tokens, mask_dict)
        pooled = bt.mean(dim=1)
        return FusionOut(tokens=bt, pooled=pooled)
```

### 配置模板

```yaml
shared:
  model:
    fusion_bottleneck:
      name: bottleneck
      d_model: 512
      num_bottleneck: 8        # latent token 数量
      nhead: 8
      num_layers: 4            # 与其他 baseline 统一深度
      dim_feedforward: 1024    # 与其他 baseline 统一
      dropout: 0.1
      write_back: false        # 是否回写增强模态 token
      max_seq_len: 2000
      pooling: mean
```

### 文献

- Nagrani et al., "Attention Bottlenecks for Multimodal Fusion" (NeurIPS 2021)
- DBA: "Domain-Separated Bottleneck Attention Fusion" (ACM TOMM 2025)
- EMBT: "Extended Multimodal Bottleneck Transformer" (Applied Sciences 2025)

---

## 方向 B: Coupled Mamba Fusion（耦合状态空间融合）

### 基本信息

| 项目 | 内容 |
|------|------|
| **改哪层** | Fusion |
| **基于谁** | EFT (替换共享 Transformer backbone 为耦合 Mamba) |
| **解决瓶颈** | B3 (O(L²) 计算 → O(L) 线性) |
| **注册名** | `coupled_mamba` |
| **实现文件** | `src/models/fusions/coupled_mamba.py` |

### 核心思想

用 State Space Model (Mamba) 替换 Transformer。每个模态维护独立的 hidden state chain，通过耦合系数矩阵实现跨模态信息流。线性复杂度天然适合 600 步序列。

### 架构图

```
EFT (现有):
  concat tokens → 共享 TransformerEncoder (O(L²))

Coupled Mamba (改进):
  各模态独立 Mamba 扫描，但 hidden state 耦合:

  t=0  t=1  t=2  ...  t=T
  ━━━━━━━━━━━━━━━━━━━━━━━━
  video_h ─→ video_h ─→ video_h ─→ ...
       ↕ C_vk    ↕ C_vk
  km_h    ─→ km_h    ─→ km_h    ─→ ...
       ↕ C_kt    ↕ C_kt
  telem_h ─→ telem_h ─→ telem_h ─→ ...

  h_video[t] = A_v · h_video[t-1] + B_v · x_video[t]
             + C_vk · h_km[t-1] + C_vt · h_telem[t-1]

  → 各模态最终 hidden states → mean/concat → pooling
```

### 关键实现细节

```python
class CoupledMambaBlock(nn.Module):
    """单个耦合 Mamba 层，多模态 hidden state 互相影响。"""
    def __init__(self, d_model, d_state, d_conv, expand):
        # 各模态独立的 Mamba 内核
        self.mamba_blocks = nn.ModuleDict()  # lazy-init
        # 耦合投影: 将其他模态的 hidden state 投影到当前模态
        self.coupling_projs = nn.ModuleDict()  # lazy-init

    def forward(self, mod_tokens_dict):
        # 1. 各模态独立 Mamba 扫描
        # 2. 对每个模态，加上其他模态的耦合贡献
        # 3. 残差连接
        ...

class CoupledMambaFusion(BaseFusion):
    def __init__(self, cfg):
        self.num_layers = cfg_get(cfg, "num_layers", 4)
        self.layers = nn.ModuleList([CoupledMambaBlock(...) for _ in range(num_layers)])
        self.pos_encoding = ...

    def forward(self, z_dict, mask_dict):
        # 对齐到最短长度
        # 逐层耦合 Mamba
        # Pooling
        ...
```

### 配置模板

```yaml
shared:
  model:
    fusion_coupled_mamba:
      name: coupled_mamba
      d_model: 512
      d_state: 16              # SSM hidden state 维度
      d_conv: 4                # 局部卷积核大小
      expand: 2                # Mamba 内部扩展系数
      num_layers: 4
      dropout: 0.1
      coupling_strength: 0.1   # 跨模态耦合系数初始化
      pooling: mean
```

### 依赖

- 需要 `mamba-ssm` 库 (`pip install mamba-ssm`)，或 `causal-conv1d` + 手写选择性扫描
- 备选: 使用 PyTorch 原生实现简化版 SSM（无 CUDA 内核优化，但可运行）

### 文献

- Coupled Mamba (NeurIPS 2024): 耦合 SSM，推理快 49%，显存降 83%
- MSAmba (AAAI 2025): Intra-Modal Mamba + Cross-Modal Hybrid Mamba
- AlignMamba (CVPR 2025): Mamba + OT 对齐 + MMD 对齐

---

## 方向 C: 跨模态对比对齐（辅助损失）

### 基本信息

| 项目 | 内容 |
|------|------|
| **改哪层** | Loss (辅助损失，不改架构) |
| **基于谁** | 任何 fusion 方法均可叠加 |
| **解决瓶颈** | B2 (模态编码器输出无语义对齐) |
| **实现文件** | `src/losses/contrastive_alignment.py` |

### 核心思想

在编码器输出、融合输入之间加 InfoNCE 对比损失。同一时刻的不同模态表示互为正样本，不同时刻的互为负样本。强制编码器将"同一情感状态"的不同模态映射到邻近区域。

### 架构图

```
不改变现有数据流，只在 loss 中增加对齐项:

  video_enc[t] ──┐
                 ├── InfoNCE: 同时刻跨模态为正对, 不同时刻为负对
  telem_enc[t] ──┘

  L_total = L_task(state + trend)
          + λ_align * L_infoNCE(video↔km, video↔telem, km↔telem)
```

### 关键实现细节

```python
class ContrastiveAlignmentLoss(nn.Module):
    """跨模态 InfoNCE 对比损失。"""
    def __init__(self, cfg):
        self.temperature = cfg.get("temperature", 0.07)
        self.lambda_align = cfg.get("lambda_align", 0.1)
        # 投影头: 将 d_model 映射到对比空间
        self.proj = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.ReLU(),
            nn.Linear(d_model, 128),
        )

    def forward(self, z_dict, mask_dict):
        """
        z_dict: {mod: EncoderOut} — 编码器输出, fusion 前
        对齐到最短时间长度, 沿时间维度采样正/负对
        """
        modalities = sorted(z_dict.keys())
        if len(modalities) < 2:
            return torch.tensor(0.0)

        # 对齐到最短长度
        min_t = min(z_dict[m]["tokens"].shape[1] for m in modalities)
        projected = {}
        for m in modalities:
            tok = z_dict[m]["tokens"][:, :min_t, :]  # [B, T, D]
            projected[m] = F.normalize(self.proj(tok), dim=-1)  # [B, T, 128]

        # 对每对模态计算 InfoNCE
        loss = 0.0
        count = 0
        for i, m1 in enumerate(modalities):
            for m2 in modalities[i+1:]:
                # [B*T, 128]
                z1 = projected[m1].reshape(-1, 128)
                z2 = projected[m2].reshape(-1, 128)
                # InfoNCE: 对角线为正对 (同时刻), 其余为负对
                sim = z1 @ z2.T / self.temperature  # [B*T, B*T]
                labels = torch.arange(sim.size(0), device=sim.device)
                loss += (F.cross_entropy(sim, labels) + F.cross_entropy(sim.T, labels)) / 2
                count += 1

        return self.lambda_align * loss / max(count, 1)
```

### 使用方式

在 runner 训练循环中，编码器输出后、融合前，计算对齐损失:

```python
# runner.py 修改
z_dict = {mod: encoder(x[mod], mask[mod]) for mod, encoder in self.model.encoders.items()}
L_align = self.align_loss(z_dict, mask_dict)  # 新增
h = self.model.fusion(z_dict, out_mask_dict)
preds = self.model.head(h)
L_task = self.loss_fn(preds, labels, masks)
L_total = L_task + L_align  # 新增
```

### 配置模板

```yaml
tasks:
  xxx_state_trend:
    train:
      contrastive_alignment:
        enabled: true
        temperature: 0.07
        lambda_align: 0.1
        proj_dim: 128
```

### 文献

- MMIM (Han et al., EMNLP 2021): 层次化互信息最大化
- DCLF (COLING 2025): 双重对比学习, IEMOCAP +4.67%
- AlignMamba (CVPR 2025): OT 局部对齐 + MMD 全局对齐

---

## 方向 D: 多尺度时序编码（编码器后增强）

### 基本信息

| 项目 | 内容 |
|------|------|
| **改哪层** | Encoder 后、Fusion 前（新增中间层） |
| **基于谁** | 所有 fusion 方法 (正交改进, 通用叠加) |
| **解决瓶颈** | B1 (编码器无时序建模) |
| **实现文件** | `src/models/components/multiscale_temporal.py` |

### 核心思想

编码器只做线性投影，完全没有时序上下文。在编码器输出后、融合前，用多尺度膨胀因果卷积捕捉不同时间尺度的模式:
- 短尺度 (0.6s): 单次按键/鼠标点击
- 中尺度 (2-3s): 一次交火
- 长尺度 (10-12s): 一轮回合

### 架构图

```
现有:
  encoder → [B, T, 512] → 直接送入 fusion

改为:
  encoder → [B, T, 512] ──┬── Conv1d(k=3, d=1)  → 短尺度 (0.6s @ 5Hz)
                           ├── Conv1d(k=3, d=5)  → 中尺度 (2.2s)
                           └── Conv1d(k=3, d=25) → 长尺度 (10.2s)
                           → concat [B, T, 512*3]
                           → Linear(512*3, 512) + LayerNorm
                           → [B, T, 512] → fusion
```

### 关键实现细节

```python
class MultiScaleTemporalEncoder(nn.Module):
    """多尺度膨胀因果卷积，在编码器输出后增强时序特征。"""
    def __init__(self, d_model=512, scales=None, dropout=0.1):
        super().__init__()
        if scales is None:
            scales = [1, 5, 25]  # 膨胀率: 对应 0.6s / 2.2s / 10.2s @ 5Hz

        self.branches = nn.ModuleList()
        for dilation in scales:
            self.branches.append(nn.Sequential(
                nn.Conv1d(d_model, d_model, kernel_size=3,
                          padding=dilation, dilation=dilation),
                nn.GELU(),
                nn.Dropout(dropout),
            ))

        self.proj = nn.Sequential(
            nn.Linear(d_model * len(scales), d_model),
            nn.LayerNorm(d_model),
        )

    def forward(self, tokens, mask=None):
        # tokens: [B, T, D]
        x = tokens.transpose(1, 2)  # [B, D, T]
        branch_outs = [branch(x).transpose(1, 2) for branch in self.branches]
        multi = torch.cat(branch_outs, dim=-1)  # [B, T, D*num_scales]
        return tokens + self.proj(multi)  # 残差连接
```

### 使用方式

在 `MultimodalModel.forward()` 中，编码器后、融合前调用:

```python
class MultimodalModel(nn.Module):
    def __init__(self, encoders, fusion, head, temporal_encoder=None):
        ...
        self.temporal_encoder = temporal_encoder  # 可选

    def forward(self, x_dict, mask_dict):
        z_dict = {}
        for mod, encoder in self.encoders.items():
            z = encoder(x_dict[mod], mask_dict.get(mod))
            if self.temporal_encoder is not None:
                z["tokens"] = self.temporal_encoder(z["tokens"], z["mask"])
            z_dict[mod] = z
        ...
```

### 配置模板

```yaml
shared:
  model:
    temporal_encoder:
      enabled: true
      scales: [1, 5, 25]      # 膨胀率
      kernel_size: 3
      dropout: 0.1
```

### 文献

- EmotionTFN (Sensors 2025): 短/中/长三尺度时间注意力, 94.2% 分类精度
- MOTIF (KBS 2025): BiMamba-X 双分支（长程+局部）
- Local-Global TCN (Applied Intelligence 2024): LC-TCN + GC-TCN
- MulT (Tsai et al., ACL 2019): Transformer 前加 1D 卷积

---

## 方向 E: Uncertainty-Aware Gated Fusion（不确定性感知门控）

### 基本信息

| 项目 | 内容 |
|------|------|
| **改哪层** | Fusion |
| **基于谁** | Gated Fusion (升级门控机制) |
| **解决瓶颈** | B5 (模态可靠性不等但被同等对待) |
| **注册名** | `uncertainty_gated` |
| **实现文件** | `src/models/fusions/uncertainty_gated.py` |

### 核心思想

当前 Gated Fusion 用 sigmoid 门控学习确定性权重。升级为: 每个编码器同时输出 mean 和 variance（不确定性估计），variance 高的模态自动降权。这给出了"我对这个模态此时此刻的信息有多大把握"的概率性回答。

### 架构图

```
Gated (现有):
  gate_m = sigmoid(W @ concat(all_tokens))     # 确定性, 无理论依据

Uncertainty-Aware (改进):
  各编码器:
    tokens_m, log_var_m = encoder_m(x)          # 同时输出均值和对数方差
    σ²_m = exp(log_var_m)                       # 方差 = 不确定性

  融合权重:
    w_m = softmax(-log_var_m)                   # 方差越小 → 越确定 → 权重越高
    fused = Σ w_m * tokens_m                    # 按置信度加权求和

  正则化:
    L_total = L_task + β * Σ_m log_var_m        # 防止方差全部坍缩到 0
```

### 关键实现细节

```python
class UncertaintyGatedFusion(BaseFusion):
    def __init__(self, cfg):
        ...
        # 每个模态一个方差预测头 (lazy-init)
        self._var_heads = nn.ModuleDict()

    def _get_var_head(self, mod, device):
        if mod not in self._var_heads:
            self._var_heads[mod] = nn.Sequential(
                nn.Linear(self.d_model, self.d_model // 4),
                nn.GELU(),
                nn.Linear(self.d_model // 4, 1),
            ).to(device)
        return self._var_heads[mod]

    def forward(self, z_dict, mask_dict):
        modality_names = sorted(z_dict.keys())
        min_t = min(z_dict[m]["tokens"].shape[1] for m in modality_names)

        tokens_list = []
        log_vars = []
        for mod in modality_names:
            tok = z_dict[mod]["tokens"][:, :min_t, :]     # [B, T, D]
            var_head = self._get_var_head(mod, tok.device)
            lv = var_head(tok)                             # [B, T, 1]
            tokens_list.append(tok)
            log_vars.append(lv)

        # 不确定性感知权重: softmax(-log_var) 沿模态维度
        log_var_stack = torch.cat(log_vars, dim=-1)        # [B, T, M]
        weights = F.softmax(-log_var_stack, dim=-1)        # [B, T, M]

        # 加权求和
        fused = sum(
            weights[..., i:i+1] * tokens_list[i]
            for i in range(len(modality_names))
        )

        # 不确定性正则项 (防坍缩, 返回供 runner 加到 loss)
        self._uncertainty_reg = log_var_stack.mean()

        # 可选 Transformer refinement
        ...
        return FusionOut(tokens=fused, pooled=pooled)
```

### 配置模板

```yaml
shared:
  model:
    fusion_uncertainty_gated:
      name: uncertainty_gated
      d_model: 512
      dropout: 0.1
      refine_layers: 4             # 与 Gated baseline 统一
      refine_nhead: 8
      refine_dim_feedforward: 1024  # 与 Gated baseline 统一
      uncertainty_reg_weight: 0.01  # β, 不确定性正则系数
      max_seq_len: 2000
      pooling: mean
```

### 文献

- Gao et al., "Embracing Unimodal Aleatoric Uncertainty for Robust Multimodal Fusion" (CVPR 2024)
- P-RMF (ACL 2025): 代理模态 + 不确定性量化
- Hierarchical MoE (arXiv 2025): 模态专家软路由

---

## 方向 F: Task-Aware Decoding（任务感知解码）

### 基本信息

| 项目 | 内容 |
|------|------|
| **改哪层** | Head |
| **基于谁** | multitask_seq head (增强 trend 分支) |
| **解决瓶颈** | B4 (state 和 trend 共用相同 token 无差异化) |
| **实现文件** | `src/models/heads/task_aware_multitask_seq.py` |

### 核心思想

state（当前水平）和 trend（变化方向）对时序信息的需求本质不同:
- **state**: 需要当前时刻的综合语境 → 直接用 fusion tokens 合适
- **trend**: 需要"近期"vs"稍早"的对比 → 时间差分特征更直接

为每个任务提供最适合的输入特征。

> **兼容性**: 方向 F 依赖逐帧 tokens [B,T,D]。LFT 多模态时返回 tokens=None（仅 pooled），因此 **F 不兼容 LFT**。F 可与 EFT/MFT/CMA/Gated/Late 配合。

### 架构图

```
现有 multitask_seq:
  tokens[B,T,D] → MLP_state → state[B,T,3]
  tokens[B,T,D] → MLP_trend → trend[B,T,3]  (同一输入)

改为 task_aware_multitask_seq:
  tokens[B,T,D] ──────────────→ MLP_state → state[B,T,3]
  tokens[B,T,D] → TemporalDiff → concat(tokens, diff) → MLP_trend → trend[B,T,3]

  TemporalDiff:
    diff[t] = tokens[t] - tokens[t-k]    (k 步时间差分, k=5 即 1 秒)
    → 编码 "最近 1 秒内表示如何变化"
```

### 关键实现细节

```python
@HEADS.register("task_aware_multitask_seq")
class TaskAwareMultiTaskSeqHead(BaseHead):
    def __init__(self, cfg):
        ...
        d_model = cfg.get("d_model", 512)
        hidden_dim = cfg.get("hidden_dim", 128)
        diff_k = cfg.get("diff_k", 5)           # 差分步长 (1s @ 5Hz)

        # State 分支: 直接用 tokens
        self.state_head = nn.Sequential(
            nn.Linear(d_model, hidden_dim), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(hidden_dim, 3),
        )
        # Trend 分支: tokens + 时间差分
        self.trend_proj = nn.Sequential(
            nn.Linear(d_model * 2, d_model), nn.LayerNorm(d_model), nn.GELU(),
        )
        self.trend_head = nn.Sequential(
            nn.Linear(d_model, hidden_dim), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(hidden_dim, 3),
        )
        self.diff_k = diff_k

    def forward(self, h):
        tokens = h["tokens"]  # [B, T, D]

        # State: 直接预测
        state_logits = self.state_head(tokens)

        # Trend: 时间差分 + 原始 token
        # diff[t] = tokens[t] - tokens[t-k], 前 k 步用零填充
        diff = tokens - F.pad(tokens[:, :-self.diff_k, :],
                              (0, 0, self.diff_k, 0))  # [B, T, D]
        trend_input = self.trend_proj(torch.cat([tokens, diff], dim=-1))
        trend_logits = self.trend_head(trend_input)

        return {"state": state_logits, "trend": trend_logits}
```

### 配置模板

```yaml
tasks:
  xxx_state_trend:
    model:
      head:
        name: task_aware_multitask_seq
        task_names: [state, trend]
        hidden_dim: 128
        num_classes: 3
        dropout: 0.1
        diff_k: 5              # 差分步长 (5步 = 1秒 @ 5Hz)
```

### 文献

- Hierarchical MoE (arXiv 2025): 不同情感维度用不同专家路由
- MMA (NAACL 2025): Mixture-of-Multimodal-Experts, 不同专家关注不同特征
- 时间差分: 经典信号处理思路, 广泛用于趋势检测

---

## 组合方案

三个候选方案，每个组合 2~3 个方向，作为论文主方法的候选。

### 方案 1: Aligned Bottleneck Fusion（对齐瓶颈融合）

**组合**: A (Bottleneck) + C (对比对齐) + F (任务感知)

```
Pipeline:
  Encoders → [跨模态对比对齐 loss] → Bottleneck Token Fusion → Task-Aware Head
                                      ↓
                        bottleneck tokens 中介跨模态交互
                                      ↓
                        state 分支: 直接用 pooled
                        trend 分支: 时间差分 + pooled
```

**论文故事**: 通过对比学习先对齐异质模态的语义空间，再用信息瓶颈提取跨模态精华，最后针对不同子任务提供定制化解码。

**创新点**: bottleneck 控制信息带宽 + 对比对齐保证语义一致 + 任务分流

### 方案 2: Multi-Scale Coupled Mamba Fusion（多尺度耦合 Mamba 融合）

**组合**: B (Coupled Mamba) + C (对比对齐) + D (多尺度时序)

```
Pipeline:
  Encoders → 多尺度 TCN → [对比对齐 loss] → Coupled Mamba → Head
              ↓
    短/中/长尺度时序特征
              ↓
    耦合 SSM 线性复杂度跨模态交互
```

**论文故事**: 先用多尺度卷积捕获不同时间粒度的压力模式，再用耦合状态空间模型以线性复杂度实现高效跨模态时序融合。

**创新点**: SSM 线性复杂度 + 多尺度时序 + 耦合跨模态状态传递

**风险**: 实现复杂度高，需要 Mamba CUDA 内核

### 方案 3: Uncertainty-Aware Multi-Scale Fusion（不确定性感知多尺度融合）

**组合**: E (不确定性门控) + C (对比对齐) + D (多尺度时序) + F (任务感知)

```
Pipeline:
  Encoders → 多尺度 TCN → [对比对齐 loss] → Uncertainty-Aware Gating → Task-Aware Head
              ↓                                  ↓
    多尺度时序特征                     各模态不确定性估计
                                        → 按置信度加权融合
                                                 ↓
                                      state: 直接预测
                                      trend: 差分特征预测
```

**论文故事**: 不同模态对游戏压力的信息量不等（视频噪声高、遥测信号直接），通过不确定性量化自适应分配模态贡献，结合多尺度时序编码和任务感知解码。

**创新点**: 不确定性驱动的自适应融合 + 多尺度 + 任务分流，且不依赖 Mamba（纯 PyTorch，易复现）

---

## 实验计划

### 独立方向消融

先单独验证每个方向的独立贡献，基于最简配置:

| 实验 | 融合方法 | 辅助 Loss | 时序增强 | Head | 对比对象 |
|------|---------|----------|---------|------|---------|
| A-only | bottleneck | 无 | 无 | multitask_seq | EFT (同为集中式融合) |
| B-only | coupled_mamba | 无 | 无 | multitask_seq | EFT (替换 Transformer) |
| C-only | eft | 对比对齐 | 无 | multitask_seq | EFT (无对齐) |
| D-only | eft | 无 | 多尺度 TCN | multitask_seq | EFT (无 TCN) |
| E-only | uncertainty_gated | 无 | 无 | multitask_seq | Gated |
| F-only | eft | 无 | 无 | task_aware_multitask_seq | EFT (原 head) |

> **注**: C/D/F 选 EFT 作为载体因为 EFT 是最简单的全交互融合且返回逐帧 tokens。LFT 返回 tokens=None，不兼容 F。

### 组合方案

消融完成后，挑选有正增益的方向组合:

| 实验 | 组合 | 对比对象 |
|------|------|---------|
| Combo-1 | A + C + F | 6 个 baseline |
| Combo-2 | B + C + D | 6 个 baseline |
| Combo-3 | E + C + D + F | 6 个 baseline |

### 每组实验配置

- 模态: video+km+telem（三模态）
- Split: cross_subject + within_subject
- Seeds: 0, 1, 2
- 评估: val_f1_mean (early stopping), test_macro_f1_state, test_macro_f1_trend, test_balanced_acc_*

### 实现优先级

| 优先级 | 方向 | 理由 |
|--------|------|------|
| 1 | F (Task-Aware Head) | 最简单, 10 行代码, 直接验证 trend 差分假设 |
| 2 | C (对比对齐) | 不改架构, 加辅助 loss, 对所有 fusion 通用 |
| 3 | D (多尺度 TCN) | 独立组件, 对所有 fusion 通用 |
| 4 | A (Bottleneck) | 新 fusion, 中等复杂度 |
| 5 | E (Uncertainty Gated) | 新 fusion, 中等复杂度 |
| 6 | B (Coupled Mamba) | 最复杂, 依赖外部库 |

---

## 参考文献

### 方向 A: Bottleneck Token Fusion

[A1] Arsha Nagrani, Shan Yang, Anurag Arnab, Aren Jansen, Cordelia Schmid, Chen Sun.
"Attention Bottlenecks for Multimodal Fusion."
**NeurIPS 2021.**
https://arxiv.org/abs/2107.00135

[A2] (DBA) — "Domain-Separated Bottleneck Attention Fusion Framework for Multimodal Sentiment Analysis."
**ACM Transactions on Multimedia Computing, Communications, and Applications (TOMM), 2025.**
https://dl.acm.org/doi/10.1145/3711865

[A3] (EMBT) — "Enhanced Emotion Recognition Through Dynamic Restrained Adaptive Loss and Extended Multimodal Bottleneck Transformer."
**Applied Sciences, 15(5):2862, 2025.**
https://www.mdpi.com/2076-3417/15/5/2862

### 方向 B: Coupled Mamba Fusion

[B1] Wenbing Li, Hang Zhou, Zhuoran Zheng, Ziqiang Liu, Jingyi Zhang.
"Coupled Mamba: Enhanced Multi-modal Fusion with Coupled State Space Model."
**NeurIPS 2024 (poster).**
https://arxiv.org/abs/2405.18014

[B2] Haijian Liang, Guancheng Wan, Jiachen Fan, Boming Yang, Jieming Zhu, Xiaoliang Xu, Zenglin Xu.
"MSAmba: Exploring Multimodal Sentiment Analysis with State Space Models."
**AAAI 2025.**
https://ojs.aaai.org/index.php/AAAI/article/view/32120

[B3] Yan Li, Yifei Xing, Xiangyuan Lan, Xin Li, Haifeng Chen, Dongmei Jiang.
"AlignMamba: Enhancing Multimodal Mamba with Local and Global Cross-modal Alignment."
**CVPR 2025.**
https://arxiv.org/abs/2412.00833

[B4] Yifei Xing et al.
"EMMA: Empowering Multi-modal Mamba with Structural and Hierarchical Alignment."
**ICLR 2025.**
https://arxiv.org/abs/2410.05938

### 方向 C: 跨模态对比对齐

[C1] Wei Han, Hui Chen, Soujanya Poria.
"Improving Multimodal Fusion with Hierarchical Mutual Information Maximization for Multimodal Sentiment Analysis (MMIM)."
**EMNLP 2021.**
https://arxiv.org/abs/2109.00412

[C2] Yunhe Xie, Chengjie Sun, Ziyi Cao, Bingquan Liu, Zhenzhou Ji, Yuanchao Liu, Lili Shan.
"A Dual Contrastive Learning Framework for Enhanced Multimodal Conversational Emotion Recognition (DCLF)."
**COLING 2025.**
https://aclanthology.org/2025.coling-main.272/

[C3] (AlignMamba 的对齐模块，见 [B3])

[C4] Mingcheng Li, Dingkang Yang, Yang Liu, Shunli Wang et al.
"Toward Robust Incomplete Multimodal Sentiment Analysis via Hierarchical Representation Learning (HRLF)."
**NeurIPS 2024.**
https://proceedings.neurips.cc/paper_files/paper/2024/hash/3209cf3312b2cbb68e33644362ddc2bd-Abstract-Conference.html

[C5] Rui Liu, Haolin Zuo, Zheng Lian, Bjorn W. Schuller, Haizhou Li.
"Contrastive Learning Based Modality-Invariant Feature Acquisition for Robust Multimodal Emotion Recognition with Missing Modalities (CIF-MMIN)."
**IEEE Transactions on Affective Computing, 2024.**
https://ieeexplore.ieee.org/document/10474146/

### 方向 D: 多尺度时序编码

[D1] (EmotionTFN) — "Multi-Scale Temporal Fusion Network for Real-Time Multimodal Emotion Recognition."
**Sensors, 25(16):5066, 2025.**
https://www.mdpi.com/1424-8220/25/16/5066

[D2] (MOTIF) — "A Text-Aware and Disentangled Mamba-based Architecture for Multimodal Sentiment Analysis."
**Knowledge-Based Systems, 2025.**
https://www.sciencedirect.com/science/article/abs/pii/S0950705125016909

[D3] Shi, Zhang, Liu.
"Local-Global Contextual TCN for Continuous Emotion Recognition."
**Applied Intelligence, 2024.**
https://link.springer.com/article/10.1007/s10489-024-05329-w

[D4] Yao-Hung Hubert Tsai, Shaojie Bai, Paul Pu Liang, J. Zico Kolter, Louis-Philippe Morency, Ruslan Salakhutdinov.
"Multimodal Transformer for Unaligned Multimodal Language Sequences (MulT)."
**ACL 2019.**
https://arxiv.org/abs/1906.00295

### 方向 E: 不确定性感知融合

[E1] Haoyu Gao, Jiayi Zhang, Xiaomin Song, Hang Su, Jun Zhu.
"Embracing Unimodal Aleatoric Uncertainty for Robust Multimodal Fusion."
**CVPR 2024.**
https://openaccess.thecvf.com/content/CVPR2024/papers/Gao_Embracing_Unimodal_Aleatoric_Uncertainty_for_Robust_Multimodal_Fusion_CVPR_2024_paper.pdf

[E2] Aoqiang Zhu, Min Hu, Xiaohua Wang, Jiaoyun Yang, Yiming Tang, Ning An.
"Proxy-Driven Robust Multimodal Sentiment Analysis with Incomplete Data (P-RMF)."
**ACL 2025.**
https://aclanthology.org/2025.acl-long.1075/

[E3] Yitong Zhu et al.
"Hierarchical MoE: Continuous Multimodal Emotion Recognition with Incomplete and Asynchronous Inputs."
**arXiv, 2025.**
https://arxiv.org/abs/2508.02133

### 方向 F: 任务感知解码

[F1] (Hierarchical MoE, 同 [E3])

[F2] Kezhou Chen, Shuo Wang, Huixia Ben, Shengeng Tang, Yanbin Hao.
"Mixture of Multimodal Adapters for Sentiment Analysis (MMA)."
**NAACL 2025.**
https://aclanthology.org/2025.naacl-long.90/

### 综述 & 通用参考

[G1] "A Comprehensive Review of Multimodal Emotion Recognition: Techniques, Challenges, and Future Directions."
**Biomimetics, 2025.**
https://pmc.ncbi.nlm.nih.gov/articles/PMC12292624/

[G2] "Recent Trends of Multimodal Affective Computing: A Survey from an NLP Perspective."
**arXiv, 2024.**
https://arxiv.org/html/2409.07388v2

[G3] "Deep Multimodal Data Fusion."
**ACM Computing Surveys, 2024.**
https://dl.acm.org/doi/full/10.1145/3649447

[G4] Waligora et al.
"Joint Multimodal Transformer for Emotion Recognition in the Wild."
**CVPR 2024 Workshop (ABAW).**
https://openaccess.thecvf.com/content/CVPR2024W/ABAW/papers/Waligora_Joint_Multimodal_Transformer_for_Emotion_Recognition_in_the_Wild_CVPRW_2024_paper.pdf

[G5] Praveen et al.
"Recursive Joint Cross-Modal Attention for Multimodal Fusion in Dimensional Emotion Recognition."
**CVPR 2024 Workshop (ABAW).**
https://openaccess.thecvf.com/content/CVPR2024W/ABAW/papers/Praveen_Recursive_Joint_Cross-Modal_Attention_for_Multimodal_Fusion_in_Dimensional_Emotion_CVPRW_2024_paper.pdf

[G6] Fanourakis & Chanel.
"AMuCS: Affective Multimodal Counter-Strike Video Game Dataset."
**2024.**

### 项目已有参考（improvement_roadmap.md 中引用）

[P1] Rahman 2020 — MAG/MAG+: Multimodal Adaptation Gate
[P2] Tang 2022 — MMT: Multi-way Multi-modal Transformer
[P3] Makantasis 2023 — 游戏 arousal 最优时间窗口 0.5-2 秒
[P4] Melhart 2022 — AGAIN 数据集, DTW 标签清洗
[P5] Epp 2011 — 键盘时序特征识别情绪 77-87%
[P6] Tahir 2022 — 89 维键盘特征达 86.95% 准确率
[P7] Yannakakis 2023 — 游戏情感计算综述
