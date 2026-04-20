# 压力识别模块改进路线图

> 基于项目现状分析 + 22篇参考文献综合形成的改进建议。
> 所有方案均限定在三模态（video/km/telem）范围内。

---

## 当前性能基线

| 指标 | 当前最佳 | 目标 |
|------|---------|------|
| CCC（唤醒度回归） | ~0.21 | >0.4 |
| Macro-F1（3类分类） | ~0.44 | >0.6 |
| 多模态增益 | +0.02 CCC | 显著正增益 |
| CMA vs LFT差异 | 几乎无 | — |

**关键观察**：玩家交火时arousal显著上升，被击杀时达峰值。arousal由离散游戏事件驱动，变化集中在2-7秒尺度。

---

## 改进方向一：标签质量优化（P0，低成本高回报）

### 1.1 标签时间平滑

**问题**：PAGAN自评标注存在标注者手部抖动噪声。

**方案**：对连续arousal标签做时间域高斯平滑。

```python
from scipy.ndimage import gaussian_filter1d

smoothed = gaussian_filter1d(arousal_values, sigma=2)  # sigma=2帧≈0.4秒@5Hz
```

**参考**：AGAIN数据集（Melhart 2022）使用DTW距离清洗标注异常值。

**实现位置**：`scripts/merge_arousal_reg_trend_labels.py` 或新建 `scripts/smooth_labels.py`

### 1.2 标注延迟校正（扩大搜索范围）

**问题**：当前lag sweep在±2秒(±10步)范围，最佳值出现在边界，提示可能需要更大范围。

**方案**：
- 扩大搜索到±5秒(±25步@5Hz)
- 尝试自适应延迟：per-participant最优lag估计
- 参考AMuCS论文baseline的7秒窗口+5秒步长设置

**实现位置**：`train.ipynb` 中lag sweep分析cell

### 1.3 标签清洗（DTW异常检测）

**参考**：Melhart 2022使用累积DTW距离检测无响应标注者，剔除约10%数据。

**方案**：
- 计算每个session的标注与其他session的DTW距离
- 剔除DTW距离异常高的session（标注质量差）
- 重新训练评估CCC变化

---

## 改进方向二：时序建模（P1，中等难度）

### 2.1 局部时序卷积（TCN）

**问题**：600帧/120秒全局self-attention粒度太粗，2-5秒的交火事件信号被稀释。

**文献支撑**：
- MulT（Tsai 2019）：Transformer前加1D时序卷积，显著提升性能
- Makantasis 2023：0.5-2秒时间窗口在游戏arousal检测中最优
- AMuCS baseline：7秒窗口+5秒步长

**方案**：在各模态编码器输出后、融合前，插入TCN层。

```python
class TemporalConvBlock(nn.Module):
    """局部时序卷积块，捕捉短时事件模式"""
    def __init__(self, d_model, kernel_size=7, num_layers=2, dropout=0.1):
        super().__init__()
        layers = []
        for _ in range(num_layers):
            layers.extend([
                nn.Conv1d(d_model, d_model, kernel_size, padding=kernel_size//2),
                nn.GELU(),
                nn.Dropout(dropout),
            ])
        self.conv = nn.Sequential(*layers)
        self.norm = nn.LayerNorm(d_model)

    def forward(self, x, mask=None):
        # x: [B, T, D]
        residual = x
        x = x.transpose(1, 2)           # [B, D, T]
        x = self.conv(x).transpose(1, 2) # [B, T, D]
        return self.norm(x + residual)    # 残差连接
```

**建议参数**：
- kernel_size=7（1.4秒@5Hz，覆盖短时交火）
- kernel_size=15（3秒，覆盖完整交战过程）
- 可做消融实验对比不同kernel size

**实现位置**：
- 新建 `src/models/components/temporal_conv.py`
- 在 `src/core/runner.py` 的 `MultimodalModel.forward()` 中，编码器输出后、融合前调用
- 或集成到各编码器内部

### 2.2 层次化注意力（局部窗口+全局）

**方案**：将全局Transformer替换为两级注意力。

```
第一级：局部窗口注意力（window_size=25帧=5秒）
    → 每个5秒窗口内做self-attention
    → 捕捉交火/击杀等事件内部的精细模式

第二级：全局注意力（在窗口级表示上）
    → 跨窗口建模长程依赖
    → 捕捉整局游戏的arousal趋势
```

**参考架构**：类似Longformer/Swin Transformer的局部+全局混合注意力。

**实现位置**：新建 `src/models/fusions/hierarchical_lft.py`，注册为 `hlft`

### 2.3 相对位置编码

**问题**：当前正弦位置编码是绝对的，不擅长建模"事件A发生后X秒arousal上升"这种相对时序关系。

**方案**：替换为ALiBi或RoPE。

**实现位置**：`src/models/components/positional_encoding.py`，新增类并注册

---

## 改进方向三：融合机制（P1，中等难度）

### 3.1 门控融合（推荐首先尝试）

**问题**：EFT简单拼接导致短模态被淹没；多模态增益几乎为零。

**文献支撑**：MAG（Rahman 2020）、MAG+（扩展版）验证门控位移机制有效。

**方案**：模态级自适应门控。

```python
class GatedFusion(BaseFusion):
    """各时间步自适应决定模态权重"""
    def __init__(self, cfg):
        super().__init__()
        d = cfg_get(cfg, "d_model", 256)
        self.modalities = None  # 延迟初始化
        # 延迟创建门控网络，因为模态集合在运行时确定

    def _init_gates(self, modality_names, d_model):
        M = len(modality_names)
        # 每个模态一个门控：输入为所有模态拼接，输出为标量门
        self.gate_nets = nn.ModuleDict({
            m: nn.Sequential(
                nn.Linear(M * d_model, d_model),
                nn.GELU(),
                nn.Linear(d_model, 1),
                nn.Sigmoid()
            ) for m in modality_names
        })
        self.modalities = modality_names

    def forward(self, z_dict, mask_dict):
        mod_names = sorted(z_dict.keys())
        if self.modalities is None:
            d = z_dict[mod_names[0]]["pooled"].shape[-1]
            self._init_gates(mod_names, d)

        # 对齐到最短时间长度
        min_t = min(z_dict[m]["tokens"].shape[1] for m in mod_names)
        trimmed = {m: z_dict[m]["tokens"][:, :min_t, :] for m in mod_names}

        # 拼接所有模态作为门控输入
        concat = torch.cat([trimmed[m] for m in mod_names], dim=-1)  # [B, T, M*D]

        # 加权融合
        fused = torch.zeros_like(trimmed[mod_names[0]])
        for m in mod_names:
            gate = self.gate_nets[m](concat)  # [B, T, 1]
            fused = fused + gate * trimmed[m]

        pooled = masked_mean_pool(fused, ...)
        return {"tokens": fused, "pooled": pooled}
```

**优势**：交火时模型可自动提高telem(伤害/击杀事件)和km(操作频率)的权重。

**实现位置**：新建 `src/models/fusions/gated.py`，注册为 `gated`

### 3.2 互信息正则化

**问题**：当前融合可能只学到模态间共性，丢弃互补信息。

**文献支撑**：MMIM（Han 2021）通过层次化MI最大化确保每个模态的信息贡献不丢失。

**方案**：在训练损失中加MI约束项。

```
L_total = L_task + λ₁ * L_MI_inter + λ₂ * L_MI_fusion

L_MI_inter: 模态对之间的互信息下界（Barber-Agakov bound）
L_MI_fusion: 融合表示与各模态表示之间的对比预测编码（CPC）损失
```

**实现要点**：
- 新建 `src/losses/mi_regularization.py`
- 使用InfoNCE/CPC loss近似MI下界
- λ₁, λ₂ 作为超参数，建议从0.01-0.1开始搜索
- GMM估计entropy（参考MMIM论文的history memory FIFO queue策略）

**实现位置**：`src/losses/mi_regularization.py`，修改 `src/core/runner.py` 训练循环以加入正则项

### 3.3 多路张量融合（MMT）

**文献支撑**：MMT（Tang 2022）用张量环分解实现O(M)复杂度的多路注意力，比MulT提升20%+ MAE。

**方案**：用张量分解替换EFT的简单拼接self-attention，3模态同时交互。

**复杂度**：较高，建议作为后期尝试。

**实现位置**：新建 `src/models/fusions/mmt.py`，注册为 `mmt`

### 3.4 渐进式融合

**方案**：按语义层次逐步融合，而非一次性拼接。

```
第一步：km + telem → 行为-游戏状态表示（行为层融合）
    理由：km(操作)和telem(游戏事件)语义最接近，都是"玩家做了什么+发生了什么"

第二步：行为表示 + video → 最终多模态表示（视觉层融合）
    理由：video提供telem无法捕捉的视觉动态（画面晃动、闪光弹等）
```

**实现位置**：新建 `src/models/fusions/progressive.py`

---

## 改进方向四：特征增强（P1-P2）

### 4.1 KM原始事件序列建模

**问题**：25维统计特征压缩过度，丢失键盘时序动态。

**文献支撑**：
- Epp 2011：键盘按键时长/间隔特征达77-87%情绪识别准确率
- Tahir 2022：89维键盘特征+文本特征达86.95%准确率
- Freihaut 2021：鼠标行为与压力关系很弱（键盘 >> 鼠标）

**方案**：
- 从原始keyboard.csv提取更丰富的时序特征：
  - 按键持续时间（dwell time）的均值/方差/分布
  - 按键间隔（flight time）的均值/方差
  - 按键频率的短时变化率（交火时WASD+鼠标点击频率骤增）
  - 操作模式变化（如从探索→交战的操作模式切换）
- 或直接用1D-CNN/小Transformer处理原始按键事件流

**实现位置**：修改 `scripts/extract_km_features.py` 增加特征维度

### 4.2 视频时序动态特征

**问题**：ResNet-50逐帧提取静态特征，丢失帧间动态信息。

**方案**：
- 帧差特征：相邻帧特征的差值/余弦距离，捕捉画面剧烈变化
- 光流统计：全局光流幅度（交火时画面剧烈晃动）
- 视觉节奏：特征变化率的统计量

**实现位置**：修改视频特征提取pipeline或在编码器中计算帧差

### 4.3 Telem事件显式编码

**问题**：当前109维统计特征已做战斗聚焦筛选，但全部作为连续特征处理，未显式建模"事件发生/未发生"的离散性。

**方案**：
- 构建二值事件通道：`is_in_combat`, `is_taking_damage`, `just_killed`, `just_died`
- 事件触发的时间衰减特征：`time_since_last_kill`, `time_since_last_death`
- 这些特征直接对应arousal变化的触发因素

**实现位置**：修改 `scripts/extract_game_telem_features.py`

---

## 改进方向五：训练策略（P2）

### 5.1 模态Dropout

**方案**：训练时随机遮蔽某个模态（置零），强制模型不依赖单一模态。

```yaml
train:
  modality_dropout: 0.15  # 15%概率遮蔽每个模态
```

**注意**：`base.yaml`已有`modality_dropout: 0.0`配置项，直接改值即可。

### 5.2 课程学习

**方案**：先在"容易"样本（高/低arousal极端值）上训练，再逐步加入中等arousal样本。

### 5.3 辅助排序损失

**方案**：在CCC/MSE之外加入pairwise ranking loss。

```
L_rank = max(0, -sign(y_i - y_j) * (pred_i - pred_j) + margin)
```

**文献支撑**：AGAIN数据集（Melhart 2022）使用preference learning（排序学习）框架，Random Forest达58-82%准确率。

**实现位置**：新建 `src/losses/ranking.py`

### 5.4 自监督预训练

**方案**：利用大量未标注游戏片段做masked autoencoding预训练，再微调。

**复杂度**：高，作为远期方向。

---

## 建议执行顺序

| 阶段 | 步骤 | 预期收益 | 实现难度 | 文献依据 |
|------|------|---------|---------|---------|
| **阶段1** | 1.1 标签平滑 | 中 | 低 | Melhart 2022 |
| **阶段1** | 1.2 延迟校正扩大范围 | 中 | 低 | AMuCS baseline |
| **阶段2** | 2.1 编码器后加TCN | 中-高 | 中 | MulT, Makantasis 2023 |
| **阶段2** | 3.1 门控融合替换LFT | 中 | 中 | MAG/MAG+ |
| **阶段3** | 4.1 KM特征增强 | 中 | 中 | Epp 2011, Tahir 2022 |
| **阶段3** | 4.3 Telem事件显式编码 | 中 | 低-中 | 项目观察 |
| **阶段4** | 3.2 MI正则化 | 中 | 中-高 | MMIM (Han 2021) |
| **阶段4** | 2.2 层次化注意力 | 中 | 中-高 | Longformer思路 |
| **阶段5** | 3.3 多路张量融合 | 中 | 高 | MMT (Tang 2022) |
| **阶段5** | 5.4 自监督预训练 | 中-高 | 高 | — |

---

## 参考文献速查

| 简称 | 全称 | 核心贡献 |
|------|------|---------|
| MulT | Tsai 2019, Multimodal Transformer | 1D时序卷积+跨模态注意力，处理非对齐序列 |
| MMT | Tang 2022, Multi-way Multi-modal Transformer | 张量分解多路注意力，O(M)复杂度 |
| MMIM | Han 2021, Hierarchical MI Maximization | 互信息正则化防止融合信息丢失 |
| MAG/MAG+ | Rahman 2020 / 扩展版 | 门控位移机制，轻量级模态融合 |
| Makantasis 2023 | Pixels and Sounds of Emotion | 游戏arousal最优时间窗口0.5-2秒 |
| AGAIN | Melhart 2022 | 游戏arousal数据集，DTW清洗+preference learning |
| AMuCS | Fanourakis 2024 | 本项目数据集，baseline CCC=0.181 |
| Epp 2011 | Keystroke Dynamics | 键盘时序特征识别情绪77-87% |
| Tahir 2022 | Non-Acted Keystrokes | 89维键盘特征达86.95%准确率 |
| Freihaut 2021 | Mouse for Stress | 鼠标行为与压力关系很弱 |
| Kang & Kim 2022 | PAO Model | 7秒窗口，DTW行为异常检测 |
| Qiu 2023 | Topic-Style Transformer | 内容感知归一化处理个体差异 |
| Yannakakis 2023 | Affective Game Computing Survey | 游戏情感计算综述 |
