# 压力分类任务定义与实施指南

## 1. 任务总览

### 1.1 目标

从连续 arousal 回归转向两个三分类任务，用于闭环游戏压力调控：

| 输出 | 类别 | 用途 |
|------|------|------|
| **stress_level** | LOW / MID / HIGH | 当前压力档位，驱动游戏参数调整 |
| **stress_trend** | DOWN / FLAT / UP | 未来压力趋势，提前触发预调整 |

### 1.2 输入

- 模态：KM + Video + Telemetry（三模态融合，各模态可单独消融）
- 窗口：seq_len 个时间步（对齐后 5Hz，seq_len=150 对应 30s）
- 延迟约束：≤ 1-2 个窗口

### 1.3 架构不变

LFT（Late Fusion Transformer）主干完全保留。仅改变：
- **Head**：regression_seq → classification_seq
- **Loss**：MSE → CrossEntropyLoss
- **Metrics**：CCC/RMSE → macro-F1 / balanced accuracy
- **Label 构建**：连续 arousal → 离散类别

---

## 2. 标签定义

### 2.1 数据来源

原始标签仍来自 ranktrace.csv 的 `VideoTime` + `arousal`/`valence` 列，通过 `build_arousal_sequence_labels.py` 插值到特征时间网格，得到 `labels_arousal_seq.json`（每个 stem 一条连续 arousal 序列）。

### 2.2 Task A: stress_level（压力状态三分类）

**标签构建流程：**

```
对每个 session (stem):
    y_raw = labels_arousal_seq[stem]["values"]   # 连续 arousal 序列

    # 1. 平滑（减少标注噪声）
    y_smooth = ema(y_raw, alpha=0.05)            # 或 gaussian_filter1d(sigma=5)

    # 2. 计算 session 内分位数阈值（仅用 train split 计算）
    q20 = percentile(y_smooth, 20)
    q80 = percentile(y_smooth, 80)

    # 3. 分档
    label[t] = 0 (LOW)  if y_smooth[t] <= q20
    label[t] = 1 (MID)  if q20 < y_smooth[t] < q80
    label[t] = 2 (HIGH) if y_smooth[t] >= q80
```

**关键设计决策：**

- **per-session 分位数**：不使用全局阈值，因为个体 arousal 尺度差异大
- **分位数只从 train split 的 session 计算**：val/test 的 session 用各自 session 内统计，不泄漏跨 split 信息
- **分位数是 session 自身的统计量**（不是 train 的全局分位数应用到其他 session），这与部署逻辑一致：每个玩家用自己的基线

### 2.3 Task B: stress_trend（压力趋势三分类）

**标签构建流程：**

```
对每个时间步 t:
    # 1. 当前窗口均值
    y_now = mean(y_smooth[t : t+W])              # W = 窗口宽度（如 5s = 25步@5Hz）

    # 2. 未来窗口均值
    y_future = mean(y_smooth[t+H : t+H+W])       # H = 预测时距（如 5s = 25步）

    # 3. 差值
    delta = y_future - y_now

    # 4. 自适应阈值
    tau = 0.25 * std(y_smooth)                    # session 内波动尺度

    # 5. 分类
    label[t] = 0 (DOWN) if delta < -tau
    label[t] = 1 (FLAT) if -tau <= delta <= tau
    label[t] = 2 (UP)   if delta > tau

    # 6. 边界处理
    如果 t+H+W 超出序列长度，该时间步标记为无效（mask=False）
```

**参数默认值：**

| 参数 | 默认值 | 说明 |
|------|--------|------|
| W | 25 步 (5s @5Hz) | 均值窗口 |
| H | 25 步 (5s @5Hz) | 预测时距 |
| tau_scale | 0.25 | tau = tau_scale × std(y_smooth) |
| smoothing_alpha | 0.05 | EMA 平滑系数 |

### 2.4 有效性掩码

最终 mask = 原始 arousal mask ∧ trend 有效 mask（趋势任务中边界无效的时间步）。

两个任务共享同一条 arousal 序列，只是标签生成逻辑不同。

---

## 3. 评估指标

### 3.1 主指标

| 任务 | 主指标 | 辅指标 |
|------|--------|--------|
| stress_level | **macro-F1** | balanced accuracy |
| stress_trend | **macro-F1** | balanced accuracy |

### 3.2 随机基线

3 分类随机基线 macro-F1 ≈ 0.33。**目标：macro-F1 > 0.40 即说明信号可用。**

### 3.3 分层评估

除全局指标外，必须报告：
- **per-session macro-F1**（判断跨人泛化）
- **per-class precision/recall**（判断类别平衡）
- **confusion matrix**（每次实验保存）

---

## 4. 代码改动清单

### 4.1 新增：标签构建脚本

**`scripts/build_stress_labels.py`**

输入：
- `labels_arousal_seq.json`（已有）
- `splits/multimodal_split.json`（已有）
- 任务参数（smoothing, quantiles, horizon 等）

输出：
- `data/labels_stress_level.json`：每个 stem → `{values: [int], mask: [bool]}`
- `data/labels_stress_trend.json`：每个 stem → `{values: [int], mask: [bool]}`
- `data/labels_stress_meta.json`：每个 stem 的阈值统计（q20, q80, tau, std）

### 4.2 修改：DataModule

**`src/data/datamodules/amucs_seq.py`**

当前 `_load_seq_label` 加载连续值。需要支持加载整数类别标签：
- 新增 config 字段 `task: level_cls | trend_cls | regression`（默认 regression 保持向后兼容）
- 分类任务时 `y` 为 `LongTensor`，回归时为 `FloatTensor`

**最小改动方案**：不改 `amucs_seq.py`，而是让 `build_stress_labels.py` 输出的 JSON 格式与现有 `labels_arousal_seq.json` 完全一致（`values` 为 int list，`mask` 为 bool list），DataModule 通过 config 中的 `labels_seq_path` 切换即可。

唯一需要改的：分类任务时 `y` 的 dtype 需要是 `long`。在 `__getitem__` 中根据值范围自动判断，或在 config 中指定 `label_dtype: long`。

### 4.3 新增：分类 Head

**`src/models/heads/classification_seq.py`**

```python
@HEADS.register("classification_seq")
class ClassificationSeqHead(BaseHead):
    """Token-wise classification: [B, T, D] -> [B, T, num_classes]"""

    def __init__(self, cfg):
        d_model = cfg.get("d_model", 256)
        hidden_dim = cfg.get("hidden_dim", 128)
        num_classes = cfg.get("num_classes", 3)
        dropout = cfg.get("dropout", 0.1)

        self.mlp = Sequential(
            Linear(d_model, hidden_dim), GELU(), Dropout(dropout),
            Linear(hidden_dim, num_classes)
        )

    def forward(self, h: FusionOut) -> Tensor:
        return self.mlp(h["tokens"])  # [B, T, num_classes] (logits)
```

### 4.4 新增：分类 Loss

**`src/losses/ce_seq_masked.py`**

```python
@LOSSES.register("ce_seq_masked")
class MaskedSequenceCELoss(nn.Module):
    """Masked cross-entropy for sequence classification."""

    def forward(self, pred, target, mask=None):
        # pred: [B, T, C], target: [B, T] (long)
        # reshape to [B*T, C] and [B*T] for cross_entropy
        # apply mask to select valid timesteps only
```

支持可选 `class_weights` 参数处理类别不平衡。

### 4.5 新增：分类 Metrics

**`src/metrics/macro_f1.py`**

```python
@METRICS.register("macro_f1")
class MacroF1Metric:
    def __call__(self, pred, target):
        # pred: [N, C] logits -> argmax -> labels
        # sklearn.metrics.f1_score(target, pred, average='macro')
```

**`src/metrics/balanced_acc.py`**

```python
@METRICS.register("balanced_acc")
class BalancedAccMetric:
    def __call__(self, pred, target):
        # sklearn.metrics.balanced_accuracy_score
```

### 4.6 修改：Runner

**`src/core/runner.py`** 需要的调整：

1. `_run_epoch` 中 metric 评估：分类任务时 `preds` 是 logits `[N, C]`，需要传给 metric 时保留 logits（由 metric 内部 argmax），或统一做 argmax 后传
2. 保存 confusion matrix 到 run_dir（分类任务时）

### 4.7 Config 示例

**`configs/experiments/stress_level_km_telem.yaml`**

```yaml
_base_: ../base.yaml

data:
  name: amucs_seq
  modalities: [km, telem]
  data_root: G:/我的云端硬盘/AmuCS_experiment/features/aligned/amucs_trial
  labels_seq_path: data/labels_stress_level.json
  split_path: data/splits/multimodal_split.json
  seq_len: 150
  train_stride: 75
  normalize: true
  label_dtype: long

model:
  d_model: 256
  encoders:
    km:
      name: stat
      feature_dim: 25
    telem:
      name: stat_pool
      feature_dim: 243
  fusion:
    name: lft
    nhead: 8
    num_layers: 2
    dim_feedforward: 512
    dropout: 0.1
  head:
    name: classification_seq
    num_classes: 3
    hidden_dim: 128
    dropout: 0.1

train:
  loss: ce_seq_masked
  epochs: 50
  optimizer:
    name: adamw
    lr: 1.0e-4
    weight_decay: 0.01
  early_stopping:
    patience: 10
    metric: val_macro_f1
    mode: max

eval:
  metrics: [macro_f1, balanced_acc]
```

---

## 5. 训练时的基线校准（模拟部署）

### 5.1 为什么需要

训练用 per-session 全程分位数构建标签，但部署时没有"未来信息"。为了让训练评估更贴近部署：

### 5.2 方案：前 T 秒估计阈值（推荐 T=60s）

```
对每个 session:
    warmup_steps = T / dt                    # 60s / 0.2s = 300 步
    y_warmup = y_smooth[:warmup_steps]

    q20_est = percentile(y_warmup, 20)
    q80_est = percentile(y_warmup, 80)

    # 用 warmup 估计的阈值对全程分档
    # 前 warmup_steps 步标记为 mask=False（不参与 loss/metric）
```

### 5.3 实现方式

在 `build_stress_labels.py` 中增加 `--warmup_sec` 参数：
- `warmup_sec=0`（默认）：用全程分位数，所有时间步有效
- `warmup_sec=60`：用前 60s 估计阈值，前 60s mask=False

**训练时建议先用 warmup_sec=0 验证信号可用性，再用 warmup_sec=60 评估部署贴近度。**

---

## 6. 部署到真实游戏：在线 warm-up

### 6.1 流程

```
Phase 1: Warm-up (前 30-60s)
    ├── 只收集 KM/Video/Telemetry 特征
    ├── 计算输入特征的 running mean/std（用于在线 z-score）
    ├── 模型正常推理但不触发游戏调参
    └── 积累 arousal 预测值分布（用于阈值估计）

Phase 2: Active (30s 后)
    ├── 模型持续输出 P(LOW/MID/HIGH) 和 P(DOWN/FLAT/UP)
    ├── EMA 平滑输出概率
    ├── Hysteresis 决策（见下）
    └── 触发游戏参数调整
```

### 6.2 Hysteresis（滞回）决策规则

避免在类别边界频繁跳动：

```
# 状态升级（进入 HIGH）
if P(HIGH) > 0.7 连续 >= 3 个窗口:
    state = HIGH

# 状态降级（退出 HIGH）
if P(HIGH) < 0.4 连续 >= 3 个窗口:
    state = MID  （或根据 P(LOW) 判断）

# 趋势同理
if P(UP) > 0.6 连续 >= 2 个窗口:
    trend = UP
```

### 6.3 在线基线更新（可选）

用滑动窗口（最近 2-5 分钟）的特征统计量更新 z-score 的 mean/std，适应玩家状态漂移（疲劳、进入手感等）。

---

## 7. 实施顺序

### Phase 1：验证信号可用性（最高优先）

1. `scripts/build_stress_labels.py` — 生成 level + trend 标签
2. `src/losses/ce_seq_masked.py` — masked CE loss
3. `src/metrics/macro_f1.py` + `balanced_acc.py` — 分类指标
4. `src/models/heads/classification_seq.py` — 分类 head
5. `src/core/runner.py` — 适配分类任务
6. `src/data/datamodules/amucs_seq.py` — 支持 long dtype
7. 实验 config + 运行：
   - 先 KM only → 检查 macro-F1 > 0.33
   - 加 Telem → 检查增益
   - 加 Video → 检查增益

### Phase 2：趋势任务

8. 用 `build_stress_labels.py --task trend` 生成趋势标签
9. 同样的消融实验

### Phase 3：部署贴近度

10. `--warmup_sec=60` 重新生成标签
11. 评估 warm-up 对指标的影响

### Phase 4：后处理集成

12. 推理脚本中加入 EMA + hysteresis

---

## 8. 文件变更总表

| 操作 | 路径 | 说明 |
|------|------|------|
| **新增** | `scripts/build_stress_labels.py` | 标签构建 |
| **新增** | `src/models/heads/classification_seq.py` | 分类 head |
| **新增** | `src/losses/ce_seq_masked.py` | masked CE loss |
| **新增** | `src/metrics/macro_f1.py` | macro-F1 |
| **新增** | `src/metrics/balanced_acc.py` | balanced accuracy |
| **新增** | `configs/experiments/stress_level_*.yaml` | 实验配置 |
| **新增** | `configs/experiments/stress_trend_*.yaml` | 实验配置 |
| **修改** | `src/data/datamodules/amucs_seq.py` | 支持 label_dtype: long |
| **修改** | `src/core/runner.py` | 分类任务适配 |
| **修改** | `src/models/heads/__init__.py` | 注册新 head |
| **修改** | `src/losses/__init__.py` | 注册新 loss |
| **修改** | `src/metrics/__init__.py` | 注册新 metrics |

---

## 9. 数据存储约定

所有原始数据和提取好的特征存放在 Google Drive 上，本地不保留副本：

| 数据 | 路径 |
|------|------|
| AMuCS 原始数据 | `G:/我的云端硬盘/AmuCS/.../researchdata/data/` |
| Video 特征 | `G:/我的云端硬盘/AmuCS_experiment/features/video/` |
| KM 特征 | `G:/我的云端硬盘/AmuCS_experiment/features/km/` |
| Telem 特征 | `G:/我的云端硬盘/AmuCS_experiment/features/telem/` |
| 对齐后特征 | `G:/我的云端硬盘/AmuCS_experiment/features/aligned/` |
| 标签文件 | `G:/我的云端硬盘/AmuCS_experiment/labels/` |

项目通过 `D:/ProjectExperiment` → `G:/我的云端硬盘/ProjectExperiment` 的 symlink 访问。
