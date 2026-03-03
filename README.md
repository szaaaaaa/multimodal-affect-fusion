# ProjectExperiment

基于 Late Fusion Transformer (LFT) 的可扩展多模态框架，用于游戏过程中的玩家情绪/压力预测。

支持两类任务：
- **回归任务**：连续 arousal/valence 预测（CCC / RMSE）
- **分类任务**：离散压力状态预测（macro-F1 / balanced accuracy），用于闭环游戏压力调控

---

## 1. 任务说明

### 1.1 回归任务：连续 Arousal 预测

输入多模态特征序列，逐时间步预测连续 arousal 值。

| 项目 | 说明 |
|------|------|
| 输入 | Video / KM / Telemetry 特征序列 `[B, T, D]` |
| 输出 | 连续 arousal 值 `[B, T, 1]` |
| 标签 | `labels_arousal_seq.json`（从 ranktrace 插值到 5Hz） |
| Loss | `mse_seq_masked` |
| 指标 | CCC, RMSE |

### 1.2 分类任务：压力状态/趋势三分类

从连续 arousal 序列离散化为三分类标签，用于实时游戏参数调整。

#### Task A: stress_level（当前压力档位）

| 类别 | 含义 | 构建规则 |
|------|------|----------|
| 0 — LOW | 低压力 | `y_smooth <= q20` |
| 1 — MID | 中等压力 | `q20 < y_smooth < q80` |
| 2 — HIGH | 高压力 | `y_smooth >= q80` |

阈值 q20/q80 为 per-session 分位数（每个玩家用自己的基线）。

#### Task B: stress_trend（未来压力趋势）

| 类别 | 含义 | 构建规则 |
|------|------|----------|
| 0 — DOWN | 压力下降 | `delta < -tau` |
| 1 — FLAT | 压力平稳 | `-tau <= delta <= tau` |
| 2 — UP | 压力上升 | `delta > tau` |

其中 `delta = mean(y_future) - mean(y_now)`，`tau = 0.25 * std(y_smooth)`。

两个任务共享同一条 arousal 序列，仅标签生成逻辑不同。详见 `docs/stress_classification_task.md`。

---

## 2. 模型架构

整体遵循 **Encoder → Fusion → Head** 三阶段流水线，所有组件通过注册表插件化。

```
                         Batch from DataModule
                 {x: {mod: [B,T,D]}, mask: {mod: [B,T]}, y}
                                   │
                 ┌─────────────────┼─────────────────┐
                 ▼                 ▼                 ▼
           ┌──────────┐     ┌──────────┐     ┌──────────┐
           │  Video    │     │   KM     │     │  Telem   │
           │  Encoder  │     │ Encoder  │     │ Encoder  │
           └────┬─────┘     └────┬─────┘     └────┬─────┘
                │                │                │
                │  EncoderOut    │  EncoderOut    │  EncoderOut
                │  {tokens,      │  {tokens,      │  {tokens,
                │   pooled,      │   pooled,      │   pooled,
                │   mask}        │   mask}        │   mask}
                └────────┬───────┴───────┬────────┘
                         ▼               ▼
                  ┌─────────────────────────┐
                  │     Fusion (LFT)        │
                  │  Late Fusion Transformer │
                  │  - Modality embedding    │
                  │  - Positional encoding   │
                  │  - Self-attention layers │
                  └───────────┬─────────────┘
                              │  FusionOut
                              │  {tokens: [B,T,D], pooled: [B,D]}
                              ▼
                  ┌─────────────────────────┐
                  │         Head            │
                  │  regression_seq →       │
                  │    [B, T, 1] 连续值     │
                  │  classification_seq →   │
                  │    [B, T, 3] logits     │
                  └───────────┬─────────────┘
                              ▼
                  ┌─────────────────────────┐
                  │         Loss            │
                  │  mse_seq_masked (回归)  │
                  │  ce_seq_masked  (分类)  │
                  └─────────────────────────┘
```

### 2.1 冻结接口

`src/core/types.py` 中定义了四个抽象基类，是所有组件的稳定契约：

| 接口 | 输入 | 输出 |
|------|------|------|
| `BaseEncoder` | `x [B,T,D_in]`, `mask [B,T]` | `EncoderOut {tokens, pooled, mask}` |
| `BaseFusion` | `z_dict {mod: EncoderOut}`, `mask_dict` | `FusionOut {tokens, pooled}` |
| `BaseHead` | `FusionOut` | `Tensor [B, T, out_dim]` |
| `BaseDataModule` | — | `DataLoader` yielding `Batch` |

### 2.2 可用组件

**Encoders（按模态注册）：**

| 模态 | 名称 | 说明 |
|------|------|------|
| video | `resnet2d` | ResNet-50 帧特征投影 + 时序 mean pooling |
| video | `emotieff` | EmotiEffNet 情绪特征编码器 |
| km | `stat` | 线性投影编码器 |
| km | `cnn1d` | 1D CNN 序列编码器 |
| telem | `stat_pool` | 统计池化编码器 |

**Fusions：**

| 名称 | 说明 |
|------|------|
| `lft` | Late Fusion Transformer（主力），支持模态嵌入 + 位置编码 + 多头自注意力 |
| `single` | 单模态直通 / 多模态拼接均值 |
| `aligned_mean` | 对齐后均值融合 |

**Heads：**

| 名称 | 输出形状 | 用途 |
|------|----------|------|
| `regression` | `[B, out_dim]` | 单步回归 |
| `regression_seq` | `[B, T, 1]` | 逐步回归 |
| `classification_seq` | `[B, T, num_classes]` | 逐步分类 |
| `va_split` | `[B, 2]` | Valence/Arousal 分路 |

**Losses：**

| 名称 | 说明 |
|------|------|
| `ccc` | 一致性相关系数损失 |
| `mse` | 均方误差 |
| `mse_seq_masked` | 带掩码的序列 MSE |
| `ce_seq_masked` | 带掩码的序列交叉熵（支持 class_weights） |

**Metrics：**

| 名称 | 说明 |
|------|------|
| `ccc` | 一致性相关系数 |
| `rmse` | 均方根误差 |
| `macro_f1` | 宏平均 F1（分类） |
| `balanced_acc` | 平衡准确率（分类） |

### 2.3 注册表系统

所有组件通过装饰器自注册，Runner 根据 YAML 配置中的字符串名查找并实例化：

```python
# 注册
@HEADS.register("classification_seq")
class ClassificationSeqHead(BaseHead): ...

# 使用（由 Runner 自动完成）
head = HEADS.build("classification_seq", head_cfg)
```

全局注册表：`FUSIONS`, `HEADS`, `LOSSES`, `METRICS`, `DATAMODULES`
按模态注册表：`get_encoder_registry("video")`, `get_encoder_registry("km")`, ...

---

## 3. 项目结构

```
ProjectExperiment/
├── src/
│   ├── core/                           # 稳定基础设施（极少修改）
│   │   ├── types.py                    #   冻结接口 + TypedDict
│   │   ├── registry.py                 #   插件注册表
│   │   ├── runner.py                   #   训练编排器
│   │   ├── config.py                   #   YAML 配置加载（支持继承）
│   │   ├── logging.py                  #   实验日志管理
│   │   └── seed.py                     #   随机种子
│   ├── data/
│   │   └── datamodules/
│   │       └── amucs_seq.py            #   序列数据集（支持 float/long 标签）
│   ├── models/
│   │   ├── encoders/
│   │   │   ├── video/                  #   resnet2d, emotieff
│   │   │   ├── km/                     #   stat, cnn1d
│   │   │   └── telem/                  #   stat_pool
│   │   ├── fusions/                    #   lft, single, aligned_mean, ...
│   │   └── heads/                      #   regression_seq, classification_seq, ...
│   ├── losses/                         #   mse_seq_masked, ce_seq_masked, ccc, ...
│   └── metrics/                        #   ccc, rmse, macro_f1, balanced_acc
├── configs/
│   ├── base.yaml                       # 全局默认配置
│   └── experiments/                    # 实验配置（继承 base.yaml）
│       ├── video_km_aligned_seq.yaml   #   Video+KM 序列回归
│       ├── video_km_telem_aligned_seq.yaml  # Video+KM+Telem 序列回归
│       ├── stress_level_km_telem.yaml  #   KM+Telem 压力分类
│       └── ...
├── scripts/
│   ├── train.py                        # 训练入口
│   ├── build_stress_labels.py          # 构建压力分类标签
│   ├── build_arousal_sequence_labels.py # 构建连续 arousal 标签
│   ├── extract_video_features.py       # 提取视频特征
│   ├── extract_km_features.py          # 提取 KM 特征
│   ├── extract_game_telem_features.py  # 提取遥测特征
│   ├── build_multimodal_split.py       # 构建多模态数据划分
│   └── summarize.py                    # 结果汇总
├── tests/                              # 形状契约测试
├── docs/                               # 技术文档
│   └── stress_classification_task.md   #   压力分类任务完整规范
├── data/                               # 本地数据目录（仅保留划分文件）
│   └── splits/                         #   train/val/test 划分
└── runs/                               # 训练输出（gitignored）
```

---

## 4. 数据格式

### 4.1 特征文件

每个 session 对应一个 `.pt` 文件，命名为 `{stem}.pt`（如 `S001_P1.pt`）。

```python
# 文件内容
{
    "features": Tensor[T, D],     # T 个时间步，D 维特征
    "mask": Tensor[T],            # bool，True = 有效
    # 可选字段：timestamps, fps, meta, ...
}
```

特征目录结构：

```
G:/我的云端硬盘/AmuCS_experiment/features/aligned/amucs_trial/
├── video/          # D=2048 (ResNet-50)
├── km/             # D=25 (统计特征)
└── telem/          # D=109 (遥测特征)
```

### 4.2 标签文件

所有标签文件共享统一的 JSON 格式：

```json
{
  "S001_P1": {
    "values": [0.45, 0.52, ...],
    "mask": [true, true, ...]
  }
}
```

- **回归标签** (`labels_arousal_seq.json`)：`values` 为连续浮点数
- **分类标签** (`labels_stress_level.json`, `labels_stress_trend.json`)：`values` 为整数 (0/1/2)

### 4.3 数据划分

```json
// data/splits/multimodal_split.json
{
  "train": ["S001_P1", "S001_P2", ...],
  "val": ["S003_P1", ...],
  "test": ["S005_P1", ...]
}
```

---

## 5. 配置系统

YAML 配置通过 `_base_` 支持层级继承。实验配置只需覆盖差异部分。

### 5.1 回归任务配置示例

```yaml
# configs/experiments/video_km_aligned_seq.yaml
_base_: ../base.yaml

data:
  name: amucs_seq
  modalities: [video, km]
  data_root: G:/我的云端硬盘/AmuCS_experiment/features/aligned/amucs_trial
  labels_seq_path: G:/我的云端硬盘/AmuCS_experiment/labels/labels_arousal_seq.json
  seq_len: 600
  train_stride: 300

model:
  head:
    name: regression_seq
    out_dim: 1

train:
  loss: mse_seq_masked
  early_stopping:
    metric: val_rmse
    mode: min

eval:
  metrics: [ccc, rmse]
```

### 5.2 分类任务配置示例

```yaml
# configs/experiments/stress_level_km_telem.yaml
_base_: ./video_km_telem_aligned_seq.yaml

task_type: classification          # 启用分类分支

data:
  modalities: [km, telem]
  labels_seq_path: G:/我的云端硬盘/AmuCS_experiment/labels/labels_stress_level.json
  label_dtype: long                # LongTensor 分类标签

model:
  head:
    name: classification_seq
    num_classes: 3
    hidden_dim: 128
    dropout: 0.1

train:
  loss: ce_seq_masked
  # loss_cfg:                      # 可选类别权重
  #   class_weights: [1.5, 1.0, 1.5]
  early_stopping:
    metric: val_macro_f1
    mode: max

eval:
  metrics: [macro_f1, balanced_acc]
```

### 5.3 关键配置字段

| 字段 | 说明 | 默认值 |
|------|------|--------|
| `task_type` | `regression` 或 `classification` | `regression` |
| `data.label_dtype` | `float` 或 `long` | `float` |
| `data.seq_len` | 序列窗口长度（时间步） | 300 |
| `data.train_stride` | 训练滑动窗口步长 | `None`（取中间窗口） |
| `model.d_model` | 共享模型维度 | 512 |
| `train.loss_cfg` | 传给 loss 构造函数的额外配置 | `None` |
| `train.modality_dropout` | 训练时随机丢弃模态的概率 | 0.0 |

---

## 6. 使用方法

### 6.1 环境准备

Python 3.10+

```bash
pip install torch torchvision pyyaml pandas numpy scikit-learn tqdm pytest
```

### 6.2 特征提取

**视频特征：**

```bash
python scripts/extract_video_features.py \
  --video_dir "/path/to/gameplay_videos" \
  --output_dir "data/features/amucs/video" \
  --session_mode subdirs \
  --name_mode amucs \
  --device cuda
```

**KM 特征：**

```bash
python scripts/extract_km_features.py \
  --root_dir "/path/to/km_raw" \
  --output_dir "data/features/amucs/km" \
  --encoder stat
```

**遥测特征：**

```bash
python scripts/extract_game_telem_features.py \
  --root_dir "/path/to/telem_raw" \
  --output_dir "data/features/amucs/telem"
```

### 6.3 构建标签

**连续 arousal 标签**（从 ranktrace 插值到特征时间网格）：

```bash
python scripts/build_arousal_sequence_labels.py \
  --ranktrace_dir "/path/to/ranktrace" \
  --output_path G:/我的云端硬盘/AmuCS_experiment/labels/labels_arousal_seq.json
```

**压力分类标签**（从连续 arousal 离散化）：

```bash
# 同时生成 stress_level 和 stress_trend 标签
python scripts/build_stress_labels.py \
  --arousal_path G:/我的云端硬盘/AmuCS_experiment/labels/labels_arousal_seq.json \
  --output_dir data \
  --task both

# 仅生成 stress_level
python scripts/build_stress_labels.py --task level

# 仅生成 stress_trend
python scripts/build_stress_labels.py --task trend

# 自定义参数
python scripts/build_stress_labels.py \
  --alpha 0.05 \
  --W 25 --H 25 \
  --tau_scale 0.25 \
  --warmup_sec 60       # 前 60s 用于估计阈值，mask=False
```

输出文件：
- `G:/我的云端硬盘/AmuCS_experiment/labels/labels_stress_level.json` — 压力状态标签
- `G:/我的云端硬盘/AmuCS_experiment/labels/labels_stress_trend.json` — 压力趋势标签
- `G:/我的云端硬盘/AmuCS_experiment/labels/labels_stress_meta.json` — 每个 session 的阈值统计和类别分布

### 6.4 构建数据划分

```bash
python scripts/build_multimodal_split.py \
  --video_dir data/features/amucs/video \
  --km_dir data/features/amucs/km \
  --labels_path G:/我的云端硬盘/AmuCS_experiment/labels/labels_arousal.json \
  --output_path data/splits/multimodal_split.json
```

### 6.5 训练

**回归任务：**

```bash
# Video + KM 序列回归
python scripts/train.py --config configs/experiments/video_km_aligned_seq.yaml

# Video + KM + Telem 序列回归
python scripts/train.py --config configs/experiments/video_km_telem_aligned_seq.yaml
```

**分类任务：**

```bash
# KM + Telem 压力状态分类
python scripts/train.py --config configs/experiments/stress_level_km_telem.yaml
```

**CLI 覆盖参数：**

```bash
python scripts/train.py \
  --config configs/experiments/stress_level_km_telem.yaml \
  --override train.epochs=100 train.seed=0 model.d_model=256 \
  data.data_root=/path/to/features
```

### 6.6 恢复训练

```bash
# 从 run 目录恢复
python scripts/train.py \
  --config configs/experiments/stress_level_km_telem.yaml \
  --resume runs/<run_dir_name>

# 从检查点恢复
python scripts/train.py \
  --config configs/experiments/stress_level_km_telem.yaml \
  --resume runs/<run_dir_name>/ckpt_last.pt
```

### 6.7 结果汇总

```bash
python scripts/summarize.py --runs_dir runs --output leaderboard.csv
```

---

## 7. 训练输出

每次训练在 `runs/` 下创建如下目录：

```
runs/{timestamp}__{dataset}__{fusion}__{modalities}__seed{seed}/
├── config.yaml          # 完整合并后的配置
├── seed.txt             # 随机种子
├── git_commit.txt       # Git commit hash
├── ckpt_best.pt         # 最佳模型权重
├── ckpt_last.pt         # 最新模型权重
└── metrics.json         # 最终指标
```

`metrics.json` 内容示例：

```json
{
  "best_val_metric": 0.452,
  "best_epoch": 23,
  "total_epochs": 50,
  "early_stopped": true,
  "val_macro_f1": 0.452,
  "val_balanced_acc": 0.48,
  "train_loss": 0.87
}
```

---

## 8. 测试

```bash
# 运行全部测试
pytest tests/ -v

# 运行形状契约测试
pytest tests/test_shapes.py
```

测试覆盖：
- Encoder/Fusion/Head 输出形状验证
- 注册表完整性
- 任意模态子集下的端到端前向传播

---

## 9. 扩展指南

### 添加新 Encoder

1. 创建 `src/models/encoders/{modality}/{name}.py`
2. 实现 `BaseEncoder`，用 `@get_encoder_registry("modality").register("name")` 注册
3. 在配置中指定 `model.encoders.{modality}.name: name`

### 添加新 Head

1. 创建 `src/models/heads/{name}.py`
2. 实现 `BaseHead`，用 `@HEADS.register("name")` 注册
3. 在 `src/models/heads/__init__.py` 添加 import
4. 在配置中指定 `model.head.name: name`

### 添加新 Loss / Metric

1. 创建 `src/losses/{name}.py` 或 `src/metrics/{name}.py`
2. 用 `@LOSSES.register("name")` 或 `@METRICS.register("name")` 注册
3. 在对应 `__init__.py` 添加 import
4. 在配置中指定 `train.loss: name` 或 `eval.metrics: [name]`

所有新组件无需修改 Runner 或其他核心文件。

---

## 10. 数据存储约定

所有原始数据和特征存放在 Google Drive 上：

| 数据 | 路径 |
|------|------|
| AMuCS 原始数据 | `G:/我的云端硬盘/AmuCS/.../researchdata/data/` |
| Video 特征 | `G:/我的云端硬盘/AmuCS_experiment/features/video/` |
| KM 特征 | `G:/我的云端硬盘/AmuCS_experiment/features/km/` |
| Telem 特征 | `G:/我的云端硬盘/AmuCS_experiment/features/telem/` |
| 对齐后特征 | `G:/我的云端硬盘/AmuCS_experiment/features/aligned/` |
| 标签文件 | `G:/我的云端硬盘/AmuCS_experiment/labels/` |

项目通过 symlink 访问：`D:/ProjectExperiment` → `G:/我的云端硬盘/ProjectExperiment`
