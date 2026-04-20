# Baseline 实验指南

## 任务定义

**唯一任务**：Arousal State（低/中/高）+ Trend（降/稳/升）多任务序列分类

- 标签：`arousal_state_trend_seq.json`（per-participant z-score → 分位数离散化）
- 评估指标：Macro-F1, Balanced Accuracy（early stopping 基于 `val_f1_mean`）
- 训练集类别均衡（各 33.3%），val/test 有自然偏移但不严重

## 6 个 Baseline 模型

所有模型共享相同的编码器、预测头、损失函数、训练配置，**唯一差异是融合方法**。

### 融合方法对比

| 模型 | 融合层级 | 核心机制 | 实现文件 |
|------|---------|---------|---------|
| **EFT** | 早期融合 | 各模态 token 沿时间轴拼接 → 共享 Transformer（跨模态 self-attention 从第 1 层开始） | `src/models/fusions/eft.py` |
| **MFT** | 中期融合 | 各模态独立私有 Transformer → cross-attention 跨模态交互 | `src/models/fusions/mft.py` |
| **LFT** | 晚期融合 | 各模态独立 Transformer（完整处理）→ 注意力加权融合 | `src/models/fusions/lft.py` |
| **Late Fusion** | 决策级 | 各模态独立 Transformer（零跨模态交互）→ 平均表示 | `src/models/fusions/late.py` |
| **CMA** | 表示级 | Anchor(video) 为基准的跨模态注意力 → 共享 Transformer | `src/models/fusions/cma.py` |
| **Gated** | 表示级 | 每时间步 sigmoid 门控决定模态权重 → 可选 Transformer 精炼 | `src/models/fusions/gated.py` |

### 架构图

```
EFT (Early Fusion Transformer):
  video → encoder ─┐
  km    → encoder ─┼─ concat(time_dim) + mod_emb → shared Transformer → head
  telem → encoder ─┘

MFT (Mid Fusion Transformer):
  video → encoder → private Transformer ─┐
  km    → encoder → private Transformer ─┼─ cross-attention → concat → head
  telem → encoder → private Transformer ─┘

LFT (Late Fusion Transformer):
  video → encoder → independent Transformer → pool ─┐
  km    → encoder → independent Transformer → pool ─┼─ attn-weighted fusion → head
  telem → encoder → independent Transformer → pool ─┘

Late Fusion:
  video → encoder → Transformer_video ─┐
  km    → encoder → Transformer_km    ─┼─ average → head
  telem → encoder → Transformer_telem ─┘

CMA:
  video(anchor) → encoder ──────────────────────┐
  km    → encoder → cross_attn(→video) ─────────┼─ concat → Transformer → head
  telem → encoder → cross_attn(→video) ─────────┘

Gated:
  video → encoder ─┐
  km    → encoder ─┼─ gate(concat) × tokens → weighted_sum → [Transformer] → head
  telem → encoder ─┘
```

## 共享配置

### 编码器

| 模态 | Encoder | 输入维度 | 输出维度 |
|------|---------|---------|---------|
| video | `resnet2d`（线性投影 CLIP 特征） | 768 | 512 |
| km | `stat`（线性投影统计特征） | 25 | 512 |
| telem | `stat_pool`（线性投影遥测特征） | 109 | 512 |

### 训练配置

| 参数 | 值 |
|------|---|
| d_model | 512 |
| optimizer | AdamW (lr=5e-5, weight_decay=0.01) |
| scheduler | cosine (warmup_epochs=3) |
| grad_clip | 1.0 |
| batch_size | 256 |
| epochs | 40 |
| early_stopping | patience=5, metric=val_f1_mean, mode=max |
| loss | multitask_ce_seq_masked (label_smoothing=0.1) |
| task_weights | state=1.0, trend=1.0 |
| seq_len | 600 (120秒 @ 5Hz) |
| stride | 300 (train/val/test) |
| AMP | bfloat16 |
| compile | true |

### 数据划分

| 模式 | 说明 |
|------|------|
| `cross_subject` | 按 session 划分 train/val/test（不同参与者） |
| `within_subject` | 同一 session 内按时间 60%/20%/20% 切分 |

### 模态组合（7 种）

| 类型 | 组合 |
|------|------|
| 单模态 | video, km, telem |
| 双模态 | video+km, video+telem, km+telem |
| 三模态 | video+km+telem |

## 运行命令

### 全量运行（7 模型 × 7 模态 × 3 seeds × 2 splits）

```bash
# Colab
python scripts/run_experiment.py \
  --sweep configs/sweeps/full_ablation.yaml \
  --data_root /content/drive/MyDrive/AmuCS_experiment/features/aligned \
  --labels_root /content/drive/MyDrive/AmuCS_experiment/labels \
  --splits_root /content/drive/MyDrive/AmuCS_experiment/splits

# 本地
python scripts/run_experiment.py \
  --sweep configs/sweeps/full_ablation.yaml \
  --data_root G:/我的云端硬盘/AmuCS_experiment/features/aligned \
  --labels_root G:/我的云端硬盘/AmuCS_experiment/labels \
  --splits_root G:/我的云端硬盘/AmuCS_experiment/splits
```

### 运行单个模型

```bash
# 只跑 EFT (Early Fusion Transformer)
python scripts/run_experiment.py \
  --sweep configs/sweeps/full_ablation.yaml \
  --tasks eft_state_trend \
  --data_root ... --labels_root ... --splits_root ...

# 只跑 MFT (Mid Fusion Transformer)
--tasks mft_state_trend

# 只跑 LFT (Late Fusion Transformer)
--tasks lft_state_trend

# 只跑 Late Fusion
--tasks late_state_trend

# 只跑 CMA
--tasks cma_state_trend

# 只跑 Gated
--tasks gated_state_trend
```

### 预览运行计划（不实际运行）

```bash
python scripts/run_experiment.py \
  --sweep configs/sweeps/full_ablation.yaml \
  --dry_run \
  --data_root ... --labels_root ... --splits_root ...
```

### 自定义输出目录

```bash
--runs_root /path/to/output/runs
```

## 输出结构

```
runs/
├── cross_subject/
│   ├── eft_state_trend_3seed/
│   │   ├── single_video/
│   │   │   ├── 2026-...__amucs_seq_multitask__eft__video__seed0/
│   │   │   │   ├── config.yaml
│   │   │   │   ├── metrics.json
│   │   │   │   ├── ckpt_best.pt
│   │   │   │   └── ckpt_last.pt
│   │   │   ├── ...seed1/
│   │   │   └── ...seed2/
│   │   ├── dual_video_km/
│   │   ├── ...
│   │   ├── results.tsv          # 所有 seed 的详细指标
│   │   └── results_summary.csv  # mean ± std 汇总
│   ├── mft_state_trend_3seed/
│   ├── lft_state_trend_3seed/
│   ├── late_state_trend_3seed/
│   ├── cma_state_trend_3seed/
│   └── gated_state_trend_3seed/
└── within_subject/
    └── (同上结构)
```

## 关键指标

每个 run 的 `metrics.json` 包含：

| 指标 | 说明 |
|------|------|
| `val_f1_mean` | val 集 state 和 trend 的 Macro-F1 均值（early stopping 依据） |
| `val_macro_f1_state` | val 集 state 分类 Macro-F1 |
| `val_macro_f1_trend` | val 集 trend 分类 Macro-F1 |
| `val_balanced_acc_state` | val 集 state Balanced Accuracy |
| `val_balanced_acc_trend` | val 集 trend Balanced Accuracy |
| `test_*` | 对应 test 集指标 |

## 恢复中断的实验

`run_experiment.py` 自动检测已完成的 run（存在 `metrics.json`），跳过不重复跑。直接重新执行同一命令即可恢复。

## Pipeline 改进记录（相对于原始 train.ipynb）

| 改进 | 文件 | 说明 |
|------|------|------|
| LR Scheduler + Warmup | `src/core/runner.py` | cosine decay, 3 epoch linear warmup |
| Gradient Clipping | `src/core/runner.py` | max_norm=1.0 |
| Label Smoothing | `src/losses/multitask_ce_seq_masked.py` | CE loss label_smoothing=0.1 |
| 视频特征升级 | 数据层 | ResNet-50 (2048-dim) → CLIP ViT-L/14 (768-dim) |

## 后续扩展

添加新的融合模型只需：

1. 在 `src/models/fusions/` 下新建文件，实现 `BaseFusion` 接口
2. 用 `@FUSIONS.register("name")` 注册
3. 在 `__init__.py` 中 import
4. 在 `full_ablation.yaml` 中添加 `fusion_xxx` 模板和对应 task 定义
5. 运行 `--tasks new_task_name` 即可
