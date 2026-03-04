# ProjectExperiment（中文版）

这是一个面向 AMuCS 类游戏情绪建模的可扩展多模态时序学习框架，核心融合器为 **LFT（Late Fusion Transformer）**。

## 当前支持的任务

- 单任务时序回归（连续 arousal）
- 单任务时序分类（三分类）
- 多任务时序分类（`state + trend`）
- **混合多任务时序学习**：
  - `arousal` 连续回归
  - `trend` 三分类

主要实验统一使用：

- `video` -> `resnet2d`
- `km` -> `stat`
- `telem` -> `stat_pool`
- 融合器 -> `lft`

## 架构概览

训练主流程：

1. 各模态 Encoder
2. LFT 融合
3. 任务 Head
4. Mask 感知 Loss 与 Metrics

核心注册表：

- `DATAMODULES`
- `FUSIONS`
- `HEADS`
- `LOSSES`
- `METRICS`

训练入口：

- `scripts/train.py`

## 新增：Arousal 回归 + Trend 分类（混合多任务）

### 1) 标签合并脚本

```bash
python scripts/merge_arousal_reg_trend_labels.py \
  --arousal /path/to/arousal_seq_z_perparticipant.json \
  --trend /path/to/arousal_3trend_seq.json \
  --output /path/to/arousal_reg_trend_seq.json
```

输出格式：

```json
{
  "<stem>": {
    "arousal": {"values": [...], "mask": [...]},
    "trend": {"values": [...], "mask": [...]}
  }
}
```

### 2) 七种模态组合配置

- `configs/amucs_seq_lft_video_multitask_arousal_trend.yaml`
- `configs/amucs_seq_lft_km_multitask_arousal_trend.yaml`
- `configs/amucs_seq_lft_telem_multitask_arousal_trend.yaml`
- `configs/amucs_seq_lft_video_km_multitask_arousal_trend.yaml`
- `configs/amucs_seq_lft_video_telem_multitask_arousal_trend.yaml`
- `configs/amucs_seq_lft_km_telem_multitask_arousal_trend.yaml`
- `configs/amucs_seq_lft_video_km_telem_multitask_arousal_trend.yaml`

### 3) 新增/扩展模块

- 混合多任务 Head：
  - `src/models/heads/multitask_mixed_seq.py`
- 混合多任务 Loss：
  - `src/losses/multitask_mixed_seq_loss.py`
- 多任务 DataModule 扩展（支持任务级 dtype）：
  - `src/data/datamodules/amucs_seq_multitask.py`
- Runner 扩展（任务级指标路由 + 可选综合指标）：
  - `src/core/runner.py`

## 已有 State+Trend 多任务分类

原有 `state + trend` 多任务分类配置保持兼容，不受影响：

- `configs/amucs_seq_lft_*_multitask_state_trend.yaml`

## Notebook 流程

主 notebook：

- `train.ipynb`

关键单元：

- `Cell 26`：`state + trend` 多任务分类
- `Cell 27`：`arousal 回归 + trend 分类`（完整 7 组合 x 3 种子）
- `Cell 28`：回归任务 Lag Sweep 分析

## 训练命令示例

示例（混合多任务，video+km）：

```bash
python -u scripts/train.py \
  --config configs/amucs_seq_lft_video_km_multitask_arousal_trend.yaml \
  --override \
    data.data_root=/path/to/features/aligned \
    data.labels_seq_path=/path/to/arousal_reg_trend_seq.json \
    data.split_path=/path/to/session_tvt.json \
    train.seed=0
```

## 运行环境

建议：

- Python 3.10+
- PyTorch + torchvision
- pyyaml, numpy, pandas, scikit-learn, tqdm, pytest

## 输出目录结构

每次训练输出：

```text
runs/{timestamp}__{dataset}__{fusion}__{modalities}__seed{seed}/
  config.yaml
  seed.txt
  git_commit.txt
  ckpt_best.pt
  ckpt_last.pt
  metrics.json
```

## 说明

- `mask=False` 的时间点不参与 loss/metric。
- 混合多任务支持任务级指标以及可选综合指标 `val_score_mixed`（可用于 early stopping）。
- 本次新增不会破坏已有单任务和已有多任务实验实现。
