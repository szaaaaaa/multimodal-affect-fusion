# 连续 Arousal 回归 + 趋势预测 多任务训练设计（LFT）

## 1. 目标

在现有 `ProjectExperiment` 框架中，新增一个混合类型多任务：

- 任务A：`arousal` 连续值时序回归（masked MSE）
- 任务B：`trend` 三分类时序预测（masked CE）

并保持以下约束：

- backbone/fusion 继续使用现有 LFT
- 不破坏现有单任务回归、单任务分类、现有 state+trend 多任务分类
- 保持现有 run_dir、日志、early stopping 主流程不变

## 2. 当前代码现状（可复用）

### 2.1 已有能力

- 多任务数据结构已支持：
  - `src/data/datamodules/amucs_seq_multitask.py`
  - batch 结构：`x`、`mod_mask`、`y(dict)`、`mask(dict)`
- 多任务 head 已支持“每任务独立分支”：
  - `src/models/heads/multitask_seq.py`
- 多任务 loss 已支持“按任务加权汇总”：
  - `src/losses/multitask_ce_seq_masked.py`
- Runner 已支持 `y` 为 dict 的多任务训练与按任务命名指标：
  - `src/core/runner.py`

### 2.2 当前缺口

当前多任务实现是“全任务分类”范式（state + trend）：

- head 默认输出分类 logits
- loss 仅支持多任务 masked CE
- DataModule 当前只有统一 `label_dtype`，不适合一个 float + 一个 long 的混合任务
- Runner 多任务评估当前使用统一 `eval.metrics` 给所有任务，不适合回归任务与分类任务混用指标

## 3. 推荐方案（最小侵入，兼容优先）

## 3.1 标签与任务命名

建议新多任务标签文件命名：

- `labels/arousal_reg_trend_seq.json`

建议 schema：

```json
{
  "<stem>": {
    "arousal": {"values": [...], "mask": [...]},
    "trend": {"values": [...], "mask": [...]} 
  }
}
```

说明：

- `arousal.values` 为连续值（建议直接用 `arousal_seq_z_perparticipant.json` 的 values）
- `trend.values` 为 0/1/2（沿用 `arousal_3trend_seq.json`）
- 两任务长度必须一致；mask 规则沿用现有逻辑

实现上可选两种方式：

1. 扩展现有 `scripts/merge_multitask_labels.py`，支持任务名参数（推荐）
2. 新增专用脚本 `scripts/merge_multitask_arousal_trend_labels.py`

## 3.2 DataModule 扩展（关键）

在 `amucs_seq_multitask.py` 增加“按任务 dtype”支持：

- 新增配置：
  - `data.task_label_dtypes: {arousal: float, trend: long}`
- 数据读取时按任务转换 dtype，而不是全局一个 `label_dtype`

目标输出不变：

- `batch["y"] = {"arousal": Tensor[B,T], "trend": Tensor[B,T]}`
- `batch["mask"] = {"arousal": Bool[B,T], "trend": Bool[B,T]}`

兼容性：

- 若未提供 `task_label_dtypes`，默认回退到当前 `label_dtype` 行为

## 3.3 Head 扩展

推荐新增 `multitask_mixed_seq`（而非改坏现有 `multitask_seq`）：

- `arousal` 分支输出 `[B,T,1]`
- `trend` 分支输出 `[B,T,3]`

建议配置驱动：

```yaml
model:
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
```

## 3.4 Loss 扩展

新增 `multitask_mixed_seq_loss`：

- `arousal`: masked MSE
- `trend`: masked CE
- 总损失：
  - `loss = w_arousal * loss_arousal + w_trend * loss_trend`

建议配置：

```yaml
train:
  loss: multitask_mixed_seq_loss
  loss_cfg:
    task_weights:
      arousal: 1.0
      trend: 1.0
    task_types:
      arousal: regression
      trend: classification
```

数值稳定性：

- 若某任务 batch 内 mask 全 False，则该任务 loss 返回 0，避免 NaN

## 3.5 评估指标扩展（Runner）

这是混合多任务的核心改造点。

当前问题：`eval.metrics` 是全局列表，会把分类指标也算到回归任务上。

建议新增配置：

```yaml
eval:
  task_metrics:
    arousal: [ccc, rmse]
    trend: [macro_f1, balanced_acc]
```

Runner 在多任务分支按任务读取各自 metrics 计算，输出：

- `val_ccc_arousal`, `val_rmse_arousal`
- `val_macro_f1_trend`, `val_balanced_acc_trend`

为 early stopping 增加一个聚合监控指标（建议）：

- `val_score_mixed = 0.5 * val_ccc_arousal + 0.5 * val_macro_f1_trend`
- 然后 `train.early_stopping.metric=val_score_mixed, mode=max`

这样避免用单一任务指标主导 early stopping。

## 3.6 配置文件建议

先优先支持你现在最常用的两种模态：

- `configs/amucs_seq_lft_video_multitask_arousal_trend.yaml`
- `configs/amucs_seq_lft_video_km_multitask_arousal_trend.yaml`

后续再批量扩展到 7 组合。

建议配置骨架：

```yaml
_base_: ./experiments/video_km_aligned_seq.yaml

task_type: multitask

data:
  name: amucs_seq_multitask
  modalities: [video, km]
  labels_seq_path: labels/arousal_reg_trend_seq.json
  task_names: [arousal, trend]
  task_label_dtypes:
    arousal: float
    trend: long

model:
  head:
    name: multitask_mixed_seq
    task_heads:
      arousal: {type: regression, hidden_dim: 128, out_dim: 1, dropout: 0.1}
      trend: {type: classification, hidden_dim: 128, num_classes: 3, dropout: 0.1}

train:
  loss: multitask_mixed_seq_loss
  loss_cfg:
    task_weights: {arousal: 1.0, trend: 1.0}
    task_types: {arousal: regression, trend: classification}
  early_stopping:
    metric: val_score_mixed
    mode: max
    patience: 10

eval:
  task_metrics:
    arousal: [ccc, rmse]
    trend: [macro_f1, balanced_acc]
```

## 4. 实施顺序（建议）

1. 标签层：先把 `arousal + trend` 合并标签做好并加校验
2. 数据层：支持 task 级 dtype
3. 模型层：新增 mixed head
4. 损失层：新增 mixed multitask loss
5. Runner：支持 task_metrics 与 `val_score_mixed`
6. 配置层：先落地 `single_video` 与 `dual_video_km`
7. 最后再扩展到 7 模态组合

## 5. Sanity Check（必须）

至少做以下检查：

1. Forward 形状检查
- `arousal` 输出 `[B,T,1]`
- `trend` 输出 `[B,T,3]`

2. mask 全 False 稳定性
- 两任务任一任务 mask 全 False，loss 不 NaN

3. 指标路由检查
- `arousal` 只产出 `ccc/rmse`
- `trend` 只产出 `macro_f1/balanced_acc`

4. 早停指标检查
- `val_score_mixed` 能随 epoch 正常记录并驱动 early stopping

## 6. 风险与注意点

- 任务尺度不一致：MSE 与 CE 数值量级可能差别较大，`task_weights` 需要调参
- 类别不均衡：trend 可能类分布偏斜，可保留 CE class weights 扩展位
- Drive I/O 抖动：建议 checkpoint 先写本地盘再异步拷贝，降低中断概率

## 7. 结论

基于现有工程，最稳妥路径是：

- 保留当前 `amucs_seq_multitask` 数据组织和 LFT 主干
- 新增“混合任务 head + 混合任务 loss + 任务级指标路由”
- 先在 `single_video` 和 `dual_video_km` 跑通，再扩展到全部 7 组合

这样改动面可控，且不会破坏现有回归/分类实验。
