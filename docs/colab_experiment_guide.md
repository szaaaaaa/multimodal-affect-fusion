# Colab 自动化实验指南

> 不再使用 ipynb，统一通过 `run_experiment.py` + sweep YAML 驱动实验。

---

## 快速开始

### Colab 环境准备（每次新开session执行一次）

```bash
# Cell 1: 挂载 + 进入项目
from google.colab import drive
drive.mount('/content/drive')

%cd /content/drive/MyDrive/ProjectExperiment
!pip install pyyaml scipy -q
!nvidia-smi
```

### 数据路径（Colab 上）

```
--data_root   /content/drive/MyDrive/AmuCS_experiment/features/aligned
--labels_root /content/drive/MyDrive/AmuCS_experiment/labels
--splits_root /content/drive/MyDrive/AmuCS_experiment/splits
```

---

## 实验阶段

### Phase 0: 现有 Baseline（已完成）

```bash
python scripts/run_experiment.py \
    --sweep configs/sweeps/full_ablation.yaml \
    --data_root /content/drive/MyDrive/AmuCS_experiment/features/aligned \
    --labels_root /content/drive/MyDrive/AmuCS_experiment/labels \
    --splits_root /content/drive/MyDrive/AmuCS_experiment/splits \
    --runs_root runs/ablations
```

结果在 `runs/ablations/` 下，每个 task 有 `results_summary.csv`。

---

### Phase 1: 标签质量优化

**Step 1**: 生成平滑标签

```bash
python scripts/smooth_labels.py \
    --arousal_path /content/drive/MyDrive/AmuCS_experiment/labels/arousal_seq_z_perparticipant.json \
    --trend_path /content/drive/MyDrive/AmuCS_experiment/labels/arousal_3trend_seq.json \
    --sigmas 1 2 3 5 \
    --output_dir /content/drive/MyDrive/AmuCS_experiment/labels/
```

**Step 2**: 跑对比实验

```bash
python scripts/run_experiment.py \
    --sweep configs/sweeps/improvement_phase1_labels.yaml \
    --data_root /content/drive/MyDrive/AmuCS_experiment/features/aligned \
    --labels_root /content/drive/MyDrive/AmuCS_experiment/labels \
    --splits_root /content/drive/MyDrive/AmuCS_experiment/splits \
    --runs_root runs/improvement_phase1
```

**Step 3**: 看结果，选最佳 sigma

```bash
# 查看各 task 的 results_summary.csv
find runs/improvement_phase1 -name "results_summary.csv" -exec echo "---" \; -exec head -5 {} \;
```

**预期结果**：5个标签变体 × 1个模态组合(VKT) × 3 seeds = 15 runs

---

### Phase 2: 时序建模 + 融合改进

**前置**：
1. Phase 1 完成，确定最佳 sigma
2. 实现新组件（见下方"待实现代码"）
3. 更新 `improvement_phase2_temporal.yaml` 中的 `labels_seq_path`

```bash
python scripts/run_experiment.py \
    --sweep configs/sweeps/improvement_phase2_temporal.yaml \
    --data_root /content/drive/MyDrive/AmuCS_experiment/features/aligned \
    --labels_root /content/drive/MyDrive/AmuCS_experiment/labels \
    --splits_root /content/drive/MyDrive/AmuCS_experiment/splits \
    --runs_root runs/improvement_phase2
```

**预期结果**：6个变体 × 4个模态组合 × 3 seeds = 72 runs

---

## 常用操作

### Dry Run（只看计划不跑）

```bash
python scripts/run_experiment.py \
    --sweep configs/sweeps/improvement_phase1_labels.yaml \
    --data_root ... --labels_root ... --splits_root ... \
    --dry_run
```

### 只跑指定 task

```bash
python scripts/run_experiment.py \
    --sweep configs/sweeps/improvement_phase2_temporal.yaml \
    --data_root ... --labels_root ... --splits_root ... \
    --tasks tcn_k7_lft gated_fusion
```

### 断点续跑（跳过已完成的）

`run_experiment.py` 默认检查 `metrics.json` 是否存在，自动跳过已完成的 run。
Colab 断线后重新执行同一命令即可续跑。

### 查看实时进度

```bash
# 查看当前有多少 run 完成
find runs/improvement_phase2 -name "metrics.json" | wc -l
```

---

## 待实现代码清单

Phase 2 需要以下新组件才能运行：

### 1. TCN 组件

文件: `src/models/components/temporal_conv.py`

功能: 在编码器输出后、融合前，对每个模态的 tokens 做局部时序卷积。

需要修改 `src/core/runner.py` 的 `MultimodalModel`:
- 读取 `cfg.model.temporal_conv` 配置
- 如果 `enabled=true`，在 encoder 后应用 TCN

### 2. 门控融合

文件: `src/models/fusions/gated.py`

功能: 模态级自适应门控加权融合。

注册: `@FUSIONS.register("gated")`

接口: 与 LFT 相同的 `BaseFusion` 接口，接受任意模态子集。

### 3. Runner 集成

修改 `MultimodalModel.forward()` 以支持可选的 temporal_conv 层。

---

## Sweep YAML 格式说明

```yaml
seeds: [0, 1, 2]          # 随机种子列表
modalities:                 # 模态组合列表
  - [video, km, telem]

shared:                     # 所有 task 共享的配置
  data: {...}
  model:
    encoders: {...}
    fusion_lft: {...}       # 以 fusion_{name} 格式定义融合模板
    fusion_cma: {...}
  train: {...}

tasks:                      # 各实验 task
  task_name:
    fusion: lft             # 选用哪个 fusion 模板
    task_type: multitask    # regression / classification / multitask
    data: {...}             # 覆盖 shared.data
    model: {...}            # 覆盖 shared.model
    train: {...}            # 覆盖 shared.train
    eval: {...}
```

`run_experiment.py` 对每个 `(task, modality_combo, seed)` 组合：
1. 从 `shared` 开始
2. 选择 `fusion_{name}` 作为 `model.fusion`
3. 深度合并 task 覆盖
4. 设置 modalities、seed、路径
5. 构建 Config → Runner → fit()
6. 结果写入 `runs_root/{task}_3seed/{modality_name}/`

---

## 结果目录结构

```
runs/improvement_phase1/
├── baseline_original_3seed/
│   └── triple_video_km_telem/
│       ├── 2026-...__seed0/metrics.json
│       ├── 2026-...__seed1/metrics.json
│       └── 2026-...__seed2/metrics.json
├── smooth_s1_3seed/
│   └── triple_video_km_telem/...
├── smooth_s2_3seed/...
├── smooth_s3_3seed/...
├── smooth_s5_3seed/...
│   ├── results.tsv              # 每个seed一行
│   └── results_summary.csv      # mean±std
```
