# AMuCS 双模态实验近期改动技术文档

本文档整理了当前仓库中与 `video + km` arousal 回归相关的关键改动，重点覆盖：
- 视频特征提取可恢复与提速改动
- 训练 batch 级断点续跑改动
- 已提取特征的时间对齐脚本
- “时间对齐后标签如何使用”的当前现状与下一步方案

---

## 1. 视频特征提取改动（`scripts/extract_video_features.py`）

### 1.1 断点续跑与会话级完成标记
- 支持按 session 子目录处理：`--session_mode subdirs`
- 每个 session 完成后写入：
  - `output_dir/.session_done/<session>.done.json`
- 重跑时默认行为：
  - 若 session 全部输出存在且 done 标记存在，则跳过该 session
  - 若输出已存在也会逐文件跳过
- 强制重提取可用：`--overwrite`

### 1.2 输出命名与冲突规避
- 新增 `--name_mode amucs`，输出 stem 统一为 `Sxxx_Py`（如 `S001_P3`）
- 在提取前检测 stem 冲突，若不同源文件会覆盖同名输出会直接报错，避免静默覆盖

### 1.3 推理速度与运行可见性
- 新增 AMP fp16 推理开关：`--amp`（CUDA 下生效）
- 进度打印已改为显式刷新（`flush=True`），并按 session/文件打印

### 1.4 Colab 中的常见坑
- `!python ... \` 多行命令容易因为缩进触发 `IndentationError`
- 推荐使用 `%%bash` cell 执行多行命令

---

## 2. 训练断点续跑改动（`src/core/runner.py`）

### 2.1 已实现能力
- 支持从 run 目录或 ckpt 文件恢复：
  - `--resume runs/<run_dir>`
  - `--resume runs/<run_dir>/ckpt_last.pt`
- checkpoint 新增 batch 级状态：
  - `epoch`
  - `batch_in_epoch`
  - `num_batches_in_epoch`
- 训练中按批次保存 `ckpt_last.pt`：
  - 频率由 `train.ckpt_every_batches` 控制（默认 `1`，即每个 batch 保存）

### 2.2 恢复逻辑
- 若中断时未完成当前 epoch，则从该 epoch 的下一个 batch 继续
- 若中断点在 epoch 末尾，则从下一 epoch 开始
- 为保持可复现性，每个 epoch 会重建 train dataloader 并固定种子偏移

---

## 3. 后处理时间对齐脚本（`scripts/sync_data.py`）

### 3.1 定位
- 用于“特征提取完成后”的时间轴对齐
- 输入是多个模态的 `.pt` 特征文件，输出为同名对齐后的 `.pt`
- 设计为可扩展，不硬编码 modality 名称

### 3.2 关键能力
- 模态自动发现或手动指定：
  - `--modalities video,km,...`
- 仅处理共同 stem（可叠加 split 过滤）：
  - `--split_path data/splits/xxx.json`
- 目标时间网格：
  - `--grid_mode uniform` + `--target_hz`
  - `--grid_mode reference` + `--reference_modality`
- 重采样：
  - `--resample nearest|linear`
- 时间原点与偏移：
  - `--time_origin zero|raw`
  - `--offsets "video=0,km=0.2"`
- 质量控制：
  - `--max_gap`（超间隔点置无效）
  - `--min_points`（过短序列跳过）
- 输出报告：
  - 默认 `output_root/alignment_report.json`

### 3.3 输出数据结构
每个对齐后的 `.pt` 保留原字段并更新：
- `features`: 对齐后的 `[T, D]`
- `timestamps`: 统一网格时间轴
- `mask`: 对齐后的有效点掩码
- `meta`: 新增 `aligned` 与对齐参数记录

---

## 4. 当前标签机制（重点）

### 4.1 `labels_arousal.json` 的来源
- 由 `encoder/km/build_arousal_labels.py` 生成
- 原始来源是每个 session 的 ranktrace CSV
- 当前策略是对每个 session 的 arousal 序列取均值，得到 session 级标量标签

### 4.2 `FULL_LABELS_PATH` 的本意
- 指向“完整标签字典”JSON（通常是 `labels_arousal.json`）
- 主要用于：
  - 过滤哪些 stem 有可用 arousal
  - 训练时给 DataModule 提供监督标签

### 4.3 你提出的问题是成立的
做了时间对齐后，如果目标是“严格时间监督”，继续用 session 均值标签在方法上不充分。

---

## 5. 时间对齐后该怎么训练

### 5.1 现阶段可直接跑通的方案（已实现）
- 任务定义：session 级 arousal 回归
- 输入：对齐后的视频/KM特征（可用）
- 标签：仍然是 `labels_arousal.json` 标量（均值）
- 结论：工程可跑通，但不是严格时间点监督

### 5.2 严格时间监督方案（当前仓库尚未完整实现）
若要让对齐真正用于监督，需要新增：
- 时间序列标签构建：
  - 从 ranktrace 读取 `(timestamp, arousal)` 序列
  - 重采样到与特征一致的时间网格
- 序列数据加载：
  - DataModule 返回 `y_seq` 与 `y_mask`
- 序列预测头与损失：
  - head 输出 `[B, T, 1]`
  - 使用 masked MSE/CCC 等序列损失

---

## 6. 推荐命令模板

### 6.1 特征时间对齐（同名文件、全量）
```bash
python scripts/sync_data.py \
  --input_root /content/drive/MyDrive/AmuCSvideo/features \
  --output_root /content/drive/MyDrive/AmuCSvideo/features_aligned \
  --modalities video,km \
  --grid_mode uniform \
  --target_hz 5 \
  --resample nearest \
  --time_origin zero \
  --report_path /content/drive/MyDrive/AmuCSvideo/output/alignment_report.json
```

### 6.2 训练（当前 session 级监督）
```bash
python scripts/train.py \
  --config configs/experiments/video_km_lft.yaml \
  --override \
    data.data_root=/content/drive/MyDrive/AmuCSvideo/features_aligned \
    data.labels_path=/content/ProjectExperiment/data/labels_arousal.json \
    runs_dir=/content/drive/MyDrive/AmuCSvideo/output/runs
```

### 6.3 中断后续跑
```bash
python scripts/train.py \
  --config configs/experiments/video_km_lft.yaml \
  --resume /content/drive/MyDrive/AmuCSvideo/output/runs/<run_dir>
```

---

## 7. 结论

- 你关心的三项工程问题已在代码层面具备可用实现：
  - 特征提取可恢复（session/file 级跳过）
  - 训练 batch 级断点续跑
  - 提取后多模态时间对齐
- 但“时间对齐后仍用均值标签”属于任务定义层面的不一致，不是代码 bug。
- 若目标是严格时序 arousal 回归，下一步应改为“序列标签 + 序列预测 + 掩码损失”链路。
