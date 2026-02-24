# ProjectExperiment（中文版）

一个可扩展的多模态情绪预测框架（Valence/Arousal），当前重点支持：
- `video` 模态（预提取的 ResNet-50 帧特征）
- `km` 模态（键盘/鼠标行为特征）

当前训练主干已完成基于「接口 + 注册表 + 配置驱动」的重构。

## 1. 当前已实现内容

### 1.1 实验方法
- 基于 AMuCS 风格会话键（如 `S001_P1`）进行多模态回归
- 支持任务：
  - Arousal 回归（`out_dim: 1`）
  - Valence/Arousal 联合回归（`out_dim: 2`）
- 已支持实验设置：
  - `km` 单模态 + `single` 融合
  - `video + km` 双模态 + `lft` 融合

### 1.2 模型架构
- 编码器（Encoders）：
  - `video/resnet2d`：先将每帧特征投影到 `d_model`，再用带 mask 的时间均值池化得到 `pooled`
  - `km/stat`：KM 统计特征线性投影编码器
  - `km/cnn1d`：KM 序列 1D CNN 编码器
- 融合器（Fusions）：
  - `single`：单模态直通；多模态时拼接 token 并做 mask 均值池化
  - `lft`：Late Fusion Transformer（位置编码 + 模态嵌入）
  - `lft_video_valence`：用于 VA 分流任务的特殊融合
- 预测头（Heads）：
  - `regression`：MLP 回归头
  - `va_split`：配合 `lft_video_valence` 的 valence/arousal 分流头
- 损失函数（Losses）：
  - `smooth_l1`、`mse`、`ccc`
- 指标（Metrics）：
  - `ccc`、`rmse`

### 1.3 可扩展机制
- 冻结接口定义：`src/core/types.py`
  - `BaseEncoder`、`BaseFusion`、`BaseHead`、`BaseDataModule`
- 注册表机制：`src/core/registry.py`
  - `DATAMODULES`、`FUSIONS`、`HEADS`、`LOSSES`、`METRICS`
  - 编码器按模态注册：`get_encoder_registry(modality)`
- 训练入口极薄：
  - `scripts/train.py` 仅做 `config -> Runner.fit()`

## 2. 仓库结构

```text
ProjectExperiment/
  configs/
    base.yaml
    experiments/
  scripts/
    train.py
    extract_video_features.py
    extract_km_features.py
    build_multimodal_split.py
    build_km_arousal_split.py
    summarize.py
  src/
    core/
    data/
    models/
    losses/
    metrics/
  tests/
  docs/
```

## 3. 数据与特征格式

### 3.1 训练期望的特征目录

```text
data/features/amucs/
  video/
    S001_P1.pt
    S001_P2.pt
    ...
  km/
    S001_P1.pt
    S001_P2.pt
    ...
```

以及：
- 标签文件：`data/labels_arousal.json`
- 划分文件：
  - 多模态：`data/splits/multimodal_split.json`
  - KM 单模态：`data/splits/km_arousal_split.json`

### 3.2 视频特征 `.pt` 内容

每个文件是一个字典，至少包含：
- `features`：`[T, D]` 张量（通常 `D=2048`）
- `timestamps`：每个特征对应时间戳（秒）
- `fps`、`sample_fps`、`stride`
- `meta`：源视频路径与提取参数信息

### 3.3 KM 特征 `.pt` 内容

从每个 session 的 CSV 日志提取：
- `keyboard.csv`
- `mousebuttons.csv`

输出为包含 `features` 和元信息的字典。

## 4. 环境准备

建议 Python 3.10+。

安装核心依赖：

```bash
pip install torch torchvision pyyaml pandas opencv-python tqdm pytest
```

## 5. 端到端使用流程

## 5.1 视频特征提取

输入为 session 子目录（如 `s001`、`s002`）时，推荐：

```bash
python scripts/extract_video_features.py \
  --video_dir "/path/to/gameplay_videos_nospeech" \
  --output_dir "data/features/amucs/video" \
  --session_mode subdirs \
  --name_mode amucs \
  --device cuda
```

说明：
- `--name_mode amucs` 会将输出命名为 `S001_P1` 这类 stem。
- 内置 session 级断点续跑：
  - 在 `output_dir/.session_done/` 写完成标记
  - 重跑时会自动跳过已完成 session/文件
- 需要强制重算时加 `--overwrite`。

## 5.2 KM 特征提取

输入目录建议是 `S*/P*` 结构：

```text
/path/to/km_raw/
  S001/
    P1/
      keyboard.csv
      mousebuttons.csv
```

执行：

```bash
python scripts/extract_km_features.py \
  --root_dir "/path/to/km_raw" \
  --output_dir "data/features/amucs/km" \
  --encoder stat
```

说明：
- 默认跳过已存在输出（支持断点续跑）
- 强制重算使用 `--overwrite`

## 5.3 构建训练/验证划分

多模态划分：

```bash
python scripts/build_multimodal_split.py \
  --video_dir data/features/amucs/video \
  --km_dir data/features/amucs/km \
  --labels_path data/labels_arousal.json \
  --output_path data/splits/multimodal_split.json
```

KM 单模态划分：

```bash
python scripts/build_km_arousal_split.py --seed 42 --train_ratio 0.8
```

## 5.4 训练实验

Video + KM LFT：

```bash
python scripts/train.py --config configs/experiments/video_km_lft.yaml
```

KM 单模态：

```bash
python scripts/train.py --config configs/experiments/km_single.yaml
```

KM 单模态（CNN 编码器）：

```bash
python scripts/train.py --config configs/experiments/km_single_cnn.yaml
```

Video + KM（VA 双输出）：

```bash
python scripts/train.py --config configs/experiments/video_km_lft_va.yaml
```

覆盖配置示例：

```bash
python scripts/train.py \
  --config configs/experiments/video_km_lft.yaml \
  --override train.epochs=100 train.seed=0 model.fusion.num_layers=2
```

## 5.5 训练断点续跑

从 run 目录恢复：

```bash
python scripts/train.py \
  --config configs/experiments/video_km_lft.yaml \
  --resume runs/<run_dir_name>
```

从 checkpoint 文件恢复：

```bash
python scripts/train.py \
  --config configs/experiments/video_km_lft.yaml \
  --resume runs/<run_dir_name>/ckpt_last.pt
```

注意：
- 训练恢复是 epoch 级（不是 batch 级）
- 如果 checkpoint 已到达 `train.epochs`，需要通过 override 调大 epoch

## 5.6 结果汇总

```bash
python scripts/summarize.py --runs_dir runs --output leaderboard.csv
```

## 6. 训练输出目录

每次训练会生成：

```text
runs/{timestamp}__{dataset}__{fusion}__{modalities}__seed{seed}/
  config.yaml (或 config.json)
  seed.txt
  git_commit.txt
  ckpt_best.pt
  ckpt_last.pt
  metrics.json
```

## 7. 当前实验配置文件

- `configs/experiments/video_km_lft.yaml`
  - 双模态（`video`、`km`）+ `lft`
- `configs/experiments/km_single.yaml`
  - KM 单模态 + `single`
- `configs/experiments/km_single_cnn.yaml`
  - KM 单模态 + `cnn1d` + `single`
- `configs/experiments/video_km_lft_va.yaml`
  - 双模态 + 2 维输出（`out_dim: 2`）

基础默认配置在 `configs/base.yaml`。

## 8. 质量验证

运行测试：

```bash
pytest -q
```

包含：
- 形状契约测试：`tests/test_shapes.py`
- 注册表完整性测试：`tests/test_registry.py`

## 9. Colab / Google Drive 实践建议

为避免 Colab 中断导致进度丢失：
- 将 `runs_dir` 放在 Google Drive
- 将特征输出目录放在 Google Drive
- 中断后重复运行同命令并使用 `--resume`

对于按 session 文件夹的视频数据，推荐：

```bash
--session_mode subdirs --name_mode amucs
```

可避免跨 session 文件名冲突，并启用 session 级断点续跑。
