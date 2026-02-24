# 数据集结构说明（中文版）

# Primary

## Observational（观测数据）

文件：
- `AMuCS_prequestionnaire_results.csv`：游戏前问卷结果

### 时间同步说明
数据记录使用 LSL（LabStreamingLayer）进行同步。原则上，同一 session 下所有 CSV 文件都来自同一时钟。

目前已发现 `keyboard.csv` 存在一些时间不一致问题，其他模态也可能存在潜在不一致。

### 数据组织

每个 session（`S001-S071`）下包含每位参与者（`P1-P4`）目录。  
例如：第 3 个 session 的第 4 位参与者数据在 `S003/P4/`。

每个 session 级别文件：
- `gameDictionary.csv`：游戏枚举值字典（用于解释 `gameInt.csv` 中编码字段）
- `gamePlayerInfo.csv`：session 内玩家信息（例如队伍编号）

每个参与者级别文件：
- `eyetracker.csv`：Tobii Pro Nano 眼动数据
- `gameFlt.csv`：游戏浮点数据（例如位置、速度等），包含部分派生列（其他玩家在屏幕上的位置）。对于玩家 `p`，对应列为 `screenXp`、`screenYp`；若 `p` 与当前参与者相同，或该玩家不在视野内，则该值为空。
- `gameInt.csv`：游戏整数数据（例如弹药、分数等）
- `gameMrk.csv`：游戏标记（例如回合开始/结束）
- `keyboard.csv`：键盘按键
- `mousebuttons.csv`：鼠标按键
- `mouseposition.csv`：鼠标位置，**不要使用**（游戏引擎会捕获鼠标光标，导致该数据不可用）
- `obsframes.csv`：游戏画面视频帧时序信息
- `physio.csv`：Bitalino 生理信号（ECG、EDA、呼吸）
- `ranktrace.csv`：PAGAN 情绪标注轨迹（Arousal 或 Valence）
- `realsenseframes.csv`：Intel RealSense 录制帧时序（人脸与深度）
- `mat.csv`：座椅压力垫数据（16x16 传感器网格展开）

### 深度视频（depth videos）
按 session（`S001-S071`）组织目录。

文件：
- `Px_realsense_depth.mkv`：参与者 `x` 的深度视频（由 rosbag 格式转换而来）

### 游戏画面视频（gameplay videos）
按 session（`S001-S071`）组织目录。

文件：
- `Px_gameplay.mp4`：参与者 `x` 的游戏画面视频。  
  注：所有视频容器分辨率均为 `1920x1080`，但部分参与者在游戏内设置了更低分辨率，导致实际游戏区域位于视频左上角。每个 session/玩家使用的分辨率见文档中的 data quality CSV。

游戏画面视频已做以下处理：
- 使用 demucs 算法过滤人声（https://github.com/facebookresearch/demucs）
- 3 个视频（`S063/P2`, `S064/P2`, `S067/P2`）为保护隐私，对部分片段做了模糊处理（当参与者面部出现在画面中时）

## Derived（派生数据）

### data
按 session（`S001-S071`）和参与者（`P1-P4`）组织。  
例如：`S003/P4/`。

文件：
- `facefeatures.csv`：从彩色人脸视频使用 OpenFace 提取的人脸特征；每帧时间与 `realsenseframe.csv` 对齐
- `screenLuminance.csv`：屏幕亮度特征（全屏平均、中心区域平均、注视点区域平均），基于按实际游戏分辨率裁剪后的 gameplay 视频计算

# Secondary

## Documentation（文档）

### docs
文件：
- consent questions：实验前知情同意中的问题列表
- data quality：各模态在每位参与者上的数据质量汇总（也包含游戏分辨率与同意相关信息）
- gamedata description：游戏数据各列字段说明
- prequestionnaire：游戏前问卷
- dataset structure：本文件

# Software

## Code（代码）

### python scripts
文件：
- read video frame (python3)：演示如何以正确编码读取深度视频帧，并将像素值转换为米制距离
- sync data (python3)：演示如何进行多模态数据对齐，最后会裁剪到仅保留游戏进行时段。使用时需修改 `basedir`（Primary 数据目录）和 `outputdir`。脚本第一个输入参数可指定用于同步的模态（默认 `'phy'`，即 Bitalino 生理数据），第二个输入参数可指定输出目录（默认 `"./synced-data/"`）。
