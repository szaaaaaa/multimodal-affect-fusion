"""
Session 时间原点工具（修正版）。

AMuCS 的 3 个模态 raw CSV 时间列 **不共用同一时钟**：
- `keyboard.csv` / `mousebuttons.csv` 用的是"键鼠采集进程时间"；
- `gameInt.csv` / `gameFlt.csv` 用的是"游戏 tick 时间"；
- `ranktrace.csv` 用的是"标注进程时间"（video 起点对应该列 VideoTime=0）。

现有 baseline (`extract_game_telem_features.py`, `KMStatEncoder`) 的做法是：
**每个模态用它自己的第一条采样作为 t0**（不做跨模态对齐），然后各自做 5Hz 栅格。
因为实验采集各子系统是同步启动的，所以 `time - t0` 足够作为相对秒数；Path B
沿用这个约定，保持与 baseline 完全一致的对齐语义。

这个模块只提供 video 视频时长工具；km/telem 的 t0 在各自抽取脚本里算。
"""

from __future__ import annotations

from pathlib import Path

import torch


VIDEO_FRAME_DT = 0.2


def get_video_T_v(video_clip_file: Path) -> int:
    """读 aligned `video_clip/{stem}.pt`，返回其 features 第 0 维长度。"""
    d = torch.load(video_clip_file, map_location="cpu", weights_only=False)
    feats = d["features"] if isinstance(d, dict) else d
    return int(feats.shape[0])
