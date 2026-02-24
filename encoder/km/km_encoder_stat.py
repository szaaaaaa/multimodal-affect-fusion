"""
Minimal KM statistical encoder (events -> binned features).

键盘/鼠标事件统计编码器：将原始事件按时间分箱为固定长度特征序列。
"""

from __future__ import annotations
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple
import math
import numpy as np
import torch
from torch import nn


@dataclass(frozen=True)
class KMEvent:
    """
    Minimal unified event schema.

    最小事件结构定义（中英文注释，NumPy 风格）。

    Parameters
    ----------
    t : float
        Timestamp in seconds.
        / 事件时间戳（秒）。
    kind : str
        Event type, e.g. "key_down", "key_up", "mouse_move".
        / 事件类型。
    x : float, optional
        Mouse X for move events.
        / 鼠标移动的 X 坐标。
    y : float, optional
        Mouse Y for move events.
        / 鼠标移动的 Y 坐标。
    button : str, optional
        Mouse button name ("left"/"right"/"middle").
        / 鼠标按钮名称。
    scroll : float, optional
        Scroll delta value.
        / 滚轮增量。
    key : str, optional
        Key name for keyboard events (e.g., "F3", "SPACE").
        / 键名（用于键盘事件）。
    """

    t: float
    kind: str
    x: Optional[float] = None
    y: Optional[float] = None
    button: Optional[str] = None
    scroll: Optional[float] = None
    key: Optional[str] = None


class KMStatEncoder:
    """
    Encode events into fixed-step statistical features.

    将事件序列编码为固定时间步长的统计特征。

    Parameters
    ----------
    dt : float, optional
        Bin size in seconds.
        / 时间分箱大小（秒）。
    device : str, optional
        Output device for tensors.
        / 输出张量所在设备。
    dtype : torch.dtype, optional
        Output dtype for tensors.
        / 输出张量的数据类型。
    """

    def __init__(self, dt: float = 0.2, device: str = "cpu", dtype: torch.dtype = torch.float32):
        if dt <= 0:
            raise ValueError("dt must be positive.")
        self.dt = float(dt)
        self.device = device
        self.dtype = dtype

    def encode(self, raw_input: Dict[str, Any]) -> Dict[str, Any]:
        """
        Encode raw KM events into binned features.

        将原始键鼠事件编码为分箱统计特征。

        Parameters
        ----------
        raw_input : dict
            Supported inputs:
            1) Direct events:
               {
                 "events": List[KMEvent] OR List[dict],
                 "t0": float,
                 "t1": float
               }
            2) Table-like data:
               {
                 "mousebuttons": List[dict],
                 "keyboard": List[dict],
                 "t0": float,
                 "t1": float
               }
            / 支持事件列表或表格输入。

        Returns
        -------
        dict
            {
              "features": Tensor[T, 25],
              "mask": Tensor[T],
              "meta": dict
            }
            / 返回特征、mask 与元信息。
        """
        if "events" in raw_input:
            events = raw_input.get("events", [])
        else:
            events = self._events_from_tables(raw_input)

        # Determine time window / 确定时间窗口
        if "t0" in raw_input and "t1" in raw_input:
            t0 = float(raw_input["t0"])
            t1 = float(raw_input["t1"])
        else:
            times = [e["t"] for e in events] if events else []
            if not times:
                raise ValueError("No events and no t0/t1 provided.")
            t0 = float(min(times))
            t1 = float(max(times)) + self.dt

        if t1 <= t0:
            raise ValueError("t1 must be > t0.")

        # Normalize events / 统一事件格式
        ev = self._coerce_events(events)
        ev = [e for e in ev if (t0 <= e.t < t1)]
        ev.sort(key=lambda e: e.t)

        # Number of bins and feature matrix / 分箱数量与特征矩阵
        T = int(math.ceil((t1 - t0) / self.dt))
        feats = np.zeros((T, 25), dtype=np.float32)

        def bin_idx(t: float) -> int:
            # Map time to bin index / 时间映射到 bin
            i = int((t - t0) // self.dt)
            return max(0, min(T - 1, i))

        # Track last move for distance/speed / 记录上一帧鼠标位置
        last_move: Optional[Tuple[float, float, float]] = None
        last_speed: Optional[float] = None
        last_key_t: Optional[float] = None
        key_intervals_by_bin: List[List[float]] = [[] for _ in range(T)]
        speed_contrib_count = np.zeros(T, dtype=np.float32)
        accel_contrib_count = np.zeros(T, dtype=np.float32)
        key_sets_by_bin: List[set[str]] = [set() for _ in range(T)]

        for e in ev:
            i = bin_idx(e.t)

            if e.kind == "key_down":
                feats[i, 0] += 1.0
                if last_key_t is not None:
                    key_intervals_by_bin[i].append(max(0.0, e.t - last_key_t))
                last_key_t = e.t
                if e.key is not None:
                    key_sets_by_bin[i].add(str(e.key))

            elif e.kind == "key_up":
                feats[i, 1] += 1.0
                if last_key_t is not None:
                    key_intervals_by_bin[i].append(max(0.0, e.t - last_key_t))
                last_key_t = e.t
                if e.key is not None:
                    key_sets_by_bin[i].add(str(e.key))

            elif e.kind == "mouse_move":
                feats[i, 2] += 1.0
                if e.x is not None and e.y is not None:
                    if last_move is not None:
                        t_prev, x_prev, y_prev = last_move
                        dx = float(e.x) - x_prev
                        dy = float(e.y) - y_prev
                        dist = math.hypot(dx, dy)
                        dt_local = max(1e-6, float(e.t) - t_prev)
                        speed = dist / dt_local
                        feats[i, 3] += dist
                        feats[i, 4] += speed
                        feats[i, 5] = max(feats[i, 5], speed)
                        speed_contrib_count[i] += 1.0
                        feats[i, 15] += dx
                        feats[i, 16] += dy
                        if last_speed is not None:
                            accel = (speed - last_speed) / dt_local
                            feats[i, 17] += accel
                            accel_contrib_count[i] += 1.0
                        last_speed = speed
                    last_move = (float(e.t), float(e.x), float(e.y))

            elif e.kind == "mouse_click":
                feats[i, 6] += 1.0
                if (e.button or "").lower() == "left":
                    feats[i, 7] += 1.0
                elif (e.button or "").lower() == "right":
                    feats[i, 8] += 1.0

            elif e.kind == "mouse_button_up":
                feats[i, 9] += 1.0
                if (e.button or "").lower() == "left":
                    feats[i, 10] += 1.0
                elif (e.button or "").lower() == "right":
                    feats[i, 11] += 1.0

            elif e.kind == "mouse_scroll":
                feats[i, 12] += 1.0
                if e.scroll is not None:
                    feats[i, 13] += float(e.scroll)

        # Average speed: sum / count / 速度均值
        speed_denom = np.maximum(speed_contrib_count, 1.0)
        feats[:, 4] = feats[:, 4] / speed_denom

        # Average acceleration / 加速度均值
        accel_denom = np.maximum(accel_contrib_count, 1.0)
        feats[:, 17] = feats[:, 17] / accel_denom

        # Inter-key interval mean / 键间隔均值
        for i in range(T):
            if key_intervals_by_bin[i]:
                feats[i, 14] = float(np.mean(key_intervals_by_bin[i]))
            else:
                feats[i, 14] = 0.0
            feats[i, 24] = float(len(key_sets_by_bin[i]))

        # Count deltas / 计数变化量
        for i in range(T):
            if i == 0:
                feats[i, 18] = feats[i, 0]
                feats[i, 19] = feats[i, 1]
            else:
                feats[i, 18] = feats[i, 0] - feats[i - 1, 0]
                feats[i, 19] = feats[i, 1] - feats[i - 1, 1]

        # Event rates (per second) / 事件率（每秒）
        feats[:, 20] = feats[:, 0] / self.dt
        feats[:, 21] = feats[:, 6] / self.dt
        feats[:, 22] = feats[:, 12] / self.dt
        feats[:, 23] = feats[:, 2] / self.dt

        # All valid (no padding) / 全部有效
        mask = torch.ones((T,), dtype=torch.bool, device=self.device)
        features = torch.tensor(feats, dtype=self.dtype, device=self.device)

        return {
            "features": features,
            "mask": mask,
            "meta": {
                "modality": "keyboard_mouse",
                "dt": self.dt,
                "t0": t0,
                "t1": t1,
                "feature_dim": int(features.shape[-1]),
                "feature_names": [
                    "key_down_count",
                    "key_up_count",
                    "mouse_move_event_count",
                    "mouse_move_distance_sum",
                    "mouse_speed_mean",
                    "mouse_speed_max",
                    "mouse_button_down_count",
                    "left_button_down_count",
                    "right_button_down_count",
                    "mouse_button_up_count",
                    "left_button_up_count",
                    "right_button_up_count",
                    "scroll_event_count",
                    "scroll_delta_sum",
                    "inter_key_interval_mean",
                    "mouse_dx_sum",
                    "mouse_dy_sum",
                    "mouse_accel_mean",
                    "key_down_delta",
                    "key_up_delta",
                    "key_down_rate",
                    "mouse_click_rate",
                    "scroll_rate",
                    "mouse_move_rate",
                    "unique_key_count",
                ],
            },
        }

    @staticmethod
    def _coerce_events(events: List[Any]) -> List[KMEvent]:
        """
        Normalize events to KMEvent list.

        将输入事件统一为 KMEvent 列表。
        """
        out: List[KMEvent] = []
        for e in events:
            if isinstance(e, KMEvent):
                out.append(e)
            elif isinstance(e, dict):
                out.append(
                    KMEvent(
                        t=float(e["t"]),
                        kind=str(e["kind"]),
                        x=e.get("x", None),
                        y=e.get("y", None),
                        button=e.get("button", None),
                        scroll=e.get("scroll", None),
                        key=e.get("key", None),
                    )
                )
            else:
                raise TypeError("events must be List[KMEvent] or List[dict].")
        return out

    @staticmethod
    def _events_from_tables(raw_input: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        Build events from table-like inputs.

        将表格格式输入转换为事件列表。
        """
        events: List[Dict[str, Any]] = []

        for row in raw_input.get("mousebuttons", []):
            t = float(row["time"])
            text = str(row.get("channel_0", "")).strip()
            text_lower = text.lower()
            button = None
            if "left" in text_lower:
                button = "left"
            elif "right" in text_lower:
                button = "right"
            elif "middle" in text_lower:
                button = "middle"
            if "pressed" in text_lower:
                events.append({"t": t, "kind": "mouse_click", "button": button})
            elif "released" in text_lower:
                events.append({"t": t, "kind": "mouse_button_up", "button": button})

        for row in raw_input.get("keyboard", []):
            t = float(row["time"])
            raw_text = str(row.get("channel_0", "")).strip()
            text = raw_text.lower()
            key_name = raw_text.split(" ")[0] if raw_text else None
            if "pressed" in text:
                events.append({"t": t, "kind": "key_down", "key": key_name})
            elif "released" in text:
                events.append({"t": t, "kind": "key_up", "key": key_name})

        return events


class KMStatTokenEncoder(nn.Module):
    """
    Lightweight token projector for stat features.

    输入: x [B, L, D] -> 输出: [B, L, d_model]
    """

    def __init__(self, d_in: int, d_model: int):
        super().__init__()
        self.proj = nn.Linear(d_in, d_model)
        self.ln = nn.LayerNorm(d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.ln(self.proj(x))


if __name__ == "__main__":
    demo = {
        "t0": 1510.0,
        "t1": 1512.0,
        "mousebuttons": [
            {"time": 1511.68, "channel_0": "MouseButtonLeft pressed"},
            {"time": 1511.75, "channel_0": "MouseButtonLeft released"},
        ],
        "keyboard": [
            {"time": 1510.959, "channel_0": "F3 pressed"},
            {"time": 1511.039, "channel_0": "F3 released"},
        ],
    }

    enc = KMStatEncoder(dt=0.2)
    out = enc.encode(demo)
    print("features shape:", tuple(out["features"].shape))
    print("first 3 bins:\n", out["features"][:3])
    print("feature_names:", out["meta"]["feature_names"])
