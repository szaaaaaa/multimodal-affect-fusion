"""
AMuCS mixed-rate sequence DataModule for Path B (raw km events + 60Hz telem).

不同模态以各自原生采样率存储，以 video 5Hz (T_v) 作为窗口基准：
- video       : [T_v, 768]      (5Hz)
- telem_60hz  : [T_v*12, D_t]   (60Hz)
- km_event    : [T_k_var, 4]    (离散事件, 变长；T_k_var 仅限窗口内)

标签仍沿用 5Hz 的 `arousal_state_trend_seq.json`，T 与 video 对齐。

km_event 单帧通道
-----------------
    channel 0 : event_type_id (long)
    channel 1 : t_rel_sec      (窗口相对秒数；已重新基于 window_start 归零)
    channel 2 : dt_sec         (窗口内相邻事件间隔)
    channel 3 : bin_id         (窗口内 0..T_v-1)
"""

from __future__ import annotations

import json
import warnings
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import torch
from torch.utils.data import DataLoader, Dataset

from src.core.registry import DATAMODULES
from src.core.types import BaseDataModule


VIDEO_FRAME_DT = 0.2
TELEM_60HZ_PER_VIDEO_FRAME = 12


def _window_frames(
    x: torch.Tensor,
    mask: torch.Tensor,
    start: int,
    end: int,
    target_len: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Slice [start:end] and right-pad to target_len."""
    x_win = x[start:end]
    m_win = mask[start:end]
    cur = x_win.shape[0]
    if cur < target_len:
        pad_shape = (target_len - cur, *x_win.shape[1:])
        x_pad = torch.zeros(pad_shape, dtype=x_win.dtype)
        m_pad = torch.zeros(target_len - cur, dtype=torch.bool)
        x_win = torch.cat([x_win, x_pad], dim=0)
        m_win = torch.cat([m_win, m_pad], dim=0)
    return x_win, m_win


class AMuCSSeqMixedRateDataset(Dataset):
    def __init__(
        self,
        modalities: List[str],
        data_root: Path,
        labels_seq_path: Path,
        split_path: Path,
        split: str,
        seq_len_video_frames: int,
        task_names: List[str],
        stride_video_frames: Optional[int] = None,
        include_tail_window: bool = True,
        normalize: bool = True,
        stats_dir: Optional[Path] = None,
        label_dtype: str = "long",
        task_label_dtypes: Optional[Dict[str, str]] = None,
        temporal_split_ratios: Optional[Dict[str, List[float]]] = None,
        modality_dir_map: Optional[Dict[str, str]] = None,
        max_events_per_window: int = 2048,
    ):
        self.modalities = modalities
        self.task_names = list(task_names)
        self.label_dtype = str(label_dtype).lower()
        self.task_label_dtypes = {str(k): str(v).lower() for k, v in dict(task_label_dtypes or {}).items()}
        self.data_root = Path(data_root)
        self.split = split
        self.seq_len = int(seq_len_video_frames)
        self.stride = int(stride_video_frames) if stride_video_frames is not None else None
        self.include_tail_window = bool(include_tail_window)
        self.normalize = normalize
        self.stats_dir = Path(stats_dir) if stats_dir else self.data_root
        self.temporal_split_ratios = temporal_split_ratios
        self.max_events_per_window = int(max_events_per_window)

        with Path(labels_seq_path).open("r", encoding="utf-8-sig") as f:
            self.labels_seq = json.load(f)
        with Path(split_path).open("r", encoding="utf-8-sig") as f:
            split_stems = json.load(f).get(split, [])

        _dir_map = dict(modality_dir_map or {})
        self.mod_dirs: Dict[str, Path] = {
            m: self.data_root / _dir_map.get(m, m) for m in self.modalities
        }

        common = set(self.labels_seq.keys()) & set(split_stems)
        for m in self.modalities:
            common &= {p.stem for p in self.mod_dirs[m].glob("*.pt")}
        self.stems = sorted(common)
        if not self.stems:
            warnings.warn(
                f"[AMuCSSeqMixedRateDataset] no stems for split={split}, mods={modalities}"
            )

        self.stats: Dict[str, Dict[str, torch.Tensor]] = {}
        if normalize:
            for m in self.modalities:
                self.stats[m] = self._load_stats(m)

        self._cache_pt: Dict[str, Dict[str, Any]] = {}  # per-stem cached raw tensors
        self._stem_base_len: Dict[str, int] = self._compute_base_lengths()
        self._index: List[Tuple[str, int, int]] = self._build_index()

    def _load_stats(self, modality: str) -> Dict[str, torch.Tensor]:
        # 优先查每模态 stats 目录下的 stats.json（新格式，telem_60hz 用）
        mod_stats = self.mod_dirs[modality] / "stats.json"
        if mod_stats.is_file():
            s = json.loads(mod_stats.read_text(encoding="utf-8"))
            if "mean" in s and "std" in s:
                return {
                    "mean": torch.tensor(s["mean"], dtype=torch.float32),
                    "std": torch.tensor(s["std"], dtype=torch.float32),
                }
        # 回退到老格式 `{modality}_input_stats.json`
        legacy = self.stats_dir / f"{modality}_input_stats.json"
        if legacy.is_file():
            s = json.loads(legacy.read_text(encoding="utf-8"))
            return {
                "mean": torch.tensor(s["mean"], dtype=torch.float32),
                "std": torch.tensor(s["std"], dtype=torch.float32),
            }
        return {}

    def _get_task_dtype(self, task: str) -> torch.dtype:
        name = self.task_label_dtypes.get(task, self.label_dtype)
        if name == "long":
            return torch.long
        if name in {"float", "float32"}:
            return torch.float32
        raise ValueError(f"unsupported dtype for task {task}: {name}")

    def _load_pt_raw(self, stem: str, modality: str) -> Dict[str, Any]:
        key = f"{stem}/{modality}"
        if key in self._cache_pt:
            return self._cache_pt[key]
        obj = torch.load(self.mod_dirs[modality] / f"{stem}.pt", map_location="cpu", weights_only=False)
        self._cache_pt[key] = obj
        return obj

    def _modality_len_in_video_frames(self, stem: str, modality: str) -> int:
        """返回该模态在该 session 下等效的 video-frame 长度上限。"""
        obj = self._load_pt_raw(stem, modality)
        if modality == "km_event":
            return int(obj["T_v"])
        if modality == "telem_60hz":
            T_t = int(obj["features"].shape[0])
            return T_t // TELEM_60HZ_PER_VIDEO_FRAME
        # 默认 5Hz 张量 (video)
        feats = obj["features"] if isinstance(obj, dict) else obj
        return int(feats.shape[0])

    def _compute_base_lengths(self) -> Dict[str, int]:
        base: Dict[str, int] = {}
        for stem in self.stems:
            item = self.labels_seq[stem]
            lengths: List[int] = []
            missing_task = False
            for task in self.task_names:
                if task not in item:
                    missing_task = True
                    break
                lengths.append(len(item[task]["values"]))
            if missing_task:
                continue
            for m in self.modalities:
                lengths.append(self._modality_len_in_video_frames(stem, m))
            L = min(lengths)
            if L > 0:
                base[stem] = int(L)
        return base

    def _get_temporal_range(self, base_len: int) -> Tuple[int, int]:
        if not self.temporal_split_ratios or self.split not in self.temporal_split_ratios:
            return 0, base_len
        lo, hi = self.temporal_split_ratios[self.split]
        return int(base_len * lo), int(base_len * hi)

    def _build_index(self) -> List[Tuple[str, int, int]]:
        index: List[Tuple[str, int, int]] = []
        use_sliding = self.stride is not None and self.stride > 0
        for stem in self.stems:
            base_len = self._stem_base_len.get(stem, 0)
            if base_len <= 0:
                continue
            rs, re = self._get_temporal_range(base_len)
            rl = re - rs
            if rl <= 0:
                continue
            if not use_sliding:
                if rl >= self.seq_len:
                    start = rs + (rl - self.seq_len) // 2
                else:
                    start = rs
                index.append((stem, int(start), int(base_len)))
                continue
            if rl <= self.seq_len:
                index.append((stem, int(rs), int(base_len)))
                continue
            assert self.stride is not None
            max_start = re - self.seq_len
            starts = list(range(rs, max_start + 1, self.stride))
            if self.include_tail_window and starts and starts[-1] != max_start:
                starts.append(max_start)
            for s in starts:
                index.append((stem, int(s), int(base_len)))
        if not index:
            warnings.warn(f"[AMuCSSeqMixedRateDataset] no windows for split={self.split}")
        return index

    def __len__(self) -> int:
        return len(self._index)

    def _load_video(self, stem: str, modality: str, start: int, end: int) -> Tuple[torch.Tensor, torch.Tensor]:
        obj = self._load_pt_raw(stem, modality)
        feats = obj["features"].float() if isinstance(obj, dict) else obj.float()
        fmask = obj.get("mask") if isinstance(obj, dict) else None
        fmask = fmask.bool() if isinstance(fmask, torch.Tensor) else torch.ones(feats.shape[0], dtype=torch.bool)
        x, m = _window_frames(feats, fmask, start, end, self.seq_len)
        if self.normalize and modality in self.stats and self.stats[modality]:
            mean = self.stats[modality]["mean"]
            std = self.stats[modality]["std"]
            x = (x - mean) / std.clamp(min=1e-8)
        return x, m

    def _load_telem_60hz(self, stem: str, start: int, end: int) -> Tuple[torch.Tensor, torch.Tensor]:
        obj = self._load_pt_raw(stem, "telem_60hz")
        feats = obj["features"].float()
        fmask = obj["mask"].bool() if "mask" in obj else torch.ones(feats.shape[0], dtype=torch.bool)
        T_t_start = start * TELEM_60HZ_PER_VIDEO_FRAME
        T_t_end = end * TELEM_60HZ_PER_VIDEO_FRAME
        target_T_t = self.seq_len * TELEM_60HZ_PER_VIDEO_FRAME
        x, m = _window_frames(feats, fmask, T_t_start, T_t_end, target_T_t)
        if self.normalize and "telem_60hz" in self.stats and self.stats["telem_60hz"]:
            mean = self.stats["telem_60hz"]["mean"]
            std = self.stats["telem_60hz"]["std"]
            x = (x - mean) / std.clamp(min=1e-8)
        return x, m

    def _load_km_event(self, stem: str, start: int, end: int) -> Tuple[torch.Tensor, torch.Tensor]:
        obj = self._load_pt_raw(stem, "km_event")
        type_ids = obj["event_type_ids"]
        t_rel = obj["t_rel"]
        bin_id_full = obj["bin_id"]
        sel = (bin_id_full >= start) & (bin_id_full < end)
        type_ids = type_ids[sel]
        t_rel = t_rel[sel] - start * VIDEO_FRAME_DT
        bin_id = bin_id_full[sel] - start

        T_k = int(type_ids.numel())
        if T_k > self.max_events_per_window:
            type_ids = type_ids[: self.max_events_per_window]
            t_rel = t_rel[: self.max_events_per_window]
            bin_id = bin_id[: self.max_events_per_window]
            T_k = self.max_events_per_window

        dt = torch.zeros_like(t_rel)
        if T_k > 1:
            dt[1:] = t_rel[1:] - t_rel[:-1]

        x = torch.stack(
            [type_ids.float(), t_rel.float(), dt.float(), bin_id.float()], dim=-1
        )  # [T_k, 4]
        m = torch.ones(T_k, dtype=torch.bool)
        return x, m

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        stem, start, base_len = self._index[idx]
        end = start + self.seq_len if base_len >= start + self.seq_len else base_len

        x: Dict[str, torch.Tensor] = {}
        mod_mask: Dict[str, torch.Tensor] = {}
        for m in self.modalities:
            if m == "km_event":
                xm, mm = self._load_km_event(stem, start, end)
            elif m == "telem_60hz":
                xm, mm = self._load_telem_60hz(stem, start, end)
            else:
                xm, mm = self._load_video(stem, m, start, end)
            x[m] = xm
            mod_mask[m] = mm

        # labels
        y_out: Dict[str, torch.Tensor] = {}
        y_mask_out: Dict[str, torch.Tensor] = {}
        item = self.labels_seq[stem]
        for task in self.task_names:
            dtype = self._get_task_dtype(task)
            values = torch.tensor(item[task]["values"], dtype=dtype)
            if dtype != torch.long and values.ndim == 1:
                values = values.unsqueeze(-1)
            lmask = torch.tensor(item[task].get("mask", [True] * values.shape[0]), dtype=torch.bool)
            values = values[:base_len]
            lmask = lmask[:base_len]
            yv, ym = _window_frames(values, lmask, start, end, self.seq_len)
            y_out[task] = yv
            y_mask_out[task] = ym

        return {
            "x": x,
            "mod_mask": mod_mask,
            "y": y_out,
            "mask": y_mask_out,
            "meta": {"stem": stem, "split": self.split, "start": int(start)},
        }


def _collate_mixed_rate(samples: List[Dict[str, Any]]) -> Dict[str, Any]:
    if not samples:
        return {}
    modalities = list(samples[0]["x"].keys())
    task_names = list(samples[0]["y"].keys())

    x: Dict[str, torch.Tensor] = {}
    mod_mask: Dict[str, torch.Tensor] = {}
    for m in modalities:
        if m == "km_event":
            # 变长：pad 到 batch max T_k
            maxT = max(int(s["x"][m].shape[0]) for s in samples)
            maxT = max(maxT, 1)  # 避免 0（encoder 不能处理空序列）
            feat_dim = samples[0]["x"][m].shape[1] if samples[0]["x"][m].ndim == 2 else 4
            xs = torch.zeros(len(samples), maxT, feat_dim, dtype=torch.float32)
            ms = torch.zeros(len(samples), maxT, dtype=torch.bool)
            for i, s in enumerate(samples):
                xi = s["x"][m]
                mi = s["mod_mask"][m]
                t = xi.shape[0]
                if t > 0:
                    xs[i, :t] = xi
                    ms[i, :t] = mi
            x[m] = xs
            mod_mask[m] = ms
        else:
            x[m] = torch.stack([s["x"][m] for s in samples], dim=0)
            mod_mask[m] = torch.stack([s["mod_mask"][m] for s in samples], dim=0)

    y = {t: torch.stack([s["y"][t] for s in samples], dim=0) for t in task_names}
    mask = {t: torch.stack([s["mask"][t] for s in samples], dim=0) for t in task_names}
    meta = {k: [s["meta"][k] for s in samples] for k in samples[0]["meta"]}
    return {"x": x, "mod_mask": mod_mask, "y": y, "mask": mask, "meta": meta}


@DATAMODULES.register("amucs_seq_mixed_rate")
class AMuCSSeqMixedRateDataModule(BaseDataModule):
    def __init__(self, cfg):
        _g = cfg.get if isinstance(cfg, dict) else (lambda k, d=None: getattr(cfg, k, d))

        self.modalities = _g("modalities", ["video"])
        data_root = Path(_g("data_root", "data/features/amucs"))
        labels_seq_path = Path(_g("labels_seq_path", "labels/arousal_state_trend_seq.json"))
        split_path = Path(_g("split_path", "data/splits/multimodal_split.json"))
        self.batch_size = _g("batch_size", 8)
        self.num_workers = _g("num_workers", 0)
        self._pin_memory = self.num_workers > 0
        self._persistent_workers = self.num_workers > 0

        seq_len_video_frames = int(_g("seq_len_video_frames", _g("seq_len", 600)))
        train_stride = _g("train_stride_video_frames", _g("train_stride", None))
        val_stride = _g("val_stride_video_frames", _g("val_stride", None))
        test_stride = _g("test_stride_video_frames", _g("test_stride", None))
        include_tail_window = _g("include_tail_window", True)
        normalize = _g("normalize", True)
        stats_dir = _g("stats_dir", None)
        label_dtype = _g("label_dtype", "long")
        task_label_dtypes = _g("task_label_dtypes", None)
        task_names = _g("task_names", ["state", "trend"])
        temporal_split_ratios = _g("temporal_split_ratios", None)
        modality_dir_map = _g("modality_dir_map", None)
        max_events_per_window = _g("max_events_per_window", 2048)

        common = dict(
            modalities=self.modalities,
            data_root=data_root,
            labels_seq_path=labels_seq_path,
            split_path=split_path,
            seq_len_video_frames=seq_len_video_frames,
            task_names=task_names,
            include_tail_window=include_tail_window,
            normalize=normalize,
            stats_dir=stats_dir,
            label_dtype=label_dtype,
            task_label_dtypes=task_label_dtypes,
            temporal_split_ratios=temporal_split_ratios,
            modality_dir_map=modality_dir_map,
            max_events_per_window=max_events_per_window,
        )

        self._train_ds = AMuCSSeqMixedRateDataset(split="train", stride_video_frames=train_stride, **common)
        self._val_ds = AMuCSSeqMixedRateDataset(split="val", stride_video_frames=val_stride, **common)
        try:
            self._test_ds = AMuCSSeqMixedRateDataset(split="test", stride_video_frames=test_stride, **common)
            if len(self._test_ds) == 0:
                self._test_ds = None
        except Exception:
            self._test_ds = None

    def train_dataloader(self) -> DataLoader:
        return DataLoader(
            self._train_ds, batch_size=self.batch_size, shuffle=True,
            num_workers=self.num_workers, collate_fn=_collate_mixed_rate,
            pin_memory=self._pin_memory, persistent_workers=self._persistent_workers,
        )

    def val_dataloader(self) -> DataLoader:
        return DataLoader(
            self._val_ds, batch_size=self.batch_size, shuffle=False,
            num_workers=self.num_workers, collate_fn=_collate_mixed_rate,
            pin_memory=self._pin_memory, persistent_workers=self._persistent_workers,
        )

    def test_dataloader(self) -> Optional[DataLoader]:
        if self._test_ds is None or len(self._test_ds) == 0:
            return None
        return DataLoader(
            self._test_ds, batch_size=self.batch_size, shuffle=False,
            num_workers=self.num_workers, collate_fn=_collate_mixed_rate,
            pin_memory=self._pin_memory, persistent_workers=self._persistent_workers,
        )
