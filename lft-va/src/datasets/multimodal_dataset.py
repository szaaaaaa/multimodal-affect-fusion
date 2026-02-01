"""
Multimodal dataset combining video and KM features.

多模态数据集，组合视频和键鼠特征。
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Optional

import torch
from torch.utils.data import Dataset


def _parse_label_value(value) -> torch.Tensor:
    if isinstance(value, dict):
        if "valence" in value and "arousal" in value:
            return torch.tensor([float(value["valence"]), float(value["arousal"])], dtype=torch.float32)
        if "v" in value and "a" in value:
            return torch.tensor([float(value["v"]), float(value["a"])], dtype=torch.float32)
        if "va" in value:
            va = value["va"]
            if isinstance(va, (list, tuple)) and len(va) == 2:
                return torch.tensor([float(va[0]), float(va[1])], dtype=torch.float32)
    if isinstance(value, (list, tuple)) and len(value) == 2:
        return torch.tensor([float(value[0]), float(value[1])], dtype=torch.float32)
    return torch.tensor([float(value)], dtype=torch.float32)


class MultimodalDataset(Dataset):
    """
    Dataset combining video and KM modalities.

    组合视频和键鼠模态的数据集。

    Aligns video and KM features by session stem. Each sample contains
    both modalities for the same session.

    Parameters
    ----------
    video_dir : Path
        Directory with video .pt files.
    km_dir : Path
        Directory with KM .pt files.
    labels_path : Path
        Path to labels JSON.
    split_path : Path
        Path to split JSON.
    split : str
        "train" or "val".
    video_win_len : int
        Video window length.
    km_win_len : int
        KM window length.
    """

    def __init__(
        self,
        video_dir: str | Path | None = None,
        km_dir: str | Path | None = None,
        labels_path: str | Path | None = None,
        split_path: str | Path | None = None,
        split: str = "train",
        video_win_len: int = 100,
        km_win_len: int = 300,
        normalize_video: bool = True,
        normalize_km: bool = True,
        video_stats_path: str | Path | None = None,
        km_stats_path: str | Path | None = None,
    ):
        lft_root = Path(__file__).resolve().parents[2]
        if video_dir is None:
            video_dir = lft_root / "data" / "features" / "amucs" / "video"
        if km_dir is None:
            km_dir = lft_root / "data" / "features" / "amucs" / "km"
        if labels_path is None:
            labels_path = lft_root / "data" / "labels_arousal.json"
        if split_path is None:
            split_path = lft_root / "data" / "splits" / "multimodal_split.json"
        if video_stats_path is None:
            video_stats_path = lft_root / "data" / "video_input_stats.json"
        if km_stats_path is None:
            km_stats_path = lft_root / "data" / "km_input_stats.json"

        self.video_dir = Path(video_dir)
        self.km_dir = Path(km_dir)
        self.video_win_len = video_win_len
        self.km_win_len = km_win_len
        self.normalize_video = normalize_video
        self.normalize_km = normalize_km
        self.split = split

        # Load labels and split
        with Path(labels_path).open("r") as f:
            self.labels = json.load(f)
        with Path(split_path).open("r") as f:
            split_stems = json.load(f).get(split, [])

        # Find common stems with both modalities
        video_stems = {p.stem for p in self.video_dir.glob("*.pt")}
        km_stems = {p.stem for p in self.km_dir.glob("*.pt")}
        common = video_stems & km_stems & set(self.labels.keys()) & set(split_stems)

        self.stems = sorted(common)
        if not self.stems:
            raise RuntimeError("No common stems found across video, km, labels, and split.")

        # Load normalization stats
        self.video_mean, self.video_std = self._load_stats(video_stats_path) if normalize_video else (None, None)
        self.km_mean, self.km_std = self._load_stats(km_stats_path) if normalize_km else (None, None)

        self._cache: Dict[str, Dict] = {}

    def _load_stats(self, path: Path) -> tuple:
        path = Path(path)
        if not path.exists():
            return None, None
        stats = json.loads(path.read_text())
        return torch.tensor(stats["mean"], dtype=torch.float32), torch.tensor(stats["std"], dtype=torch.float32)

    def _load_pt(self, stem: str, modality: str) -> torch.Tensor:
        key = f"{stem}_{modality}"
        if key in self._cache:
            return self._cache[key]
        path = (self.video_dir if modality == "video" else self.km_dir) / f"{stem}.pt"
        data = torch.load(path, map_location="cpu", weights_only=False)
        feats = data["features"].float()
        self._cache[key] = feats
        return feats

    def _window_and_pad(self, feats: torch.Tensor, win_len: int) -> tuple:
        T, D = feats.shape
        if T >= win_len:
            # Center crop
            start = (T - win_len) // 2
            feats = feats[start:start + win_len]
            mask = torch.ones(win_len, dtype=torch.bool)
        else:
            # Pad
            pad = torch.zeros(win_len - T, D)
            mask = torch.cat([torch.ones(T), torch.zeros(win_len - T)]).bool()
            feats = torch.cat([feats, pad], dim=0)
        return feats, mask

    def __len__(self) -> int:
        return len(self.stems)

    def __getitem__(self, idx: int) -> Dict:
        stem = self.stems[idx]

        # Video
        video_feats = self._load_pt(stem, "video")
        video, video_mask = self._window_and_pad(video_feats, self.video_win_len)
        if self.video_mean is not None:
            video = (video - self.video_mean) / self.video_std

        # KM
        km_feats = self._load_pt(stem, "km")
        km, km_mask = self._window_and_pad(km_feats, self.km_win_len)
        if self.km_mean is not None:
            km = (km - self.km_mean) / self.km_std

        # Label
        y = _parse_label_value(self.labels[stem])

        return {
            "video": video,
            "video_mask": video_mask,
            "km": km,
            "km_mask": km_mask,
            "y": y,
            "stem": stem,
        }
