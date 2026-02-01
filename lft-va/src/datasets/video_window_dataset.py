"""
Video window dataset for face features.

视频面部特征的切窗数据集。
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Tuple

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


class VideoWindDataset(Dataset):
    """
    Windowed dataset for video (face) feature sequences.

    视频特征序列的切窗数据集。

    Parameters
    ----------
    data_dir : Path
        Directory containing *.pt feature files.
    labels_path : Path
        Path to labels JSON file.
    split_path : Path
        Path to split JSON file.
    split : str
        "train" or "val".
    win_len : int
        Window length.
    stride : int
        Window stride.
    """

    def __init__(
        self,
        data_dir: str | Path | None = None,
        labels_path: str | Path | None = None,
        split_path: str | Path | None = None,
        split: str = "train",
        normalize_inputs: bool = True,
        input_stats_path: str | Path | None = None,
        win_len: int = 100,
        stride: int = 50,
        cache: bool = True,
    ):
        self.win_len = int(win_len)
        self.stride = int(stride)
        self.cache = bool(cache)

        lft_root = Path(__file__).resolve().parents[2]
        if data_dir is None:
            data_dir = lft_root / "data" / "features" / "amucs" / "video"
        if labels_path is None:
            labels_path = lft_root / "data" / "labels_arousal.json"
        if split_path is None:
            split_path = lft_root / "data" / "splits" / "video_arousal_split.json"
        if input_stats_path is None:
            input_stats_path = lft_root / "data" / "video_input_stats.json"

        self.data_dir = Path(data_dir)
        self.labels_path = Path(labels_path)
        self.split_path = Path(split_path)
        self.split = split
        self.normalize_inputs = bool(normalize_inputs)
        self.input_stats_path = Path(input_stats_path)

        files = sorted(self.data_dir.glob("*.pt"))
        if not files:
            raise FileNotFoundError(f"No .pt files found in {self.data_dir}")

        with self.labels_path.open("r", encoding="utf-8") as f:
            self.labels = json.load(f)
        with self.split_path.open("r", encoding="utf-8") as f:
            split_obj = json.load(f)

        split_stems = list(split_obj.get(self.split, []))
        file_map = {p.stem: p for p in files}

        self.files = []
        self.stems: List[str] = []
        for stem in split_stems:
            if stem in self.labels and stem in file_map:
                self.stems.append(stem)
                self.files.append(file_map[stem])

        if not self.files:
            raise RuntimeError("No valid stems after applying labels and split.")

        self._cache: Dict[int, Dict] = {}
        self._index: List[Tuple[int, int]] = []
        self._build_index()

        self.input_mean: torch.Tensor | None = None
        self.input_std: torch.Tensor | None = None
        if self.normalize_inputs:
            self._load_or_compute_stats()

    def _load_file(self, file_idx: int) -> Dict:
        if self.cache and file_idx in self._cache:
            return self._cache[file_idx]
        data = torch.load(self.files[file_idx], map_location="cpu", weights_only=False)
        if self.cache:
            self._cache[file_idx] = data
        return data

    def _load_or_compute_stats(self):
        if self.split == "train":
            sum_x, sum_x2, count = None, None, 0
            for i in range(len(self.files)):
                feats = self._load_file(i)["features"].float()
                if sum_x is None:
                    sum_x = feats.sum(dim=0)
                    sum_x2 = (feats ** 2).sum(dim=0)
                else:
                    sum_x += feats.sum(dim=0)
                    sum_x2 += (feats ** 2).sum(dim=0)
                count += feats.shape[0]
            self.input_mean = sum_x / max(count, 1)
            self.input_std = torch.sqrt(torch.clamp(sum_x2 / max(count, 1) - self.input_mean ** 2, min=1e-8))
            stats = {"mean": self.input_mean.tolist(), "std": self.input_std.tolist()}
            self.input_stats_path.parent.mkdir(parents=True, exist_ok=True)
            with self.input_stats_path.open("w", encoding="utf-8") as f:
                json.dump(stats, f)
        else:
            if not self.input_stats_path.exists():
                raise FileNotFoundError(f"Input stats not found: {self.input_stats_path}")
            stats = json.loads(self.input_stats_path.read_text())
            self.input_mean = torch.tensor(stats["mean"], dtype=torch.float32)
            self.input_std = torch.tensor(stats["std"], dtype=torch.float32)

    def _build_index(self):
        for i in range(len(self.files)):
            T = self._load_file(i)["features"].shape[0]
            for start in range(0, max(0, T - self.win_len) + 1, self.stride):
                self._index.append((i, start))

    def __len__(self) -> int:
        return len(self._index)

    def __getitem__(self, idx: int) -> Dict:
        file_idx, start = self._index[idx]
        data = self._load_file(file_idx)
        feats = data["features"][start:start + self.win_len].float()

        # Pad if needed
        if feats.shape[0] < self.win_len:
            pad = torch.zeros(self.win_len - feats.shape[0], feats.shape[1])
            feats = torch.cat([feats, pad], dim=0)
            mask = torch.cat([torch.ones(feats.shape[0]), torch.zeros(self.win_len - feats.shape[0])]).bool()
        else:
            mask = torch.ones(self.win_len, dtype=torch.bool)

        if self.normalize_inputs and self.input_mean is not None:
            feats = (feats - self.input_mean) / self.input_std

        stem = self.stems[file_idx]
        y = _parse_label_value(self.labels[stem])

        return {"video": feats, "video_mask": mask, "y": y, "stem": stem}
