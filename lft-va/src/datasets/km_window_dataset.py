"""
KM window dataset for AMuCS features.

用于 AMuCS 键鼠特征切窗的数据集（中英文注释，NumPy 风格）。
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Tuple
import json

import torch
from torch.utils.data import Dataset


class KMWindDataset(Dataset):
    """
    Windowed dataset for KM feature sequences.

    键鼠特征序列的切窗数据集。

    Parameters
    ----------
    data_dir : str or pathlib.Path, optional
        Directory containing *.pt feature files.
        / 包含 *.pt 特征文件的目录。
    labels_path : str or pathlib.Path, optional
        Path to labels_arousal.json.
        / 标签文件路径。
    split_path : str or pathlib.Path, optional
        Path to km_arousal_split.json.
        / 切分文件路径。
    split : str, optional
        Split name: "train" or "val".
        / 切分名称。
    win_len : int, optional
        Window length (L).
        / 窗口长度。
    stride : int, optional
        Window stride.
        / 窗口步长。
    cache : bool, optional
        Cache per-file torch.load results.
        / 是否缓存每个文件的加载结果。
    """
    def __init__(
        self,
        data_dir: str | Path | None = None,
        labels_path: str | Path | None = None,
        split_path: str | Path | None = None,
        split: str = "train",
        normalize_inputs: bool = True,
        input_stats_path: str | Path | None = None,
        win_len: int = 300,
        stride: int = 150,
        cache: bool = True,
    ):
        """
        Build index for all windows.

        构建所有窗口的索引。
        """
        self.win_len = int(win_len)
        self.stride = int(stride)
        self.cache = bool(cache)

        if self.win_len <= 0 or self.stride <= 0:
            raise ValueError("win_len and stride must be positive.")

        lft_root = Path(__file__).resolve().parents[3]
        if data_dir is None:
            data_dir = lft_root / "data" / "features" / "amucs" / "km"
        if labels_path is None:
            labels_path = lft_root / "data" / "labels_arousal.json"
        if split_path is None:
            split_path = lft_root / "data" / "splits" / "km_arousal_split.json"
        if input_stats_path is None:
            input_stats_path = lft_root / "data" / "km_input_stats.json"
        self.data_dir = Path(data_dir)
        self.labels_path = Path(labels_path)
        self.split_path = Path(split_path)
        self.split = split
        self.normalize_inputs = bool(normalize_inputs)
        self.input_stats_path = Path(input_stats_path)

        files = sorted(self.data_dir.glob("*.pt"))
        if not files:
            raise FileNotFoundError(f"No .pt files found in {self.data_dir}")

        if not self.labels_path.exists():
            raise FileNotFoundError(f"labels_arousal.json not found: {self.labels_path}")
        if not self.split_path.exists():
            raise FileNotFoundError(f"km_arousal_split.json not found: {self.split_path}")

        with self.labels_path.open("r", encoding="utf-8") as f:
            self.labels = json.load(f)
        with self.split_path.open("r", encoding="utf-8") as f:
            split_obj = json.load(f)
        if self.split not in split_obj:
            raise KeyError(f"Split '{self.split}' not found in {self.split_path}")

        split_stems = list(split_obj[self.split])
        if not split_stems:
            raise RuntimeError(f"Split '{self.split}' is empty.")

        file_map = {p.stem: p for p in files}
        self.files = []
        self.stems: List[str] = []
        skipped_no_label = 0
        skipped_no_feat = 0
        for stem in split_stems:
            if stem not in self.labels:
                skipped_no_label += 1
                continue
            if stem not in file_map:
                skipped_no_feat += 1
                continue
            self.stems.append(stem)
            self.files.append(file_map[stem])

        if skipped_no_label or skipped_no_feat:
            print(f"[KMWindDataset] skipped_no_label={skipped_no_label}, skipped_no_feat={skipped_no_feat}")
        if not self.files:
            raise RuntimeError("No valid stems after applying labels and split.")

        self._cache: Dict[int, Dict] = {}
        self._index: List[Tuple[int, int]] = []
        self._build_index()

        if not self._index:
            raise RuntimeError("No windows could be generated with current win_len/stride.")

        self.input_mean: torch.Tensor | None = None
        self.input_std: torch.Tensor | None = None
        if self.normalize_inputs:
            if self.split == "train":
                self.input_mean, self.input_std = self._compute_input_stats()
                stats = {
                    "mean": self.input_mean.tolist(),
                    "std": self.input_std.tolist(),
                }
                with self.input_stats_path.open("w", encoding="utf-8") as f:
                    json.dump(stats, f, ensure_ascii=False, indent=2)
            else:
                if not self.input_stats_path.exists():
                    raise FileNotFoundError(f"Input stats not found: {self.input_stats_path}")
                stats = json.loads(self.input_stats_path.read_text(encoding="utf-8"))
                self.input_mean = torch.tensor(stats["mean"], dtype=torch.float32)
                self.input_std = torch.tensor(stats["std"], dtype=torch.float32)

    def _load_file(self, file_idx: int) -> Dict:
        """
        Load one feature file (with optional cache).

        加载单个特征文件（可选缓存）。
        """
        if self.cache and file_idx in self._cache:
            return self._cache[file_idx]
        data = torch.load(self.files[file_idx], map_location="cpu")
        if self.cache:
            self._cache[file_idx] = data
        return data

    def _compute_input_stats(self) -> tuple[torch.Tensor, torch.Tensor]:
        sum_x = None
        sum_x2 = None
        count = 0
        for i in range(len(self.files)):
            data = self._load_file(i)
            feats = data["features"].to(torch.float32)
            mask = data.get("mask", None)
            if mask is not None:
                feats = feats[mask]
            if sum_x is None:
                sum_x = feats.sum(dim=0)
                sum_x2 = (feats * feats).sum(dim=0)
            else:
                sum_x += feats.sum(dim=0)
                sum_x2 += (feats * feats).sum(dim=0)
            count += feats.shape[0]
        mean = sum_x / max(count, 1)
        var = sum_x2 / max(count, 1) - mean * mean
        std = torch.sqrt(torch.clamp(var, min=1e-8))
        return mean, std

    def _build_index(self) -> None:
        """
        Build (file_idx, start) index for all windows.

        构建所有窗口的 (file_idx, start) 索引。
        """
        for i, p in enumerate(self.files):
            data = self._load_file(i)
            feats = data["features"]
            T = int(feats.shape[0])
            max_start = T - self.win_len
            if max_start < 0:
                continue
            for start in range(0, max_start + 1, self.stride):
                self._index.append((i, start))

    def __len__(self) -> int:
        """
        Return number of windows.

        返回窗口总数。
        """
        return len(self._index)

    def __getitem__(self, idx: int) -> Dict:
        """
        Get one windowed sample.

        获取一个窗口样本。

        Returns
        -------
        dict
            {
              "km": Tensor[win_len, 15] float32,
              "km_mask": Tensor[win_len] bool,
              "y": Tensor[2] float32,
              "id": str
            }
            / 返回特征、mask、dummy 标签与样本 ID。
        """
        file_idx, start = self._index[idx]
        data = self._load_file(file_idx)
        feats = data["features"]
        mask = data["mask"]

        km = feats[start : start + self.win_len].to(torch.float32)
        km_mask = mask[start : start + self.win_len].to(torch.bool)
        if self.normalize_inputs and self.input_mean is not None and self.input_std is not None:
            km = (km - self.input_mean) / self.input_std

        stem = self.stems[file_idx]
        y_val = float(self.labels[stem])
        y = torch.tensor([y_val], dtype=torch.float32)

        return {
            "km": km,
            "km_mask": km_mask,
            "y": y,
            "stem": stem,
        }
