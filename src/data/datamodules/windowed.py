"""
Windowed single-modality DataModules (video or KM).

Reproduces legacy sliding-window datasets while emitting the unified Batch schema.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import torch
from torch.utils.data import DataLoader, Dataset

from src.core.registry import DATAMODULES
from src.core.types import BaseDataModule


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


class VideoWindowDataset(Dataset):
    """
    Sliding-window dataset for video features.
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

        root = Path(__file__).resolve().parents[3]
        if data_dir is None:
            data_dir = root / "data" / "features" / "amucs" / "video"
        if labels_path is None:
            labels_path = Path("G:/我的云端硬盘/AmuCS_experiment/labels/labels_arousal.json")
        if split_path is None:
            split_path = root / "data" / "splits" / "video_arousal_split.json"
        if input_stats_path is None:
            input_stats_path = root / "data" / "features" / "amucs" / "video_input_stats.json"

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
                json.dump(stats, f, ensure_ascii=False, indent=2)
        else:
            if not self.input_stats_path.exists():
                raise FileNotFoundError(f"Input stats not found: {self.input_stats_path}")
            stats = json.loads(self.input_stats_path.read_text(encoding="utf-8"))
            self.input_mean = torch.tensor(stats["mean"], dtype=torch.float32)
            self.input_std = torch.tensor(stats["std"], dtype=torch.float32)

    def _build_index(self):
        for i in range(len(self.files)):
            T = self._load_file(i)["features"].shape[0]
            for start in range(0, max(0, T - self.win_len) + 1, self.stride):
                self._index.append((i, start))

    def __len__(self) -> int:
        return len(self._index)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        file_idx, start = self._index[idx]
        data = self._load_file(file_idx)
        feats = data["features"][start:start + self.win_len].float()

        # Pad if needed
        if feats.shape[0] < self.win_len:
            valid_len = feats.shape[0]
            pad = torch.zeros(self.win_len - valid_len, feats.shape[1])
            feats = torch.cat([feats, pad], dim=0)
            mask = torch.cat([torch.ones(valid_len), torch.zeros(self.win_len - valid_len)]).bool()
        else:
            mask = torch.ones(self.win_len, dtype=torch.bool)

        if self.normalize_inputs and self.input_mean is not None:
            feats = (feats - self.input_mean) / self.input_std

        stem = self.stems[file_idx]
        y = _parse_label_value(self.labels[stem])

        return {
            "x": {"video": feats},
            "mask": {"video": mask},
            "y": y,
            "meta": {"stem": stem, "start": start},
        }


class KMWindowDataset(Dataset):
    """
    Sliding-window dataset for KM features.
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
        self.win_len = int(win_len)
        self.stride = int(stride)
        self.cache = bool(cache)

        if self.win_len <= 0 or self.stride <= 0:
            raise ValueError("win_len and stride must be positive.")

        root = Path(__file__).resolve().parents[3]
        if data_dir is None:
            data_dir = root / "data" / "features" / "amucs" / "km"
        if labels_path is None:
            labels_path = Path("G:/我的云端硬盘/AmuCS_experiment/labels/labels_arousal.json")
        if split_path is None:
            split_path = root / "data" / "splits" / "km_arousal_split.json"
        if input_stats_path is None:
            input_stats_path = root / "data" / "features" / "amucs" / "km_input_stats.json"

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
        for stem in split_stems:
            if stem not in self.labels:
                continue
            if stem not in file_map:
                continue
            self.stems.append(stem)
            self.files.append(file_map[stem])

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
                self.input_stats_path.parent.mkdir(parents=True, exist_ok=True)
                with self.input_stats_path.open("w", encoding="utf-8") as f:
                    json.dump(stats, f, ensure_ascii=False, indent=2)
            else:
                if not self.input_stats_path.exists():
                    raise FileNotFoundError(f"Input stats not found: {self.input_stats_path}")
                stats = json.loads(self.input_stats_path.read_text(encoding="utf-8"))
                self.input_mean = torch.tensor(stats["mean"], dtype=torch.float32)
                self.input_std = torch.tensor(stats["std"], dtype=torch.float32)

    def _load_file(self, file_idx: int) -> Dict:
        if self.cache and file_idx in self._cache:
            return self._cache[file_idx]
        data = torch.load(self.files[file_idx], map_location="cpu", weights_only=False)
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
        for i in range(len(self.files)):
            data = self._load_file(i)
            feats = data["features"]
            T = int(feats.shape[0])
            max_start = T - self.win_len
            if max_start < 0:
                continue
            for start in range(0, max_start + 1, self.stride):
                self._index.append((i, start))

    def __len__(self) -> int:
        return len(self._index)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        file_idx, start = self._index[idx]
        data = self._load_file(file_idx)
        feats = data["features"]
        mask = data["mask"]

        km = feats[start: start + self.win_len].to(torch.float32)
        km_mask = mask[start: start + self.win_len].to(torch.bool)
        if self.normalize_inputs and self.input_mean is not None and self.input_std is not None:
            km = (km - self.input_mean) / self.input_std

        stem = self.stems[file_idx]
        y = _parse_label_value(self.labels[stem])

        return {
            "x": {"km": km},
            "mask": {"km": km_mask},
            "y": y,
            "meta": {"stem": stem, "start": start},
        }


def _collate_batch(samples: List[Dict[str, Any]]) -> Dict[str, Any]:
    if not samples:
        return {}

    modalities = list(samples[0]["x"].keys())
    x: Dict[str, torch.Tensor] = {}
    mask_dict: Dict[str, torch.Tensor] = {}

    for mod in modalities:
        x[mod] = torch.stack([s["x"][mod] for s in samples])
        mask_dict[mod] = torch.stack([s["mask"][mod] for s in samples])

    y = torch.stack([s["y"] for s in samples])
    meta = {k: [s["meta"][k] for s in samples] for k in samples[0]["meta"]}
    return {"x": x, "mask": mask_dict, "y": y, "meta": meta}


@DATAMODULES.register("video_window")
class VideoWindowDataModule(BaseDataModule):
    """
    DataModule for sliding-window video features.
    """

    def __init__(self, cfg):
        if isinstance(cfg, dict):
            _g = cfg.get
        else:
            _g = lambda k, d=None: getattr(cfg, k, d)

        self.batch_size = _g("batch_size", 8)
        self.num_workers = _g("num_workers", 0)

        self._train_ds = VideoWindowDataset(
            data_dir=_g("data_dir", None),
            labels_path=_g("labels_path", None),
            split_path=_g("split_path", None),
            split="train",
            normalize_inputs=_g("normalize", True),
            input_stats_path=_g("input_stats_path", None),
            win_len=_g("win_len", 100),
            stride=_g("stride", 50),
            cache=_g("cache", True),
        )
        self._val_ds = VideoWindowDataset(
            data_dir=_g("data_dir", None),
            labels_path=_g("labels_path", None),
            split_path=_g("split_path", None),
            split="val",
            normalize_inputs=_g("normalize", True),
            input_stats_path=_g("input_stats_path", None),
            win_len=_g("win_len", 100),
            stride=_g("stride", 50),
            cache=_g("cache", True),
        )

    def train_dataloader(self) -> DataLoader:
        return DataLoader(
            self._train_ds,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            collate_fn=_collate_batch,
        )

    def val_dataloader(self) -> DataLoader:
        return DataLoader(
            self._val_ds,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            collate_fn=_collate_batch,
        )


@DATAMODULES.register("km_window")
class KMWindowDataModule(BaseDataModule):
    """
    DataModule for sliding-window KM features.
    """

    def __init__(self, cfg):
        if isinstance(cfg, dict):
            _g = cfg.get
        else:
            _g = lambda k, d=None: getattr(cfg, k, d)

        self.batch_size = _g("batch_size", 8)
        self.num_workers = _g("num_workers", 0)

        self._train_ds = KMWindowDataset(
            data_dir=_g("data_dir", None),
            labels_path=_g("labels_path", None),
            split_path=_g("split_path", None),
            split="train",
            normalize_inputs=_g("normalize", True),
            input_stats_path=_g("input_stats_path", None),
            win_len=_g("win_len", 300),
            stride=_g("stride", 150),
            cache=_g("cache", True),
        )
        self._val_ds = KMWindowDataset(
            data_dir=_g("data_dir", None),
            labels_path=_g("labels_path", None),
            split_path=_g("split_path", None),
            split="val",
            normalize_inputs=_g("normalize", True),
            input_stats_path=_g("input_stats_path", None),
            win_len=_g("win_len", 300),
            stride=_g("stride", 150),
            cache=_g("cache", True),
        )

    def train_dataloader(self) -> DataLoader:
        return DataLoader(
            self._train_ds,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            collate_fn=_collate_batch,
        )

    def val_dataloader(self) -> DataLoader:
        return DataLoader(
            self._val_ds,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            collate_fn=_collate_batch,
        )
