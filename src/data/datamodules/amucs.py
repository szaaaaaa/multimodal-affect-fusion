"""
AMuCS DataModule — produces unified Batch dicts for multimodal training.

AMuCS 数据模块 — 产出统一 Batch 字典，用于多模态训练。

Migrated from legacy/lft_va_src/datasets/multimodal_dataset.py.
Supports arbitrary modality subsets controlled by config.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional

import torch
from torch.utils.data import DataLoader, Dataset

from src.core.registry import DATAMODULES
from src.core.types import BaseDataModule


# ──────────────────────────────────────────────
# Label parsing (preserved from original)
# ──────────────────────────────────────────────

def _parse_label_value(value) -> torch.Tensor:
    """Parse flexible label formats into a tensor."""
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


# ──────────────────────────────────────────────
# Unified Dataset
# ──────────────────────────────────────────────

class AMuCSDataset(Dataset):
    """
    Unified AMuCS dataset supporting arbitrary modality subsets.

    统一 AMuCS 数据集，支持任意模态子集。

    Outputs the standardised Batch schema:
        {"x": {mod: Tensor}, "mask": {mod: Tensor}, "y": Tensor, "meta": {...}}
    """

    def __init__(
        self,
        modalities: List[str],
        data_root: Path,
        labels_path: Path,
        split_path: Path,
        split: str,
        win_lens: Dict[str, int],
        normalize: bool = True,
        stats_dir: Optional[Path] = None,
    ):
        self.modalities = modalities
        self.data_root = Path(data_root)
        self.split = split
        self.win_lens = win_lens
        self.normalize = normalize
        self.stats_dir = Path(stats_dir) if stats_dir else self.data_root

        # Load labels and split
        with Path(labels_path).open("r", encoding="utf-8") as f:
            self.labels = json.load(f)
        with Path(split_path).open("r", encoding="utf-8") as f:
            split_stems = json.load(f).get(split, [])

        # Find stems that exist across ALL requested modalities + labels + split
        stem_sets = []
        self.mod_dirs: Dict[str, Path] = {}
        for mod in modalities:
            mod_dir = self.data_root / mod
            self.mod_dirs[mod] = mod_dir
            if mod_dir.exists():
                stem_sets.append({p.stem for p in mod_dir.glob("*.pt")})
            else:
                stem_sets.append(set())

        common = set(self.labels.keys()) & set(split_stems)
        for s in stem_sets:
            common &= s

        self.stems = sorted(common)
        if not self.stems:
            import warnings
            warnings.warn(
                f"[AMuCSDataset] No common stems found for split={split}, "
                f"modalities={modalities}. Dataset will be empty."
            )

        # Normalization stats per modality
        self.stats: Dict[str, Dict[str, torch.Tensor]] = {}
        if normalize:
            for mod in modalities:
                self.stats[mod] = self._load_stats(mod)

        self._cache: Dict[str, torch.Tensor] = {}

    def _load_stats(self, modality: str) -> Dict[str, torch.Tensor]:
        """Load pre-computed mean/std for a modality."""
        stats_path = self.stats_dir / f"{modality}_input_stats.json"
        if stats_path.exists():
            stats = json.loads(stats_path.read_text(encoding="utf-8"))
            return {
                "mean": torch.tensor(stats["mean"], dtype=torch.float32),
                "std": torch.tensor(stats["std"], dtype=torch.float32),
            }
        return {}

    def _load_pt(self, stem: str, modality: str) -> torch.Tensor:
        cache_key = f"{stem}_{modality}"
        if cache_key in self._cache:
            return self._cache[cache_key]
        path = self.mod_dirs[modality] / f"{stem}.pt"
        data = torch.load(path, map_location="cpu", weights_only=False)
        feats = data["features"].float() if isinstance(data, dict) else data.float()
        self._cache[cache_key] = feats
        return feats

    def _window_and_pad(self, feats: torch.Tensor, win_len: int):
        """Center-crop or pad to fixed window length."""
        T, D = feats.shape
        if T >= win_len:
            start = (T - win_len) // 2
            feats = feats[start:start + win_len]
            mask = torch.ones(win_len, dtype=torch.bool)
        else:
            pad = torch.zeros(win_len - T, D)
            mask = torch.cat([torch.ones(T), torch.zeros(win_len - T)]).bool()
            feats = torch.cat([feats, pad], dim=0)
        return feats, mask

    def __len__(self) -> int:
        return len(self.stems)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        stem = self.stems[idx]

        x: Dict[str, torch.Tensor] = {}
        mask: Dict[str, torch.Tensor] = {}

        for mod in self.modalities:
            feats = self._load_pt(stem, mod)
            win_len = self.win_lens.get(mod, feats.shape[0])
            feat_win, feat_mask = self._window_and_pad(feats, win_len)

            # Normalize
            if self.normalize and mod in self.stats and self.stats[mod]:
                mean = self.stats[mod]["mean"]
                std = self.stats[mod]["std"]
                feat_win = (feat_win - mean) / std.clamp(min=1e-8)

            x[mod] = feat_win
            mask[mod] = feat_mask

        y = _parse_label_value(self.labels[stem])

        return {
            "x": x,
            "mask": mask,
            "y": y,
            "meta": {"stem": stem, "split": self.split},
        }


def _collate_batch(samples: List[Dict]) -> Dict[str, Any]:
    """
    Custom collate function for the unified Batch schema.

    将样本列表合并为 Batch 字典。
    """
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


# ──────────────────────────────────────────────
# DataModule (registered)
# ──────────────────────────────────────────────

@DATAMODULES.register("amucs")
class AMuCSDataModule(BaseDataModule):
    """
    AMuCS DataModule — reads config and builds train/val/test DataLoaders.

    Parameters (via cfg dict)
    -------------------------
    modalities : list[str]
    data_root : str
    labels_path : str
    split_path : str
    win_lens : dict  (modality -> int)
    batch_size : int
    num_workers : int
    normalize : bool
    stats_dir : str (optional)
    """

    def __init__(self, cfg):
        if isinstance(cfg, dict):
            _g = cfg.get
        else:
            _g = lambda k, d=None: getattr(cfg, k, d)

        self.modalities = _g("modalities", ["video", "km"])
        data_root = Path(_g("data_root", "data/features/amucs"))
        labels_path = Path(_g("labels_path", "data/labels_arousal.json"))
        split_path = Path(_g("split_path", "data/splits/multimodal_split.json"))
        self.batch_size = _g("batch_size", 8)
        self.num_workers = _g("num_workers", 0)
        normalize = _g("normalize", True)
        stats_dir = _g("stats_dir", None)

        # Window lengths per modality
        default_wins = {"video": 24, "km": 300}
        win_lens_cfg = _g("win_lens", {})
        self.win_lens = {mod: win_lens_cfg.get(mod, default_wins.get(mod, 200))
                         for mod in self.modalities}

        common_kwargs = dict(
            modalities=self.modalities,
            data_root=data_root,
            labels_path=labels_path,
            split_path=split_path,
            win_lens=self.win_lens,
            normalize=normalize,
            stats_dir=stats_dir,
        )

        self._train_ds = AMuCSDataset(split="train", **common_kwargs)
        self._val_ds = AMuCSDataset(split="val", **common_kwargs)

        # Test split is optional
        try:
            self._test_ds: Optional[AMuCSDataset] = AMuCSDataset(split="test", **common_kwargs)
            if len(self._test_ds) == 0:
                self._test_ds = None
        except Exception:
            self._test_ds = None

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

    def test_dataloader(self) -> Optional[DataLoader]:
        if self._test_ds is None or len(self._test_ds) == 0:
            return None
        return DataLoader(
            self._test_ds,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            collate_fn=_collate_batch,
        )

