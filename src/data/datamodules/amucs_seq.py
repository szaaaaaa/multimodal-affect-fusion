"""
AMuCS sequence DataModule for strict temporal regression.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional

import torch
from torch.utils.data import DataLoader, Dataset

from src.core.registry import DATAMODULES
from src.core.types import BaseDataModule


class AMuCSSeqDataset(Dataset):
    """
    Sequence dataset for aligned multimodal features + aligned arousal labels.
    """

    def __init__(
        self,
        modalities: List[str],
        data_root: Path,
        labels_seq_path: Path,
        split_path: Path,
        split: str,
        seq_len: int,
        normalize: bool = True,
        stats_dir: Optional[Path] = None,
    ):
        self.modalities = modalities
        self.data_root = Path(data_root)
        self.split = split
        self.seq_len = int(seq_len)
        self.normalize = normalize
        self.stats_dir = Path(stats_dir) if stats_dir else self.data_root

        with Path(labels_seq_path).open("r", encoding="utf-8") as f:
            self.labels_seq = json.load(f)
        with Path(split_path).open("r", encoding="utf-8") as f:
            split_stems = json.load(f).get(split, [])

        self.mod_dirs: Dict[str, Path] = {m: self.data_root / m for m in self.modalities}
        stem_sets = []
        for m in self.modalities:
            stem_sets.append({p.stem for p in self.mod_dirs[m].glob("*.pt")})

        common = set(self.labels_seq.keys()) & set(split_stems)
        for s in stem_sets:
            common &= s
        self.stems = sorted(common)

        self.stats: Dict[str, Dict[str, torch.Tensor]] = {}
        if normalize:
            for m in self.modalities:
                self.stats[m] = self._load_stats(m)

        self._cache_feats: Dict[str, torch.Tensor] = {}
        self._cache_masks: Dict[str, torch.Tensor] = {}

    def _load_stats(self, modality: str) -> Dict[str, torch.Tensor]:
        stats_path = self.stats_dir / f"{modality}_input_stats.json"
        if not stats_path.exists():
            return {}
        stats = json.loads(stats_path.read_text(encoding="utf-8"))
        return {
            "mean": torch.tensor(stats["mean"], dtype=torch.float32),
            "std": torch.tensor(stats["std"], dtype=torch.float32),
        }

    def _load_pt(self, stem: str, modality: str) -> tuple[torch.Tensor, torch.Tensor]:
        feat_key = f"{stem}_{modality}_feat"
        mask_key = f"{stem}_{modality}_mask"
        if feat_key in self._cache_feats and mask_key in self._cache_masks:
            return self._cache_feats[feat_key], self._cache_masks[mask_key]

        obj = torch.load(self.mod_dirs[modality] / f"{stem}.pt", map_location="cpu", weights_only=False)
        if isinstance(obj, dict):
            feats = obj["features"].float()
            if "mask" in obj:
                mask = obj["mask"].bool()
            else:
                mask = torch.ones(feats.shape[0], dtype=torch.bool)
        else:
            feats = obj.float()
            mask = torch.ones(feats.shape[0], dtype=torch.bool)

        self._cache_feats[feat_key] = feats
        self._cache_masks[mask_key] = mask
        return feats, mask

    def _load_seq_label(self, stem: str) -> tuple[torch.Tensor, torch.Tensor]:
        item = self.labels_seq[stem]
        values = torch.tensor(item["values"], dtype=torch.float32)
        if values.ndim == 1:
            values = values.unsqueeze(-1)
        if "mask" in item:
            y_mask = torch.tensor(item["mask"], dtype=torch.bool)
        else:
            y_mask = torch.ones(values.shape[0], dtype=torch.bool)
        return values, y_mask

    @staticmethod
    def _window_with_shared_start(
        x: torch.Tensor,
        mask: torch.Tensor,
        start: int,
        end: int,
        seq_len: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        x_win = x[start:end]
        m_win = mask[start:end]
        cur_len = x_win.shape[0]
        if cur_len < seq_len:
            pad_shape = (seq_len - cur_len, *x_win.shape[1:])
            x_pad = torch.zeros(pad_shape, dtype=x_win.dtype)
            m_pad = torch.zeros(seq_len - cur_len, dtype=torch.bool)
            x_win = torch.cat([x_win, x_pad], dim=0)
            m_win = torch.cat([m_win, m_pad], dim=0)
        return x_win, m_win

    def __len__(self) -> int:
        return len(self.stems)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        stem = self.stems[idx]

        mod_feats: Dict[str, torch.Tensor] = {}
        mod_masks: Dict[str, torch.Tensor] = {}
        lengths = []
        for m in self.modalities:
            feats, feat_mask = self._load_pt(stem, m)
            mod_feats[m] = feats
            mod_masks[m] = feat_mask
            lengths.append(feats.shape[0])

        y_seq, y_mask = self._load_seq_label(stem)
        lengths.append(y_seq.shape[0])
        base_len = min(lengths)

        if base_len >= self.seq_len:
            start = (base_len - self.seq_len) // 2
            end = start + self.seq_len
        else:
            start = 0
            end = base_len

        x: Dict[str, torch.Tensor] = {}
        mask: Dict[str, torch.Tensor] = {}
        for m in self.modalities:
            feats = mod_feats[m][:base_len]
            feat_mask = mod_masks[m][:base_len]
            feat_win, mask_win = self._window_with_shared_start(feats, feat_mask, start, end, self.seq_len)
            if self.normalize and m in self.stats and self.stats[m]:
                mean = self.stats[m]["mean"]
                std = self.stats[m]["std"]
                feat_win = (feat_win - mean) / std.clamp(min=1e-8)
            x[m] = feat_win
            mask[m] = mask_win

        y_seq = y_seq[:base_len]
        y_mask = y_mask[:base_len]
        y_win, y_mask_win = self._window_with_shared_start(y_seq, y_mask, start, end, self.seq_len)

        return {
            "x": x,
            "mask": mask,
            "y": y_win,
            "y_mask": y_mask_win,
            "meta": {"stem": stem, "split": self.split},
        }


def _collate_batch(samples: List[Dict[str, Any]]) -> Dict[str, Any]:
    if not samples:
        return {}

    modalities = list(samples[0]["x"].keys())
    x = {m: torch.stack([s["x"][m] for s in samples], dim=0) for m in modalities}
    mask = {m: torch.stack([s["mask"][m] for s in samples], dim=0) for m in modalities}
    y = torch.stack([s["y"] for s in samples], dim=0)
    y_mask = torch.stack([s["y_mask"] for s in samples], dim=0)
    meta = {k: [s["meta"][k] for s in samples] for k in samples[0]["meta"]}
    return {"x": x, "mask": mask, "y": y, "y_mask": y_mask, "meta": meta}


@DATAMODULES.register("amucs_seq")
class AMuCSSeqDataModule(BaseDataModule):
    """
    DataModule for strict temporal regression with aligned labels.
    """

    def __init__(self, cfg):
        if isinstance(cfg, dict):
            _g = cfg.get
        else:
            _g = lambda k, d=None: getattr(cfg, k, d)

        self.modalities = _g("modalities", ["video", "km"])
        data_root = Path(_g("data_root", "data/features_aligned/amucs"))
        labels_seq_path = Path(_g("labels_seq_path", "data/labels_arousal_seq.json"))
        split_path = Path(_g("split_path", "data/splits/multimodal_split.json"))
        self.batch_size = _g("batch_size", 8)
        self.num_workers = _g("num_workers", 0)
        seq_len = int(_g("seq_len", 300))
        normalize = _g("normalize", True)
        stats_dir = _g("stats_dir", None)

        common_kwargs = dict(
            modalities=self.modalities,
            data_root=data_root,
            labels_seq_path=labels_seq_path,
            split_path=split_path,
            seq_len=seq_len,
            normalize=normalize,
            stats_dir=stats_dir,
        )

        self._train_ds = AMuCSSeqDataset(split="train", **common_kwargs)
        self._val_ds = AMuCSSeqDataset(split="val", **common_kwargs)
        try:
            self._test_ds: Optional[AMuCSSeqDataset] = AMuCSSeqDataset(split="test", **common_kwargs)
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
