"""
AMuCS sequence DataModule for strict temporal regression.
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
        stride: Optional[int] = None,
        include_tail_window: bool = True,
        normalize: bool = True,
        stats_dir: Optional[Path] = None,
        label_dtype: str = "float",
        temporal_split_ratios: Optional[Dict[str, List[float]]] = None,
        modality_dir_map: Optional[Dict[str, str]] = None,
    ):
        self.modalities = modalities
        self.label_dtype = torch.long if label_dtype == "long" else torch.float32
        self.data_root = Path(data_root)
        self.split = split
        self.seq_len = int(seq_len)
        self.stride = int(stride) if stride is not None else None
        self.include_tail_window = bool(include_tail_window)
        self.normalize = normalize
        self.stats_dir = Path(stats_dir) if stats_dir else self.data_root
        self.temporal_split_ratios = temporal_split_ratios

        # Use utf-8-sig to be robust to BOM-prefixed JSON files generated on Windows.
        with Path(labels_seq_path).open("r", encoding="utf-8-sig") as f:
            self.labels_seq = json.load(f)
        with Path(split_path).open("r", encoding="utf-8-sig") as f:
            split_stems = json.load(f).get(split, [])

        # modality_dir_map allows modality "video" to read from subdir "video_clip"
        _dir_map = dict(modality_dir_map or {})
        self.mod_dirs: Dict[str, Path] = {
            m: self.data_root / _dir_map.get(m, m) for m in self.modalities
        }
        stem_sets = []
        for m in self.modalities:
            stem_sets.append({p.stem for p in self.mod_dirs[m].glob("*.pt")})

        common = set(self.labels_seq.keys()) & set(split_stems)
        for s in stem_sets:
            common &= s
        self.stems = sorted(common)
        if not self.stems:
            warnings.warn(
                f"[AMuCSSeqDataset] No common stems found for split={split}, "
                f"modalities={modalities}. Dataset will be empty."
            )

        self.stats: Dict[str, Dict[str, torch.Tensor]] = {}
        if normalize:
            for m in self.modalities:
                self.stats[m] = self._load_stats(m)

        self._stem_base_len: Dict[str, int] = self._compute_base_lengths()
        self._index: List[Tuple[str, int, int]] = self._build_index()

        self._cache_feats: Dict[str, torch.Tensor] = {}
        self._cache_masks: Dict[str, torch.Tensor] = {}
        self._cache_labels: Dict[str, torch.Tensor] = {}
        self._cache_label_masks: Dict[str, torch.Tensor] = {}

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
        if stem in self._cache_labels and stem in self._cache_label_masks:
            return self._cache_labels[stem], self._cache_label_masks[stem]

        item = self.labels_seq[stem]
        values = torch.tensor(item["values"], dtype=self.label_dtype)
        if values.ndim == 1 and self.label_dtype != torch.long:
            values = values.unsqueeze(-1)
        if "mask" in item:
            y_mask = torch.tensor(item["mask"], dtype=torch.bool)
        else:
            y_mask = torch.ones(values.shape[0], dtype=torch.bool)
        self._cache_labels[stem] = values
        self._cache_label_masks[stem] = y_mask
        return values, y_mask

    def _load_feat_len(self, stem: str, modality: str) -> int:
        obj = torch.load(self.mod_dirs[modality] / f"{stem}.pt", map_location="cpu", weights_only=False)
        feats = obj["features"] if isinstance(obj, dict) else obj
        return int(feats.shape[0])

    def _compute_base_lengths(self) -> Dict[str, int]:
        base_lens: Dict[str, int] = {}
        for stem in self.stems:
            y_len = len(self.labels_seq[stem]["values"])
            lengths = [y_len]
            for m in self.modalities:
                try:
                    lengths.append(self._load_feat_len(stem, m))
                except Exception:
                    lengths = []
                    break
            if not lengths:
                continue
            base_len = min(lengths)
            if base_len > 0:
                base_lens[stem] = int(base_len)
        return base_lens

    def _get_temporal_range(self, base_len: int) -> Tuple[int, int]:
        """Return (range_start, range_end) for windowing based on temporal_split_ratios."""
        if not self.temporal_split_ratios or self.split not in self.temporal_split_ratios:
            return 0, base_len
        lo_ratio, hi_ratio = self.temporal_split_ratios[self.split]
        return int(base_len * lo_ratio), int(base_len * hi_ratio)

    def _build_index(self) -> List[Tuple[str, int, int]]:
        index: List[Tuple[str, int, int]] = []
        use_sliding = self.stride is not None and self.stride > 0

        for stem in self.stems:
            base_len = self._stem_base_len.get(stem, 0)
            if base_len <= 0:
                continue

            range_start, range_end = self._get_temporal_range(base_len)
            range_len = range_end - range_start
            if range_len <= 0:
                continue

            if not use_sliding:
                if range_len >= self.seq_len:
                    start = range_start + (range_len - self.seq_len) // 2
                else:
                    start = range_start
                index.append((stem, int(start), int(base_len)))
                continue

            if range_len <= self.seq_len:
                index.append((stem, int(range_start), int(base_len)))
                continue

            assert self.stride is not None
            max_start = range_end - self.seq_len
            starts = list(range(range_start, max_start + 1, self.stride))
            if self.include_tail_window and starts and starts[-1] != max_start:
                starts.append(max_start)
            for start in starts:
                index.append((stem, int(start), int(base_len)))

        if not index:
            warnings.warn(
                f"[AMuCSSeqDataset] No windows generated for split={self.split}. "
                "Check seq_len/stride or data availability."
            )
        return index

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
        return len(self._index)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        stem, start, base_len = self._index[idx]

        mod_feats: Dict[str, torch.Tensor] = {}
        mod_masks: Dict[str, torch.Tensor] = {}
        for m in self.modalities:
            feats, feat_mask = self._load_pt(stem, m)
            mod_feats[m] = feats
            mod_masks[m] = feat_mask

        y_seq, y_mask = self._load_seq_label(stem)

        if base_len >= self.seq_len:
            end = start + self.seq_len
        else:
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
            "meta": {"stem": stem, "split": self.split, "start": int(start)},
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
        data_root = Path(_g("data_root", "G:/我的云端硬盘/AmuCS_experiment/features/aligned/amucs_trial"))
        labels_seq_path = Path(_g("labels_seq_path", "G:/我的云端硬盘/AmuCS_experiment/labels/labels_arousal_seq.json"))
        split_path = Path(_g("split_path", "data/splits/multimodal_split.json"))
        self.batch_size = _g("batch_size", 8)
        self.num_workers = _g("num_workers", 0)
        self._pin_memory = self.num_workers > 0
        self._persistent_workers = self.num_workers > 0
        seq_len = int(_g("seq_len", 300))
        train_stride = _g("train_stride", None)
        val_stride = _g("val_stride", None)
        test_stride = _g("test_stride", None)
        include_tail_window = _g("include_tail_window", True)
        normalize = _g("normalize", True)
        stats_dir = _g("stats_dir", None)
        label_dtype = _g("label_dtype", "float")
        temporal_split_ratios = _g("temporal_split_ratios", None)
        modality_dir_map = _g("modality_dir_map", None)

        common_kwargs = dict(
            modalities=self.modalities,
            data_root=data_root,
            labels_seq_path=labels_seq_path,
            split_path=split_path,
            seq_len=seq_len,
            include_tail_window=include_tail_window,
            normalize=normalize,
            stats_dir=stats_dir,
            label_dtype=label_dtype,
            temporal_split_ratios=temporal_split_ratios,
            modality_dir_map=modality_dir_map,
        )

        self._train_ds = AMuCSSeqDataset(split="train", stride=train_stride, **common_kwargs)
        self._val_ds = AMuCSSeqDataset(split="val", stride=val_stride, **common_kwargs)
        try:
            self._test_ds = AMuCSSeqDataset(split="test", stride=test_stride, **common_kwargs)
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
            pin_memory=self._pin_memory,
            persistent_workers=self._persistent_workers,
        )

    def val_dataloader(self) -> DataLoader:
        return DataLoader(
            self._val_ds,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            collate_fn=_collate_batch,
            pin_memory=self._pin_memory,
            persistent_workers=self._persistent_workers,
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
            pin_memory=self._pin_memory,
            persistent_workers=self._persistent_workers,
        )
