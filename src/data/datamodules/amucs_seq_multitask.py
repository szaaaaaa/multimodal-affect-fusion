"""
AMuCS sequence DataModule for multitask temporal classification.

Expected label schema:
{
  "<stem>": {
    "state": {"values": [...], "mask": [...]},
    "trend": {"values": [...], "mask": [...]}
  }
}
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


def _window_with_shared_start(
    x: torch.Tensor,
    mask: torch.Tensor,
    start: int,
    end: int,
    seq_len: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Slice [start:end] and right-pad to seq_len."""
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


class AMuCSSeqMultitaskDataset(Dataset):
    """Aligned multimodal features + multitask sequence labels."""

    def __init__(
        self,
        modalities: List[str],
        data_root: Path,
        labels_seq_path: Path,
        split_path: Path,
        split: str,
        seq_len: int,
        task_names: List[str],
        stride: Optional[int] = None,
        include_tail_window: bool = True,
        normalize: bool = True,
        stats_dir: Optional[Path] = None,
        label_dtype: str = "long",
        task_label_dtypes: Optional[Dict[str, str]] = None,
    ):
        self.modalities = modalities
        self.task_names = list(task_names)
        self.label_dtype = str(label_dtype).lower()
        self.task_label_dtypes = {
            str(k): str(v).lower()
            for k, v in dict(task_label_dtypes or {}).items()
        }
        self.data_root = Path(data_root)
        self.split = split
        self.seq_len = int(seq_len)
        self.stride = int(stride) if stride is not None else None
        self.include_tail_window = bool(include_tail_window)
        self.normalize = normalize
        self.stats_dir = Path(stats_dir) if stats_dir else self.data_root

        with Path(labels_seq_path).open("r", encoding="utf-8-sig") as f:
            self.labels_seq = json.load(f)
        with Path(split_path).open("r", encoding="utf-8-sig") as f:
            split_stems = json.load(f).get(split, [])

        self.mod_dirs: Dict[str, Path] = {m: self.data_root / m for m in self.modalities}
        stem_sets = [{p.stem for p in self.mod_dirs[m].glob("*.pt")} for m in self.modalities]
        common = set(self.labels_seq.keys()) & set(split_stems)
        for s in stem_sets:
            common &= s
        self.stems = sorted(common)
        if not self.stems:
            warnings.warn(
                f"[AMuCSSeqMultitaskDataset] No common stems for split={split}, "
                f"modalities={modalities}. Dataset will be empty."
            )

        self.stats: Dict[str, Dict[str, torch.Tensor]] = {}
        if normalize:
            for m in self.modalities:
                self.stats[m] = self._load_stats(m)

        self._stem_base_len: Dict[str, int] = self._compute_base_lengths()
        self._index: List[Tuple[str, int, int]] = self._build_index()

        self._cache_feats: Dict[str, torch.Tensor] = {}
        self._cache_feat_masks: Dict[str, torch.Tensor] = {}
        self._cache_labels: Dict[str, Dict[str, torch.Tensor]] = {}
        self._cache_label_masks: Dict[str, Dict[str, torch.Tensor]] = {}

    def _get_task_dtype(self, task: str) -> torch.dtype:
        task_dtype_name = self.task_label_dtypes.get(task, self.label_dtype)
        if task_dtype_name == "long":
            return torch.long
        if task_dtype_name in {"float", "float32"}:
            return torch.float32
        raise ValueError(
            f"Unsupported dtype for task '{task}': {task_dtype_name}. "
            "Use 'long' or 'float'."
        )

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
        if feat_key in self._cache_feats and mask_key in self._cache_feat_masks:
            return self._cache_feats[feat_key], self._cache_feat_masks[mask_key]

        obj = torch.load(self.mod_dirs[modality] / f"{stem}.pt", map_location="cpu", weights_only=False)
        if isinstance(obj, dict):
            feats = obj["features"].float()
            feat_mask = obj.get("mask")
            feat_mask = feat_mask.bool() if feat_mask is not None else torch.ones(feats.shape[0], dtype=torch.bool)
        else:
            feats = obj.float()
            feat_mask = torch.ones(feats.shape[0], dtype=torch.bool)

        self._cache_feats[feat_key] = feats
        self._cache_feat_masks[mask_key] = feat_mask
        return feats, feat_mask

    def _load_seq_label(self, stem: str) -> tuple[Dict[str, torch.Tensor], Dict[str, torch.Tensor]]:
        if stem in self._cache_labels and stem in self._cache_label_masks:
            return self._cache_labels[stem], self._cache_label_masks[stem]

        item = self.labels_seq[stem]
        y_dict: Dict[str, torch.Tensor] = {}
        y_mask_dict: Dict[str, torch.Tensor] = {}
        for task in self.task_names:
            if task not in item:
                raise KeyError(f"Missing task '{task}' for stem={stem}")
            task_item = item[task]
            task_dtype = self._get_task_dtype(task)
            values = torch.tensor(task_item["values"], dtype=task_dtype)
            if task_dtype != torch.long and values.ndim == 1:
                values = values.unsqueeze(-1)
            mask = torch.tensor(
                task_item.get("mask", [True] * values.shape[0]),
                dtype=torch.bool,
            )
            if mask.shape[0] != values.shape[0]:
                raise ValueError(
                    f"Label length mismatch for stem={stem}, task={task}: "
                    f"values={values.shape[0]}, mask={mask.shape[0]}"
                )
            y_dict[task] = values
            y_mask_dict[task] = mask

        self._cache_labels[stem] = y_dict
        self._cache_label_masks[stem] = y_mask_dict
        return y_dict, y_mask_dict

    def _load_feat_len(self, stem: str, modality: str) -> int:
        obj = torch.load(self.mod_dirs[modality] / f"{stem}.pt", map_location="cpu", weights_only=False)
        feats = obj["features"] if isinstance(obj, dict) else obj
        return int(feats.shape[0])

    def _compute_base_lengths(self) -> Dict[str, int]:
        base_lens: Dict[str, int] = {}
        for stem in self.stems:
            item = self.labels_seq[stem]
            lengths = []
            for task in self.task_names:
                if task not in item:
                    lengths = []
                    break
                lengths.append(len(item[task]["values"]))
            if not lengths:
                continue
            for m in self.modalities:
                try:
                    lengths.append(self._load_feat_len(stem, m))
                except Exception:
                    lengths = []
                    break
            if lengths:
                base_len = min(lengths)
                if base_len > 0:
                    base_lens[stem] = int(base_len)
        return base_lens

    def _build_index(self) -> List[Tuple[str, int, int]]:
        index: List[Tuple[str, int, int]] = []
        use_sliding = self.stride is not None and self.stride > 0

        for stem in self.stems:
            base_len = self._stem_base_len.get(stem, 0)
            if base_len <= 0:
                continue

            if not use_sliding:
                start = (base_len - self.seq_len) // 2 if base_len >= self.seq_len else 0
                index.append((stem, int(start), int(base_len)))
                continue

            if base_len <= self.seq_len:
                index.append((stem, 0, int(base_len)))
                continue

            assert self.stride is not None
            max_start = base_len - self.seq_len
            starts = list(range(0, max_start + 1, self.stride))
            if self.include_tail_window and starts and starts[-1] != max_start:
                starts.append(max_start)
            for start in starts:
                index.append((stem, int(start), int(base_len)))

        if not index:
            warnings.warn(
                f"[AMuCSSeqMultitaskDataset] No windows generated for split={self.split}. "
                "Check seq_len/stride or data availability."
            )
        return index

    def __len__(self) -> int:
        return len(self._index)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        stem, start, base_len = self._index[idx]
        end = start + self.seq_len if base_len >= self.seq_len else base_len

        x: Dict[str, torch.Tensor] = {}
        mod_mask: Dict[str, torch.Tensor] = {}
        for m in self.modalities:
            feats, feat_mask = self._load_pt(stem, m)
            feats = feats[:base_len]
            feat_mask = feat_mask[:base_len]
            feat_win, mask_win = _window_with_shared_start(feats, feat_mask, start, end, self.seq_len)
            if self.normalize and m in self.stats and self.stats[m]:
                mean = self.stats[m]["mean"]
                std = self.stats[m]["std"]
                feat_win = (feat_win - mean) / std.clamp(min=1e-8)
            x[m] = feat_win
            mod_mask[m] = mask_win

        y_dict, y_mask_dict = self._load_seq_label(stem)
        y_out: Dict[str, torch.Tensor] = {}
        task_mask: Dict[str, torch.Tensor] = {}
        for task in self.task_names:
            y_seq = y_dict[task][:base_len]
            y_m = y_mask_dict[task][:base_len]
            y_win, y_m_win = _window_with_shared_start(y_seq, y_m, start, end, self.seq_len)
            y_out[task] = y_win
            task_mask[task] = y_m_win

        return {
            "x": x,
            "mod_mask": mod_mask,  # modality masks for encoders/fusion
            "y": y_out,             # task labels, each [T]
            "mask": task_mask,      # task label masks, each [T]
            "meta": {"stem": stem, "split": self.split, "start": int(start)},
        }


def _collate_batch_multitask(samples: List[Dict[str, Any]]) -> Dict[str, Any]:
    if not samples:
        return {}

    modalities = list(samples[0]["x"].keys())
    task_names = list(samples[0]["y"].keys())

    x = {m: torch.stack([s["x"][m] for s in samples], dim=0) for m in modalities}
    mod_mask = {m: torch.stack([s["mod_mask"][m] for s in samples], dim=0) for m in modalities}
    y = {t: torch.stack([s["y"][t] for s in samples], dim=0) for t in task_names}
    mask = {t: torch.stack([s["mask"][t] for s in samples], dim=0) for t in task_names}
    meta = {k: [s["meta"][k] for s in samples] for k in samples[0]["meta"]}

    return {"x": x, "mod_mask": mod_mask, "y": y, "mask": mask, "meta": meta}


@DATAMODULES.register("amucs_seq_multitask")
class AMuCSSeqMultitaskDataModule(BaseDataModule):
    """DataModule for multitask sequence classification on aligned AMuCS features."""

    def __init__(self, cfg):
        if isinstance(cfg, dict):
            _g = cfg.get
        else:
            _g = lambda k, d=None: getattr(cfg, k, d)

        self.modalities = _g("modalities", ["video"])
        data_root = Path(_g("data_root", "data/features/amucs"))
        labels_seq_path = Path(_g("labels_seq_path", "labels/arousal_state_trend_seq.json"))
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
        label_dtype = _g("label_dtype", "long")
        task_label_dtypes = _g("task_label_dtypes", None)
        task_names = _g("task_names", ["state", "trend"])

        common_kwargs = dict(
            modalities=self.modalities,
            data_root=data_root,
            labels_seq_path=labels_seq_path,
            split_path=split_path,
            seq_len=seq_len,
            task_names=task_names,
            include_tail_window=include_tail_window,
            normalize=normalize,
            stats_dir=stats_dir,
            label_dtype=label_dtype,
            task_label_dtypes=task_label_dtypes,
        )

        self._train_ds = AMuCSSeqMultitaskDataset(split="train", stride=train_stride, **common_kwargs)
        self._val_ds = AMuCSSeqMultitaskDataset(split="val", stride=val_stride, **common_kwargs)
        try:
            self._test_ds = AMuCSSeqMultitaskDataset(split="test", stride=test_stride, **common_kwargs)
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
            collate_fn=_collate_batch_multitask,
            pin_memory=self._pin_memory,
            persistent_workers=self._persistent_workers,
        )

    def val_dataloader(self) -> DataLoader:
        return DataLoader(
            self._val_ds,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            collate_fn=_collate_batch_multitask,
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
            collate_fn=_collate_batch_multitask,
            pin_memory=self._pin_memory,
            persistent_workers=self._persistent_workers,
        )
