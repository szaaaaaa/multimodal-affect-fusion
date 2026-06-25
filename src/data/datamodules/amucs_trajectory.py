"""AMuCS trajectory DataModule.

Task: given a past 20s multimodal window, predict the next 10s of ranktrace
(51 points @ 5Hz).

Sampling: uniform stride on anchors, weighted by cumulative event proximity
  w(t) = Σ_i exp(-|t - e_i| / tau_s)

The weighted sampling is applied to the training split via a
`WeightedRandomSampler`; val/test use the same uniform anchors without
weighting (so evaluation sees the natural distribution).
"""

from __future__ import annotations

import json
import math
import warnings
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset, RandomSampler, WeightedRandomSampler

from src.core.registry import DATAMODULES
from src.core.types import BaseDataModule


class AMuCSTrajectoryDataset(Dataset):
    """Per-anchor samples for future-trajectory regression.

    A sample is defined by (stem, t_anchor_idx) where t_anchor_idx is the index
    into the 5Hz grid. Input spans [t-pre_s, t), output spans [t, t+post_s].
    """

    def __init__(
        self,
        modalities: List[str],
        data_root: Path,
        labels_seq_path: Path,
        events_path: Path,
        split_path: Path,
        split: str,
        pre_s: float = 20.0,
        post_s: float = 10.0,
        dt: float = 0.2,
        stride_s: float = 2.0,
        normalize: bool = True,
        stats_dir: Optional[Path] = None,
        modality_dir_map: Optional[Dict[str, str]] = None,
        delta_sigma: Optional[float] = None,
        within_stem_split: bool = False,
        within_stem_ratios: Tuple[float, float, float] = (0.7, 0.15, 0.15),
        target_mode: str = "trajectory",
        target_offset_s: float = 5.0,
    ):
        self.modalities = modalities
        self.data_root = Path(data_root)
        self.split = split
        self.dt = float(dt)
        self.pre_steps = int(round(pre_s / self.dt))   # 20s @ 5Hz = 100
        # post includes endpoint t+post_s → 51 points for post_s=10, dt=0.2
        self.post_steps = int(round(post_s / self.dt)) + 1
        self.stride_steps = max(1, int(round(stride_s / self.dt)))
        self.normalize = normalize
        self.stats_dir = Path(stats_dir) if stats_dir else self.data_root

        with Path(labels_seq_path).open("r", encoding="utf-8-sig") as f:
            self.labels_seq = json.load(f)
        with Path(events_path).open("r", encoding="utf-8-sig") as f:
            self.events = json.load(f)
        with Path(split_path).open("r", encoding="utf-8-sig") as f:
            split_stems = set(json.load(f).get(split, []))

        _dir_map = dict(modality_dir_map or {})
        self.mod_dirs: Dict[str, Path] = {
            m: self.data_root / _dir_map.get(m, m) for m in self.modalities
        }
        stem_sets = [{p.stem for p in self.mod_dirs[m].glob("*.pt")} for m in self.modalities]
        common = set(self.labels_seq.keys()) & set(self.events.keys()) & split_stems
        for s in stem_sets:
            common &= s
        self.stems = sorted(common)
        if not self.stems:
            warnings.warn(
                f"[AMuCSTrajectoryDataset] empty split={split}, modalities={modalities}"
            )

        self.stats: Dict[str, Dict[str, torch.Tensor]] = {}
        if normalize:
            for m in self.modalities:
                self.stats[m] = self._load_stats(m)

        self._cache_feats: Dict[Tuple[str, str], torch.Tensor] = {}
        self._cache_masks: Dict[Tuple[str, str], torch.Tensor] = {}
        self._cache_labels: Dict[str, torch.Tensor] = {}
        self._cache_label_masks: Dict[str, torch.Tensor] = {}

        self.within_stem_split = bool(within_stem_split)
        r = tuple(within_stem_ratios)
        if len(r) != 3 or abs(sum(r) - 1.0) > 1e-6:
            raise ValueError(f"within_stem_ratios must be 3 floats summing to 1.0, got {within_stem_ratios}")
        self.within_stem_ratios = r

        if target_mode not in {"trajectory", "single_point"}:
            raise ValueError(f"target_mode must be 'trajectory' or 'single_point', got {target_mode!r}")
        self.target_mode = target_mode
        self.target_offset_s = float(target_offset_s)
        self.target_offset_steps = int(round(self.target_offset_s / self.dt))
        if self.target_mode == "single_point" and not (0 < self.target_offset_steps < self.post_steps):
            raise ValueError(
                f"target_offset_s={self.target_offset_s} must map to a step in (0, {self.post_steps - 1}] "
                f"given dt={self.dt} and post_s"
            )

        self._anchors: List[Tuple[str, int]] = self._build_anchors()
        self.delta_sigma: float = float(delta_sigma) if delta_sigma is not None else 1.0

    def _load_stats(self, modality: str) -> Dict[str, torch.Tensor]:
        p = self.stats_dir / f"{modality}_input_stats.json"
        if not p.exists():
            return {}
        s = json.loads(p.read_text(encoding="utf-8"))
        return {
            "mean": torch.tensor(s["mean"], dtype=torch.float32),
            "std": torch.tensor(s["std"], dtype=torch.float32),
        }

    def _load_pt(self, stem: str, modality: str) -> Tuple[torch.Tensor, torch.Tensor]:
        key = (stem, modality)
        if key in self._cache_feats:
            return self._cache_feats[key], self._cache_masks[key]
        obj = torch.load(self.mod_dirs[modality] / f"{stem}.pt", map_location="cpu", weights_only=False)
        if isinstance(obj, dict):
            feats = obj["features"].float()
            mask = obj["mask"].bool() if "mask" in obj else torch.ones(feats.shape[0], dtype=torch.bool)
        else:
            feats = obj.float()
            mask = torch.ones(feats.shape[0], dtype=torch.bool)
        self._cache_feats[key] = feats
        self._cache_masks[key] = mask
        return feats, mask

    def _load_label(self, stem: str) -> Tuple[torch.Tensor, torch.Tensor]:
        if stem in self._cache_labels:
            return self._cache_labels[stem], self._cache_label_masks[stem]
        item = self.labels_seq[stem]
        y = torch.tensor(item["values"], dtype=torch.float32)
        if y.ndim == 1:
            y = y.unsqueeze(-1)  # [T, 1]
        m = torch.tensor(item.get("mask", [True] * y.shape[0]), dtype=torch.bool)
        self._cache_labels[stem] = y
        self._cache_label_masks[stem] = m
        return y, m

    def _stem_length(self, stem: str) -> int:
        lengths = [len(self.labels_seq[stem]["values"])]
        for m in self.modalities:
            feats, _ = self._load_pt(stem, m)
            lengths.append(int(feats.shape[0]))
        return min(lengths)

    def _build_anchors(self) -> List[Tuple[str, int]]:
        """Uniform anchors over valid range per stem.

        If within_stem_split is True, each stem's [i_min, i_max] timeline is
        further partitioned into train/val/test segments by `within_stem_ratios`,
        with a `post_steps` gap between segments so output windows never overlap
        (= no label leak across splits, even when the same stems are in all splits).
        """
        anchors: List[Tuple[str, int]] = []
        r_tr, r_va, _ = self.within_stem_ratios
        for stem in self.stems:
            L = self._stem_length(stem)
            i_min = self.pre_steps
            i_max = L - self.post_steps
            if i_max <= i_min:
                continue

            if self.within_stem_split:
                span = i_max - i_min
                train_end = i_min + int(span * r_tr)
                val_start = train_end + self.post_steps
                val_end = val_start + int(span * r_va)
                test_start = val_end + self.post_steps
                if self.split == "train":
                    lo, hi = i_min, train_end
                elif self.split == "val":
                    lo, hi = val_start, val_end
                else:  # test
                    lo, hi = test_start, i_max
                if hi <= lo:
                    continue
            else:
                lo, hi = i_min, i_max

            for i in range(lo, hi + 1, self.stride_steps):
                anchors.append((stem, i))
        if not anchors:
            warnings.warn(f"[AMuCSTrajectoryDataset] no anchors for split={self.split}")
        return anchors

    # --- public: anchor list + event weights ---------------------------------

    def anchors(self) -> List[Tuple[str, int]]:
        return self._anchors

    def compute_delta_sigma(self) -> float:
        """Global σ of Δy across all train anchors.

        For target_mode='trajectory':    Δy = y[i:i+post] - y[i]   (pooled over 51 points × N anchors)
        For target_mode='single_point':  Δy = y[i+k] - y[i]        (N scalars)

        Called on the training dataset only; val/test inherit the value via
        `set_delta_sigma` to keep normalisation consistent (no test-time leakage).
        """
        deltas: List[np.ndarray] = []
        for stem, i in self._anchors:
            y_all, _ = self._load_label(stem)
            anchor_val = float(y_all[i].squeeze().item())
            if self.target_mode == "single_point":
                k = self.target_offset_steps
                delta = float(y_all[i + k].squeeze().item()) - anchor_val
                deltas.append(np.array([delta], dtype=np.float64))
            else:
                y_seg = y_all[i : i + self.post_steps].squeeze(-1).numpy()
                deltas.append(y_seg - anchor_val)
        if not deltas:
            return 1.0
        all_deltas = np.concatenate(deltas)
        sigma = float(np.std(all_deltas))
        return sigma if sigma > 1e-9 else 1.0

    def set_delta_sigma(self, sigma: float) -> None:
        self.delta_sigma = float(sigma) if sigma > 1e-9 else 1.0

    def compute_event_weights(self, tau_s: float = 10.0) -> torch.Tensor:
        """Per-anchor weight = Σ_i exp(-|t - e_i| / tau_s), events of all types."""
        tau_steps = tau_s / self.dt
        w = torch.zeros(len(self._anchors), dtype=torch.float64)
        for k, (stem, i) in enumerate(self._anchors):
            ev_by_type = self.events.get(stem, {})
            all_events_s = [t for evs in ev_by_type.values() for t in evs]
            if not all_events_s:
                continue
            ev_steps = torch.tensor(all_events_s, dtype=torch.float64) / self.dt
            d = (ev_steps - float(i)).abs()
            w[k] = torch.exp(-d / tau_steps).sum().item()
        # avoid all-zero weights (e.g. empty events): fall back to uniform
        if w.sum() == 0:
            w[:] = 1.0
        return w

    def __len__(self) -> int:
        return len(self._anchors)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        stem, i = self._anchors[idx]
        start = i - self.pre_steps
        end = i  # input window is [start, end), length pre_steps

        x: Dict[str, torch.Tensor] = {}
        mask: Dict[str, torch.Tensor] = {}
        for m in self.modalities:
            feats, feat_mask = self._load_pt(stem, m)
            feat_win = feats[start:end]
            mask_win = feat_mask[start:end]
            if feat_win.shape[0] < self.pre_steps:
                pad_shape = (self.pre_steps - feat_win.shape[0], *feat_win.shape[1:])
                feat_win = torch.cat([feat_win, torch.zeros(pad_shape, dtype=feat_win.dtype)], dim=0)
                mask_win = torch.cat([mask_win, torch.zeros(self.pre_steps - mask_win.shape[0], dtype=torch.bool)], dim=0)
            if self.normalize and m in self.stats and self.stats[m]:
                mean = self.stats[m]["mean"]
                std = self.stats[m]["std"]
                feat_win = (feat_win - mean) / std.clamp(min=1e-8)
            x[m] = feat_win
            mask[m] = mask_win

        y_all, y_mask_all = self._load_label(stem)
        anchor_val = y_all[i].squeeze().detach()
        if self.target_mode == "single_point":
            k = self.target_offset_steps
            y_win = (y_all[i + k : i + k + 1, 0] - anchor_val) / self.delta_sigma  # [1]
            y_mask_win = y_mask_all[i + k : i + k + 1]                              # [1]
        else:
            y_seg = y_all[i : i + self.post_steps].squeeze(-1)   # [51]
            y_win = (y_seg - anchor_val) / self.delta_sigma      # [51]
            y_mask_win = y_mask_all[i : i + self.post_steps]
            if y_win.shape[0] < self.post_steps:
                pad = self.post_steps - y_win.shape[0]
                y_win = torch.cat([y_win, torch.zeros(pad)], dim=0)
                y_mask_win = torch.cat([y_mask_win, torch.zeros(pad, dtype=torch.bool)], dim=0)

        return {
            "x": x,
            "mask": mask,
            "y": y_win,                     # [post_steps]
            "y_mask": y_mask_win,           # [post_steps]
            "meta": {"stem": stem, "split": self.split, "t_idx": int(i)},
        }


def _collate(samples: List[Dict[str, Any]]) -> Dict[str, Any]:
    if not samples:
        return {}
    modalities = list(samples[0]["x"].keys())
    x = {m: torch.stack([s["x"][m] for s in samples], dim=0) for m in modalities}
    mask = {m: torch.stack([s["mask"][m] for s in samples], dim=0) for m in modalities}
    y = torch.stack([s["y"] for s in samples], dim=0)
    y_mask = torch.stack([s["y_mask"] for s in samples], dim=0)
    meta = {k: [s["meta"][k] for s in samples] for k in samples[0]["meta"]}
    return {"x": x, "mask": mask, "y": y, "y_mask": y_mask, "meta": meta}


@DATAMODULES.register("amucs_trajectory")
class AMuCSTrajectoryDataModule(BaseDataModule):
    """DataModule for future-trajectory regression with event-weighted sampling."""

    def __init__(self, cfg):
        _g = cfg.get if isinstance(cfg, dict) else (lambda k, d=None: getattr(cfg, k, d))

        self.modalities = _g("modalities", ["video", "km", "telem"])
        data_root = Path(_g("data_root", "features"))
        labels_seq_path = Path(_g("labels_seq_path", "labels/labels_arousal_seq.json"))
        events_path = Path(_g("events_path", "labels/events_index.json"))
        split_path = Path(_g("split_path", "splits/session_tvt.json"))

        self.batch_size = int(_g("batch_size", 32))
        self.num_workers = int(_g("num_workers", 0))
        self._pin = self.num_workers > 0
        self._persistent = self.num_workers > 0

        pre_s = float(_g("pre_s", 20.0))
        post_s = float(_g("post_s", 10.0))
        dt = float(_g("dt", 0.2))
        train_stride_s = float(_g("train_stride_s", 2.0))
        val_stride_s = float(_g("val_stride_s", 2.0))
        test_stride_s = float(_g("test_stride_s", 1.0))
        self.tau_s = float(_g("tau_s", 10.0))
        self.num_train_samples = _g("num_train_samples", None)  # int or None (=len dataset)
        self.event_weighted = bool(_g("event_weighted", True))
        self.within_stem_split = bool(_g("within_stem_split", False))
        self.within_stem_ratios = tuple(_g("within_stem_ratios", [0.7, 0.15, 0.15]))
        self.target_mode = str(_g("target_mode", "trajectory"))
        self.target_offset_s = float(_g("target_offset_s", 5.0))

        normalize = _g("normalize", True)
        stats_dir = _g("stats_dir", None)
        modality_dir_map = _g("modality_dir_map", None)

        common = dict(
            modalities=self.modalities,
            data_root=data_root,
            labels_seq_path=labels_seq_path,
            events_path=events_path,
            split_path=split_path,
            pre_s=pre_s,
            post_s=post_s,
            dt=dt,
            normalize=normalize,
            stats_dir=stats_dir,
            modality_dir_map=modality_dir_map,
            within_stem_split=self.within_stem_split,
            within_stem_ratios=self.within_stem_ratios,
            target_mode=self.target_mode,
            target_offset_s=self.target_offset_s,
        )

        self._train_ds = AMuCSTrajectoryDataset(split="train", stride_s=train_stride_s, **common)
        # Compute Δσ on training split, share with val/test (no test-time leakage).
        self.delta_sigma = self._train_ds.compute_delta_sigma()
        self._train_ds.set_delta_sigma(self.delta_sigma)
        print(f"[AMuCSTrajectoryDataModule] train-set Δσ = {self.delta_sigma:.4f}")

        self._val_ds = AMuCSTrajectoryDataset(
            split="val", stride_s=val_stride_s, delta_sigma=self.delta_sigma, **common,
        )
        try:
            self._test_ds = AMuCSTrajectoryDataset(
                split="test", stride_s=test_stride_s, delta_sigma=self.delta_sigma, **common,
            )
            if len(self._test_ds) == 0:
                self._test_ds = None
        except Exception:
            self._test_ds = None

        n = len(self._train_ds)
        n_samples = int(self.num_train_samples) if self.num_train_samples is not None else n
        if self.event_weighted:
            weights = self._train_ds.compute_event_weights(tau_s=self.tau_s)
            self._train_sampler = WeightedRandomSampler(
                weights=weights.double(),
                num_samples=n_samples,
                replacement=True,
            )
            print(f"[AMuCSTrajectoryDataModule] train sampler = event-weighted (tau_s={self.tau_s})")
        else:
            self._train_sampler = RandomSampler(
                self._train_ds, replacement=True, num_samples=n_samples,
            )
            print(f"[AMuCSTrajectoryDataModule] train sampler = uniform (event_weighted=false)")

    def train_dataloader(self) -> DataLoader:
        return DataLoader(
            self._train_ds,
            batch_size=self.batch_size,
            sampler=self._train_sampler,
            num_workers=self.num_workers,
            collate_fn=_collate,
            pin_memory=self._pin,
            persistent_workers=self._persistent,
        )

    def val_dataloader(self) -> DataLoader:
        return DataLoader(
            self._val_ds,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            collate_fn=_collate,
            pin_memory=self._pin,
            persistent_workers=self._persistent,
        )

    def test_dataloader(self) -> Optional[DataLoader]:
        if self._test_ds is None or len(self._test_ds) == 0:
            return None
        return DataLoader(
            self._test_ds,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            collate_fn=_collate,
            pin_memory=self._pin,
            persistent_workers=self._persistent,
        )
