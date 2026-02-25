#!/usr/bin/env python3
"""
Diagnostics for strict sequence-regression data pipeline.

Checks:
1) split/stem intersections across modalities and labels
2) label scale stats (min/max/mean/std) per split
3) sampled window time range + label variation
"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch


def _load_json(path: Path) -> dict:
    with path.open("r", encoding="utf-8-sig") as f:
        return json.load(f)


def _split_session(stem: str) -> str:
    return stem.split("_", 1)[0] if "_" in stem else stem


def _label_stats(labels: dict, stems: Sequence[str]) -> Optional[dict]:
    vals_all: List[float] = []
    valid_ratios: List[float] = []
    used = 0
    for stem in stems:
        item = labels.get(stem)
        if not item:
            continue
        vals = item.get("values", [])
        if not vals:
            continue
        vals_all.extend(vals)
        mask = item.get("mask")
        if mask:
            valid = sum(1 for x in mask if x)
            valid_ratios.append(valid / max(len(mask), 1))
        used += 1

    if not vals_all:
        return None

    arr = np.asarray(vals_all, dtype=np.float64)
    out = {
        "stems_used": used,
        "count_values": int(arr.size),
        "min": float(arr.min()),
        "max": float(arr.max()),
        "mean": float(arr.mean()),
        "std": float(arr.std()),
        "p1": float(np.percentile(arr, 1)),
        "p50": float(np.percentile(arr, 50)),
        "p99": float(np.percentile(arr, 99)),
    }
    if valid_ratios:
        vr = np.asarray(valid_ratios, dtype=np.float64)
        out["valid_mask_ratio_mean"] = float(vr.mean())
        out["valid_mask_ratio_min"] = float(vr.min())
        out["valid_mask_ratio_max"] = float(vr.max())
    return out


def _load_feat_len(feat_path: Path) -> int:
    obj = torch.load(feat_path, map_location="cpu", weights_only=False)
    feats = obj["features"] if isinstance(obj, dict) else obj
    return int(feats.shape[0])


def _base_len_for_stem(stem: str, labels: dict, mod_dirs: Dict[str, Path]) -> Optional[int]:
    item = labels.get(stem)
    if not item:
        return None
    y_len = len(item.get("values", []))
    if y_len <= 0:
        return None
    lens = [y_len]
    for _, d in mod_dirs.items():
        p = d / f"{stem}.pt"
        if not p.exists():
            return None
        lens.append(_load_feat_len(p))
    return min(lens)


def _build_window_starts(base_len: int, seq_len: int, stride: Optional[int], include_tail: bool) -> List[int]:
    if base_len <= 0:
        return []
    if not stride or stride <= 0:
        if base_len >= seq_len:
            return [(base_len - seq_len) // 2]
        return [0]
    if base_len <= seq_len:
        return [0]
    max_start = base_len - seq_len
    starts = list(range(0, max_start + 1, stride))
    if include_tail and starts and starts[-1] != max_start:
        starts.append(max_start)
    return starts


def _sample_window_checks(
    stems: Sequence[str],
    labels: dict,
    ref_mod_dir: Path,
    seq_len: int,
    stride: Optional[int],
    include_tail: bool,
    n: int,
    rng: random.Random,
) -> List[dict]:
    candidates: List[Tuple[str, int, int]] = []
    for stem in stems:
        feat_path = ref_mod_dir / f"{stem}.pt"
        item = labels.get(stem)
        if not feat_path.exists() or not item:
            continue
        obj = torch.load(feat_path, map_location="cpu", weights_only=False)
        ts = obj.get("timestamps", None) if isinstance(obj, dict) else None
        vals = item.get("values", [])
        if ts is None or not vals:
            continue
        base_len = min(len(ts), len(vals))
        starts = _build_window_starts(base_len, seq_len, stride, include_tail)
        for s in starts:
            e = min(s + seq_len, base_len)
            if e > s:
                candidates.append((stem, s, e))

    if not candidates:
        return []

    picks = candidates if len(candidates) <= n else rng.sample(candidates, n)
    out = []
    for stem, s, e in picks:
        feat_path = ref_mod_dir / f"{stem}.pt"
        obj = torch.load(feat_path, map_location="cpu", weights_only=False)
        ts = np.asarray(obj["timestamps"], dtype=np.float64)
        vals = np.asarray(labels[stem]["values"], dtype=np.float64)
        y = vals[s:e]
        out.append(
            {
                "stem": stem,
                "start": int(s),
                "end": int(e),
                "t0": float(ts[s]),
                "t1": float(ts[e - 1]),
                "y_min": float(y.min()),
                "y_max": float(y.max()),
                "y_std": float(y.std()),
            }
        )
    return out


def main() -> None:
    p = argparse.ArgumentParser(description="Diagnose strict sequence data pipeline.")
    p.add_argument("--features_root", required=True, type=str)
    p.add_argument("--labels_seq_path", required=True, type=str)
    p.add_argument("--split_path", required=True, type=str)
    p.add_argument("--modalities", type=str, default="video,km")
    p.add_argument("--reference_modality", type=str, default="video")
    p.add_argument("--seq_len", type=int, default=600)
    p.add_argument("--train_stride", type=int, default=None)
    p.add_argument("--val_stride", type=int, default=None)
    p.add_argument("--test_stride", type=int, default=None)
    p.add_argument("--include_tail_window", action="store_true")
    p.add_argument("--sample_windows", type=int, default=5)
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()

    features_root = Path(args.features_root)
    labels_path = Path(args.labels_seq_path)
    split_path = Path(args.split_path)
    modalities = [m.strip() for m in args.modalities.split(",") if m.strip()]
    mod_dirs = {m: features_root / m for m in modalities}
    ref_mod_dir = features_root / args.reference_modality
    rng = random.Random(args.seed)

    labels = _load_json(labels_path)
    split = _load_json(split_path)

    feat_stems = {}
    for m, d in mod_dirs.items():
        feat_stems[m] = {x.stem for x in d.glob("*.pt")} if d.exists() else set()

    print("=== Split Presence ===")
    print(f"split keys: {list(split.keys())}")
    for k in ("train", "val", "test"):
        print(f"{k}: {len(split.get(k, []))}")

    print("\n=== Session Overlap (by stem prefix) ===")
    sess = {}
    for k in ("train", "val", "test"):
        arr = split.get(k, [])
        sess[k] = {_split_session(x) for x in arr}
        print(f"{k}: sessions={len(sess[k])}")
    print(f"train∩val={len(sess['train'] & sess['val'])}")
    print(f"train∩test={len(sess['train'] & sess['test'])}")
    print(f"val∩test={len(sess['val'] & sess['test'])}")

    print("\n=== Stem Intersections by Split ===")
    label_stems = set(labels.keys())
    for k in ("train", "val", "test"):
        split_stems = set(split.get(k, []))
        inter = split_stems & label_stems
        for m in modalities:
            inter &= feat_stems[m]
        print(
            f"{k}: split={len(split_stems)} labels_inter={len(split_stems & label_stems)} "
            f"final_intersection={len(inter)}"
        )

    print("\n=== Label Scale Stats ===")
    all_stats = _label_stats(labels, labels.keys())
    print(f"all: {all_stats}")
    for k in ("train", "val", "test"):
        stats = _label_stats(labels, split.get(k, []))
        print(f"{k}: {stats}")

    print("\n=== Window Counts ===")
    strides = {"train": args.train_stride, "val": args.val_stride, "test": args.test_stride}
    for k in ("train", "val", "test"):
        split_stems = set(split.get(k, []))
        inter = split_stems & label_stems
        for m in modalities:
            inter &= feat_stems[m]
        total_windows = 0
        stem_used = 0
        for stem in sorted(inter):
            base_len = _base_len_for_stem(stem, labels, mod_dirs)
            if base_len is None:
                continue
            starts = _build_window_starts(
                base_len=base_len,
                seq_len=args.seq_len,
                stride=strides[k],
                include_tail=args.include_tail_window,
            )
            total_windows += len(starts)
            stem_used += 1
        print(
            f"{k}: stems={stem_used} stride={strides[k]} seq_len={args.seq_len} "
            f"windows={total_windows}"
        )

    print("\n=== Sample Window Checks ===")
    for k in ("train", "val", "test"):
        split_stems = set(split.get(k, []))
        inter = split_stems & label_stems
        for m in modalities:
            inter &= feat_stems[m]
        samples = _sample_window_checks(
            stems=sorted(inter),
            labels=labels,
            ref_mod_dir=ref_mod_dir,
            seq_len=args.seq_len,
            stride=strides[k],
            include_tail=args.include_tail_window,
            n=args.sample_windows,
            rng=rng,
        )
        print(f"{k}:")
        if not samples:
            print("  (no samples)")
            continue
        for item in samples:
            print(" ", item)


if __name__ == "__main__":
    main()

