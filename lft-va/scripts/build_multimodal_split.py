"""
Build train/val split for multimodal (video + KM) dataset.

构建多模态数据集的训练/验证划分。
"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--video_dir", type=str, default=None, help="Video features directory")
    parser.add_argument("--km_dir", type=str, default=None, help="KM features directory")
    parser.add_argument("--labels_path", type=str, default=None, help="Labels JSON path")
    parser.add_argument("--output_path", type=str, default=None, help="Output split JSON path")
    parser.add_argument("--val_ratio", type=float, default=0.2, help="Validation ratio")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    args = parser.parse_args()

    lft_root = Path(__file__).resolve().parents[1]

    video_dir = Path(args.video_dir) if args.video_dir else lft_root / "data" / "features" / "amucs" / "video"
    km_dir = Path(args.km_dir) if args.km_dir else lft_root / "data" / "features" / "amucs" / "km"
    labels_path = Path(args.labels_path) if args.labels_path else lft_root / "data" / "labels_arousal.json"
    output_path = Path(args.output_path) if args.output_path else lft_root / "data" / "splits" / "multimodal_split.json"

    # Find common stems
    video_stems = {p.stem for p in video_dir.glob("*.pt")} if video_dir.exists() else set()
    km_stems = {p.stem for p in km_dir.glob("*.pt")} if km_dir.exists() else set()

    if labels_path.exists():
        with labels_path.open("r") as f:
            labels = json.load(f)
        label_stems = set(labels.keys())
    else:
        label_stems = video_stems | km_stems

    common_stems = sorted(video_stems & km_stems & label_stems)

    if not common_stems:
        print(f"Warning: No common stems found.")
        print(f"  video_dir: {video_dir} ({len(video_stems)} files)")
        print(f"  km_dir: {km_dir} ({len(km_stems)} files)")
        print(f"  labels: {labels_path} ({len(label_stems)} entries)")
        common_stems = sorted(video_stems | km_stems)  # Fallback to union

    random.seed(args.seed)
    random.shuffle(common_stems)

    n_val = int(len(common_stems) * args.val_ratio)
    val_stems = common_stems[:n_val]
    train_stems = common_stems[n_val:]

    split = {"train": train_stems, "val": val_stems}

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as f:
        json.dump(split, f, ensure_ascii=False, indent=2)

    print(f"Split saved to: {output_path}")
    print(f"  train: {len(train_stems)} sessions")
    print(f"  val: {len(val_stems)} sessions")


if __name__ == "__main__":
    main()
