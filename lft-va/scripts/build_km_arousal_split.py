"""
Build train/val split for KM arousal sessions.

为 KM Arousal 生成固定种子 80/20 的 train/val 切分（中英文注释，NumPy 风格）。
"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path


def _project_root() -> Path:
    return Path(__file__).resolve().parents[1]


def main() -> None:
    """
    Create km_arousal_split.json with fixed seed.

    使用固定种子生成 km_arousal_split.json。
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--train_ratio", type=float, default=0.8)
    args = parser.parse_args()

    root = _project_root()
    labels_path = root / "data" / "labels_arousal.json"
    feats_dir = root / "data" / "features" / "amucs" / "km"
    out_path = root / "data" / "splits" / "km_arousal_split.json"

    if not labels_path.exists():
        raise FileNotFoundError(f"labels_arousal.json not found: {labels_path}")
    if not feats_dir.exists():
        raise FileNotFoundError(f"features dir not found: {feats_dir}")

    with labels_path.open("r", encoding="utf-8") as f:
        labels = json.load(f)

    feat_stems = {p.stem for p in feats_dir.glob("*.pt")}
    stems = [s for s in labels.keys() if s in feat_stems]

    if not stems:
        raise RuntimeError("No overlapping stems between labels and features.")

    random.seed(args.seed)
    random.shuffle(stems)

    n_train = int(len(stems) * args.train_ratio)
    train = stems[:n_train]
    val = stems[n_train:]

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as f:
        json.dump({"train": train, "val": val}, f, ensure_ascii=False, indent=2)

    print(f"Saved split: train={len(train)}, val={len(val)} -> {out_path}")


if __name__ == "__main__":
    main()
