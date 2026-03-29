"""
Build within-subject split: all stems appear in train, val, and test.

Temporal partitioning is handled by the DataModule via temporal_split_ratios,
not by the split file. This script simply puts all valid stems in all sets.

Usage:
    python scripts/build_within_subject_split.py \
        --modality_dirs ".../features/aligned/video,.../features/aligned/km,.../features/aligned/telem" \
        --labels_path ".../labels/arousal_3cls_seq.json" \
        --output_path ".../splits/within_subject.json"
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--modality_dirs", type=str, required=True,
                        help="Comma-separated modality feature dirs")
    parser.add_argument("--labels_path", type=str, required=True,
                        help="Labels JSON path (any label file to get valid stems)")
    parser.add_argument("--output_path", type=str, required=True,
                        help="Output split JSON path")
    args = parser.parse_args()

    mod_dirs = [Path(d.strip()) for d in args.modality_dirs.split(",") if d.strip()]
    labels_path = Path(args.labels_path)
    output_path = Path(args.output_path)

    with labels_path.open("r", encoding="utf-8-sig") as f:
        label_stems = set(json.load(f).keys())

    stem_sets = []
    for d in mod_dirs:
        stems = {p.stem for p in d.glob("*.pt")} if d.exists() else set()
        stem_sets.append(stems)
        print(f"  {d.name}: {len(stems)} files")

    common = label_stems
    for s in stem_sets:
        common &= s
    all_stems = sorted(common)

    split = {"train": all_stems, "val": all_stems, "test": all_stems}

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as f:
        json.dump(split, f, ensure_ascii=False, indent=2)

    print(f"Within-subject split saved: {output_path}")
    print(f"  All sets contain {len(all_stems)} stems (temporal partitioning via DataModule)")


if __name__ == "__main__":
    main()
