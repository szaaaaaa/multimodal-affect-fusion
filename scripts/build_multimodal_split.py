"""
Build train/val/test split for multimodal dataset.

构建多模态数据集的训练/验证/测试划分。
Supports arbitrary modality directories via --modality_dirs.
"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from typing import Dict, List


def _session_of(stem: str) -> str:
    return stem.split("_", 1)[0] if "_" in stem else stem


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--video_dir", type=str, default=None, help="Video features directory")
    parser.add_argument("--km_dir", type=str, default=None, help="KM features directory")
    parser.add_argument("--telem_dir", type=str, default=None, help="Telemetry features directory")
    parser.add_argument(
        "--modality_dirs",
        type=str,
        default=None,
        help="Comma-separated modality dirs (overrides --video_dir/--km_dir/--telem_dir)",
    )
    parser.add_argument("--labels_path", type=str, default=None, help="Labels JSON path")
    parser.add_argument("--output_path", type=str, default=None, help="Output split JSON path")
    parser.add_argument("--val_ratio", type=float, default=0.2, help="Validation ratio")
    parser.add_argument("--test_ratio", type=float, default=0.0, help="Test ratio")
    parser.add_argument(
        "--split_by_session",
        action="store_true",
        help="Split by session prefix (e.g., S001) to avoid leakage across phases.",
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    args = parser.parse_args()

    if args.val_ratio < 0 or args.test_ratio < 0:
        raise ValueError("val_ratio and test_ratio must be non-negative.")
    if args.val_ratio + args.test_ratio >= 1.0:
        raise ValueError("val_ratio + test_ratio must be < 1.0.")

    project_root = Path(__file__).resolve().parents[1]

    # Build list of modality directories
    if args.modality_dirs:
        mod_dirs = [Path(d.strip()) for d in args.modality_dirs.split(",") if d.strip()]
    else:
        mod_dirs = []
        if args.video_dir:
            mod_dirs.append(Path(args.video_dir))
        if args.km_dir:
            mod_dirs.append(Path(args.km_dir))
        if args.telem_dir:
            mod_dirs.append(Path(args.telem_dir))
        if not mod_dirs:
            mod_dirs = [
                project_root / "data" / "features" / "amucs" / "video",
                project_root / "data" / "features" / "amucs" / "km",
            ]

    labels_path = Path(args.labels_path) if args.labels_path else Path("G:/我的云端硬盘/AmuCS_experiment/labels/labels_arousal.json")
    output_path = Path(args.output_path) if args.output_path else project_root / "data" / "splits" / "multimodal_split.json"

    # Find common stems across all modality directories
    mod_stem_sets = []
    for d in mod_dirs:
        stems = {p.stem for p in d.glob("*.pt")} if d.exists() else set()
        mod_stem_sets.append(stems)
        print(f"  {d.name}: {len(stems)} files")

    if labels_path.exists():
        with labels_path.open("r", encoding="utf-8-sig") as f:
            labels = json.load(f)
        label_stems = set(labels.keys())
    else:
        label_stems = set.union(*mod_stem_sets) if mod_stem_sets else set()

    common_stems = sorted(set.intersection(*mod_stem_sets, label_stems) if mod_stem_sets else [])

    if not common_stems:
        print("Warning: No common stems found.")
        for i, d in enumerate(mod_dirs):
            print(f"  dir[{i}]: {d} ({len(mod_stem_sets[i])} files)")
        print(f"  labels: {labels_path} ({len(label_stems)} entries)")
        common_stems = sorted(set.union(*mod_stem_sets) if mod_stem_sets else [])  # Fallback to union

    random.seed(args.seed)

    if args.split_by_session:
        session_to_stems: Dict[str, List[str]] = {}
        for stem in common_stems:
            session_to_stems.setdefault(_session_of(stem), []).append(stem)

        sessions = sorted(session_to_stems.keys())
        random.shuffle(sessions)

        n_test_sess = int(len(sessions) * args.test_ratio)
        n_val_sess = int(len(sessions) * args.val_ratio)

        test_sessions = set(sessions[:n_test_sess])
        val_sessions = set(sessions[n_test_sess:n_test_sess + n_val_sess])
        train_sessions = set(sessions[n_test_sess + n_val_sess:])

        def collect(sess_set):
            out = []
            for s in sorted(sess_set):
                out.extend(sorted(session_to_stems[s]))
            return out

        train_stems = collect(train_sessions)
        val_stems = collect(val_sessions)
        test_stems = collect(test_sessions)
    else:
        random.shuffle(common_stems)
        n_test = int(len(common_stems) * args.test_ratio)
        n_val = int(len(common_stems) * args.val_ratio)
        test_stems = common_stems[:n_test]
        val_stems = common_stems[n_test:n_test + n_val]
        train_stems = common_stems[n_test + n_val:]

    split = {"train": train_stems, "val": val_stems, "test": test_stems}

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as f:
        json.dump(split, f, ensure_ascii=False, indent=2)

    print(f"Split saved to: {output_path}")
    print(f"  train: {len(train_stems)} stems")
    print(f"  val: {len(val_stems)} stems")
    print(f"  test: {len(test_stems)} stems")
    if args.split_by_session:
        print(f"  unique sessions: {len(sessions)}")
        print(
            f"  session split -> train: {len(train_sessions)}, "
            f"val: {len(val_sessions)}, test: {len(test_sessions)}"
        )


if __name__ == "__main__":
    main()

