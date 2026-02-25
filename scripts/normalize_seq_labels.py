#!/usr/bin/env python3
"""
Normalize sequence labels with train-split statistics (z-score).
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def _load_json(path: Path) -> dict:
    with path.open("r", encoding="utf-8-sig") as f:
        return json.load(f)


def main() -> None:
    p = argparse.ArgumentParser(description="Normalize sequence labels by train split stats.")
    p.add_argument("--labels_seq_path", required=True, type=str)
    p.add_argument("--split_path", required=True, type=str)
    p.add_argument("--output_path", required=True, type=str)
    p.add_argument(
        "--stats_output_path",
        default=None,
        type=str,
        help="Optional output path for {mean,std,count}. Defaults to <output>_stats.json",
    )
    p.add_argument(
        "--use_mask",
        action="store_true",
        help="Use label mask when computing train statistics.",
    )
    args = p.parse_args()

    labels_path = Path(args.labels_seq_path)
    split_path = Path(args.split_path)
    out_path = Path(args.output_path)
    stats_path = Path(args.stats_output_path) if args.stats_output_path else out_path.with_name(
        out_path.stem + "_stats.json"
    )

    labels = _load_json(labels_path)
    split = _load_json(split_path)
    train_stems = split.get("train", [])
    if not train_stems:
        raise RuntimeError(f"Split has empty or missing 'train': {split_path}")

    train_vals = []
    for stem in train_stems:
        item = labels.get(stem)
        if not item:
            continue
        vals = np.asarray(item.get("values", []), dtype=np.float64)
        if vals.size == 0:
            continue
        if args.use_mask and "mask" in item:
            m = np.asarray(item["mask"], dtype=bool)
            if m.size == vals.size:
                vals = vals[m]
        if vals.size > 0:
            train_vals.append(vals)

    if not train_vals:
        raise RuntimeError("No valid train label values found.")

    train_cat = np.concatenate(train_vals, axis=0)
    mean = float(train_cat.mean())
    std = float(train_cat.std())
    if std <= 0:
        raise RuntimeError(f"Invalid std={std}; cannot normalize.")

    normalized = {}
    for stem, item in labels.items():
        vals = np.asarray(item.get("values", []), dtype=np.float64)
        z = (vals - mean) / std if vals.size > 0 else vals
        out_item = dict(item)
        out_item["values"] = z.astype(np.float32).tolist()
        normalized[stem] = out_item

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(normalized, ensure_ascii=False, indent=2), encoding="utf-8")
    stats = {"mean": mean, "std": std, "count": int(train_cat.size)}
    stats_path.write_text(json.dumps(stats, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"saved normalized labels: {out_path} ({len(normalized)} stems)")
    print(f"saved stats: {stats_path} -> mean={mean:.6f}, std={std:.6f}, count={train_cat.size}")


if __name__ == "__main__":
    main()

