#!/usr/bin/env python3
"""
Smooth arousal labels with Gaussian filter, then rebuild multitask label files.

对 arousal 标签做高斯平滑，然后重建 arousal_reg_trend 多任务标签文件。

Usage:
    # 生成 sigma=1,2,3,5 四个版本
    python scripts/smooth_labels.py \
        --arousal_path labels/arousal_seq_z_perparticipant.json \
        --trend_path labels/arousal_3trend_seq.json \
        --sigmas 1 2 3 5 \
        --output_dir labels/

    # 输出文件:
    #   labels/arousal_reg_trend_seq_smooth_s1.json
    #   labels/arousal_reg_trend_seq_smooth_s2.json
    #   labels/arousal_reg_trend_seq_smooth_s3.json
    #   labels/arousal_reg_trend_seq_smooth_s5.json
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from scipy.ndimage import gaussian_filter1d


def load_json(path: Path) -> dict:
    with path.open("r", encoding="utf-8-sig") as f:
        return json.load(f)


def smooth_arousal(arousal_labels: dict, sigma: float) -> dict:
    """Apply Gaussian smoothing to arousal values, respecting masks."""
    smoothed = {}
    for stem, item in arousal_labels.items():
        values = np.array(item["values"], dtype=np.float64)
        mask = np.array(item.get("mask", [True] * len(values)), dtype=bool)

        if mask.sum() < 2 or sigma <= 0:
            # Not enough valid points to smooth
            smoothed[stem] = {
                "values": item["values"],
                "mask": [bool(m) for m in mask],
            }
            continue

        # Only smooth valid regions; leave invalid as-is
        smoothed_values = values.copy()
        smoothed_values[mask] = gaussian_filter1d(values[mask], sigma=sigma)

        smoothed[stem] = {
            "values": smoothed_values.tolist(),
            "mask": [bool(m) for m in mask],
        }

    return smoothed


def rebuild_trend(arousal_labels: dict, k: int = 5) -> dict:
    """
    Rebuild 3-class trend labels from (smoothed) arousal.
    0=decreasing, 1=stable, 2=increasing.
    Thresholds: ±0.05 on the delta over k steps.
    """
    trend_labels = {}
    for stem, item in arousal_labels.items():
        values = np.array(item["values"], dtype=np.float64)
        mask = np.array(item["mask"], dtype=bool)
        T = len(values)

        trend_values = np.ones(T, dtype=np.int64)  # default: stable
        trend_mask = np.zeros(T, dtype=bool)

        for t in range(k, T):
            if mask[t] and mask[t - k]:
                delta = values[t] - values[t - k]
                trend_mask[t] = True
                if delta > 0.05:
                    trend_values[t] = 2  # increasing
                elif delta < -0.05:
                    trend_values[t] = 0  # decreasing
                else:
                    trend_values[t] = 1  # stable

        trend_labels[stem] = {
            "values": trend_values.tolist(),
            "mask": trend_mask.tolist(),
        }

    return trend_labels


def merge_multitask(arousal_labels: dict, trend_labels: dict) -> dict:
    """Merge arousal + trend into multitask format."""
    merged = {}
    for stem in sorted(arousal_labels.keys()):
        if stem not in trend_labels:
            continue
        arousal = arousal_labels[stem]
        trend = trend_labels[stem]
        # Ensure same length
        min_len = min(len(arousal["values"]), len(trend["values"]))
        merged[stem] = {
            "arousal": {
                "values": arousal["values"][:min_len],
                "mask": arousal["mask"][:min_len],
            },
            "trend": {
                "values": trend["values"][:min_len],
                "mask": trend["mask"][:min_len],
            },
        }
    return merged


def main():
    parser = argparse.ArgumentParser(description="Smooth arousal labels")
    parser.add_argument("--arousal_path", required=True, help="Path to arousal seq json")
    parser.add_argument("--trend_path", default=None,
                        help="Path to trend seq json (if None, recompute from smoothed arousal)")
    parser.add_argument("--sigmas", nargs="+", type=float, default=[1, 2, 3, 5],
                        help="Gaussian sigma values to try")
    parser.add_argument("--output_dir", default="labels/", help="Output directory")
    parser.add_argument("--rebuild_trend", action="store_true",
                        help="Recompute trend from smoothed arousal (default: use original trend)")
    args = parser.parse_args()

    arousal_path = Path(args.arousal_path)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    arousal_labels = load_json(arousal_path)
    print(f"Loaded arousal labels: {len(arousal_labels)} stems from {arousal_path}")

    # Load or prepare trend labels
    if args.trend_path:
        original_trend = load_json(Path(args.trend_path))
        print(f"Loaded trend labels: {len(original_trend)} stems")
    else:
        args.rebuild_trend = True
        original_trend = None

    for sigma in args.sigmas:
        print(f"\nSmoothing with sigma={sigma} ({sigma * 0.2:.1f}s @5Hz)...")

        smoothed_arousal = smooth_arousal(arousal_labels, sigma)

        if args.rebuild_trend:
            trend = rebuild_trend(smoothed_arousal)
            print(f"  Rebuilt trend labels from smoothed arousal")
        else:
            trend = original_trend

        merged = merge_multitask(smoothed_arousal, trend)

        out_path = output_dir / f"arousal_reg_trend_seq_smooth_s{int(sigma)}.json"
        out_path.write_text(
            json.dumps(merged, ensure_ascii=False),
            encoding="utf-8",
        )
        print(f"  Saved: {out_path} ({len(merged)} stems)")


if __name__ == "__main__":
    main()
