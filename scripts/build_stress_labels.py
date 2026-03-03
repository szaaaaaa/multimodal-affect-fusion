#!/usr/bin/env python3
"""
Build discrete stress labels from continuous arousal sequences.

Reads labels_arousal_seq.json and produces:
  - labels_stress_level.json  (LOW=0, MID=1, HIGH=2)
  - labels_stress_trend.json  (DOWN=0, FLAT=1, UP=2)
  - labels_stress_meta.json   (per-stem threshold statistics)
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def ema(values: np.ndarray, alpha: float) -> np.ndarray:
    """Exponential moving average (forward pass)."""
    out = np.empty_like(values)
    out[0] = values[0]
    for i in range(1, len(values)):
        out[i] = alpha * values[i] + (1 - alpha) * out[i - 1]
    return out


# ---------------------------------------------------------------------------
# Task A: stress_level
# ---------------------------------------------------------------------------

def build_level_labels(
    arousal: Dict[str, dict],
    alpha: float,
    warmup_sec: float,
    dt: float = 0.2,
) -> Tuple[Dict[str, dict], Dict[str, dict]]:
    """Return (labels_dict, meta_dict) for stress_level task."""
    labels: Dict[str, dict] = {}
    meta: Dict[str, dict] = {}

    warmup_steps = int(warmup_sec / dt) if warmup_sec > 0 else 0

    for stem, item in arousal.items():
        raw = np.array(item["values"], dtype=np.float64)
        orig_mask = np.array(item.get("mask", [True] * len(raw)), dtype=bool)

        y_smooth = ema(raw, alpha)

        # Per-session quantiles (from valid steps only)
        if warmup_steps > 0 and warmup_steps < len(y_smooth):
            basis = y_smooth[:warmup_steps][orig_mask[:warmup_steps]]
        else:
            basis = y_smooth[orig_mask]

        if len(basis) == 0:
            basis = y_smooth  # fallback

        q20 = float(np.percentile(basis, 20))
        q80 = float(np.percentile(basis, 80))

        cls = np.ones(len(y_smooth), dtype=int)  # default MID=1
        cls[y_smooth <= q20] = 0   # LOW
        cls[y_smooth >= q80] = 2   # HIGH

        mask = orig_mask.copy()
        if warmup_steps > 0:
            mask[:warmup_steps] = False

        labels[stem] = {
            "values": cls.tolist(),
            "mask": mask.tolist(),
        }
        meta[stem] = {"q20": q20, "q80": q80}

    return labels, meta


# ---------------------------------------------------------------------------
# Task B: stress_trend
# ---------------------------------------------------------------------------

def build_trend_labels(
    arousal: Dict[str, dict],
    alpha: float,
    W: int,
    H: int,
    tau_scale: float,
    warmup_sec: float,
    dt: float = 0.2,
) -> Tuple[Dict[str, dict], Dict[str, dict]]:
    """Return (labels_dict, meta_dict) for stress_trend task."""
    labels: Dict[str, dict] = {}
    meta: Dict[str, dict] = {}

    warmup_steps = int(warmup_sec / dt) if warmup_sec > 0 else 0

    for stem, item in arousal.items():
        raw = np.array(item["values"], dtype=np.float64)
        orig_mask = np.array(item.get("mask", [True] * len(raw)), dtype=bool)

        y_smooth = ema(raw, alpha)
        T = len(y_smooth)

        tau = tau_scale * float(np.std(y_smooth[orig_mask]) if orig_mask.any() else np.std(y_smooth))

        cls = np.ones(T, dtype=int)  # default FLAT=1
        mask = orig_mask.copy()

        for t in range(T):
            end_now = t + W
            start_future = t + H
            end_future = t + H + W

            if end_future > T or end_now > T:
                mask[t] = False
                continue

            y_now = np.mean(y_smooth[t:end_now])
            y_future = np.mean(y_smooth[start_future:end_future])
            delta = y_future - y_now

            if delta < -tau:
                cls[t] = 0  # DOWN
            elif delta > tau:
                cls[t] = 2  # UP
            # else stays FLAT=1

        if warmup_steps > 0:
            mask[:warmup_steps] = False

        labels[stem] = {
            "values": cls.tolist(),
            "mask": mask.tolist(),
        }
        meta[stem] = {"tau": tau, "std": float(np.std(y_smooth))}

    return labels, meta


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Build stress classification labels")
    parser.add_argument("--arousal_path", type=str,
                        default="G:/我的云端硬盘/AmuCS_experiment/labels/labels_arousal_seq.json",
                        help="Path to labels_arousal_seq.json")
    parser.add_argument("--output_dir", type=str,
                        default="G:/我的云端硬盘/AmuCS_experiment/labels",
                        help="Output directory for label files")
    parser.add_argument("--task", type=str, default="both", choices=["level", "trend", "both"],
                        help="Which task labels to build")
    parser.add_argument("--alpha", type=float, default=0.05,
                        help="EMA smoothing coefficient")
    parser.add_argument("--W", type=int, default=25,
                        help="Window width for trend (steps)")
    parser.add_argument("--H", type=int, default=25,
                        help="Horizon for trend (steps)")
    parser.add_argument("--tau_scale", type=float, default=0.25,
                        help="tau = tau_scale * std(y_smooth)")
    parser.add_argument("--warmup_sec", type=float, default=0.0,
                        help="Warm-up seconds (0 = use full session)")

    args = parser.parse_args()

    arousal_path = Path(args.arousal_path)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    with arousal_path.open("r", encoding="utf-8-sig") as f:
        arousal = json.load(f)

    all_meta: Dict[str, dict] = {}

    if args.task in ("level", "both"):
        level_labels, level_meta = build_level_labels(
            arousal, alpha=args.alpha, warmup_sec=args.warmup_sec,
        )
        out_path = output_dir / "labels_stress_level.json"
        with out_path.open("w", encoding="utf-8") as f:
            json.dump(level_labels, f, indent=2)
        print(f"Wrote {out_path} ({len(level_labels)} stems)")

        # Class distribution
        for stem, item in level_labels.items():
            vals = np.array(item["values"])
            mask = np.array(item["mask"])
            valid = vals[mask]
            dist = {int(c): int((valid == c).sum()) for c in range(3)}
            level_meta[stem]["class_dist"] = dist

        for stem in level_meta:
            all_meta.setdefault(stem, {})["level"] = level_meta[stem]

    if args.task in ("trend", "both"):
        trend_labels, trend_meta = build_trend_labels(
            arousal, alpha=args.alpha, W=args.W, H=args.H,
            tau_scale=args.tau_scale, warmup_sec=args.warmup_sec,
        )
        out_path = output_dir / "labels_stress_trend.json"
        with out_path.open("w", encoding="utf-8") as f:
            json.dump(trend_labels, f, indent=2)
        print(f"Wrote {out_path} ({len(trend_labels)} stems)")

        # Class distribution
        for stem, item in trend_labels.items():
            vals = np.array(item["values"])
            mask = np.array(item["mask"])
            valid = vals[mask]
            dist = {int(c): int((valid == c).sum()) for c in range(3)}
            trend_meta[stem]["class_dist"] = dist

        for stem in trend_meta:
            all_meta.setdefault(stem, {})["trend"] = trend_meta[stem]

    meta_path = output_dir / "labels_stress_meta.json"
    with meta_path.open("w", encoding="utf-8") as f:
        json.dump(all_meta, f, indent=2)
    print(f"Wrote {meta_path}")


if __name__ == "__main__":
    main()
