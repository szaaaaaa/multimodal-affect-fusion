#!/usr/bin/env python3
"""
Build time-aligned arousal sequence labels for strict temporal regression.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np
import torch


def _project_root() -> Path:
    return Path(__file__).resolve().parents[1]


def _stem_to_ranktrace(stem: str, ranktrace_root: Path) -> Path:
    session, phase = stem.split("_", 1)
    return ranktrace_root / session / phase / "ranktrace.csv"


def _read_ranktrace(csv_path: Path, time_col: str, value_col: str, time_unit: str) -> tuple[np.ndarray, np.ndarray]:
    times = []
    values = []
    with csv_path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            if time_col not in row or value_col not in row:
                continue
            try:
                t = float(row[time_col])
                v = float(row[value_col])
            except (TypeError, ValueError):
                continue
            times.append(t)
            values.append(v)

    if not times:
        raise RuntimeError(f"No valid rows in {csv_path}")

    t_np = np.asarray(times, dtype=np.float64)
    v_np = np.asarray(values, dtype=np.float64)
    if time_unit == "ms":
        t_np = t_np / 1000.0
    return t_np, v_np


def _load_grid_timestamps(features_path: Path) -> np.ndarray:
    obj = torch.load(features_path, map_location="cpu", weights_only=False)
    ts = obj.get("timestamps", None)
    if ts is None:
        raise KeyError(f"Missing 'timestamps' in {features_path}")
    return np.asarray(ts, dtype=np.float64)


def main() -> None:
    parser = argparse.ArgumentParser(description="Build aligned arousal sequence labels from ranktrace.csv")
    parser.add_argument(
        "--features_root",
        type=str,
        default=str(_project_root() / "data" / "features_aligned" / "amucs_trial"),
        help="Aligned feature root containing modality subdirs.",
    )
    parser.add_argument(
        "--reference_modality",
        type=str,
        default="video",
        help="Use this modality's timestamps as target label timeline.",
    )
    parser.add_argument(
        "--ranktrace_root",
        type=str,
        default=str(_project_root() / "data"),
        help="Dataset root containing Sxxx/Py/ranktrace.csv.",
    )
    parser.add_argument(
        "--output_path",
        type=str,
        default=str(_project_root() / "data" / "labels_arousal_seq.json"),
        help="Output sequence labels json path.",
    )
    parser.add_argument("--time_col", type=str, default="VideoTime")
    parser.add_argument("--value_col", type=str, default="arousal")
    parser.add_argument("--time_unit", type=str, default="ms", choices=["ms", "s"])
    parser.add_argument(
        "--stats_path",
        type=str,
        default=None,
        help="Optional z-score stats json with keys {mean,std}.",
    )
    args = parser.parse_args()

    features_dir = Path(args.features_root) / args.reference_modality
    ranktrace_root = Path(args.ranktrace_root)
    out_path = Path(args.output_path)

    if not features_dir.exists():
        raise FileNotFoundError(f"features dir not found: {features_dir}")
    if not ranktrace_root.exists():
        raise FileNotFoundError(f"ranktrace root not found: {ranktrace_root}")

    z_mean = None
    z_std = None
    if args.stats_path:
        stats = json.loads(Path(args.stats_path).read_text(encoding="utf-8"))
        z_mean = float(stats["mean"])
        z_std = float(stats["std"])
        if z_std == 0:
            raise ValueError("stats std must be non-zero")

    labels = {}
    errors = {}
    stems = sorted([p.stem for p in features_dir.glob("*.pt")])
    for i, stem in enumerate(stems, start=1):
        feat_path = features_dir / f"{stem}.pt"
        csv_path = _stem_to_ranktrace(stem, ranktrace_root)
        if not csv_path.exists():
            errors[stem] = f"missing ranktrace: {csv_path}"
            continue

        try:
            grid_t = _load_grid_timestamps(feat_path)
            src_t, src_v = _read_ranktrace(csv_path, args.time_col, args.value_col, args.time_unit)

            uniq_t, uniq_idx = np.unique(src_t, return_index=True)
            uniq_v = src_v[uniq_idx]

            interp_v = np.interp(grid_t, uniq_t, uniq_v, left=uniq_v[0], right=uniq_v[-1])
            valid_mask = (grid_t >= uniq_t.min()) & (grid_t <= uniq_t.max())

            if z_mean is not None and z_std is not None:
                interp_v = (interp_v - z_mean) / z_std

            labels[stem] = {
                "timestamps": grid_t.tolist(),
                "values": interp_v.astype(np.float32).tolist(),
                "mask": valid_mask.astype(bool).tolist(),
            }
        except Exception as e:
            errors[stem] = str(e)

        if i % 50 == 0 or i == len(stems):
            print(f"[{i}/{len(stems)}] processed")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(labels, ensure_ascii=False, indent=2), encoding="utf-8")

    report_path = out_path.with_name(out_path.stem + "_report.json")
    report = {
        "features_root": str(args.features_root),
        "reference_modality": args.reference_modality,
        "ranktrace_root": str(ranktrace_root),
        "output_path": str(out_path),
        "total_stems": len(stems),
        "saved_stems": len(labels),
        "error_stems": len(errors),
        "errors": errors,
    }
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"saved labels: {out_path} ({len(labels)})")
    print(f"saved report: {report_path} ({len(errors)} errors)")


if __name__ == "__main__":
    main()
