#!/usr/bin/env python3
"""
Align pre-extracted multimodal features by timestamps.

This script is designed for already extracted modality features (e.g. video/km)
and is extensible to more modalities without hard-coding modality names.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, Iterable, Tuple

import numpy as np
import torch


def _parse_modalities(arg: str | None, input_root: Path) -> list[str]:
    if arg:
        return [m.strip() for m in arg.split(",") if m.strip()]
    return sorted([p.name for p in input_root.iterdir() if p.is_dir()])


def _parse_offsets(arg: str | None) -> Dict[str, float]:
    if not arg:
        return {}
    offsets: Dict[str, float] = {}
    for item in arg.split(","):
        item = item.strip()
        if not item:
            continue
        key, value = item.split("=", 1)
        offsets[key.strip()] = float(value.strip())
    return offsets


def _load_pt(path: Path) -> dict:
    obj = torch.load(path, map_location="cpu", weights_only=False)
    if not isinstance(obj, dict):
        raise TypeError(f"Expected dict .pt content, got {type(obj)} for {path}")
    if "features" not in obj:
        raise KeyError(f"Missing 'features' in {path}")
    return obj


def _extract_time_axis(obj: dict, n_steps: int) -> np.ndarray:
    if "timestamps" in obj:
        ts = np.asarray(obj["timestamps"], dtype=np.float64)
        if ts.shape[0] != n_steps:
            raise ValueError(f"timestamps length mismatch: {ts.shape[0]} vs {n_steps}")
        return ts

    meta = obj.get("meta", {})
    if isinstance(meta, dict) and "t0" in meta and "dt" in meta:
        t0 = float(meta["t0"])
        dt = float(meta["dt"])
        return t0 + np.arange(n_steps, dtype=np.float64) * dt

    if "sample_fps" in obj and float(obj["sample_fps"]) > 0:
        sample_fps = float(obj["sample_fps"])
        return np.arange(n_steps, dtype=np.float64) / sample_fps

    if "fps" in obj and "stride" in obj and float(obj["fps"]) > 0:
        sample_fps = float(obj["fps"]) / float(obj["stride"])
        if sample_fps > 0:
            return np.arange(n_steps, dtype=np.float64) / sample_fps

    raise KeyError("No usable timestamp info found (timestamps or meta[t0,dt] or sample_fps).")


def _extract_mask(obj: dict, n_steps: int) -> np.ndarray:
    if "mask" not in obj:
        return np.ones(n_steps, dtype=bool)
    mask = obj["mask"]
    if isinstance(mask, torch.Tensor):
        mask_np = mask.detach().cpu().numpy().astype(bool)
    else:
        mask_np = np.asarray(mask, dtype=bool)
    if mask_np.shape[0] != n_steps:
        raise ValueError(f"mask length mismatch: {mask_np.shape[0]} vs {n_steps}")
    return mask_np


def _nearest_resample(
    src_t: np.ndarray,
    src_x: np.ndarray,
    src_mask: np.ndarray,
    grid_t: np.ndarray,
    max_gap: float | None,
) -> Tuple[np.ndarray, np.ndarray]:
    idx_right = np.searchsorted(src_t, grid_t, side="left")
    idx_right = np.clip(idx_right, 0, len(src_t) - 1)
    idx_left = np.maximum(idx_right - 1, 0)
    choose_left = np.abs(src_t[idx_left] - grid_t) <= np.abs(src_t[idx_right] - grid_t)
    idx = np.where(choose_left, idx_left, idx_right)

    out_x = src_x[idx]
    out_mask = src_mask[idx].copy()
    if max_gap is not None:
        out_mask &= (np.abs(src_t[idx] - grid_t) <= max_gap)
    return out_x, out_mask


def _linear_resample(
    src_t: np.ndarray,
    src_x: np.ndarray,
    src_mask: np.ndarray,
    grid_t: np.ndarray,
    max_gap: float | None,
) -> Tuple[np.ndarray, np.ndarray]:
    # np.interp needs monotonic times and handles 1D arrays only; apply per feature dim.
    uniq_t, uniq_idx = np.unique(src_t, return_index=True)
    uniq_x = src_x[uniq_idx]
    uniq_mask = src_mask[uniq_idx]

    out_x = np.empty((grid_t.shape[0], uniq_x.shape[1]), dtype=np.float32)
    for d in range(uniq_x.shape[1]):
        out_x[:, d] = np.interp(
            grid_t,
            uniq_t,
            uniq_x[:, d],
            left=uniq_x[0, d],
            right=uniq_x[-1, d],
        ).astype(np.float32)

    _, out_mask = _nearest_resample(uniq_t, uniq_x, uniq_mask, grid_t, max_gap)
    return out_x, out_mask


def _build_grid(
    mode: str,
    start_t: float,
    end_t: float,
    target_hz: float,
    ref_t: np.ndarray | None,
) -> np.ndarray:
    if mode == "reference":
        if ref_t is None:
            raise ValueError("reference grid mode needs ref_t.")
        m = (ref_t >= start_t) & (ref_t <= end_t)
        return ref_t[m]

    step = 1.0 / target_hz
    return np.arange(start_t, end_t + 1e-12, step, dtype=np.float64)


def _iter_stems(modality_dirs: Dict[str, Path], split_path: Path | None) -> Iterable[str]:
    stems_per_mod = []
    for d in modality_dirs.values():
        stems_per_mod.append({p.stem for p in d.glob("*.pt")})
    common = set.intersection(*stems_per_mod) if stems_per_mod else set()

    if split_path is not None:
        split = json.loads(split_path.read_text(encoding="utf-8"))
        allowed = set(split.get("train", [])) | set(split.get("val", [])) | set(split.get("test", []))
        common &= allowed

    return sorted(common)


def main() -> None:
    parser = argparse.ArgumentParser(description="Align pre-extracted features by timestamps")
    parser.add_argument("--input_root", type=str, required=True, help="Root dir with modality subdirs")
    parser.add_argument("--output_root", type=str, required=True, help="Root dir to save aligned features")
    parser.add_argument(
        "--modalities",
        type=str,
        default=None,
        help="Comma-separated modalities (default: all subdirs under input_root)",
    )
    parser.add_argument(
        "--split_path",
        type=str,
        default=None,
        help="Optional split json. If set, only align stems in train/val/test union.",
    )
    parser.add_argument(
        "--grid_mode",
        type=str,
        default="uniform",
        choices=["uniform", "reference"],
        help="Time grid mode: uniform or use timestamps from reference modality.",
    )
    parser.add_argument("--target_hz", type=float, default=5.0, help="Grid frequency for uniform mode")
    parser.add_argument(
        "--reference_modality",
        type=str,
        default=None,
        help="Reference modality for grid_mode=reference (default: first modality).",
    )
    parser.add_argument(
        "--resample",
        type=str,
        default="nearest",
        choices=["nearest", "linear"],
        help="Resampling method to map each modality to the target grid.",
    )
    parser.add_argument(
        "--time_origin",
        type=str,
        default="zero",
        choices=["zero", "raw"],
        help="zero: shift each modality timeline by its first timestamp; raw: keep raw timestamps.",
    )
    parser.add_argument(
        "--offsets",
        type=str,
        default=None,
        help="Optional modality offsets in seconds, e.g. 'video=0,km=1.25'.",
    )
    parser.add_argument(
        "--max_gap",
        type=float,
        default=None,
        help="If set, mark aligned mask=False when nearest source timestamp is farther than max_gap.",
    )
    parser.add_argument(
        "--min_points",
        type=int,
        default=2,
        help="Minimum aligned points required to save a stem.",
    )
    parser.add_argument("--overwrite", action="store_true", help="Overwrite existing aligned .pt")
    parser.add_argument(
        "--report_path",
        type=str,
        default=None,
        help="Optional report json path (default: output_root/alignment_report.json).",
    )
    args = parser.parse_args()

    input_root = Path(args.input_root)
    output_root = Path(args.output_root)
    output_root.mkdir(parents=True, exist_ok=True)

    modalities = _parse_modalities(args.modalities, input_root)
    if not modalities:
        raise ValueError("No modalities found.")
    modality_dirs = {m: input_root / m for m in modalities}
    for m, d in modality_dirs.items():
        if not d.exists():
            raise FileNotFoundError(f"Modality dir not found: {m} -> {d}")

    offsets = _parse_offsets(args.offsets)
    split_path = Path(args.split_path) if args.split_path else None
    stems = list(_iter_stems(modality_dirs, split_path))
    if not stems:
        raise RuntimeError("No common stems found across modalities (and split, if provided).")

    ref_mod = args.reference_modality or modalities[0]
    if args.grid_mode == "reference" and ref_mod not in modalities:
        raise ValueError(f"reference_modality '{ref_mod}' not in modalities {modalities}")

    report = {
        "input_root": str(input_root),
        "output_root": str(output_root),
        "modalities": modalities,
        "grid_mode": args.grid_mode,
        "target_hz": args.target_hz,
        "reference_modality": ref_mod,
        "resample": args.resample,
        "time_origin": args.time_origin,
        "offsets": offsets,
        "max_gap": args.max_gap,
        "total_stems": len(stems),
        "saved_stems": 0,
        "skipped_no_overlap": 0,
        "skipped_too_short": 0,
        "skipped_exists": 0,
        "errors": [],
    }

    for i, stem in enumerate(stems, start=1):
        out_paths = {m: output_root / m / f"{stem}.pt" for m in modalities}
        if not args.overwrite and all(p.exists() for p in out_paths.values()):
            report["skipped_exists"] += 1
            if i % 20 == 0 or i == len(stems):
                print(f"[{i}/{len(stems)}] skip existing: {stem}")
            continue

        try:
            payloads: Dict[str, dict] = {}
            times: Dict[str, np.ndarray] = {}
            feats: Dict[str, np.ndarray] = {}
            masks: Dict[str, np.ndarray] = {}

            for m in modalities:
                obj = _load_pt(modality_dirs[m] / f"{stem}.pt")
                x = obj["features"]
                if isinstance(x, torch.Tensor):
                    x_np = x.detach().cpu().numpy().astype(np.float32)
                else:
                    x_np = np.asarray(x, dtype=np.float32)
                if x_np.ndim != 2:
                    raise ValueError(f"{m}/{stem}: features must be 2D [T,D], got shape {x_np.shape}")

                t_np = _extract_time_axis(obj, x_np.shape[0])
                if args.time_origin == "zero":
                    t_np = t_np - t_np[0]
                t_np = t_np + offsets.get(m, 0.0)

                payloads[m] = obj
                feats[m] = x_np
                times[m] = t_np
                masks[m] = _extract_mask(obj, x_np.shape[0])

            start_t = max(float(times[m].min()) for m in modalities)
            end_t = min(float(times[m].max()) for m in modalities)
            if end_t <= start_t:
                report["skipped_no_overlap"] += 1
                continue

            ref_t = times[ref_mod] if args.grid_mode == "reference" else None
            grid_t = _build_grid(args.grid_mode, start_t, end_t, args.target_hz, ref_t)
            if grid_t.shape[0] < args.min_points:
                report["skipped_too_short"] += 1
                continue

            for m in modalities:
                if args.resample == "linear":
                    x_aligned, m_aligned = _linear_resample(
                        times[m], feats[m], masks[m], grid_t, args.max_gap
                    )
                else:
                    x_aligned, m_aligned = _nearest_resample(
                        times[m], feats[m], masks[m], grid_t, args.max_gap
                    )

                out_obj = dict(payloads[m])
                out_obj["features"] = torch.from_numpy(x_aligned)
                out_obj["mask"] = torch.from_numpy(m_aligned.astype(bool))
                out_obj["timestamps"] = grid_t.tolist()

                meta = dict(out_obj.get("meta", {}))
                meta["aligned"] = True
                meta["align_grid_mode"] = args.grid_mode
                meta["align_target_hz"] = float(args.target_hz)
                meta["align_reference_modality"] = ref_mod
                meta["align_resample"] = args.resample
                meta["align_time_origin"] = args.time_origin
                meta["align_offset_sec"] = float(offsets.get(m, 0.0))
                meta["align_max_gap"] = args.max_gap
                out_obj["meta"] = meta

                out_paths[m].parent.mkdir(parents=True, exist_ok=True)
                tmp_path = out_paths[m].with_suffix(out_paths[m].suffix + ".tmp")
                torch.save(out_obj, tmp_path)
                tmp_path.replace(out_paths[m])

            report["saved_stems"] += 1
            if i % 20 == 0 or i == len(stems):
                print(f"[{i}/{len(stems)}] aligned: {stem} (T={len(grid_t)})")

        except Exception as e:
            report["errors"].append({"stem": stem, "error": str(e)})
            print(f"[{i}/{len(stems)}] error: {stem} -> {e}")

    report_path = Path(args.report_path) if args.report_path else (output_root / "alignment_report.json")
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")

    print("Alignment done.")
    print(
        f"  total={report['total_stems']}, saved={report['saved_stems']}, "
        f"skip_exists={report['skipped_exists']}, "
        f"skip_no_overlap={report['skipped_no_overlap']}, "
        f"skip_too_short={report['skipped_too_short']}, errors={len(report['errors'])}"
    )
    print(f"  report: {report_path}")


if __name__ == "__main__":
    main()
