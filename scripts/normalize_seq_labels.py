#!/usr/bin/env python3
"""
Normalize sequence labels with per-participant z-score,
following the AMuCS paper methodology.

Each participant (stem) is normalized independently using its own
mean and std, so that inter-participant annotation scale differences
(inherent to RankTrace's relative & unbounded method) are removed.
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
    p = argparse.ArgumentParser(
        description="Normalize sequence labels per participant (z-score)."
    )
    p.add_argument("--labels_seq_path", required=True, type=str)
    p.add_argument("--output_path", required=True, type=str)
    p.add_argument(
        "--stats_output_path",
        default=None,
        type=str,
        help="Optional output path for per-participant stats. "
             "Defaults to <output>_stats.json",
    )
    p.add_argument(
        "--use_mask",
        action="store_true",
        help="Use label mask when computing per-participant statistics.",
    )
    p.add_argument(
        "--min_std",
        default=1e-6,
        type=float,
        help="Minimum std threshold; participants below this are skipped "
             "(constant annotation).",
    )
    # Keep --split_path for backward compatibility but it is no longer
    # required since normalization is per-participant.
    p.add_argument("--split_path", default=None, type=str,
                   help="(Unused, kept for backward compatibility)")
    args = p.parse_args()

    labels_path = Path(args.labels_seq_path)
    out_path = Path(args.output_path)
    stats_path = (
        Path(args.stats_output_path)
        if args.stats_output_path
        else out_path.with_name(out_path.stem + "_stats.json")
    )

    labels = _load_json(labels_path)

    normalized = {}
    per_participant_stats = {}
    skipped = []

    for stem, item in labels.items():
        vals = np.asarray(item.get("values", []), dtype=np.float64)
        if vals.size == 0:
            normalized[stem] = dict(item)
            continue

        # Compute stats using valid (masked) values only
        if args.use_mask and "mask" in item:
            mask = np.asarray(item["mask"], dtype=bool)
            if mask.size == vals.size:
                valid_vals = vals[mask]
            else:
                valid_vals = vals
        else:
            valid_vals = vals

        if valid_vals.size == 0:
            normalized[stem] = dict(item)
            continue

        mean = float(valid_vals.mean())
        std = float(valid_vals.std())

        if std < args.min_std:
            skipped.append(stem)
            # Still store with zero-mean but no scaling
            out_item = dict(item)
            out_item["values"] = (vals - mean).astype(np.float32).tolist()
            normalized[stem] = out_item
            per_participant_stats[stem] = {
                "mean": mean, "std": std, "count": int(valid_vals.size),
                "skipped": True,
            }
            continue

        z = (vals - mean) / std
        out_item = dict(item)
        out_item["values"] = z.astype(np.float32).tolist()
        normalized[stem] = out_item
        per_participant_stats[stem] = {
            "mean": mean, "std": std, "count": int(valid_vals.size),
            "skipped": False,
        }

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(
        json.dumps(normalized, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    stats_path.write_text(
        json.dumps(per_participant_stats, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    n_total = len(labels)
    n_ok = n_total - len(skipped)
    print(f"saved normalized labels: {out_path} ({n_total} stems)")
    print(f"saved stats: {stats_path} ({n_ok} normalized, {len(skipped)} skipped)")
    if skipped:
        print(f"  skipped (std < {args.min_std}): {skipped}")


if __name__ == "__main__":
    main()
