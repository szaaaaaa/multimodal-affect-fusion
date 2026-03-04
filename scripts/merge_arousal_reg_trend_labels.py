#!/usr/bin/env python3
"""
Merge continuous arousal sequence labels + trend sequence labels into one multitask JSON.

Input:
  - arousal json: {stem: {"values": [...], "mask": [...]}, ...}
  - trend json:   {stem: {"values": [...], "mask": [...]}, ...}

Output:
  {
    "<stem>": {
      "arousal": {"values": [...], "mask": [...]},
      "trend": {"values": [...], "mask": [...]}
    },
    ...
  }
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List, Tuple


def _load_json(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8-sig") as f:
        return json.load(f)


def _validate_single_task_record(
    stem: str,
    task_name: str,
    item: Dict[str, Any],
) -> Tuple[List[Any], List[bool]]:
    if "values" not in item:
        raise KeyError(f"[{task_name}] stem={stem} missing key: values")
    values = item["values"]
    mask = item.get("mask", [True] * len(values))

    if len(values) != len(mask):
        raise ValueError(
            f"[{task_name}] stem={stem} values/mask length mismatch: "
            f"{len(values)} vs {len(mask)}"
        )

    return values, [bool(v) for v in mask]


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Merge arousal(regression) + trend(classification) sequence labels."
    )
    parser.add_argument("--arousal", required=True, help="Path to arousal_seq(_z)_*.json")
    parser.add_argument("--trend", required=True, help="Path to arousal_3trend_seq.json")
    parser.add_argument(
        "--output",
        default="labels/arousal_reg_trend_seq.json",
        help="Output path (default: labels/arousal_reg_trend_seq.json)",
    )
    args = parser.parse_args()

    arousal_path = Path(args.arousal)
    trend_path = Path(args.trend)
    output_path = Path(args.output)

    arousal_labels = _load_json(arousal_path)
    trend_labels = _load_json(trend_path)

    arousal_stems = set(arousal_labels.keys())
    trend_stems = set(trend_labels.keys())
    if arousal_stems != trend_stems:
        only_arousal = sorted(arousal_stems - trend_stems)[:10]
        only_trend = sorted(trend_stems - arousal_stems)[:10]
        raise ValueError(
            "Arousal/Trend stems mismatch. "
            f"arousal_only(sample)={only_arousal}, trend_only(sample)={only_trend}"
        )

    merged: Dict[str, Dict[str, Dict[str, Any]]] = {}
    for stem in sorted(arousal_stems):
        arousal_values, arousal_mask = _validate_single_task_record(
            stem, "arousal", arousal_labels[stem]
        )
        trend_values, trend_mask = _validate_single_task_record(
            stem, "trend", trend_labels[stem]
        )

        if len(arousal_values) != len(trend_values):
            raise ValueError(
                f"stem={stem} arousal/trend sequence length mismatch: "
                f"{len(arousal_values)} vs {len(trend_values)}"
            )

        merged[stem] = {
            "arousal": {"values": arousal_values, "mask": arousal_mask},
            "trend": {"values": trend_values, "mask": trend_mask},
        }

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(merged, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    print(
        f"Saved multitask labels: {output_path} "
        f"(stems={len(merged)}, tasks=arousal+trend)"
    )


if __name__ == "__main__":
    main()
