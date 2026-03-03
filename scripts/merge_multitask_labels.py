#!/usr/bin/env python3
"""
Merge state/trend single-task sequence labels into multitask label JSON.

Input:
  - state json: {stem: {"values": [...], "mask": [...]}, ...}
  - trend json: {stem: {"values": [...], "mask": [...]}, ...}

Output:
  {
    "<stem>": {
      "state": {"values": [...], "mask": [...]},
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
) -> Tuple[List[int], List[bool]]:
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
        description="Merge state + trend sequence labels for multitask training."
    )
    parser.add_argument("--state", required=True, help="Path to arousal_3cls_seq.json")
    parser.add_argument("--trend", required=True, help="Path to arousal_3trend_seq.json")
    parser.add_argument(
        "--output",
        default="labels/arousal_state_trend_seq.json",
        help="Output path (default: labels/arousal_state_trend_seq.json)",
    )
    args = parser.parse_args()

    state_path = Path(args.state)
    trend_path = Path(args.trend)
    output_path = Path(args.output)

    state_labels = _load_json(state_path)
    trend_labels = _load_json(trend_path)

    state_stems = set(state_labels.keys())
    trend_stems = set(trend_labels.keys())
    if state_stems != trend_stems:
        only_state = sorted(state_stems - trend_stems)[:10]
        only_trend = sorted(trend_stems - state_stems)[:10]
        raise ValueError(
            "State/Trend stems mismatch. "
            f"state_only(sample)={only_state}, trend_only(sample)={only_trend}"
        )

    merged: Dict[str, Dict[str, Dict[str, Any]]] = {}
    for stem in sorted(state_stems):
        state_values, state_mask = _validate_single_task_record(
            stem, "state", state_labels[stem]
        )
        trend_values, trend_mask = _validate_single_task_record(
            stem, "trend", trend_labels[stem]
        )

        if len(state_values) != len(trend_values):
            raise ValueError(
                f"stem={stem} state/trend sequence length mismatch: "
                f"{len(state_values)} vs {len(trend_values)}"
            )

        merged[stem] = {
            "state": {"values": state_values, "mask": state_mask},
            "trend": {"values": trend_values, "mask": trend_mask},
        }

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(merged, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    print(
        f"Saved multitask labels: {output_path} "
        f"(stems={len(merged)}, tasks=state+trend)"
    )


if __name__ == "__main__":
    main()
