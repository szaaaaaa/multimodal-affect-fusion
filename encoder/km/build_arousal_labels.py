"""
Build file-level arousal labels from ranktrace CSVs.

从 ranktrace CSV 生成文件级 Arousal 标签（中英文注释，NumPy 风格）。
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import statistics


def _project_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _read_arousal_values(csv_path: Path) -> list[float]:
    """
    Read arousal values from a CSV file.

    从 CSV 中读取 arousal 数值（尽量兼容不同列名）。
    """
    values: list[float] = []
    with csv_path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.reader(f)
        header = next(reader, None)
        arousal_idx = None
        if header:
            for i, h in enumerate(header):
                if "arousal" in str(h).lower():
                    arousal_idx = i
                    break

        def push_row(row: list[str]) -> None:
            nums: list[float] = []
            for cell in row:
                try:
                    nums.append(float(cell))
                except ValueError:
                    continue
            if nums:
                values.append(nums[-1])

        if header and arousal_idx is not None:
            for row in reader:
                if arousal_idx < len(row):
                    try:
                        values.append(float(row[arousal_idx]))
                    except ValueError:
                        continue
        else:
            if header:
                push_row(header)
            for row in reader:
                push_row(row)

    return values


def main() -> None:
    """
    Build labels_arousal.json from arousal_sessions.json.

    根据 arousal_sessions.json 生成 labels_arousal.json。
    """
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--sessions",
        type=str,
        default=str(_project_root() / "data" / "arousal_sessions.json"),
    )
    parser.add_argument("--zscore", action="store_true", help="Apply z-score normalization")
    args = parser.parse_args()

    sessions_path = Path(args.sessions)
    if not sessions_path.exists():
        raise FileNotFoundError(f"arousal_sessions.json not found: {sessions_path}")

    with sessions_path.open("r", encoding="utf-8") as f:
        sessions = json.load(f)

    labels: dict[str, float] = {}
    for stem, csv_file in sessions.items():
        csv_path = Path(csv_file)
        if not csv_path.exists():
            continue
        values = _read_arousal_values(csv_path)
        if not values:
            continue
        labels[stem] = float(statistics.mean(values))

    out_path = _project_root() / "data" / "labels_arousal.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    if args.zscore:
        vals = list(labels.values())
        if not vals:
            raise RuntimeError("No labels to normalize.")
        mean_val = float(statistics.mean(vals))
        std_val = float(statistics.pstdev(vals))
        if std_val == 0.0:
            raise RuntimeError("Standard deviation is zero; cannot z-score.")
        labels = {k: (v - mean_val) / std_val for k, v in labels.items()}
        stats_path = _project_root() / "data" / "labels_arousal_stats.json"
        with stats_path.open("w", encoding="utf-8") as f:
            json.dump({"mean": mean_val, "std": std_val}, f, ensure_ascii=False, indent=2)
        print(f"Saved z-score stats to {stats_path}")

    with out_path.open("w", encoding="utf-8") as f:
        json.dump(labels, f, ensure_ascii=False, indent=2)

    print(f"Saved {len(labels)} labels to {out_path}")


if __name__ == "__main__":
    main()
