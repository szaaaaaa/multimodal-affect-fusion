"""
Filter AMuCS sessions that have arousal ranktrace.

筛选包含 arousal ranktrace 的 AMuCS session（中英文注释，NumPy 风格）。
"""

import argparse
import csv
import json
from pathlib import Path


def _project_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _has_arousal_column(csv_path: Path) -> bool:
    """
    Check whether a CSV file has an arousal column.

    判断 CSV 是否包含 arousal 列。
    """
    try:
        with csv_path.open("r", encoding="utf-8", newline="") as f:
            reader = csv.reader(f)
            header = next(reader, None)
            if not header:
                return False
            for h in header:
                if "arousal" in str(h).lower():
                    return True
    except Exception:
        return False
    return False


def main() -> None:
    """
    Traverse AMuCS raw data and collect arousal ranktrace paths.

    遍历 AMuCS 原始数据，收集 arousal ranktrace 文件路径。
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=str, required=True, help="AMuCS raw data root directory")
    args = parser.parse_args()

    data_root = Path(args.root)
    if not data_root.exists():
        raise FileNotFoundError(f"Root not found: {data_root}")

    out = {}

    # First pass: strict Sxxx/Px/ranktrace layout
    for s_dir in sorted(data_root.glob("S*")):
        if not s_dir.is_dir():
            continue
        for p_dir in sorted(s_dir.glob("P*")):
            if not p_dir.is_dir():
                continue
            rank_dir = p_dir / "ranktrace"
            if not rank_dir.is_dir():
                continue
            csv_files = sorted(rank_dir.glob("*.csv"))
            arousal_files = [p for p in csv_files if _has_arousal_column(p)]
            if not arousal_files:
                continue
            key = f"{s_dir.name}_{p_dir.name}"
            out[key] = arousal_files[0].as_posix()

    # Fallback: recursive search for any ranktrace CSV that has arousal column
    if not out:
        for csv_path in data_root.rglob("*.csv"):
            if "ranktrace" not in csv_path.name.lower():
                continue
            if not _has_arousal_column(csv_path):
                continue
            parts = list(csv_path.parts)
            key = None
            for i in range(len(parts) - 1):
                s = parts[i]
                p = parts[i + 1]
                if s.startswith("S") and s[1:].isdigit() and p.startswith("P") and p[1:].isdigit():
                    key = f"{s}_{p}"
                    break
            if key is None:
                continue
            out[key] = csv_path.as_posix()

    out_path = _project_root() / "lft-va" / "data" / "arousal_sessions.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)

    print(f"Saved {len(out)} sessions to {out_path}")
    if not out:
        print(f"No arousal ranktrace found under: {data_root}")


if __name__ == "__main__":
    main()
