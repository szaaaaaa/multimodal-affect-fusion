"""
Check KM feature tensors in a directory.

检查键鼠特征文件的形状与元信息（中英文注释，NumPy 风格）。
"""

import argparse
from pathlib import Path
import statistics

import torch


def _default_dir() -> Path:
    """
    Get default feature directory.

    获取默认特征目录。

    Returns
    -------
    pathlib.Path
        Default KM feature directory.
        默认键鼠特征目录。
    """
    root = Path(__file__).resolve().parents[2]
    return root / "lft-va" / "data" / "features" / "amucs" / "km"


def main() -> None:
    """
    CLI entry: scan .pt files and print stats.

    命令行入口：扫描 .pt 文件并打印统计信息。
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("--dir", type=str, default=str(_default_dir()))
    args = parser.parse_args()

    data_dir = Path(args.dir)
    if not data_dir.exists():
        raise FileNotFoundError(f"Directory not found: {data_dir}")

    # Only scan current directory (no recursion). / 仅扫描当前目录
    files = sorted(data_dir.glob("*.pt"))
    print(f"Found {len(files)} files")
    if not files:
        return

    d_values = set()
    t_values = []

    printed_names = False
    for p in files:
        data = torch.load(p, map_location="cpu")
        feats = data["features"]
        t = int(feats.shape[0])
        d = int(feats.shape[1]) if feats.ndim == 2 else -1
        t_values.append(t)
        d_values.add(d)
        print(f"{p.name}: T={t}, D={d}")

        meta = data.get("meta", {})
        if isinstance(meta, dict) and "dt" in meta:
            print(f"  meta.dt={meta['dt']}")
        if not printed_names and isinstance(meta, dict) and "feature_names" in meta:
            print(f"  meta.feature_names={meta['feature_names']}")
            printed_names = True

    print(f"D unique values: {sorted(d_values)}")
    if len(d_values) != 1:
        print("WARNING: D is not consistent across files.")

    if t_values:
        t_min = min(t_values)
        t_max = max(t_values)
        t_mean = statistics.mean(t_values)
        print(f"T min/max/mean: {t_min}/{t_max}/{t_mean:.2f}")


if __name__ == "__main__":
    main()
