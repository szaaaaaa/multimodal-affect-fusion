#!/usr/bin/env python3
"""
Summarize all experiment runs into a leaderboard CSV.

汇总所有实验运行为排行榜 CSV。

Usage:
    python scripts/summarize.py --runs_dir runs/ --output leaderboard.csv
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

try:
    import yaml
    YAML_AVAILABLE = True
except ImportError:
    YAML_AVAILABLE = False


def load_run(run_dir: Path) -> dict:
    """Load metrics and config from a single run directory."""
    row = {"run_name": run_dir.name}

    # Load metrics
    metrics_path = run_dir / "metrics.json"
    if metrics_path.exists():
        with metrics_path.open("r") as f:
            metrics = json.load(f)
        row.update(metrics)

    # Load config to extract experiment info
    config = None
    config_yaml = run_dir / "config.yaml"
    config_json = run_dir / "config.json"
    if YAML_AVAILABLE and config_yaml.exists():
        with config_yaml.open("r") as f:
            config = yaml.safe_load(f)
    elif config_json.exists():
        with config_json.open("r") as f:
            config = json.load(f)

    if config:
        data_cfg = config.get("data", {})
        model_cfg = config.get("model", {})
        train_cfg = config.get("train", {})

        row["dataset"] = data_cfg.get("name", "")
        row["modalities"] = "+".join(sorted(data_cfg.get("modalities", [])))
        row["fusion"] = model_cfg.get("fusion", {}).get("name", "")
        row["seed"] = train_cfg.get("seed", "")

        # Encoder names
        encoders = model_cfg.get("encoders", {})
        for mod, enc_cfg in encoders.items():
            row[f"encoder_{mod}"] = enc_cfg.get("name", "")

    # Seed from file
    seed_path = run_dir / "seed.txt"
    if seed_path.exists() and "seed" not in row:
        row["seed"] = seed_path.read_text().strip()

    return row


def main():
    parser = argparse.ArgumentParser(description="Summarize runs")
    parser.add_argument("--runs_dir", default="runs", help="Runs directory")
    parser.add_argument("--output", default="leaderboard.csv", help="Output CSV path")
    args = parser.parse_args()

    runs_dir = Path(args.runs_dir)
    if not runs_dir.exists():
        print(f"Runs directory not found: {runs_dir}", file=sys.stderr)
        sys.exit(1)

    # Collect all runs that have metrics.json
    runs = []
    for d in sorted(runs_dir.iterdir()):
        if d.is_dir() and (d / "metrics.json").exists():
            runs.append(load_run(d))

    if not runs:
        print("No completed runs found.")
        return

    # Determine CSV columns (union of all keys)
    all_keys = []
    seen = set()
    # Priority columns first
    priority = ["run_name", "dataset", "fusion", "modalities", "seed",
                "best_val_metric", "best_epoch", "total_params"]
    for k in priority:
        if any(k in r for r in runs):
            all_keys.append(k)
            seen.add(k)
    for r in runs:
        for k in r:
            if k not in seen:
                all_keys.append(k)
                seen.add(k)

    # Write CSV
    output_path = Path(args.output)
    with output_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=all_keys, extrasaction="ignore")
        writer.writeheader()
        for r in runs:
            writer.writerow(r)

    print(f"Wrote {len(runs)} runs to {output_path}")


if __name__ == "__main__":
    main()
