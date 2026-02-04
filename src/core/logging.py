"""
Experiment logging and run directory management.

实验日志与运行目录管理。
"""

from __future__ import annotations

import json
import subprocess
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional


def create_run_dir(
    base_dir: str | Path,
    dataset: str,
    fusion: str,
    modalities: list,
    seed: int,
) -> Path:
    """
    Create a standardised run output directory.

    创建标准化的运行输出目录。

    Format: {timestamp}__{dataset}__{fusion}__{modalities}__{seed}
    """
    base_dir = Path(base_dir)
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    mods = "_".join(sorted(modalities))
    name = f"{timestamp}__{dataset}__{fusion}__{mods}__seed{seed}"
    run_dir = base_dir / name
    run_dir.mkdir(parents=True, exist_ok=True)
    return run_dir


def save_run_metadata(
    run_dir: Path,
    config: dict,
    seed: int,
) -> None:
    """
    Save reproducibility metadata to run directory.

    保存可复现性元数据到运行目录。

    Saves: config.yaml (or .json), seed.txt, git_commit.txt
    """
    run_dir = Path(run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)

    # Config
    try:
        import yaml
        with (run_dir / "config.yaml").open("w", encoding="utf-8") as f:
            yaml.dump(dict(config), f, default_flow_style=False, allow_unicode=True, sort_keys=False)
    except ImportError:
        with (run_dir / "config.json").open("w", encoding="utf-8") as f:
            json.dump(dict(config), f, indent=2, ensure_ascii=False, default=str)

    # Seed
    (run_dir / "seed.txt").write_text(str(seed))

    # Git commit hash
    try:
        commit = subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            stderr=subprocess.DEVNULL,
        ).decode().strip()
        (run_dir / "git_commit.txt").write_text(commit)
    except (subprocess.CalledProcessError, FileNotFoundError):
        pass


def save_metrics(run_dir: Path, metrics: Dict[str, Any]) -> None:
    """
    Save metrics dict to metrics.json.

    保存指标到 metrics.json。
    """
    path = Path(run_dir) / "metrics.json"
    with path.open("w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2, ensure_ascii=False, default=str)
