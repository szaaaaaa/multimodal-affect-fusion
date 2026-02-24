#!/usr/bin/env python3
"""
Thin training entry point — cfg → build → runner.fit()

极薄训练入口 — 不包含任何业务逻辑。

Usage:
    python scripts/train.py --config configs/base.yaml
    python scripts/train.py --config configs/experiments/km_single.yaml
    python scripts/train.py --config configs/base.yaml --override model.fusion.name=mult train.seed=0
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Add project root to path
ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.core.config import load_config
from src.core.runner import Runner


def main():
    parser = argparse.ArgumentParser(description="Train multimodal model")
    parser.add_argument("--config", required=True, help="Path to YAML config file")
    parser.add_argument("--override", nargs="*", default=[], help="key=value overrides")
    parser.add_argument(
        "--resume",
        default=None,
        help="Resume from checkpoint file or run directory containing ckpt_last.pt",
    )
    args = parser.parse_args()

    cfg = load_config(args.config, overrides=args.override)
    runner = Runner(cfg, resume=args.resume)
    runner.fit()


if __name__ == "__main__":
    main()
