#!/usr/bin/env python3
"""
Automated experiment runner — sweep config → batch training → results aggregation.

自动实验执行器 — 从 sweep config 批量训练并汇总结果。

Reads a sweep YAML that declares {tasks × modalities × seeds}, builds a flat
config for each combination, calls Runner.fit(), and writes per-task
results.tsv + results_summary.csv.

Usage (Colab):
    python scripts/run_experiment.py \
        --sweep configs/sweeps/full_ablation.yaml \
        --data_root /content/drive/MyDrive/AmuCS_experiment/features/aligned \
        --labels_root /content/drive/MyDrive/AmuCS_experiment/labels \
        --splits_root /content/drive/MyDrive/AmuCS_experiment/splits

Usage (local):
    python scripts/run_experiment.py \
        --sweep configs/sweeps/full_ablation.yaml \
        --data_root G:/我的云端硬盘/AmuCS_experiment/features/aligned \
        --labels_root G:/我的云端硬盘/AmuCS_experiment/labels \
        --splits_root G:/我的云端硬盘/AmuCS_experiment/splits

Options:
    --runs_root   Override output root (default: runs/)
    --tasks       Run only specified tasks, e.g. --tasks arousal_3cls state_trend_multitask
    --dry_run     Print plan without running
"""

from __future__ import annotations

import argparse
import copy
import csv
import json
import sys
import traceback
from pathlib import Path
from typing import Any, Dict, List, Optional

# Add project root to path
ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import yaml

from src.core.config import Config
from src.core.runner import Runner


# ── Helpers ───────────────────────────────────────────────


def deep_merge(base: dict, override: dict) -> dict:
    """Recursively merge override into a copy of base."""
    result = copy.deepcopy(base)
    for k, v in override.items():
        if k in result and isinstance(result[k], dict) and isinstance(v, dict):
            result[k] = deep_merge(result[k], v)
        else:
            result[k] = copy.deepcopy(v)
    return result


def modality_exp_name(mods: List[str]) -> str:
    """[video] → single_video, [video, km] → dual_video_km, etc."""
    n = len(mods)
    prefix = {1: "single", 2: "dual", 3: "triple"}[n]
    # Keep original order from sweep config (matches existing run dirs)
    return f"{prefix}_{'_'.join(mods)}"


def find_completed_run(exp_dir: Path, seed: int) -> bool:
    """Check if a run with the given seed already finished in exp_dir."""
    if not exp_dir.is_dir():
        return False
    for d in exp_dir.iterdir():
        if not d.is_dir():
            continue
        if f"seed{seed}" in d.name and (d / "metrics.json").exists():
            return True
    return False


def collect_metrics(exp_dir: Path) -> List[dict]:
    """Collect metrics.json from all seed runs under exp_dir."""
    rows = []
    if not exp_dir.is_dir():
        return rows
    for d in sorted(exp_dir.iterdir()):
        mf = d / "metrics.json"
        if d.is_dir() and mf.exists():
            with mf.open("r", encoding="utf-8") as f:
                m = json.load(f)
            # Extract seed from dir name (... __seed0)
            seed_str = d.name.rsplit("seed", 1)[-1] if "seed" in d.name else "?"
            m["seed"] = seed_str
            rows.append(m)
    return rows


# ── Config assembly ───────────────────────────────────────


def build_run_config(
    shared: dict,
    task_def: dict,
    mods: List[str],
    seed: int,
    data_root: str,
    labels_root: str,
    splits_root: str,
    runs_dir: str,
    split_mode_overrides: Optional[Dict[str, Any]] = None,
) -> Config:
    """Assemble a complete Config for one (task, modality, seed) combination."""

    # Start from shared
    cfg = copy.deepcopy(shared)

    # Apply split_mode overrides (split_path, temporal_split_ratios, etc.)
    if split_mode_overrides:
        cfg["data"] = deep_merge(cfg.get("data", {}), split_mode_overrides)

    # Select fusion template
    fusion_key = task_def.get("fusion", "lft")
    fusion_template_key = f"fusion_{fusion_key}"
    fusion_cfg = cfg.get("model", {}).pop(fusion_template_key, {})
    # Remove all fusion templates from model
    model = cfg.get("model", {})
    for k in list(model.keys()):
        if k.startswith("fusion_"):
            model.pop(k)
    model["fusion"] = fusion_cfg

    # Merge task-specific overrides (data, model, train, eval, task_type)
    for section in ("data", "model", "train", "eval"):
        if section in task_def:
            cfg[section] = deep_merge(cfg.get(section, {}), task_def[section])
    if "task_type" in task_def:
        cfg["task_type"] = task_def["task_type"]

    # Set modalities
    cfg["data"]["modalities"] = list(mods)

    # Filter encoders to only active modalities
    all_encoders = cfg.get("model", {}).get("encoders", {})
    cfg["model"]["encoders"] = {m: all_encoders[m] for m in mods if m in all_encoders}

    # Resolve paths — replace relative with absolute using CLI roots
    data = cfg["data"]
    data["data_root"] = data_root

    labels_seq = data.get("labels_seq_path", "")
    if labels_seq and not Path(labels_seq).is_absolute():
        data["labels_seq_path"] = str(Path(labels_root) / Path(labels_seq).name)

    split = data.get("split_path", "")
    if split and not Path(split).is_absolute():
        data["split_path"] = str(Path(splits_root) / Path(split).name)

    # Seed
    cfg.setdefault("train", {})["seed"] = seed

    # Runs dir
    cfg["runs_dir"] = runs_dir

    # Device
    cfg.setdefault("device", "auto")

    return Config(cfg)


# ── Results aggregation ───────────────────────────────────


def write_results(task_dir: Path, exp_names: List[str]):
    """Write results.tsv and results_summary.csv for a task group."""

    # Collect all rows
    all_rows = []
    for exp_name in sorted(exp_names):
        exp_dir = task_dir / exp_name
        for row in collect_metrics(exp_dir):
            row["exp"] = exp_name
            all_rows.append(row)

    if not all_rows:
        return

    # ── results.tsv (one row per seed run) ──
    # Determine metric columns (exclude metadata)
    meta_keys = {"exp", "seed", "best_epoch", "total_epochs", "early_stopped",
                 "total_params", "train_time_s", "best_val_metric"}
    metric_keys = []
    for r in all_rows:
        for k in r:
            if k not in meta_keys and k not in metric_keys:
                metric_keys.append(k)

    tsv_path = task_dir / "results.tsv"
    tsv_cols = ["exp", "seed"] + metric_keys
    with tsv_path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=tsv_cols, delimiter="\t", extrasaction="ignore")
        w.writeheader()
        for r in sorted(all_rows, key=lambda x: (x.get("exp", ""), x.get("seed", ""))):
            w.writerow(r)

    # ── results_summary.csv (mean ± std per experiment) ──
    import statistics

    # Identify numeric metric columns that vary across seeds
    summary_metrics = []
    for k in metric_keys:
        vals = [r[k] for r in all_rows if k in r and isinstance(r[k], (int, float))]
        if vals:
            summary_metrics.append(k)

    summary_cols = ["exp"]
    for m in summary_metrics:
        summary_cols.extend([m, m])  # mean, std columns
    header_row_1 = [""]
    header_row_2 = ["exp"]
    for m in summary_metrics:
        header_row_1.extend([m, m])
        header_row_2.extend(["mean", "std"])

    csv_path = task_dir / "results_summary.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(header_row_1)
        w.writerow(header_row_2)
        for exp_name in sorted(exp_names):
            exp_rows = [r for r in all_rows if r.get("exp") == exp_name]
            if not exp_rows:
                continue
            row = [exp_name]
            for m in summary_metrics:
                vals = [r[m] for r in exp_rows if m in r and isinstance(r[m], (int, float))]
                if len(vals) >= 2:
                    row.extend([statistics.mean(vals), statistics.stdev(vals)])
                elif len(vals) == 1:
                    row.extend([vals[0], 0.0])
                else:
                    row.extend(["", ""])
            w.writerow(row)

    print(f"  Results written: {tsv_path}")
    print(f"  Summary written: {csv_path}")


# ── Main ──────────────────────────────────────────────────


def main():
    parser = argparse.ArgumentParser(description="Run sweep experiments")
    parser.add_argument("--sweep", required=True, help="Path to sweep YAML")
    parser.add_argument("--data_root", required=True, help="Absolute path to features/aligned")
    parser.add_argument("--labels_root", required=True, help="Absolute path to labels dir")
    parser.add_argument("--splits_root", required=True, help="Absolute path to splits dir")
    parser.add_argument("--runs_root", default="runs", help="Output root for runs (default: runs/)")
    parser.add_argument("--tasks", nargs="*", default=None, help="Only run these tasks")
    parser.add_argument("--dry_run", action="store_true", help="Print plan, don't run")
    args = parser.parse_args()

    with open(args.sweep, "r", encoding="utf-8") as f:
        sweep = yaml.safe_load(f)

    seeds = sweep["seeds"]
    modality_combos = sweep["modalities"]
    shared = sweep.get("shared", {})
    all_tasks = sweep["tasks"]

    # split_modes: optional top-level dict of named split configurations.
    # Each mode defines data overrides (split_path, temporal_split_ratios, etc.)
    # When absent, a single anonymous mode is used with no overrides.
    raw_split_modes = sweep.get("split_modes", None)
    if raw_split_modes:
        split_modes: List[tuple] = [(name, cfg) for name, cfg in raw_split_modes.items()]
    else:
        split_modes = [(None, None)]

    # Filter tasks if specified
    task_names = list(all_tasks.keys())
    if args.tasks:
        task_names = [t for t in args.tasks if t in all_tasks]
        if not task_names:
            print(f"Error: none of {args.tasks} found in sweep. Available: {list(all_tasks.keys())}")
            sys.exit(1)

    # Build run plan
    plan = []
    for split_mode_name, split_mode_cfg in split_modes:
        for task_name in task_names:
            task_def = all_tasks[task_name]
            task_dir_name = f"{task_name}_3seed"
            if split_mode_name:
                task_dir_name = f"{split_mode_name}/{task_dir_name}"
            for mods in modality_combos:
                exp_name = modality_exp_name(mods)
                # Append task suffix for multitask/cma naming
                if "multitask" in task_name:
                    suffix = "_multitask"
                    if "cma" in task_name:
                        suffix += "_cma" if "reg_trend" not in task_name else "_regtrend_cma"
                    elif "reg_trend" in task_name:
                        suffix += "_regtrend"
                    exp_name += suffix
                task_dir = Path(args.runs_root) / task_dir_name
                for seed in seeds:
                    plan.append({
                        "task": task_name,
                        "mods": mods,
                        "seed": seed,
                        "task_dir": task_dir,
                        "exp_name": exp_name,
                        "exp_dir": task_dir / exp_name,
                        "split_mode_name": split_mode_name,
                        "split_mode_cfg": split_mode_cfg,
                    })

    # Print plan
    total = len(plan)
    skip = sum(1 for p in plan if find_completed_run(p["exp_dir"], p["seed"]))
    print(f"Sweep plan: {total} runs total, {skip} already completed, {total - skip} to run")
    print()

    if args.dry_run:
        for p in plan:
            done = find_completed_run(p["exp_dir"], p["seed"])
            status = "SKIP" if done else "RUN"
            print(f"  [{status}] {p['task']} / {p['exp_name']} / seed{p['seed']}")
        return

    # Execute
    completed_tasks = set()
    failed = []
    for i, p in enumerate(plan):
        task_name = p["task"]
        exp_name = p["exp_name"]
        seed = p["seed"]
        exp_dir = p["exp_dir"]

        if find_completed_run(exp_dir, seed):
            print(f"[{i+1}/{total}] SKIP {task_name}/{exp_name}/seed{seed} (already done)")
            completed_tasks.add(task_name)
            continue

        split_label = f" [{p['split_mode_name']}]" if p.get("split_mode_name") else ""
        print(f"\n{'='*60}")
        print(f"[{i+1}/{total}]{split_label} {task_name} / {exp_name} / seed{seed}")
        print(f"{'='*60}")

        cfg = build_run_config(
            shared=shared,
            task_def=all_tasks[task_name],
            mods=p["mods"],
            seed=seed,
            data_root=args.data_root,
            labels_root=args.labels_root,
            splits_root=args.splits_root,
            runs_dir=str(exp_dir),
            split_mode_overrides=p.get("split_mode_cfg"),
        )

        try:
            runner = Runner(cfg)
            runner.fit()
            completed_tasks.add(task_name)
        except Exception:
            traceback.print_exc()
            failed.append(f"{task_name}/{exp_name}/seed{seed}")
            print(f"FAILED: {task_name}/{exp_name}/seed{seed}")

    # Aggregate results per completed task (per split_mode)
    print(f"\n{'='*60}")
    print("Aggregating results...")
    print(f"{'='*60}")
    for split_mode_name, _ in split_modes:
        for task_name in task_names:
            task_dir_name = f"{task_name}_3seed"
            if split_mode_name:
                task_dir_name = f"{split_mode_name}/{task_dir_name}"
            task_dir = Path(args.runs_root) / task_dir_name
            if not task_dir.is_dir():
                continue
            exp_names = [d.name for d in sorted(task_dir.iterdir()) if d.is_dir()]
            if exp_names:
                label = f" [{split_mode_name}]" if split_mode_name else ""
                print(f"\nTask: {task_name}{label}")
                write_results(task_dir, exp_names)

    # Summary
    print(f"\n{'='*60}")
    print(f"Done. {total - len(failed)}/{total} succeeded.")
    if failed:
        print(f"Failed ({len(failed)}):")
        for f_name in failed:
            print(f"  - {f_name}")


if __name__ == "__main__":
    main()
