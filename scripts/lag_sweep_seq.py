#!/usr/bin/env python3
"""
Run lag sweep on val/test for sequence regression checkpoints.

Purpose:
- Check whether low CCC is mainly caused by label delay misalignment.
- Evaluate RMSE/CCC under temporal shifts (lags).
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.core.config import load_config
from src.core.runner import Runner


def _ccc(x: np.ndarray, y: np.ndarray) -> float:
    if x.size == 0 or y.size == 0:
        return float("nan")
    mx, my = x.mean(), y.mean()
    vx, vy = x.var(), y.var()
    cov = np.mean((x - mx) * (y - my))
    denom = vx + vy + (mx - my) ** 2
    if denom == 0:
        return 0.0
    return float(2.0 * cov / denom)


def _rmse(x: np.ndarray, y: np.ndarray) -> float:
    if x.size == 0 or y.size == 0:
        return float("nan")
    return float(np.sqrt(np.mean((x - y) ** 2)))


def _align_by_lag(
    pred: torch.Tensor,
    target: torch.Tensor,
    mask: torch.Tensor,
    lag: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    lag > 0: compare pred[t] with target[t + lag]
    lag < 0: compare pred[t] with target[t - |lag|]
    """
    if lag == 0:
        return pred, target, mask

    if lag > 0:
        p = pred[:, :-lag]
        t = target[:, lag:]
        m = mask[:, :-lag] & mask[:, lag:]
        return p, t, m

    k = -lag
    p = pred[:, k:]
    t = target[:, :-k]
    m = mask[:, k:] & mask[:, :-k]
    return p, t, m


@torch.no_grad()
def evaluate_lag(
    runner: Runner,
    split: str,
    lag: int,
) -> Dict[str, float]:
    if split == "val":
        loader = runner.dm.val_dataloader()
    elif split == "test":
        loader = runner.dm.test_dataloader()
        if loader is None:
            return {"lag": lag, "rmse": float("nan"), "ccc": float("nan"), "n": 0}
    else:
        raise ValueError(f"Unsupported split: {split}")

    model = runner.model
    model.eval()

    all_p: List[np.ndarray] = []
    all_t: List[np.ndarray] = []

    for batch in loader:
        x_dict = {mod: batch["x"][mod].to(runner.device) for mod in batch["x"]}
        mask_dict = {mod: batch["mask"][mod].to(runner.device) for mod in batch["mask"]}
        y = batch["y"].to(runner.device)
        y_mask = batch["y_mask"].to(runner.device).bool()

        y_hat = model(x_dict, mask_dict)
        if y_hat.ndim == 3 and y_hat.shape[-1] == 1:
            y_hat = y_hat.squeeze(-1)
        if y.ndim == 3 and y.shape[-1] == 1:
            y = y.squeeze(-1)
        if y_mask.ndim == 3 and y_mask.shape[-1] == 1:
            y_mask = y_mask.squeeze(-1)

        p, t, m = _align_by_lag(y_hat, y, y_mask, lag)
        if m.numel() == 0:
            continue

        pv = p[m].detach().cpu().numpy()
        tv = t[m].detach().cpu().numpy()
        if pv.size > 0:
            all_p.append(pv)
            all_t.append(tv)

    if not all_p:
        return {"lag": lag, "rmse": float("nan"), "ccc": float("nan"), "n": 0}

    p_arr = np.concatenate(all_p, axis=0)
    t_arr = np.concatenate(all_t, axis=0)
    return {"lag": lag, "rmse": _rmse(p_arr, t_arr), "ccc": _ccc(p_arr, t_arr), "n": int(p_arr.size)}


def main() -> None:
    parser = argparse.ArgumentParser(description="Lag sweep for sequence regression runs.")
    parser.add_argument("--config", required=True, help="Config path")
    parser.add_argument("--ckpt", required=True, help="Checkpoint path or run dir")
    parser.add_argument("--split", default="val", choices=["val", "test"], help="Which split to sweep")
    parser.add_argument("--max_lag_steps", type=int, default=None, help="Sweep from -k..+k in steps")
    parser.add_argument("--max_lag_sec", type=float, default=5.0, help="Used if max_lag_steps is not set")
    parser.add_argument("--step_sec", type=float, default=0.2, help="Seconds per sequence step")
    parser.add_argument("--override", nargs="*", default=[], help="Config overrides, key=value")
    parser.add_argument("--output_json", default=None, help="Optional output json path")
    args = parser.parse_args()

    cfg = load_config(args.config, overrides=args.override)
    runner = Runner(cfg, resume=args.ckpt)

    if args.max_lag_steps is not None:
        k = int(args.max_lag_steps)
    else:
        k = int(round(float(args.max_lag_sec) / float(args.step_sec)))
    lags = list(range(-k, k + 1))

    rows = [evaluate_lag(runner, args.split, lag) for lag in lags]
    rows_valid = [r for r in rows if not np.isnan(r["ccc"])]
    if not rows_valid:
        raise RuntimeError("No valid lag results; check split/data/ckpt.")

    best_ccc = max(rows_valid, key=lambda r: r["ccc"])
    best_rmse = min(rows_valid, key=lambda r: r["rmse"])

    print(f"split={args.split} steps_per_sec={1.0/args.step_sec:.3f}")
    print(f"best_ccc: lag={best_ccc['lag']} ({best_ccc['lag']*args.step_sec:+.2f}s), ccc={best_ccc['ccc']:.6f}, rmse={best_ccc['rmse']:.6f}")
    print(f"best_rmse: lag={best_rmse['lag']} ({best_rmse['lag']*args.step_sec:+.2f}s), ccc={best_rmse['ccc']:.6f}, rmse={best_rmse['rmse']:.6f}")

    out = {
        "split": args.split,
        "step_sec": float(args.step_sec),
        "max_lag_steps": int(k),
        "best_ccc": best_ccc,
        "best_rmse": best_rmse,
        "rows": rows,
    }

    if args.output_json:
        out_path = Path(args.output_json)
    else:
        ckpt_path = Path(args.ckpt)
        run_dir = ckpt_path if ckpt_path.is_dir() else ckpt_path.parent
        out_path = run_dir / f"lag_sweep_{args.split}.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"saved: {out_path}")


if __name__ == "__main__":
    main()
