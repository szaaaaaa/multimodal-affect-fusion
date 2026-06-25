#!/usr/bin/env python3
"""Post-hoc stratified evaluation for trajectory pilot runs.

Loads a completed run's best checkpoint, runs predictions over val and test
splits, then reports stratified metrics:

    * event (anchor within ±stratify_s of any event)
    * quiet (anchor farther than stratify_s from every event)
    * overall

Metrics: MSE, CCC, Peak-F1, Lead time, Amplitude error, Event-triggered
correlation, plus predictor variance (to distinguish "constant 0 output"
from "noisy output" when CCC≈0).

Usage:
  python scripts/evaluate_trajectory.py \
      --run_dir G:/我的云端硬盘/AmuCS_experiment/runs/trajectory_pilot/cross_subject/eft_trajectory_pilot_3seed/triple_video_km_telem/<timestamp>__...seed0 \
      --splits val test \
      --out_path <run_dir>/stratified_eval.json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import torch
import yaml

# Ensure project root is on path
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.core.runner import Runner  # noqa: E402
from src.metrics.event_locked import (  # noqa: E402
    EventLockedConfig,
    compute_event_locked_metrics,
)


def load_config(run_dir: Path) -> Dict[str, Any]:
    p = run_dir / "config.yaml"
    with p.open("r", encoding="utf-8-sig") as f:
        return yaml.safe_load(f)


def collect_predictions(runner: Runner, split: str) -> Tuple[np.ndarray, np.ndarray, List[str], List[int]]:
    """Return (preds [N,T], targets [N,T], stems [N], t_idxs [N])."""
    loader = {
        "val": runner.dm.val_dataloader(),
        "test": runner.dm.test_dataloader(),
    }[split]
    assert loader is not None, f"no {split} loader"

    runner.model.eval()
    preds_all: List[torch.Tensor] = []
    targets_all: List[torch.Tensor] = []
    stems: List[str] = []
    t_idxs: List[int] = []

    with torch.no_grad():
        for batch in loader:
            x = {m: batch["x"][m].to(runner.device) for m in batch["x"]}
            mask = {m: batch["mask"][m].to(runner.device) for m in batch["mask"]}
            y_hat = runner.model(x, mask)         # [B, 51]
            preds_all.append(y_hat.cpu())
            targets_all.append(batch["y"])
            stems.extend(batch["meta"]["stem"])
            t_idxs.extend(int(i) for i in batch["meta"]["t_idx"])
    return (
        torch.cat(preds_all, 0).numpy(),
        torch.cat(targets_all, 0).numpy(),
        stems,
        t_idxs,
    )


def compute_anchor_event_info(
    stems: List[str],
    t_idxs: List[int],
    events_by_stem: Dict[str, Dict[str, List[float]]],
    dt: float,
    post_s: float,
) -> Tuple[np.ndarray, np.ndarray]:
    """Return (distance_to_nearest_event_s [N], has_event_in_post_window [N]).

    Anchor time t_s = t_idx * dt. Distance is min over all event types.
    """
    N = len(stems)
    distances = np.full(N, np.inf)
    has_event = np.zeros(N, dtype=bool)

    for k in range(N):
        t_s = t_idxs[k] * dt
        stem = stems[k]
        ev_by_type = events_by_stem.get(stem, {})
        all_events = [e for evs in ev_by_type.values() for e in evs]
        if not all_events:
            continue
        arr = np.asarray(all_events, dtype=np.float64)
        distances[k] = float(np.min(np.abs(arr - t_s)))
        # event lands within post window [t_s, t_s + post_s]
        has_event[k] = bool(((arr >= t_s) & (arr <= t_s + post_s)).any())
    return distances, has_event


def load_events_index(events_path: Path) -> Dict[str, Dict[str, List[float]]]:
    with events_path.open("r", encoding="utf-8-sig") as f:
        return json.load(f)


def extra_diagnostics(preds: np.ndarray, targets: np.ndarray) -> Dict[str, float]:
    """Predictor-vs-target variance diagnostics.

    Helps distinguish 'predicts constant 0' (pred_std≈0, mse≈gt_var)
    from 'predicts noise' (pred_std>0 but uncorrelated).
    """
    return {
        "pred_std": float(preds.std()),
        "pred_mean": float(preds.mean()),
        "gt_std": float(targets.std()),
        "gt_mean": float(targets.mean()),
        "pred_std_over_gt_std": float(preds.std() / (targets.std() + 1e-9)),
    }


def stratified_eval(
    preds: np.ndarray,
    targets: np.ndarray,
    distances: np.ndarray,
    has_event: np.ndarray,
    cfg: EventLockedConfig,
    gt_sigma: float,
) -> Dict[str, Dict[str, float]]:
    """Run event-locked metrics AND add pred/gt variance diagnostics per stratum."""
    el = compute_event_locked_metrics(
        y_pred=preds,
        y_true=targets,
        anchor_event_distance_s=distances,
        is_near_event=has_event,
        gt_sigma=gt_sigma,
        cfg=cfg,
    )
    # Add variance diagnostics per stratum
    for stratum_name, stratum_mask in [
        ("event", (distances <= cfg.stratify_s)),
        ("quiet", (distances > cfg.stratify_s)),
        ("overall", np.ones(len(preds), dtype=bool)),
    ]:
        if not stratum_mask.any():
            continue
        el[stratum_name].update(extra_diagnostics(preds[stratum_mask], targets[stratum_mask]))
    return el


def print_table(results: Dict[str, Dict[str, Dict[str, float]]]):
    cols = [
        ("n", "n", 6, "d"),
        ("mse", "mse", 9, ".4f"),
        ("ccc", "ccc", 8, ".4f"),
        ("pred_std", "pred_std", 9, ".4f"),
        ("gt_std", "gt_std", 8, ".4f"),
        ("peak_f1", "peak_f1", 9, ".4f"),
        ("lead_s", "lead_time_median_s", 8, ".2f"),
        ("amp_err", "amplitude_err_median", 9, ".3f"),
        ("ev_corr", "event_corr_mean", 9, ".4f"),
    ]
    for split, strata in results.items():
        print(f"\n=== {split.upper()} ===")
        header = f"{'stratum':<10}" + "".join(f"{c[0]:>{c[2]}}" for c in cols)
        print(header)
        print("-" * len(header))
        for stratum in ("event", "quiet", "overall"):
            m = strata.get(stratum)
            if m is None or m.get("n", 0) == 0:
                print(f"{stratum:<10}  (no samples)")
                continue
            row = f"{stratum:<10}"
            for label, key, width, fmt_spec in cols:
                v = m.get(key)
                if v is None or (isinstance(v, float) and np.isnan(v)):
                    row += f"{'nan':>{width}}"
                else:
                    row += f"{v:>{width}{fmt_spec}}"
            print(row)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run_dir", type=str, required=True)
    ap.add_argument("--splits", type=str, nargs="+", default=["val", "test"])
    ap.add_argument("--ckpt_name", type=str, default="ckpt_best.pt",
                    help="Checkpoint filename under run_dir (default: ckpt_best.pt)")
    ap.add_argument("--out_path", type=str, default=None,
                    help="Path for stratified_eval.json (default: <run_dir>/stratified_eval[_<ckpt_stem>].json)")
    ap.add_argument("--stratify_s", type=float, default=10.0)
    ap.add_argument("--peak_eps_sigma", type=float, default=None,
                    help="Override EventLockedConfig default (1.0).")
    ap.add_argument("--tau_s", type=float, default=2.0)
    args = ap.parse_args()

    run_dir = Path(args.run_dir)
    assert run_dir.exists(), f"run_dir not found: {run_dir}"

    cfg = load_config(run_dir)
    # Force compile=False for eval: training-time torch.compile prefixes state_dict
    # keys with "_orig_mod."; we strip those below so we can load into a plain model.
    cfg.setdefault("train", {})["compile"] = False

    ckpt_path = run_dir / args.ckpt_name
    if not ckpt_path.exists():
        alt = run_dir / "ckpt_last.pt"
        assert alt.exists(), f"no checkpoint found at {ckpt_path} or {alt}"
        ckpt_path = alt
    print(f"[load] config from {run_dir / 'config.yaml'}")
    print(f"[load] checkpoint from {ckpt_path}")

    # Build pipeline WITHOUT resume (we handle ckpt loading manually to strip
    # the "_orig_mod." prefix left behind by torch.compile during training).
    runner = Runner(cfg)
    ckpt = torch.load(ckpt_path, map_location=runner.device, weights_only=False)
    state = ckpt.get("model", ckpt)
    state = {k.replace("_orig_mod.", "", 1) if k.startswith("_orig_mod.") else k: v
             for k, v in state.items()}
    # Prepare lazy per-modality fusion layers so their trained weights can
    # actually load. EFT builds `_modality_emb` lazily on first forward, and
    # `prepare_lazy_layers_from_state_dict` only handles merge layers — so the
    # modality embedding must be allocated explicitly here, otherwise its
    # ckpt weights get silently dropped under strict=False and multi-modality
    # evaluation would fall back to random init.
    fusion = getattr(runner.model, "fusion", None)
    if fusion is not None:
        if hasattr(fusion, "prepare_lazy_layers_from_state_dict"):
            fusion.prepare_lazy_layers_from_state_dict(state)
        if hasattr(fusion, "_get_modality_emb"):
            fusion._get_modality_emb(sorted(runner.modalities))
    missing, unexpected = runner.model.load_state_dict(state, strict=False)
    if missing:
        print(f"[load] missing keys (often lazy modality embeddings): {len(missing)} — {missing[:3]}...")
    if unexpected:
        print(f"[load] unexpected keys: {len(unexpected)} — {unexpected[:3]}...")

    events_path = Path(cfg["data"]["events_path"])
    events_by_stem = load_events_index(events_path)
    dt = float(cfg["data"].get("dt", 0.2))
    post_s = float(cfg["data"].get("post_s", 10.0))

    el_cfg_kwargs = {"dt": dt, "post_s": post_s, "stratify_s": args.stratify_s, "tau_s": args.tau_s}
    if args.peak_eps_sigma is not None:
        el_cfg_kwargs["peak_eps_sigma"] = args.peak_eps_sigma
    el_cfg = EventLockedConfig(**el_cfg_kwargs)

    # GT σ for Peak threshold: use training-set Δσ stored in the datamodule
    # (the targets we've just collected are already normalised by this σ, so GT std ≈ 1)
    gt_sigma = 1.0

    results: Dict[str, Dict[str, Dict[str, float]]] = {}
    for split in args.splits:
        if split == "test" and runner.dm.test_dataloader() is None:
            print(f"[{split}] no loader — skipping")
            continue
        print(f"[{split}] collecting predictions ...")
        preds, targets, stems, t_idxs = collect_predictions(runner, split)
        print(f"[{split}] {len(preds)} anchors  pred_shape={preds.shape}  target_shape={targets.shape}")

        distances, has_event = compute_anchor_event_info(
            stems, t_idxs, events_by_stem, dt=dt, post_s=post_s,
        )
        n_event = int((distances <= args.stratify_s).sum())
        n_quiet = int((distances > args.stratify_s).sum())
        print(f"[{split}] strata: event={n_event} ({n_event/len(preds):.1%}) "
              f"quiet={n_quiet} ({n_quiet/len(preds):.1%})")

        results[split] = stratified_eval(preds, targets, distances, has_event, el_cfg, gt_sigma)

    print_table(results)

    if args.out_path:
        out_path = Path(args.out_path)
    elif args.ckpt_name == "ckpt_best.pt":
        out_path = run_dir / "stratified_eval.json"  # preserve legacy filename
    else:
        out_path = run_dir / f"stratified_eval_{Path(args.ckpt_name).stem}.json"
    with out_path.open("w", encoding="utf-8") as f:
        json.dump({
            "run_dir": str(run_dir),
            "checkpoint": str(ckpt_path),
            "config": {
                "stratify_s": el_cfg.stratify_s,
                "peak_eps_sigma": el_cfg.peak_eps_sigma,
                "tau_s": el_cfg.tau_s,
                "post_s": el_cfg.post_s,
                "dt": el_cfg.dt,
            },
            "results": results,
        }, f, indent=2)
    print(f"\n[write] {out_path}")


if __name__ == "__main__":
    main()
