"""Event-locked evaluation metrics for future-trajectory regression.

Given per-anchor predictions `ŷ(t) ∈ ℝ^{post_steps}` and GT `y(t)` plus the
event-timestamp index, compute four event-response-aware metrics:

    Peak-F1(±τ)                  — does the predicted trajectory contain a
                                   peak whose time is within ±τ of the GT peak
                                   after each event onset?
    Lead time (median, IQR)      — sign of predicted-peak-time minus GT-peak-time
    Amplitude error (median)     — relative error of predicted-peak vs GT-peak
    Event-triggered correlation  — Pearson r between ŷ and y over event window

Reports three strata:
    event    — anchors within ±stratify_s of any event
    quiet    — anchors > stratify_s from every event
    overall  — all anchors
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np


@dataclass
class EventLockedConfig:
    dt: float = 0.2
    pre_s: float = 5.0       # baseline (ignored for trajectory case since anchor = window start)
    post_s: float = 10.0     # prediction horizon
    peak_eps_sigma: float = 1.0   # a "peak" requires Δ > eps·σ above baseline (per decision doc §8.3)
    tau_s: float = 2.0       # tolerance window for Peak-F1
    stratify_s: float = 10.0 # "event stratum" = anchor within this of any event


def _find_peak(signal: np.ndarray, dt: float) -> Tuple[float, float]:
    """Return (amplitude, latency_s) of the max of signal."""
    idx = int(np.argmax(signal))
    return float(signal[idx]), float(idx * dt)


def _pearson(a: np.ndarray, b: np.ndarray) -> float:
    if len(a) < 2:
        return float("nan")
    sa = a.std()
    sb = b.std()
    if sa < 1e-9 or sb < 1e-9:
        return float("nan")
    return float(np.corrcoef(a, b)[0, 1])


def _has_peak(signal: np.ndarray, eps: float) -> bool:
    """Simple peak criterion: max - min > eps."""
    return bool(signal.max() - signal.min() > eps)


def compute_event_locked_metrics(
    y_pred: np.ndarray,             # [N, post_steps]
    y_true: np.ndarray,             # [N, post_steps]
    anchor_event_distance_s: np.ndarray,  # [N]  distance from anchor to nearest event (s)
    is_near_event: np.ndarray,      # [N] bool  whether anchor has *any event in post window*
    gt_sigma: float,                # session-level (or global) σ for amplitude normalisation
    cfg: Optional[EventLockedConfig] = None,
) -> Dict[str, Dict[str, float]]:
    """Compute metrics + stratified reports."""
    cfg = cfg or EventLockedConfig()
    eps = cfg.peak_eps_sigma * gt_sigma
    tau_steps = int(round(cfg.tau_s / cfg.dt))

    # Overall arrays
    N = y_pred.shape[0]
    results: Dict[str, Dict[str, float]] = {}

    for stratum_name, stratum_mask in [
        ("event", (anchor_event_distance_s <= cfg.stratify_s)),
        ("quiet", (anchor_event_distance_s > cfg.stratify_s)),
        ("overall", np.ones(N, dtype=bool)),
    ]:
        n = int(stratum_mask.sum())
        if n == 0:
            results[stratum_name] = {"n": 0}
            continue
        yp = y_pred[stratum_mask]
        yt = y_true[stratum_mask]
        near = is_near_event[stratum_mask]

        # MSE, CCC
        mse = float(np.mean((yp - yt) ** 2))
        ccc = _concordance(yp.flatten(), yt.flatten())

        # Peak-F1 (only makes sense when anchor is near an event — GT has a peak)
        tp = fp = fn = 0
        leads: List[float] = []
        amp_errs: List[float] = []
        corrs: List[float] = []
        for k in range(n):
            pred_seg = yp[k]
            true_seg = yt[k]
            # event-triggered correlation — all anchors
            corrs.append(_pearson(pred_seg, true_seg))

            pred_has = _has_peak(pred_seg, eps)
            true_has = _has_peak(true_seg, eps)
            if true_has and pred_has:
                p_amp, p_lat = _find_peak(pred_seg, cfg.dt)
                t_amp, t_lat = _find_peak(true_seg, cfg.dt)
                if abs(p_lat - t_lat) / cfg.dt <= tau_steps:
                    tp += 1
                    leads.append(t_lat - p_lat)  # >0 = predicted earlier
                    if abs(t_amp) > 1e-6:
                        amp_errs.append(abs(p_amp - t_amp) / max(abs(t_amp), 0.1 * gt_sigma))
                else:
                    fp += 1
                    fn += 1
            elif pred_has and not true_has:
                fp += 1
            elif true_has and not pred_has:
                fn += 1

        if tp + fp > 0 and tp + fn > 0:
            precision = tp / (tp + fp)
            recall = tp / (tp + fn)
            f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
        else:
            precision = recall = f1 = float("nan")

        corrs_valid = [c for c in corrs if not np.isnan(c)]
        results[stratum_name] = {
            "n": n,
            "mse": mse,
            "ccc": ccc,
            "peak_f1": float(f1),
            "peak_precision": float(precision),
            "peak_recall": float(recall),
            "tp": int(tp), "fp": int(fp), "fn": int(fn),
            "lead_time_median_s": float(np.median(leads)) if leads else float("nan"),
            "lead_time_iqr_s": float(np.subtract(*np.percentile(leads, [75, 25]))) if len(leads) > 1 else float("nan"),
            "amplitude_err_median": float(np.median(amp_errs)) if amp_errs else float("nan"),
            "event_corr_mean": float(np.mean(corrs_valid)) if corrs_valid else float("nan"),
        }

    return results


def _concordance(pred: np.ndarray, target: np.ndarray) -> float:
    """Lin's CCC."""
    if len(pred) < 2:
        return float("nan")
    mp, mt = pred.mean(), target.mean()
    vp, vt = pred.var(), target.var()
    cov = ((pred - mp) * (target - mt)).mean()
    denom = vp + vt + (mp - mt) ** 2
    if denom < 1e-12:
        return float("nan")
    return float(2 * cov / denom)


# --- helpers for callers ----------------------------------------------------

def nearest_event_distance(
    anchor_t_s: float,
    event_times_s: Sequence[float],
) -> float:
    if not event_times_s:
        return float("inf")
    arr = np.asarray(event_times_s, dtype=np.float64)
    return float(np.min(np.abs(arr - anchor_t_s)))


def anchor_has_event_in_window(
    anchor_t_s: float,
    event_times_s: Sequence[float],
    window_s: float,
) -> bool:
    if not event_times_s:
        return False
    arr = np.asarray(event_times_s, dtype=np.float64)
    return bool(((arr >= anchor_t_s) & (arr <= anchor_t_s + window_s)).any())
