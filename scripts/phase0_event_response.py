#!/usr/bin/env python3
"""Phase 0: event-locked response diagnostic.

Question: does the AMuCS ranktrace systematically respond to in-game
events (deathVictim / deathAttacker)?

For every event we cut a window around the event time, per-event detrend
with pre-event baseline, normalise by pre-event std, and average across
events. A shuffled-event control (500 trials) gives a 95% CI band for the
null hypothesis of no event-locked response.

Outputs:
  phase0_response.pdf     main figure (one subplot per event type)
  phase0_stats.json       per-type peak amplitude, latency, p-value, n

Usage:
  python scripts/phase0_event_response.py \
      --amucs_root "G:/我的云端硬盘/AmuCS/Affective Multimodal Counter-Strike video game dataset (AMuCS) - Public/researchdata/data" \
      --labels_path "G:/我的云端硬盘/AmuCS_experiment/labels/labels_arousal_seq.json" \
      --out_dir "G:/我的云端硬盘/AmuCS_experiment/phase0"
"""

from __future__ import annotations

import argparse
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

EVENT_COLS = ("deathVictim", "deathAttacker")  # extend later if needed
STEM_RE = re.compile(r"^S(\d+)_P(\d+)$")


@dataclass
class Params:
    pre_s: float = 5.0          # baseline window length (seconds)
    post_s: float = 15.0        # response window length (seconds)
    grid_dt: float = 0.2        # resample ranktrace to this grid (5 Hz)
    isolation_s: float = 5.0    # event is "isolated" if no other event within +/- isolation_s
    n_shuffle: int = 500        # number of shuffled control trials
    rng_seed: int = 0


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_stem_ranktrace(amucs_root: Path, session: str, participant: str) -> Optional[pd.DataFrame]:
    """Load raw ranktrace.csv for a stem. Returns sorted DataFrame with ['time','value'] or None.

    Column name for the annotation is either 'arousal' or 'valence' depending on
    which dimension was traced for that session.
    """
    p = amucs_root / session / participant / "ranktrace.csv"
    if not p.exists():
        return None
    df = pd.read_csv(p)
    if "time" not in df.columns:
        return None
    affect_col = "arousal" if "arousal" in df.columns else ("valence" if "valence" in df.columns else None)
    if affect_col is None:
        return None
    df = df[["time", affect_col]].dropna().sort_values("time").reset_index(drop=True)
    df = df.rename(columns={affect_col: "value"})
    if len(df) < 2:
        return None
    return df


def load_stem_events(amucs_root: Path, session: str, participant: str,
                     cols: Tuple[str, ...] = EVENT_COLS) -> Dict[str, np.ndarray]:
    """Load event onset timestamps (absolute seconds) for each event column.

    deathVictim / deathAttacker are single-tick events (value == -1 most of the time,
    non-minus-one exactly at the onset tick).
    """
    p = amucs_root / session / participant / "gameInt.csv"
    if not p.exists():
        return {c: np.array([]) for c in cols}
    df = pd.read_csv(p, usecols=["time", *cols])
    out: Dict[str, np.ndarray] = {}
    for c in cols:
        mask = df[c].values != -1
        if not mask.any():
            out[c] = np.array([])
            continue
        times = df.loc[mask, "time"].values
        # merge consecutive ticks within 1s (defensive; single-tick expected)
        times = np.sort(times)
        if len(times) > 1:
            keep = np.concatenate(([True], np.diff(times) > 1.0))
            times = times[keep]
        out[c] = times
    return out


def resample_ranktrace(rt: pd.DataFrame, t_start: float, t_end: float, dt: float) -> Tuple[np.ndarray, np.ndarray]:
    """Linear-interp ranktrace onto uniform grid [t_start, t_end] with step dt.

    Out-of-range grid points get np.nan so they can be excluded later.
    """
    grid = np.arange(t_start, t_end + dt / 2, dt)
    rt_t = rt["time"].values
    rt_v = rt["value"].values
    in_range = (grid >= rt_t[0]) & (grid <= rt_t[-1])
    values = np.full_like(grid, np.nan, dtype=np.float64)
    values[in_range] = np.interp(grid[in_range], rt_t, rt_v)
    return grid, values


# ---------------------------------------------------------------------------
# Event-locked analysis
# ---------------------------------------------------------------------------

def filter_isolated(events: np.ndarray, all_other: np.ndarray, iso_s: float) -> np.ndarray:
    """Keep event timestamps with no 'other' event within +/- iso_s."""
    if len(events) == 0 or len(all_other) == 0:
        return events
    other_sorted = np.sort(all_other)
    keep = []
    for t in events:
        idx = np.searchsorted(other_sorted, t)
        # nearest neighbour on either side
        nbrs = []
        if idx > 0:
            nbrs.append(abs(t - other_sorted[idx - 1]))
        if idx < len(other_sorted):
            nbrs.append(abs(t - other_sorted[idx]))
        if not nbrs or min(nbrs) > iso_s:
            keep.append(t)
    return np.array(keep)


def extract_response(values: np.ndarray, grid: np.ndarray, t_event: float,
                     pre_s: float, post_s: float, dt: float,
                     session_sd: float) -> Optional[np.ndarray]:
    """Return standardised Δ-response [n_samples] or None if window invalid.

    Detrend: subtract per-event pre-window baseline mean (removes session drift).
    Normalise: divide by *session-level* std (removes per-individual scale,
    but avoids blow-up when pre-window is locally flat).
    """
    n_pre = int(round(pre_s / dt))
    n_post = int(round(post_s / dt))
    i_event = int(round((t_event - grid[0]) / dt))
    i_start = i_event - n_pre
    i_end = i_event + n_post + 1
    if i_start < 0 or i_end > len(values):
        return None
    window = values[i_start:i_end]
    if np.isnan(window).any():
        return None
    baseline_mu = window[:n_pre].mean()
    if session_sd < 1e-6:
        return None
    return (window - baseline_mu) / session_sd


def build_response_matrix(stem_records: List[Dict], event_type: str, p: Params) -> Tuple[np.ndarray, int, int]:
    """Aggregate standardised responses across all stems for one event type.

    Returns:
        matrix [n_events_used, n_samples],
        n_events_used,
        n_events_total (after isolation filter).
    """
    responses = []
    n_total = 0
    for rec in stem_records:
        grid = rec["grid"]
        values = rec["values"]
        # isolation: keep events of this type that have no OTHER event within iso_s
        all_other = np.concatenate([rec["events"][c] for c in EVENT_COLS if c != event_type])
        evs = filter_isolated(rec["events"][event_type], all_other, p.isolation_s)
        n_total += len(evs)
        session_sd = rec["session_sd"]
        for t_ev in evs:
            r = extract_response(values, grid, t_ev, p.pre_s, p.post_s, p.grid_dt, session_sd)
            if r is not None:
                responses.append(r)
    if not responses:
        return np.empty((0, 0)), 0, n_total
    mat = np.stack(responses, axis=0)
    return mat, mat.shape[0], n_total


def shuffled_control(stem_records: List[Dict], event_type: str, p: Params) -> np.ndarray:
    """Re-compute mean response n_shuffle times with event times drawn uniformly
    from each stem's in-range grid. Returns [n_shuffle, n_samples]."""
    rng = np.random.default_rng(p.rng_seed)
    n_pre = int(round(p.pre_s / p.grid_dt))
    n_post = int(round(p.post_s / p.grid_dt))
    n_samples = n_pre + n_post + 1

    # Pre-compute per-stem valid index range and event count of this type
    per_stem = []
    for rec in stem_records:
        grid = rec["grid"]
        values = rec["values"]
        valid_idx = np.where(~np.isnan(values))[0]
        if len(valid_idx) == 0:
            per_stem.append(None)
            continue
        i_min = max(valid_idx[0] + n_pre, n_pre)
        i_max = min(valid_idx[-1] - n_post, len(values) - n_post - 1)
        if i_max <= i_min:
            per_stem.append(None)
            continue
        # how many events of this type (after isolation) in this stem
        all_other = np.concatenate([rec["events"][c] for c in EVENT_COLS if c != event_type])
        n_events = len(filter_isolated(rec["events"][event_type], all_other, p.isolation_s))
        if n_events == 0:
            per_stem.append(None)
            continue
        session_sd = rec["session_sd"]
        if session_sd < 1e-6:
            per_stem.append(None)
            continue
        per_stem.append((values, i_min, i_max, n_events, n_pre, n_post, session_sd))

    shuffled_means = np.full((p.n_shuffle, n_samples), np.nan)
    for k in range(p.n_shuffle):
        trial_responses = []
        for entry in per_stem:
            if entry is None:
                continue
            values, i_min, i_max, n_events, n_pre_, n_post_, session_sd = entry
            picks = rng.integers(i_min, i_max + 1, size=n_events)
            for i_event in picks:
                window = values[i_event - n_pre_: i_event + n_post_ + 1]
                if np.isnan(window).any():
                    continue
                mu = window[:n_pre_].mean()
                trial_responses.append((window - mu) / session_sd)
        if trial_responses:
            shuffled_means[k] = np.mean(trial_responses, axis=0)
    return shuffled_means


# ---------------------------------------------------------------------------
# Plotting + stats
# ---------------------------------------------------------------------------

def summarise(mat: np.ndarray, shuffled: np.ndarray, pre_s: float, post_s: float,
              grid_dt: float) -> Dict:
    """Peak amplitude / latency / p-value (fraction of shuffles with peak >= real)."""
    n_pre = int(round(pre_s / grid_dt))
    t_axis = np.arange(mat.shape[1]) * grid_dt - pre_s
    mean_resp = mat.mean(axis=0)
    post_slice = slice(n_pre, mat.shape[1])
    peak_idx_rel = np.argmax(mean_resp[post_slice])
    peak_amp = float(mean_resp[post_slice][peak_idx_rel])
    peak_lat = float(t_axis[post_slice][peak_idx_rel])

    # Shuffled peak dist: per-trial max in post window
    trial_peaks = np.nanmax(shuffled[:, post_slice], axis=1)
    trial_peaks = trial_peaks[~np.isnan(trial_peaks)]
    if len(trial_peaks) == 0:
        p_val = float("nan")
    else:
        p_val = float((trial_peaks >= peak_amp).mean())
    return {
        "n_events_used": int(mat.shape[0]),
        "peak_amplitude_sigma": peak_amp,
        "peak_latency_s": peak_lat,
        "shuffle_p_value": p_val,
        "shuffle_trials_ok": int(len(trial_peaks)),
    }


def plot_responses(results: Dict[str, Dict], out_path: Path, pre_s: float, post_s: float, grid_dt: float):
    import matplotlib.pyplot as plt
    event_types = [et for et in EVENT_COLS if et in results and results[et]["mat"].shape[0] > 0]
    if not event_types:
        print("[plot] No event types with data. Skipping figure.")
        return
    n = len(event_types)
    fig, axes = plt.subplots(1, n, figsize=(5.5 * n, 4.2), sharey=True)
    if n == 1:
        axes = [axes]
    for ax, et in zip(axes, event_types):
        res = results[et]
        mat = res["mat"]
        shuf = res["shuffled"]
        t_axis = np.arange(mat.shape[1]) * grid_dt - pre_s
        mean_resp = mat.mean(axis=0)
        sem = mat.std(axis=0, ddof=1) / np.sqrt(mat.shape[0])
        ax.fill_between(t_axis, mean_resp - sem, mean_resp + sem, alpha=0.25, color="C0")
        ax.plot(t_axis, mean_resp, color="C0", lw=2.0, label=f"real (n={mat.shape[0]})")
        if shuf.size:
            lo = np.nanpercentile(shuf, 2.5, axis=0)
            hi = np.nanpercentile(shuf, 97.5, axis=0)
            ax.fill_between(t_axis, lo, hi, alpha=0.20, color="gray", label="shuffled 95% CI")
        ax.axvline(0, color="k", lw=0.8, ls="--")
        ax.axhline(0, color="k", lw=0.5)
        s = res["stats"]
        ax.set_title(f"{et}\npeak={s['peak_amplitude_sigma']:+.2f}σ @ {s['peak_latency_s']:+.1f}s  p={s['shuffle_p_value']:.3f}")
        ax.set_xlabel("time relative to event (s)")
        ax.legend(loc="upper right", fontsize=8)
    axes[0].set_ylabel("Δ ranktrace (σ, pre-event baseline)")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    fig.savefig(out_path.with_suffix(".png"), dpi=150)
    plt.close(fig)
    print(f"[plot] Saved {out_path} (+ .png)")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Phase 0 event-locked response diagnostic")
    parser.add_argument("--amucs_root", type=str, required=True)
    parser.add_argument("--labels_path", type=str, required=True,
                        help="labels_arousal_seq.json — used only to filter arousal-annotated stems")
    parser.add_argument("--out_dir", type=str, required=True)
    parser.add_argument("--pre_s", type=float, default=Params.pre_s)
    parser.add_argument("--post_s", type=float, default=Params.post_s)
    parser.add_argument("--grid_dt", type=float, default=Params.grid_dt)
    parser.add_argument("--isolation_s", type=float, default=Params.isolation_s)
    parser.add_argument("--n_shuffle", type=int, default=Params.n_shuffle)
    parser.add_argument("--rng_seed", type=int, default=Params.rng_seed)
    args = parser.parse_args()

    p = Params(
        pre_s=args.pre_s, post_s=args.post_s, grid_dt=args.grid_dt,
        isolation_s=args.isolation_s, n_shuffle=args.n_shuffle, rng_seed=args.rng_seed,
    )

    amucs_root = Path(args.amucs_root)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Stem list: keys of labels_arousal_seq.json (these are arousal-annotated)
    with Path(args.labels_path).open("r", encoding="utf-8-sig") as f:
        labels = json.load(f)
    stems = sorted(labels.keys())
    print(f"[stems] total arousal-annotated: {len(stems)}")

    # Build per-stem records (ranktrace resampled + event timestamps)
    stem_records: List[Dict] = []
    skipped = 0
    for stem in stems:
        m = STEM_RE.match(stem)
        if not m:
            skipped += 1
            continue
        session = f"S{int(m.group(1)):03d}"
        participant = f"P{int(m.group(2))}"
        rt = load_stem_ranktrace(amucs_root, session, participant)
        if rt is None:
            skipped += 1
            continue
        events = load_stem_events(amucs_root, session, participant)
        t_start = rt["time"].iloc[0]
        t_end = rt["time"].iloc[-1]
        grid, values = resample_ranktrace(rt, t_start, t_end, p.grid_dt)
        valid_vals = values[~np.isnan(values)]
        session_sd = float(np.std(valid_vals)) if len(valid_vals) > 1 else 0.0
        stem_records.append({
            "stem": stem,
            "grid": grid,
            "values": values,
            "events": events,
            "session_sd": session_sd,
        })
    print(f"[stems] usable: {len(stem_records)}  skipped: {skipped}")

    # Per-event-type analysis
    results: Dict[str, Dict] = {}
    stats_out: Dict[str, Dict] = {}
    for et in EVENT_COLS:
        print(f"\n=== {et} ===")
        mat, n_used, n_total = build_response_matrix(stem_records, et, p)
        if n_used == 0:
            print(f"  no usable events ({n_total} after isolation filter, 0 with valid window)")
            continue
        print(f"  events: total={n_total}  with valid window={n_used}")
        print(f"  running {p.n_shuffle} shuffled control trials…")
        shuffled = shuffled_control(stem_records, et, p)
        stats = summarise(mat, shuffled, p.pre_s, p.post_s, p.grid_dt)
        print(f"  peak={stats['peak_amplitude_sigma']:+.3f}σ @ {stats['peak_latency_s']:+.2f}s   "
              f"shuffle p={stats['shuffle_p_value']:.4f}")
        results[et] = {"mat": mat, "shuffled": shuffled, "stats": stats}
        stats_out[et] = {**stats, "n_events_after_isolation": n_total}

    # Save outputs
    fig_path = out_dir / "phase0_response.pdf"
    plot_responses(results, fig_path, p.pre_s, p.post_s, p.grid_dt)

    stats_path = out_dir / "phase0_stats.json"
    with stats_path.open("w", encoding="utf-8") as f:
        json.dump({
            "params": p.__dict__,
            "n_stems_usable": len(stem_records),
            "results": stats_out,
        }, f, indent=2)
    print(f"\n[stats] Saved {stats_path}")


if __name__ == "__main__":
    main()
