#!/usr/bin/env python3
"""
Extract combat-focused game telemetry features from gameFlt.csv + gameInt.csv.

Only keeps features with high arousal relevance: enemy detection, combat
engagement, and damage/death events.  Low-signal columns (position, velocity,
eye direction, latency, armor, equipped histogram, isDucking, isJumping) are
excluded to improve signal-to-noise ratio given the small dataset (175 sessions).

Fixed global column set ensures consistent feature_dim across all sessions.
Only processes stems that have all required modalities (km + telem + arousal).

Usage:
    python scripts/extract_game_telem_features.py \
        --root_dir <AMuCS_data_root> \
        --output_dir data/features/amucs/telem \
        --dt 0.2
"""

from __future__ import annotations

import argparse
import math
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[1]

# ── Fixed global column definitions (combat-focused) ───────────────────
# Only columns with strong arousal relevance are kept.

# gameFlt target slots (-1 = no target -> NaN)
# Enemy proximity & aim accuracy — triggers "spot enemy" arousal
FLT_TARGET_SLOTS = {
    "distance": ["distance1", "distance2", "distance3", "distance4"],
    "degreesError": ["degreesError1", "degreesError2", "degreesError3", "degreesError4"],
    "distanceError": ["distanceError1", "distanceError2", "distanceError3", "distanceError4"],
}

# gameInt continuous state — combat-relevant only
INT_CONTINUOUS = [
    "health",                      # health drop -> damage taken
    "magazineAmmo", "reserveAmmo", # ammo consumption during firefight
    "num_enemies_in_FOV",          # enemy awareness
    "num_enemies_in_close_range",  # close-range threat
    "num_enemies_in_mid_range",    # mid-range threat
    "health_danger",               # low-health urgency
]

# gameInt binary/count events — combat actions
INT_EVENT = [
    "isReloading",
    "bulletShots", "bulletHits",
    "combat",
]

# gameInt sparse value events (-1 = no event -> 0) — damage & death
INT_SPARSE_VALUE = [
    "damage",
    "damageToEnemy", "damageFromEnemy",
    "deathVictim", "deathAttacker",
]

# gameInt categorical -> fixed-size histogram — aim & hit info
INT_CATEGORICAL = {
    "aimTarget": 5,
    "aimBodyPart": 6,
    "hitGroup": 8,
}

# gameInt FOV slots (-1 -> 0)
INT_FOV_SLOTS = ["inFOV1", "inFOV2", "inFOV3", "inFOV4"]

# ── Pre-compute fixed feature names (deterministic order) ───────────────

def _build_fixed_feature_names() -> List[str]:
    names = []
    stats7 = ["mean", "std", "min", "max", "last", "delta", "valid_ratio"]
    for c in INT_CONTINUOUS:
        for s in stats7:
            names.append(f"{c}_{s}")
    for group_name in FLT_TARGET_SLOTS:
        for s in ["min_mean", "min_min", "valid_count_mean", "any_valid_ratio"]:
            names.append(f"target_{group_name}_{s}")
    for c in INT_EVENT + INT_SPARSE_VALUE:
        for s in ["sum", "any", "rate"]:
            names.append(f"{c}_{s}")
    names.append("fov_any_count_sum")
    names.append("fov_any_count_mean")
    for c, max_bins in INT_CATEGORICAL.items():
        for i in range(max_bins):
            names.append(f"{c}_bin{i}")
    return names

FIXED_FEATURE_NAMES = _build_fixed_feature_names()
FIXED_FEAT_DIM = len(FIXED_FEATURE_NAMES)


# ── Helpers ─────────────────────────────────────────────────────────────

def _find_common_stems(root: Path, km_dir: Optional[Path] = None) -> Set[str]:
    """Find stems that have gameFlt + gameInt + ranktrace (+ km if dir given)."""
    telem_stems = set()
    arousal_stems = set()
    for sd in sorted(root.glob("S*")):
        if not sd.is_dir():
            continue
        for pd_ in sorted(sd.glob("P*")):
            stem = f"{sd.name}_{pd_.name}"
            if (pd_ / "gameFlt.csv").exists() and (pd_ / "gameInt.csv").exists():
                telem_stems.add(stem)
            if (pd_ / "ranktrace.csv").exists():
                arousal_stems.add(stem)
    common = telem_stems & arousal_stems
    if km_dir and km_dir.exists():
        km_stems = {p.stem for p in km_dir.glob("*.pt")}
        common &= km_stems
    return common


def _load_and_merge(session_path: Path) -> Optional[Tuple[pd.DataFrame, float, float]]:
    """Load gameFlt + gameInt, merge by nearest timestamp, return (df, t0, t1)."""
    flt_path = session_path / "gameFlt.csv"
    int_path = session_path / "gameInt.csv"
    if not flt_path.exists() or not int_path.exists():
        return None

    flt = pd.read_csv(flt_path, index_col=0)
    gi = pd.read_csv(int_path, index_col=0)

    gi_drop = {"session", "participant", "tick", "player_idx"}
    gi_cols = [c for c in gi.columns if c not in gi_drop and c not in flt.columns]
    gi_merge = gi[["time"] + gi_cols].copy()

    flt = flt.sort_values("time").reset_index(drop=True)
    gi_merge = gi_merge.sort_values("time").reset_index(drop=True)

    merged = pd.merge_asof(flt, gi_merge, on="time", tolerance=0.02, direction="nearest")

    t = merged["time"].values.astype(np.float64)
    return merged, float(t[0]), float(t[-1])


def _clean_sentinels(df: pd.DataFrame) -> pd.DataFrame:
    """Convert -1 sentinels to NaN/0 as appropriate."""
    for group_cols in FLT_TARGET_SLOTS.values():
        for c in group_cols:
            if c in df.columns:
                df[c] = df[c].replace(-1, np.nan)
    for c in INT_SPARSE_VALUE:
        if c in df.columns:
            df.loc[df[c] == -1, c] = 0
    for c in INT_FOV_SLOTS:
        if c in df.columns:
            df.loc[df[c] == -1, c] = 0
    return df


def _stat_pool_continuous(values: np.ndarray, valid: np.ndarray) -> np.ndarray:
    """[N] -> [7]: mean, std, min, max, last, delta, valid_ratio."""
    n_valid = valid.sum()
    if n_valid == 0:
        return np.zeros(7, dtype=np.float32)
    v = values[valid]
    idx = np.where(valid)[0]
    return np.array([
        v.mean(),
        v.std() if n_valid > 1 else 0.0,
        v.min(), v.max(),
        values[idx[-1]],
        values[idx[-1]] - values[idx[0]],
        n_valid / len(values),
    ], dtype=np.float32)


def _stat_pool_target_slots(slot_values: np.ndarray, valid: np.ndarray) -> np.ndarray:
    """[N, slots] -> [4]: min_mean, min_min, valid_count_mean, any_valid_ratio."""
    n = slot_values.shape[0]
    if n == 0:
        return np.zeros(4, dtype=np.float32)
    min_per_step = np.full(n, np.nan, dtype=np.float32)
    count_per_step = np.zeros(n, dtype=np.float32)
    for i in range(n):
        v = slot_values[i][valid[i]]
        if len(v) > 0:
            min_per_step[i] = v.min()
            count_per_step[i] = len(v)
    has_any = ~np.isnan(min_per_step)
    if has_any.sum() == 0:
        return np.zeros(4, dtype=np.float32)
    min_vals = min_per_step[has_any]
    return np.array([min_vals.mean(), min_vals.min(), count_per_step.mean(), has_any.mean()],
                    dtype=np.float32)


def _event_pool(values: np.ndarray, dt: float) -> np.ndarray:
    """[N] -> [3]: sum, any, rate."""
    s = float(values.sum())
    return np.array([s, 1.0 if s > 0 else 0.0, s / dt], dtype=np.float32)


def _categorical_histogram(values: np.ndarray, max_bins: int) -> np.ndarray:
    """[N] -> [max_bins] counts (0-indexed, -1 ignored)."""
    hist = np.zeros(max_bins, dtype=np.float32)
    for v in values:
        if np.isnan(v):
            continue
        iv = int(v)
        if 0 <= iv < max_bins:
            hist[iv] += 1
    return hist


# ── Main encoder ────────────────────────────────────────────────────────

def encode_session(df: pd.DataFrame, t0: float, t1: float, dt: float) -> Dict:
    """Encode one session into [n_bins, FIXED_FEAT_DIM] features."""
    time_col = df["time"].values.astype(np.float64)
    n_bins = int(math.ceil((t1 - t0) / dt))
    if n_bins <= 0:
        raise ValueError(f"Invalid time range: t0={t0}, t1={t1}")

    bin_idx = np.clip(((time_col - t0) / dt).astype(int), 0, n_bins - 1)
    df = _clean_sentinels(df.copy())

    features = np.zeros((n_bins, FIXED_FEAT_DIM), dtype=np.float32)
    mask = np.zeros(n_bins, dtype=np.bool_)

    all_continuous = INT_CONTINUOUS
    all_events = INT_EVENT + INT_SPARSE_VALUE

    for b in range(n_bins):
        rows = np.where(bin_idx == b)[0]
        if len(rows) == 0:
            continue
        mask[b] = True
        off = 0

        # Continuous stats (7 per col)
        for c in all_continuous:
            if c in df.columns:
                vals = df[c].values[rows].astype(np.float32)
                valid = ~np.isnan(vals)
                features[b, off:off + 7] = _stat_pool_continuous(vals, valid)
            off += 7

        # Target slot aggregates (4 per group)
        for slot_cols in FLT_TARGET_SLOTS.values():
            existing = [c for c in slot_cols if c in df.columns]
            if existing:
                sv = np.column_stack([df[c].values[rows].astype(np.float32) for c in existing])
                features[b, off:off + 4] = _stat_pool_target_slots(sv, ~np.isnan(sv))
            off += 4

        # Event columns (3 per col)
        for c in all_events:
            if c in df.columns:
                features[b, off:off + 3] = _event_pool(df[c].values[rows].astype(np.float32), dt)
            off += 3

        # FOV aggregate (2)
        fov_sum = 0.0
        for c in INT_FOV_SLOTS:
            if c in df.columns:
                fov_sum += df[c].values[rows].astype(np.float32).sum()
        features[b, off] = fov_sum
        features[b, off + 1] = fov_sum / max(len(rows), 1)
        off += 2

        # Categorical histograms
        for c, max_bins in INT_CATEGORICAL.items():
            if c in df.columns:
                features[b, off:off + max_bins] = _categorical_histogram(df[c].values[rows], max_bins)
            off += max_bins

    return {
        "features": torch.tensor(features, dtype=torch.float32),
        "mask": torch.tensor(mask, dtype=torch.bool),
        "meta": {
            "modality": "telemetry",
            "dt": dt,
            "t0": t0,
            "t1": t1,
            "feature_dim": FIXED_FEAT_DIM,
            "feature_names": FIXED_FEATURE_NAMES,
        },
    }


def encode_all_sessions(
    root_dir: str,
    output_dir: str,
    km_dir: Optional[str] = None,
    dt: float = 0.2,
    skip_existing: bool = True,
) -> None:
    root = Path(root_dir)
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    km_path = Path(km_dir) if km_dir else ROOT / "data" / "features" / "amucs" / "km"
    common = _find_common_stems(root, km_path)
    print(f"Common stems (km ∩ telem ∩ arousal): {len(common)}")

    saved = 0
    skipped = 0
    errors = {}

    for stem in sorted(common):
        save_path = out / f"{stem}.pt"
        if skip_existing and save_path.exists():
            skipped += 1
            continue

        session_id, player_id = stem.split("_", 1)
        session_path = root / session_id / player_id
        print(f"Processing {stem}...", end=" ")

        try:
            result = _load_and_merge(session_path)
            if result is None:
                print("Skipped (no data)")
                continue
            df, t0, t1 = result
            encoded = encode_session(df, t0, t1, dt)
            tmp_path = save_path.with_suffix(save_path.suffix + ".tmp")
            torch.save(encoded, tmp_path)
            tmp_path.replace(save_path)
            print(f"Saved [{tuple(encoded['features'].shape)}]")
            saved += 1
        except Exception as e:
            errors[stem] = str(e)
            print(f"Error: {e}")

    print(f"\nDone: saved={saved}, skipped={skipped}, errors={len(errors)}")
    print(f"Fixed feature_dim: {FIXED_FEAT_DIM}")
    if errors:
        for k, v in list(errors.items())[:10]:
            print(f"  {k}: {v}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Extract game telemetry features")
    parser.add_argument("--root_dir", type=str, required=True)
    parser.add_argument("--output_dir", type=str,
                        default=str(ROOT / "data" / "features" / "amucs" / "telem"))
    parser.add_argument("--km_dir", type=str, default=None,
                        help="KM features dir for stem filtering")
    parser.add_argument("--dt", type=float, default=0.2)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    encode_all_sessions(args.root_dir, args.output_dir, args.km_dir, args.dt,
                        skip_existing=not args.overwrite)
