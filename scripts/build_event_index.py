#!/usr/bin/env python3
"""Extract in-game event timestamps and write a per-stem index.

Output schema:
  {
    "<stem>": {
      "deathVictim":   [t_rel_1, t_rel_2, ...],    # seconds, relative to telem.t0
      "deathAttacker": [t_rel_1, ...]
    },
    ...
  }

Time alignment: all event timestamps are converted to seconds *relative to
telem.meta.t0* — matching the time base of labels_arousal_seq.json (which is
also resampled onto telem's time grid starting at t0).
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd
import torch


EVENT_COLS = ("deathVictim", "deathAttacker")
STEM_RE = re.compile(r"^S(\d+)_P(\d+)$")


def extract_event_times_abs(gameint_path: Path) -> Dict[str, np.ndarray]:
    """Read event columns from gameInt.csv, return absolute-seconds onset times per type."""
    df = pd.read_csv(gameint_path, usecols=["time", *EVENT_COLS])
    out: Dict[str, np.ndarray] = {}
    for col in EVENT_COLS:
        mask = df[col].values != -1
        if not mask.any():
            out[col] = np.array([])
            continue
        times = np.sort(df.loc[mask, "time"].values)
        # merge consecutive ticks (<1s apart) into single onset
        if len(times) > 1:
            keep = np.concatenate(([True], np.diff(times) > 1.0))
            times = times[keep]
        out[col] = times
    return out


def telem_t0(telem_path: Path) -> float:
    obj = torch.load(telem_path, map_location="cpu", weights_only=False)
    return float(obj["meta"]["t0"])


def main():
    ap = argparse.ArgumentParser(description="Build per-stem event timestamp index")
    ap.add_argument("--amucs_root", type=str, required=True,
                    help="AMuCS raw data root (contains S001/ ... S071/)")
    ap.add_argument("--telem_dir", type=str, required=True,
                    help="Directory with telem .pt files (for t0 reference)")
    ap.add_argument("--stems_from", type=str, required=True,
                    help="labels_arousal_seq.json — stem list filter")
    ap.add_argument("--out_path", type=str, required=True)
    args = ap.parse_args()

    amucs_root = Path(args.amucs_root)
    telem_dir = Path(args.telem_dir)

    with Path(args.stems_from).open("r", encoding="utf-8-sig") as f:
        stems = sorted(json.load(f).keys())
    print(f"[stems] total: {len(stems)}")

    index: Dict[str, Dict[str, List[float]]] = {}
    missing: List[str] = []
    for stem in stems:
        m = STEM_RE.match(stem)
        if not m:
            missing.append(stem)
            continue
        session = f"S{int(m.group(1)):03d}"
        participant = f"P{int(m.group(2))}"
        gi_path = amucs_root / session / participant / "gameInt.csv"
        telem_path = telem_dir / f"{stem}.pt"
        if not gi_path.exists() or not telem_path.exists():
            missing.append(stem)
            continue
        t0 = telem_t0(telem_path)
        abs_events = extract_event_times_abs(gi_path)
        rel_events = {
            col: sorted(float(t - t0) for t in abs_events[col])
            for col in EVENT_COLS
        }
        index[stem] = rel_events

    print(f"[events] stems indexed: {len(index)}  missing: {len(missing)}")
    if missing:
        print(f"  missing examples: {missing[:5]}")

    # stats
    for col in EVENT_COLS:
        counts = [len(index[s][col]) for s in index]
        if counts:
            print(f"  {col}: total={sum(counts)}  median/stem={int(np.median(counts))}  "
                  f"min={min(counts)}  max={max(counts)}")

    out_path = Path(args.out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as f:
        json.dump(index, f, indent=2)
    print(f"[write] {out_path}")


if __name__ == "__main__":
    main()
