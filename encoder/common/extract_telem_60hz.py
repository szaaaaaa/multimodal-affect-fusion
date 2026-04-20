"""
抽取 telem 60Hz 连续流，写到 `features/aligned/telem_60hz/{stem}.pt`。

不做 0.2s 聚合。把 `gameInt.csv` 的 23 个战斗相关列重采样到固定 60Hz 网格
`t_g[i] = (i + 0.5) / 60` 秒（i = 0..T_t-1，T_t = T_v * 12），用最近邻前向填充。
这样所有 session 的 `T_t = T_v * 12`，与 video 5Hz 的 T_v 精确对齐为 12 倍，
encoder 内部用 stride=12 下采样即可回到 T_v，保留原始粒度。

列顺序（23 维）
-------------
- continuous(7) : health, magazineAmmo, reserveAmmo, num_enemies_in_FOV,
                  num_enemies_in_close_range, num_enemies_in_mid_range, health_danger
- event(4)      : isReloading, bulletShots, bulletHits, combat
- sparse(5)     : damage, damageToEnemy, damageFromEnemy, deathVictim, deathAttacker  [-1→0]
- fov(4)        : inFOV1, inFOV2, inFOV3, inFOV4  [-1→0]
- categorical(3): aimTarget+1 (原 -1=none → 0)；
                  aimBodyPart、hitGroup 保持原值（数据自带 0=none 语义）

输出字段（每个 session 的 .pt）
--------------------------------
- features : FloatTensor[T_t, 23]
- mask     : BoolTensor[T_t]            （几乎全 True；若 t_g 落在首个 gameInt 采样
                                         之前，则该时间步 features=0、mask=False）
- bin_id   : LongTensor[T_t]            floor(t_g / 0.2)，用于 encoder 下采样到 T_v
- T_v      : int

同时在输出目录生成：
- stats.json : {"mean": [23], "std": [23]}，用于 normalize

CLI
---
    python -m encoder.common.extract_telem_60hz \
        --raw-root   ".../researchdata/data" \
        --video-root ".../features/aligned/video_clip" \
        --out-root   ".../features/aligned/telem_60hz"
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import List

import numpy as np
import pandas as pd
import torch

from encoder.common.time_origin import VIDEO_FRAME_DT


SAMPLE_RATE = 60  # Hz, upsample/resample target
BIN_PER_FRAME = int(SAMPLE_RATE * VIDEO_FRAME_DT)  # = 12

CONTINUOUS_COLS = [
    "health", "magazineAmmo", "reserveAmmo",
    "num_enemies_in_FOV", "num_enemies_in_close_range", "num_enemies_in_mid_range",
    "health_danger",
]
EVENT_COLS = ["isReloading", "bulletShots", "bulletHits", "combat"]
SPARSE_COLS = ["damage", "damageToEnemy", "damageFromEnemy", "deathVictim", "deathAttacker"]
FOV_COLS = ["inFOV1", "inFOV2", "inFOV3", "inFOV4"]
CATEGORICAL_COLS = ["aimTarget", "aimBodyPart", "hitGroup"]

ALL_COLS = CONTINUOUS_COLS + EVENT_COLS + SPARSE_COLS + FOV_COLS + CATEGORICAL_COLS
FEATURE_DIM = len(ALL_COLS)  # 23


def _session_root_from_stem(raw_root: Path, stem: str) -> Path:
    subj, part = stem.split("_")
    return raw_root / subj / part


def list_sessions(km_aligned_root: Path) -> List[str]:
    return sorted(p.stem for p in km_aligned_root.glob("*.pt"))


def _load_and_clean(csv_path: Path) -> pd.DataFrame:
    """读 gameInt.csv 并做 schema 清洗；时间列保留为绝对秒（未减 t0）。"""
    df = pd.read_csv(csv_path)
    df = df.loc[:, ["time"] + ALL_COLS].copy()
    df = df.sort_values("time", kind="mergesort").reset_index(drop=True)

    # ammo 哨兵：未持武器/死亡时 raw 列存 -1 或 uint8 翻转 255，统一视为"0 弹"
    for c in ("magazineAmmo", "reserveAmmo"):
        df.loc[(df[c] < 0) | (df[c] >= 200), c] = 0
    # sparse/fov 中 -1 表示"无事件"→ 0
    for c in SPARSE_COLS + FOV_COLS:
        df[c] = df[c].where(df[c] >= 0, 0)
    # aimTarget 原值 ∈ {-1,0,1}：-1=none，+1 shift 得 {0,1,2}
    df["aimTarget"] = df["aimTarget"].astype(int) + 1
    # aimBodyPart / hitGroup 原值 0 已是 "none/unknown"，保持原值
    for c in ("aimBodyPart", "hitGroup"):
        df[c] = df[c].astype(int)
    return df


def extract_session(raw_root: Path, stem: str, T_v_cap: int) -> dict | None:
    """gameInt 自身 t0 + 外部 T_v_cap（= baseline aligned km T）确定 60Hz 栅格。

    T_v_cap 保证 km/telem 在数据层共享同一 T_v，与 baseline 的对齐约定一致。
    """
    sroot = _session_root_from_stem(raw_root, stem)
    gi_path = sroot / "gameInt.csv"
    if not gi_path.is_file():
        return None

    df = _load_and_clean(gi_path)
    if len(df) == 0 or T_v_cap <= 0:
        return None

    t0 = float(df["time"].iloc[0])
    T_v_telem = T_v_cap
    T_t = T_v_telem * BIN_PER_FRAME
    t_grid = (np.arange(T_t, dtype=np.float64) + 0.5) / SAMPLE_RATE  # 相对 t0 秒

    t_rel = df["time"].to_numpy(dtype=np.float64) - t0
    idx = np.searchsorted(t_rel, t_grid, side="right") - 1
    mask_np = idx >= 0
    idx_clipped = np.clip(idx, 0, len(df) - 1)

    feats_np = df[ALL_COLS].to_numpy(dtype=np.float32)  # [N, 23]
    sampled = feats_np[idx_clipped]
    sampled[~mask_np] = 0.0

    features = torch.from_numpy(sampled).float()
    mask = torch.from_numpy(mask_np).bool()
    bin_id = torch.arange(T_t) // BIN_PER_FRAME

    return {
        "features": features,
        "mask": mask,
        "bin_id": bin_id.long(),
        "T_v": T_v_telem,
        "t0_sec": t0,
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw-root", type=Path, required=True)
    ap.add_argument("--km-aligned-root", type=Path, required=True,
                    help="用于取 session stem 列表（例如 features/aligned/km）")
    ap.add_argument("--out-root", type=Path, required=True)
    args = ap.parse_args()

    stems = list_sessions(args.km_aligned_root)
    print(f"[telem_60hz] {len(stems)} sessions; feature_dim={FEATURE_DIM}")

    args.out_root.mkdir(parents=True, exist_ok=True)

    # 两遍：先算均值方差（在线 Welford），再写文件
    sums = np.zeros(FEATURE_DIM, dtype=np.float64)
    sqs = np.zeros(FEATURE_DIM, dtype=np.float64)
    total = 0
    cache: dict[str, dict] = {}
    missing: List[str] = []

    for i, stem in enumerate(stems, 1):
        aligned_km = torch.load(
            args.km_aligned_root / f"{stem}.pt", map_location="cpu", weights_only=False
        )
        T_v_cap = int(aligned_km["features"].shape[0])
        feats = extract_session(args.raw_root, stem, T_v_cap)
        if feats is None:
            missing.append(stem)
            print(f"[{i}/{len(stems)}] {stem}: MISSING")
            continue
        cache[stem] = feats
        v = feats["features"].numpy()[feats["mask"].numpy()]  # 只统计有效栅格点
        sums += v.sum(axis=0)
        sqs += (v * v).sum(axis=0)
        total += v.shape[0]
        print(f"[{i}/{len(stems)}] {stem}: T_t={feats['features'].shape[0]} T_v={feats['T_v']}")

    mean = (sums / max(total, 1)).tolist()
    var = (sqs / max(total, 1)) - np.array(mean) ** 2
    std = np.sqrt(np.maximum(var, 1e-6)).tolist()

    (args.out_root / "stats.json").write_text(
        json.dumps(
            {"columns": ALL_COLS, "feature_dim": FEATURE_DIM, "mean": mean, "std": std, "total_samples": total},
            indent=2,
        ),
        encoding="utf-8",
    )
    (args.out_root / "missing.json").write_text(
        json.dumps(missing, indent=2), encoding="utf-8"
    )

    for stem, feats in cache.items():
        torch.save(feats, args.out_root / f"{stem}.pt")
    print(f"[done] wrote {len(cache)} sessions, missing={len(missing)}, out={args.out_root}")


if __name__ == "__main__":
    main()
