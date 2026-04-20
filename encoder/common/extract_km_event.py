"""
抽取 km 原始离散事件序列，写到 `features/aligned/km_event/{stem}.pt`。

每个 session 合并 `keyboard.csv` 与 `mousebuttons.csv` 的事件，按绝对秒数排序，
过滤到视频窗口 `[t_origin, t_origin + T_video_seconds]`，再映射到相对秒数。

输出字段（每个 session 的 .pt）
--------------------------------
- event_type_ids : LongTensor[T_k]         事件类型 id（来自 vocab）
- t_rel          : FloatTensor[T_k]        相对视频起点的秒数
- dt             : FloatTensor[T_k]        相邻事件间隔（秒，首个为 0）
- bin_id         : LongTensor[T_k]         所属 5Hz bin（floor(t_rel/0.2)），
                                           用于 encoder 内部 scatter-mean pool 到 T_v
- T_v            : int                     视频帧数（= 该 session 的 5Hz 长度）

同时在输出目录生成：
- vocab.json       : {token: id, ...}，`<pad>`=0, `<unk>`=1
- stats.json       : {"dt_log_mean": ..., "dt_log_std": ...} 用于 normalize
- missing.json     : 列出没有任何事件的 session（通常是 keyboard + mousebuttons 都空）

CLI
---
    python -m encoder.common.extract_km_event \
        --raw-root   "G:/我的云端硬盘/AmuCS/.../researchdata/data" \
        --video-root "G:/我的云端硬盘/AmuCS_experiment/features/aligned/video_clip" \
        --out-root   "G:/我的云端硬盘/AmuCS_experiment/features/aligned/km_event" \
        [--dry-run]
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from collections import Counter
from pathlib import Path
from typing import Dict, List, Tuple

import torch

from encoder.common.time_origin import VIDEO_FRAME_DT


PAD_TOKEN = "<pad>"
UNK_TOKEN = "<unk>"


def _session_root_from_stem(raw_root: Path, stem: str) -> Path:
    """`S001_P3` -> `raw_root/S001/P3`。"""
    subj, part = stem.split("_")
    return raw_root / subj / part


def _read_events(csv_path: Path) -> List[Tuple[float, str]]:
    """读事件 CSV，返回 [(time_abs_sec, channel_0), ...]，跳过 0.0 dummy 行。"""
    out: List[Tuple[float, str]] = []
    if not csv_path.is_file():
        return out
    with csv_path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            t_raw = row.get("time", "").strip()
            ch = row.get("channel_0", "").strip()
            if t_raw == "" or ch == "":
                continue
            try:
                t = float(t_raw)
            except ValueError:
                continue
            if t == 0.0:
                continue
            out.append((t, ch))
    return out


def list_sessions(km_aligned_root: Path) -> List[str]:
    """从 `features/aligned/km/*.pt` 拿 70 个 stem 列表。"""
    return sorted(p.stem for p in km_aligned_root.glob("*.pt"))


def build_vocab(
    raw_root: Path, stems: List[str], min_count: int = 1
) -> Tuple[Dict[str, int], Counter]:
    """扫所有 session 的键鼠事件，按频次排序构建 vocab。"""
    counter: Counter = Counter()
    for stem in stems:
        sroot = _session_root_from_stem(raw_root, stem)
        for fn in ("keyboard.csv", "mousebuttons.csv"):
            for _, ch in _read_events(sroot / fn):
                counter[ch] += 1
    vocab: Dict[str, int] = {PAD_TOKEN: 0, UNK_TOKEN: 1}
    for tok, cnt in counter.most_common():
        if cnt < min_count:
            break
        if tok in vocab:
            continue
        vocab[tok] = len(vocab)
    return vocab, counter


def extract_session(
    raw_root: Path,
    stem: str,
    vocab: Dict[str, int],
    T_v_cap: int,
) -> Dict[str, torch.Tensor] | None:
    """抽单个 session 的事件序列。

    t0 沿用 baseline 的 `min(times)` 约定（= 首个事件时间）。
    T_v 强制对齐 baseline aligned km 的 T（传入 `T_v_cap`），
    过滤掉 bin_id >= T_v_cap 的事件 —— 这会丢弃 raw CSV 里
    少数 session 的离群时间跳变（如 S069_P1 有 67864s 的 gap）。
    """
    sroot = _session_root_from_stem(raw_root, stem)
    events = _read_events(sroot / "keyboard.csv") + _read_events(sroot / "mousebuttons.csv")
    events.sort(key=lambda e: e[0])
    if not events or T_v_cap <= 0:
        return None

    t0 = events[0][0]
    window_end_sec = T_v_cap * VIDEO_FRAME_DT

    types: List[int] = []
    t_rel_list: List[float] = []
    for t_abs, ch in events:
        t_rel = t_abs - t0
        if t_rel >= window_end_sec:
            continue
        types.append(vocab.get(ch, vocab[UNK_TOKEN]))
        t_rel_list.append(t_rel)

    if not types:
        return None

    type_ids = torch.tensor(types, dtype=torch.long)
    t_rel = torch.tensor(t_rel_list, dtype=torch.float32)
    dt = torch.zeros_like(t_rel)
    if len(t_rel) > 1:
        dt[1:] = t_rel[1:] - t_rel[:-1]
    bin_id = torch.clamp((t_rel / VIDEO_FRAME_DT).floor().long(), min=0, max=T_v_cap - 1)

    return {
        "event_type_ids": type_ids,
        "t_rel": t_rel,
        "dt": dt,
        "bin_id": bin_id,
        "T_v": T_v_cap,
        "t0_sec": float(t0),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw-root", type=Path, required=True)
    ap.add_argument("--km-aligned-root", type=Path, required=True,
                    help="用于取 session stem 列表（例如 features/aligned/km）")
    ap.add_argument("--out-root", type=Path, required=True)
    ap.add_argument("--dry-run", action="store_true", help="只构建并打印 vocab，不写特征")
    args = ap.parse_args()

    stems = list_sessions(args.km_aligned_root)
    print(f"[vocab] scanning {len(stems)} sessions under {args.raw_root} ...")

    vocab, counter = build_vocab(args.raw_root, stems)
    print(f"[vocab] size = {len(vocab)} (incl. <pad>, <unk>)")
    print("[vocab] top 30 by frequency:")
    for tok, cnt in counter.most_common(30):
        print(f"  {tok!r:40s}  {cnt}")

    if args.dry_run:
        print("[dry-run] skipping feature write")
        return

    args.out_root.mkdir(parents=True, exist_ok=True)
    (args.out_root / "vocab.json").write_text(
        json.dumps(vocab, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    missing: List[str] = []
    dt_log_vals: List[float] = []
    for i, stem in enumerate(stems, 1):
        aligned_km = torch.load(
            args.km_aligned_root / f"{stem}.pt", map_location="cpu", weights_only=False
        )
        T_v_cap = int(aligned_km["features"].shape[0])
        feats = extract_session(args.raw_root, stem, vocab, T_v_cap)
        if feats is None:
            missing.append(stem)
            print(f"[{i}/{len(stems)}] {stem}: MISSING (no events)")
            continue
        torch.save(feats, args.out_root / f"{stem}.pt")
        T_k = int(feats["event_type_ids"].numel())
        dt = feats["dt"]
        if dt.numel() > 1:
            dt_log_vals.extend(torch.log1p(dt[1:]).tolist())
        print(f"[{i}/{len(stems)}] {stem}: T_k={T_k} T_v={feats['T_v']}")

    if dt_log_vals:
        mean = float(sum(dt_log_vals) / len(dt_log_vals))
        var = float(sum((v - mean) ** 2 for v in dt_log_vals) / len(dt_log_vals))
        std = math.sqrt(var) if var > 0 else 1.0
    else:
        mean, std = 0.0, 1.0
    (args.out_root / "stats.json").write_text(
        json.dumps({"dt_log1p_mean": mean, "dt_log1p_std": std}, indent=2),
        encoding="utf-8",
    )
    (args.out_root / "missing.json").write_text(
        json.dumps(missing, indent=2), encoding="utf-8"
    )
    print(f"[done] vocab_size={len(vocab)} missing={len(missing)} out={args.out_root}")


if __name__ == "__main__":
    main()
