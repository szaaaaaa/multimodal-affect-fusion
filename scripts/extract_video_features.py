"""
Extract gameplay visual features from video frames.

Supports multiple backbones:
  - resnet50: ResNet-50 (ImageNet), output 2048-dim
  - clip_vit_l14: CLIP ViT-L/14, output 768-dim

使用预训练视觉模型从游戏画面视频帧提取特征。
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from datetime import datetime
from contextlib import nullcontext

import cv2
import torch

try:
    from torchvision.models import resnet50, ResNet50_Weights
    TORCHVISION_AVAILABLE = True
except ImportError:
    TORCHVISION_AVAILABLE = False

try:
    import open_clip
    OPEN_CLIP_AVAILABLE = True
except ImportError:
    OPEN_CLIP_AVAILABLE = False


IMAGENET_MEAN = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
IMAGENET_STD = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
VIDEO_EXTS = {".mp4", ".avi", ".mov", ".mkv"}


# ── Backbone builders ────────────────────────────────────


def _build_resnet50(device: str, pretrained: bool, freeze: bool) -> torch.nn.Module:
    if not TORCHVISION_AVAILABLE:
        raise ImportError("torchvision is required. Install it with: pip install torchvision")

    weights = ResNet50_Weights.DEFAULT if pretrained else None
    model = resnet50(weights=weights)
    model.fc = torch.nn.Identity()
    model.to(device)
    model.eval()

    if freeze:
        for p in model.parameters():
            p.requires_grad = False

    return model


def _build_clip_vit_l14(device: str) -> tuple:
    """Build CLIP ViT-L/14 visual encoder. Returns (model, preprocess_transform)."""
    if not OPEN_CLIP_AVAILABLE:
        raise ImportError("open_clip is required. Install with: pip install open_clip_torch")

    model, _, preprocess = open_clip.create_model_and_transforms(
        "ViT-L-14", pretrained="openai", device=device,
    )
    model.eval()
    for p in model.parameters():
        p.requires_grad = False

    return model, preprocess


# ── Frame preprocessing ──────────────────────────────────


def _preprocess_frame_resnet(frame_bgr, frame_size: int) -> torch.Tensor:
    frame = cv2.resize(frame_bgr, (frame_size, frame_size))
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    tensor = torch.from_numpy(frame).permute(2, 0, 1).float() / 255.0
    return tensor.unsqueeze(0)


def _preprocess_frame_clip(frame_bgr, preprocess) -> torch.Tensor:
    """Preprocess frame using CLIP's own transform pipeline."""
    from PIL import Image
    frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    pil_img = Image.fromarray(frame_rgb)
    return preprocess(pil_img).unsqueeze(0)


def extract_video_features(
    video_path: Path,
    model: torch.nn.Module,
    device: str,
    target_fps: int = 8,
    frame_size: int = 224,
    batch_size: int = 32,
    use_amp: bool = False,
    backbone: str = "resnet50",
    clip_preprocess=None,
) -> dict:
    """
    Extract per-frame visual features from a video.

    Parameters
    ----------
    backbone : str
        "resnet50" or "clip_vit_l14"
    clip_preprocess : callable, optional
        CLIP preprocessing transform (required when backbone is clip_*)

    Returns
    -------
    dict
        {"features": Tensor[T, D], "timestamps": list, "fps": float, "sample_fps": float}
    """
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS) or 0.0
    if fps <= 0:
        fps = float(target_fps)

    stride = max(int(round(fps / target_fps)), 1)
    sample_fps = fps / stride

    is_clip = backbone.startswith("clip")

    features_chunks = []
    timestamps = []
    batch = []
    batch_indices = []

    if not is_clip:
        mean = IMAGENET_MEAN.to(device)
        std = IMAGENET_STD.to(device)

    amp_ctx = (
        torch.autocast(device_type="cuda", dtype=torch.float16)
        if use_amp and str(device).startswith("cuda")
        else nullcontext()
    )

    frame_idx = 0
    with torch.no_grad():
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            if frame_idx % stride == 0:
                if is_clip:
                    batch.append(_preprocess_frame_clip(frame, clip_preprocess))
                else:
                    batch.append(_preprocess_frame_resnet(frame, frame_size))
                batch_indices.append(frame_idx)

                if len(batch) >= batch_size:
                    x = torch.cat(batch, dim=0).to(device, non_blocking=True)
                    if not is_clip:
                        x = (x - mean) / std
                    with amp_ctx:
                        if is_clip:
                            feats = model.encode_image(x).detach().float().cpu()
                        else:
                            feats = model(x).detach().cpu()
                    features_chunks.append(feats)
                    timestamps.extend([i / fps for i in batch_indices])
                    batch.clear()
                    batch_indices.clear()

            frame_idx += 1

        if batch:
            x = torch.cat(batch, dim=0).to(device, non_blocking=True)
            if not is_clip:
                x = (x - mean) / std
            with amp_ctx:
                if is_clip:
                    feats = model.encode_image(x).detach().float().cpu()
                else:
                    feats = model(x).detach().cpu()
            features_chunks.append(feats)
            timestamps.extend([i / fps for i in batch_indices])

    cap.release()

    if not features_chunks:
        raise RuntimeError(f"No features extracted from {video_path}")

    features = torch.cat(features_chunks, dim=0)

    return {
        "features": features,
        "timestamps": timestamps,
        "fps": fps,
        "sample_fps": sample_fps,
        "stride": stride,
    }


def _list_videos(root: Path) -> list[Path]:
    return sorted([p for p in root.rglob("*") if p.is_file() and p.suffix.lower() in VIDEO_EXTS])


def _discover_sessions(video_dir: Path, session_mode: str) -> list[tuple[str, Path, list[Path]]]:
    subdirs = sorted([p for p in video_dir.iterdir() if p.is_dir()])
    use_subdirs = session_mode == "subdirs" or (session_mode == "auto" and len(subdirs) > 0)

    if not use_subdirs:
        videos = sorted([p for p in video_dir.iterdir() if p.is_file() and p.suffix.lower() in VIDEO_EXTS])
        return [("__root__", video_dir, videos)]

    sessions = []
    for session_dir in subdirs:
        videos = _list_videos(session_dir)
        sessions.append((session_dir.name, session_dir, videos))
    return sessions


def _session_done_path(output_dir: Path, done_dir_name: str, session_name: str) -> Path:
    return output_dir / done_dir_name / f"{session_name}.done.json"


def _save_session_done(done_path: Path, session_name: str, session_dir: Path, expected_outputs: list[Path]) -> None:
    done_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "session": session_name,
        "source_dir": str(session_dir),
        "num_outputs": len(expected_outputs),
        "timestamp": datetime.now().isoformat(timespec="seconds"),
    }
    tmp_path = done_path.with_suffix(done_path.suffix + ".tmp")
    tmp_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    tmp_path.replace(done_path)


def _normalize_session_name(session_name: str) -> str:
    m = re.fullmatch(r"[sS](\d+)", session_name)
    if m:
        return f"S{int(m.group(1)):03d}"
    return session_name.upper()


def _extract_phase_token(video_stem: str) -> str | None:
    # AMuCS file pattern typically starts with P1/P2/P3/P4
    m = re.match(r"(?i)^(p\d+)(?:\b|_)", video_stem)
    if m:
        return m.group(1).upper()
    return None


def _build_output_stem(video_path: Path, session_name: str, name_mode: str) -> str:
    raw_stem = video_path.stem
    if name_mode == "raw":
        return raw_stem
    if name_mode == "session_prefix":
        return f"{_normalize_session_name(session_name)}_{raw_stem}"
    if name_mode == "amucs":
        phase = _extract_phase_token(raw_stem)
        if phase:
            return f"{_normalize_session_name(session_name)}_{phase}"
        return f"{_normalize_session_name(session_name)}_{raw_stem}"

    # auto
    if session_name == "__root__":
        return raw_stem
    phase = _extract_phase_token(raw_stem)
    if phase:
        return f"{_normalize_session_name(session_name)}_{phase}"
    return f"{_normalize_session_name(session_name)}_{raw_stem}"


def main() -> None:
    parser = argparse.ArgumentParser(description="Extract gameplay visual features")
    parser.add_argument("--video_dir", type=str, required=True, help="Directory with video files")
    parser.add_argument(
        "--output_dir",
        type=str,
        default=str(Path(__file__).resolve().parents[1] / "data" / "features" / "amucs" / "video"),
        help="Output directory for .pt files",
    )
    parser.add_argument("--backbone", type=str, default="resnet50",
                        choices=["resnet50", "clip_vit_l14"],
                        help="Visual backbone: resnet50 (2048-d) or clip_vit_l14 (768-d)")
    parser.add_argument("--target_fps", type=int, default=8, help="Target sampling FPS")
    parser.add_argument("--frame_size", type=int, default=224, help="Frame size (square, resnet50 only)")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size for inference")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--amp", action="store_true", help="Enable CUDA AMP fp16 inference")
    parser.add_argument("--no_pretrained", action="store_true", help="Disable pretrained weights (resnet50 only)")
    parser.add_argument("--no_freeze", action="store_true", help="Do not freeze weights (resnet50 only)")
    parser.add_argument(
        "--session_mode",
        type=str,
        default="auto",
        choices=["auto", "flat", "subdirs"],
        help="Session discovery mode: auto (prefer subdirs), flat (only top-level videos), subdirs (each top-level dir is a session).",
    )
    parser.add_argument(
        "--done_dir",
        type=str,
        default=".session_done",
        help="Directory under output_dir to store per-session completion markers.",
    )
    parser.add_argument("--overwrite", action="store_true", help="Overwrite existing output .pt files")
    parser.add_argument(
        "--name_mode",
        type=str,
        default="auto",
        choices=["auto", "raw", "session_prefix", "amucs"],
        help="Output stem naming mode. 'amucs' generates Sxxx_Py style stems from session folder and video name.",
    )
    args = parser.parse_args()

    video_dir = Path(args.video_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    clip_preprocess = None
    if args.backbone == "resnet50":
        if not TORCHVISION_AVAILABLE:
            print("torchvision not installed. Install with: pip install torchvision")
            return
        pretrained = not args.no_pretrained
        freeze = not args.no_freeze
        model = _build_resnet50(args.device, pretrained=pretrained, freeze=freeze)
    elif args.backbone == "clip_vit_l14":
        if not OPEN_CLIP_AVAILABLE:
            print("open_clip not installed. Install with: pip install open_clip_torch")
            return
        print("Loading CLIP ViT-L/14 model (first run downloads ~900MB)...", flush=True)
        model, clip_preprocess = _build_clip_vit_l14(args.device)
        print("CLIP model loaded.", flush=True)
        pretrained = True
    else:
        raise ValueError(f"Unknown backbone: {args.backbone}")

    sessions = _discover_sessions(video_dir, args.session_mode)
    total_videos = sum(len(videos) for _, _, videos in sessions)
    print(f"Found {len(sessions)} session(s), total videos: {total_videos}")

    # Build planned output stems and detect collisions to avoid silent overwrite.
    stem_to_sources: dict[str, list[str]] = {}
    per_session_items: list[tuple[str, Path, list[tuple[Path, str]]]] = []
    for session_name, session_dir, videos in sessions:
        items: list[tuple[Path, str]] = []
        for video_path in videos:
            out_stem = _build_output_stem(video_path, session_name, args.name_mode)
            items.append((video_path, out_stem))
            stem_to_sources.setdefault(out_stem, []).append(str(video_path))
        per_session_items.append((session_name, session_dir, items))
    collision_stems = {k: v for k, v in stem_to_sources.items() if len(v) > 1}
    if collision_stems:
        preview = next(iter(collision_stems.items()))
        stem, sources = preview
        raise RuntimeError(
            "Duplicate video stems detected across sessions; output filenames would collide. "
            f"Example stem='{stem}', sources={sources[:3]}"
        )

    # Build flat list of all (session_name, session_dir, video_path, out_stem) for progress bar
    all_items = []
    for session_name, session_dir, items in per_session_items:
        if not items:
            continue
        expected_outputs = [output_dir / f"{out_stem}.pt" for _, out_stem in items]
        done_path = _session_done_path(output_dir, args.done_dir, session_name)
        already_complete = all(p.exists() for p in expected_outputs)
        if not args.overwrite and (done_path.exists() or already_complete):
            if already_complete and not done_path.exists():
                _save_session_done(done_path, session_name, session_dir, expected_outputs)
            continue
        for video_path, out_stem in items:
            all_items.append((session_name, session_dir, video_path, out_stem, items))

    try:
        from tqdm import tqdm
        pbar = tqdm(total=len(all_items), desc="Extracting", unit="video")
    except ImportError:
        pbar = None

    done_sessions: set = set()
    for session_name, session_dir, video_path, out_stem, items in all_items:
            output_path = output_dir / f"{out_stem}.pt"
            if output_path.exists() and not args.overwrite:
                if pbar:
                    pbar.update(1)
                continue

            try:
                if pbar:
                    pbar.set_postfix_str(f"{session_name}/{output_path.name}")
                result = extract_video_features(
                    video_path=video_path,
                    model=model,
                    device=args.device,
                    target_fps=args.target_fps,
                    frame_size=args.frame_size,
                    batch_size=args.batch_size,
                    use_amp=args.amp,
                    backbone=args.backbone,
                    clip_preprocess=clip_preprocess,
                )
                result["meta"] = {
                    "source": str(video_path),
                    "session": session_name,
                    "output_stem": out_stem,
                    "backbone": args.backbone,
                    "feature_dim": int(result["features"].shape[1]),
                    "pretrained": pretrained,
                    "target_fps": args.target_fps,
                    "frame_size": args.frame_size,
                }
                tmp_path = output_path.with_suffix(output_path.suffix + ".tmp")
                torch.save(result, tmp_path)
                tmp_path.replace(output_path)
            except Exception as e:
                print(f"\nError processing {video_path}: {e}", flush=True)

            if pbar:
                pbar.update(1)

            # Mark session done if all its outputs exist
            if session_name not in done_sessions:
                expected = [output_dir / f"{s}.pt" for _, s in items]
                done_p = _session_done_path(output_dir, args.done_dir, session_name)
                if all(p.exists() for p in expected):
                    _save_session_done(done_p, session_name, session_dir, expected)
                    done_sessions.add(session_name)

    if pbar:
        pbar.close()


if __name__ == "__main__":
    main()

