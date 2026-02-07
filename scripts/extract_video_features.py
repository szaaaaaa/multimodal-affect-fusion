"""
Extract gameplay visual features from video frames using ResNet-50.

使用 ResNet-50 从游戏画面视频帧提取视觉特征。
"""

from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import torch
from tqdm import tqdm

try:
    from torchvision.models import resnet50, ResNet50_Weights
    TORCHVISION_AVAILABLE = True
except ImportError:
    TORCHVISION_AVAILABLE = False


IMAGENET_MEAN = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
IMAGENET_STD = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)


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


def _preprocess_frame(frame_bgr, frame_size: int, device: str) -> torch.Tensor:
    frame = cv2.resize(frame_bgr, (frame_size, frame_size))
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    tensor = torch.from_numpy(frame).permute(2, 0, 1).float() / 255.0
    tensor = tensor.unsqueeze(0).to(device)
    tensor = (tensor - IMAGENET_MEAN.to(device)) / IMAGENET_STD.to(device)
    return tensor


def extract_video_features(
    video_path: Path,
    model: torch.nn.Module,
    device: str,
    target_fps: int = 8,
    frame_size: int = 224,
    batch_size: int = 32,
) -> dict:
    """
    Extract per-frame ResNet-50 features from a video.

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

    features_chunks = []
    timestamps = []
    batch = []
    batch_indices = []

    frame_idx = 0
    with torch.no_grad():
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            if frame_idx % stride == 0:
                batch.append(_preprocess_frame(frame, frame_size, device))
                batch_indices.append(frame_idx)

                if len(batch) >= batch_size:
                    x = torch.cat(batch, dim=0)
                    feats = model(x).detach().cpu()
                    features_chunks.append(feats)
                    timestamps.extend([i / fps for i in batch_indices])
                    batch.clear()
                    batch_indices.clear()

            frame_idx += 1

        if batch:
            x = torch.cat(batch, dim=0)
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


def main() -> None:
    parser = argparse.ArgumentParser(description="Extract gameplay features using ResNet-50")
    parser.add_argument("--video_dir", type=str, required=True, help="Directory with video files")
    parser.add_argument(
        "--output_dir",
        type=str,
        default=str(Path(__file__).resolve().parents[1] / "data" / "features" / "amucs" / "video"),
        help="Output directory for .pt files",
    )
    parser.add_argument("--target_fps", type=int, default=8, help="Target sampling FPS")
    parser.add_argument("--frame_size", type=int, default=224, help="Frame size (square)")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size for ResNet inference")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--no_pretrained", action="store_true", help="Disable ImageNet pretrained weights")
    parser.add_argument("--no_freeze", action="store_true", help="Do not freeze ResNet weights")
    args = parser.parse_args()

    if not TORCHVISION_AVAILABLE:
        print("torchvision not installed. Install with: pip install torchvision")
        return

    video_dir = Path(args.video_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    pretrained = not args.no_pretrained
    freeze = not args.no_freeze
    model = _build_resnet50(args.device, pretrained=pretrained, freeze=freeze)

    video_exts = {".mp4", ".avi", ".mov", ".mkv"}
    video_files = [f for f in video_dir.iterdir() if f.suffix.lower() in video_exts]

    print(f"Found {len(video_files)} videos in {video_dir}")

    for video_path in tqdm(video_files, desc="Extracting features"):
        output_path = output_dir / f"{video_path.stem}.pt"
        if output_path.exists():
            continue

        try:
            result = extract_video_features(
                video_path=video_path,
                model=model,
                device=args.device,
                target_fps=args.target_fps,
                frame_size=args.frame_size,
                batch_size=args.batch_size,
            )
            result["meta"] = {
                "source": str(video_path),
                "backbone": "resnet50",
                "feature_dim": int(result["features"].shape[1]),
                "pretrained": pretrained,
                "target_fps": args.target_fps,
                "frame_size": args.frame_size,
            }
            torch.save(result, output_path)
        except Exception as e:
            print(f"Error processing {video_path}: {e}")


if __name__ == "__main__":
    main()

