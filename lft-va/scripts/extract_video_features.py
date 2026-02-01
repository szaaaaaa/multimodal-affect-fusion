"""
Extract face features from video frames using EmotiEffLib.

使用 EmotiEffLib 从视频帧提取面部特征。
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Optional

import cv2
import numpy as np
import torch
from tqdm import tqdm

try:
    from emotiefflib.facial_emotions import HSEmotionRecognizer
    EMOTIEFFLIB_AVAILABLE = True
except ImportError:
    EMOTIEFFLIB_AVAILABLE = False


def extract_video_features(
    video_path: Path,
    recognizer,
    sample_rate: int = 5,
    face_detector=None,
) -> dict:
    """
    Extract features from video file.

    Parameters
    ----------
    video_path : Path
        Path to video file.
    recognizer : HSEmotionRecognizer
        EmotiEffLib recognizer instance.
    sample_rate : int
        Extract every N-th frame.
    face_detector : optional
        Face detector (e.g., cv2.CascadeClassifier). If None, use full frame.

    Returns
    -------
    dict
        {"features": Tensor[T, D], "timestamps": list, "fps": float}
    """
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    features_list = []
    timestamps = []
    frame_idx = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        if frame_idx % sample_rate == 0:
            # Detect face or use full frame
            face_img = frame
            if face_detector is not None:
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                faces = face_detector.detectMultiScale(gray, 1.1, 4)
                if len(faces) > 0:
                    x, y, w, h = faces[0]
                    face_img = frame[y:y+h, x:x+w]

            # Resize for model input
            face_img = cv2.resize(face_img, (224, 224))

            # Extract features
            feat = recognizer.extract_features(face_img)
            features_list.append(feat)
            timestamps.append(frame_idx / fps)

        frame_idx += 1

    cap.release()

    if not features_list:
        raise RuntimeError(f"No features extracted from {video_path}")

    features = torch.tensor(np.array(features_list), dtype=torch.float32)

    return {
        "features": features,
        "timestamps": timestamps,
        "fps": fps,
        "sample_rate": sample_rate,
    }


def main():
    parser = argparse.ArgumentParser(description="Extract video features using EmotiEffLib")
    parser.add_argument("--video_dir", type=str, required=True, help="Directory with video files")
    parser.add_argument("--output_dir", type=str, required=True, help="Output directory for .pt files")
    parser.add_argument("--model_name", type=str, default="enet_b0_8_best_afew")
    parser.add_argument("--sample_rate", type=int, default=5, help="Extract every N-th frame")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--use_face_detector", action="store_true", help="Use face detection")
    args = parser.parse_args()

    if not EMOTIEFFLIB_AVAILABLE:
        print("EmotiEffLib not installed. Install with: pip install emotiefflib[torch]")
        return

    video_dir = Path(args.video_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Initialize recognizer
    recognizer = HSEmotionRecognizer(model_name=args.model_name, device=args.device)

    # Face detector (optional)
    face_detector = None
    if args.use_face_detector:
        cascade_path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
        face_detector = cv2.CascadeClassifier(cascade_path)

    # Find video files
    video_exts = {".mp4", ".avi", ".mov", ".mkv"}
    video_files = [f for f in video_dir.iterdir() if f.suffix.lower() in video_exts]

    print(f"Found {len(video_files)} videos in {video_dir}")

    for video_path in tqdm(video_files, desc="Extracting features"):
        output_path = output_dir / f"{video_path.stem}.pt"
        if output_path.exists():
            continue

        try:
            result = extract_video_features(
                video_path, recognizer, args.sample_rate, face_detector
            )
            result["meta"] = {
                "source": str(video_path),
                "model_name": args.model_name,
                "feature_dim": int(result["features"].shape[1]),
            }
            torch.save(result, output_path)
        except Exception as e:
            print(f"Error processing {video_path}: {e}")


if __name__ == "__main__":
    main()
