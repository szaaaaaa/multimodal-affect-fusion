"""
Train Late Fusion Transformer for VA prediction.

训练晚期融合 Transformer 进行 VA 预测。
"""

from __future__ import annotations

import argparse
import json
import random
from datetime import datetime
from pathlib import Path
import sys

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def _add_src_to_path() -> None:
    lft_root = Path(__file__).resolve().parents[1]
    src_dir = lft_root / "src"
    proj_root = lft_root.parent
    if str(src_dir) not in sys.path:
        sys.path.insert(0, str(src_dir))
    if str(proj_root) not in sys.path:
        sys.path.insert(0, str(proj_root))


def _set_seed(seed: int = 42) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _run_epoch(
    loader: DataLoader,
    model: torch.nn.Module,
    optimizer=None,
    device: str = "cpu",
    phase: str = "train",
) -> float:
    total, count = 0.0, 0
    train_mode = optimizer is not None
    model.train(train_mode)

    for batch in loader:
        video = batch["video"].to(device)
        video_mask = batch["video_mask"].to(device)
        km = batch["km"].to(device)
        km_mask = batch["km_mask"].to(device)
        y = batch["y"].to(device)

        y_hat = model(video, km, video_mask, km_mask)
        loss = F.smooth_l1_loss(y_hat, y)

        if train_mode:
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        bs = video.shape[0]
        total += loss.item() * bs
        count += bs

    return total / max(1, count)


def main():
    parser = argparse.ArgumentParser(description="Train LFT model")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--d_model", type=int, default=256)
    parser.add_argument("--nhead", type=int, default=8)
    parser.add_argument("--num_layers", type=int, default=4)
    parser.add_argument("--video_feature_dim", type=int, default=1280)
    parser.add_argument("--km_feature_dim", type=int, default=25)
    parser.add_argument("--video_win_len", type=int, default=100)
    parser.add_argument("--km_win_len", type=int, default=300)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    _add_src_to_path()
    _set_seed(args.seed)

    from lft_va.datasets import MultimodalDataset
    from lft_va.models.late_fusion_transformer import LateFusionTransformer, LFTConfig

    # Dataset
    train_ds = MultimodalDataset(
        split="train",
        video_win_len=args.video_win_len,
        km_win_len=args.km_win_len,
    )
    val_ds = MultimodalDataset(
        split="val",
        video_win_len=args.video_win_len,
        km_win_len=args.km_win_len,
    )

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=0)

    print(f"Train samples: {len(train_ds)}, Val samples: {len(val_ds)}")

    # Model
    config = LFTConfig(
        d_model=args.d_model,
        nhead=args.nhead,
        num_encoder_layers=args.num_layers,
        video_feature_dim=args.video_feature_dim,
        km_feature_dim=args.km_feature_dim,
        output_dim=1,  # Arousal only for now
    )
    model = LateFusionTransformer(config).to(args.device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)

    total_params = sum(p.numel() for p in model.parameters())
    print(f"Model parameters: {total_params:,}")

    # Output directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = Path(__file__).resolve().parents[1] / "outputs" / "lft" / timestamp
    out_dir.mkdir(parents=True, exist_ok=True)

    # Training loop
    metrics = {"train_loss": [], "val_loss": []}
    best_val = float("inf")

    for epoch in range(1, args.epochs + 1):
        train_loss = _run_epoch(train_loader, model, optimizer, args.device, "train")
        val_loss = _run_epoch(val_loader, model, None, args.device, "val")

        metrics["train_loss"].append(train_loss)
        metrics["val_loss"].append(val_loss)

        print(f"Epoch {epoch}/{args.epochs} | train_loss: {train_loss:.6f} | val_loss: {val_loss:.6f}")

        if val_loss < best_val:
            best_val = val_loss
            torch.save({"model": model.state_dict(), "epoch": epoch, "config": config}, out_dir / "best.pt")

    # Save final model and metrics
    torch.save({"model": model.state_dict(), "epoch": args.epochs, "config": config}, out_dir / "last.pt")
    with (out_dir / "metrics.json").open("w") as f:
        json.dump(metrics, f, indent=2)

    # Plot loss curve
    plt.figure()
    plt.plot(metrics["train_loss"], label="train")
    plt.plot(metrics["val_loss"], label="val")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.savefig(out_dir / "loss_curve.png")
    plt.close()

    print(f"Training complete. Best val loss: {best_val:.6f}")
    print(f"Outputs saved to: {out_dir}")


if __name__ == "__main__":
    main()
