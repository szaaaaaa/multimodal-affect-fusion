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
    mode: str = "fusion",
) -> tuple[float, float]:
    total, count = 0.0, 0
    preds = []
    targets = []
    train_mode = optimizer is not None
    model.train(train_mode)

    with torch.set_grad_enabled(train_mode):
        for batch in loader:
            video = batch["video"].to(device)
            video_mask = batch["video_mask"].to(device)
            y = batch["y"].to(device)

            if mode == "fusion":
                km = batch["km"].to(device)
                km_mask = batch["km_mask"].to(device)
                y_hat = model(video, km, video_mask, km_mask)
            else:
                y_hat = model(video, None, video_mask, None)

            if train_mode:
                loss = F.smooth_l1_loss(y_hat, y)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                bs = video.shape[0]
                total += loss.item() * bs
                count += bs

            preds.append(y_hat.detach().cpu())
            targets.append(y.detach().cpu())

    ccc = _ccc_metric(torch.cat(preds, dim=0), torch.cat(targets, dim=0))
    loss_avg = total / max(1, count) if train_mode else 0.0
    return loss_avg, ccc


def _ccc_metric(pred: torch.Tensor, target: torch.Tensor, eps: float = 1e-8) -> float:
    pred = pred.float()
    target = target.float()
    if pred.ndim == 1:
        pred = pred.unsqueeze(1)
    if target.ndim == 1:
        target = target.unsqueeze(1)

    ccc_vals = []
    for i in range(pred.size(1)):
        x = pred[:, i]
        y = target[:, i]
        mean_x = x.mean()
        mean_y = y.mean()
        var_x = x.var(unbiased=False)
        var_y = y.var(unbiased=False)
        cov = ((x - mean_x) * (y - mean_y)).mean()
        ccc = (2 * cov) / (var_x + var_y + (mean_x - mean_y) ** 2 + eps)
        ccc_vals.append(ccc.item())
    return float(sum(ccc_vals) / len(ccc_vals))


def main():
    parser = argparse.ArgumentParser(description="Train LFT model")
    parser.add_argument("--mode", type=str, default="fusion", choices=["fusion", "video"])
    parser.add_argument("--task", type=str, default="arousal", choices=["arousal", "va"])
    parser.add_argument("--va_mode", type=str, default="shared", choices=["shared", "video_valence"])
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
    parser.add_argument("--video_dir", type=str, default=None)
    parser.add_argument("--km_dir", type=str, default=None)
    parser.add_argument("--labels_path", type=str, default=None)
    parser.add_argument("--split_path", type=str, default=None)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    _add_src_to_path()
    _set_seed(args.seed)

    from models.late_fusion_transformer import LateFusionTransformer, LFTConfig

    # Dataset
    if args.mode == "fusion":
        from datasets import MultimodalDataset
        train_ds = MultimodalDataset(
            video_dir=args.video_dir,
            km_dir=args.km_dir,
            labels_path=args.labels_path,
            split_path=args.split_path,
            split="train",
            video_win_len=args.video_win_len,
            km_win_len=args.km_win_len,
        )
        val_ds = MultimodalDataset(
            video_dir=args.video_dir,
            km_dir=args.km_dir,
            labels_path=args.labels_path,
            split_path=args.split_path,
            split="val",
            video_win_len=args.video_win_len,
            km_win_len=args.km_win_len,
        )
    else:
        from datasets import VideoWindDataset
        train_ds = VideoWindDataset(
            data_dir=args.video_dir,
            labels_path=args.labels_path,
            split_path=args.split_path,
            split="train",
            win_len=args.video_win_len,
        )
        val_ds = VideoWindDataset(
            data_dir=args.video_dir,
            labels_path=args.labels_path,
            split_path=args.split_path,
            split="val",
            win_len=args.video_win_len,
        )

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=0)

    print(f"Train samples: {len(train_ds)}, Val samples: {len(val_ds)}")
    task_dim = 2 if args.task == "va" else 1
    sample_dim = int(train_ds[0]["y"].numel())
    if sample_dim != task_dim:
        raise ValueError(
            f"Label dim mismatch: task expects {task_dim}, dataset returns {sample_dim}. "
            "Use a VA label file for --task va."
        )

    # Model
    config = LFTConfig(
        d_model=args.d_model,
        nhead=args.nhead,
        num_encoder_layers=args.num_layers,
        video_feature_dim=args.video_feature_dim,
        km_feature_dim=args.km_feature_dim,
        output_dim=task_dim,
        va_mode=args.va_mode,
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
    metrics = {"train_loss": [], "train_ccc": [], "val_ccc": []}
    best_val = float("-inf")

    for epoch in range(1, args.epochs + 1):
        train_loss, train_ccc = _run_epoch(train_loader, model, optimizer, args.device, "train", args.mode)
        _, val_ccc = _run_epoch(val_loader, model, None, args.device, "val", args.mode)

        metrics["train_loss"].append(train_loss)
        metrics["train_ccc"].append(train_ccc)
        metrics["val_ccc"].append(val_ccc)

        print(
            f"Epoch {epoch}/{args.epochs} | train_loss: {train_loss:.6f} "
            f"| train_ccc: {train_ccc:.4f} | val_ccc: {val_ccc:.4f}"
        )

        if val_ccc > best_val:
            best_val = val_ccc
            torch.save({"model": model.state_dict(), "epoch": epoch, "config": config}, out_dir / "best.pt")

    # Save final model and metrics
    torch.save({"model": model.state_dict(), "epoch": args.epochs, "config": config}, out_dir / "last.pt")
    with (out_dir / "metrics.json").open("w") as f:
        json.dump(metrics, f, indent=2)

    # Plot loss curve
    plt.figure()
    plt.plot(metrics["train_loss"], label="train_loss")
    plt.plot(metrics["train_ccc"], label="train_ccc")
    plt.plot(metrics["val_ccc"], label="val_ccc")
    plt.xlabel("Epoch")
    plt.ylabel("Metric")
    plt.legend()
    plt.savefig(out_dir / "loss_curve.png")
    plt.close()

    print(f"Training complete. Best val CCC: {best_val:.4f}")
    print(f"Outputs saved to: {out_dir}")


if __name__ == "__main__":
    main()
