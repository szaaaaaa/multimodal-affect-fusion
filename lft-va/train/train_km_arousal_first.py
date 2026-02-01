"""
Train KM-only arousal regression (session-level).

"""

from __future__ import annotations

import json
import random
import argparse
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
    if str(src_dir) not in sys.path:
        sys.path.insert(0, str(src_dir))


def _set_seed(seed: int = 42) -> None:
    random.seed(seed)
    torch.manual_seed(seed)


def _run_epoch(
    loader: DataLoader,
    model: torch.nn.Module,
    optimizer=None,
    log_fh=None,
    phase: str = "train",
    log_interval: int = 50,
    print_y_stats: bool = False,
) -> float:
    total = 0.0
    count = 0
    train_mode = optimizer is not None
    model.train(train_mode)

    for step, batch in enumerate(loader, start=1):
        km = batch["km"]
        km_mask = batch["km_mask"]
        y = batch["y"]

        y_hat = model(km, km_mask)
        loss = F.smooth_l1_loss(y_hat, y)

        if train_mode:
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        bs = km.shape[0]
        total += loss.item() * bs
        count += bs

        if print_y_stats and step == 1:
            y_min = float(y.min().item())
            y_max = float(y.max().item())
            y_mean = float(y.mean().item())
            stems = batch.get("stem", [])
            stem_preview = list(stems)[:5]
            msg = f"{phase} batch1 y min/max/mean {y_min:.6f}/{y_max:.6f}/{y_mean:.6f} stems {stem_preview}"
            print(msg)
            if log_fh is not None:
                log_fh.write(msg + "\n")

        if log_interval > 0 and (step % log_interval == 0 or step == 1):
            msg = f"{phase} step {step}/{len(loader)} loss {loss.item():.6f}"
            print(msg)
            if log_fh is not None:
                log_fh.write(msg + "\n")

    return total / max(1, count)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--km_encoder", type=str, default="stat", choices=["stat", "cnn"])
    args = parser.parse_args()

    _add_src_to_path()
    _set_seed(123)

    from lft_va.datasets.km_window_dataset import KMWindDataset
    from lft_va.models.km_transformer_min import KMTransformerRegressor

    train_ds = KMWindDataset(split="train")
    val_ds = KMWindDataset(split="val")

    train_loader = DataLoader(train_ds, batch_size=8, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=8, shuffle=False, num_workers=0)

    sample_pt = torch.load(train_ds.files[0], map_location="cpu")
    d_in = int(sample_pt["features"].shape[1])
    model = KMTransformerRegressor(d_in=d_in, km_encoder=args.km_encoder)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_root = "km_arousal_cnn" if args.km_encoder == "cnn" else "km_arousal_first"
    out_dir = Path(__file__).resolve().parents[1] / "outputs" / out_root / timestamp
    out_dir.mkdir(parents=True, exist_ok=True)

    metrics = {"train_loss": [], "val_loss": []}
    best_val = float("inf")

    epochs = 3
    log_path = out_dir / "train.log"
    with log_path.open("w", encoding="utf-8") as log_fh:
        for epoch in range(1, epochs + 1):
            print(f"epoch {epoch} start")
            log_fh.write(f"epoch {epoch} start\n")

            train_loss = _run_epoch(
                train_loader,
                model,
                optimizer,
                log_fh=log_fh,
                phase="train",
                log_interval=50,
                print_y_stats=(epoch == 1),
            )
            val_loss = _run_epoch(
                val_loader, model, optimizer=None, log_fh=log_fh, phase="val", log_interval=50
            )

            metrics["train_loss"].append(train_loss)
            metrics["val_loss"].append(val_loss)

            print(
                f"epoch {epoch} | train loss {train_loss:.6f} | val loss {val_loss:.6f} "
                f"| train sessions {len(train_ds.stems)} | val sessions {len(val_ds.stems)} "
                f"| train windows {len(train_ds)} | val windows {len(val_ds)}"
            )

            if val_loss < best_val:
                best_val = val_loss
                torch.save({"model": model.state_dict(), "epoch": epoch}, out_dir / "best.pt")

        torch.save({"model": model.state_dict(), "epoch": epochs}, out_dir / "last.pt")
        with (out_dir / "metrics.json").open("w", encoding="utf-8") as f:
            json.dump(metrics, f, ensure_ascii=False, indent=2)

        x = list(range(1, epochs + 1))
        plt.figure()
        plt.plot(x, metrics["train_loss"], label="train")
        plt.plot(x, metrics["val_loss"], label="val")
        plt.xlabel("epoch")
        plt.ylabel("loss")
        plt.legend()
        plt.tight_layout()
        plt.savefig(out_dir / "loss_curve.png")
        plt.close()


if __name__ == "__main__":
    main()
