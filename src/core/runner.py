"""
Training runner — the stable entry point that never changes.

训练运行器 — 稳定的入口点，未来扩展不改此文件。

Builds all modules from config via registries, then runs the training loop.
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any, Dict, List, Optional

import torch
from torch import nn

from src.core.config import Config
from src.core.logging import create_run_dir, save_metrics, save_run_metadata
from src.core.registry import (
    DATAMODULES,
    FUSIONS,
    HEADS,
    LOSSES,
    METRICS,
    get_encoder_registry,
)
from src.core.seed import set_seed

# Ensure all modules are imported so registrations happen
import src.models.encoders.km      # noqa: F401
import src.models.encoders.video   # noqa: F401
import src.models.fusions          # noqa: F401
import src.models.heads            # noqa: F401
import src.losses                  # noqa: F401
import src.metrics                 # noqa: F401
import src.data.datamodules        # noqa: F401


class MultimodalModel(nn.Module):
    """
    Thin wrapper that composes encoders + fusion + head into a single nn.Module.

    将 encoder + fusion + head 组合为单一 nn.Module 的薄包装。
    """

    def __init__(self, encoders: nn.ModuleDict, fusion: nn.Module, head: nn.Module):
        super().__init__()
        self.encoders = encoders
        self.fusion = fusion
        self.head = head

    def forward(self, x_dict, mask_dict):
        z_dict = {}
        for mod, encoder in self.encoders.items():
            z_dict[mod] = encoder(x_dict[mod], mask_dict.get(mod))

        out_mask_dict = {mod: z_dict[mod]["mask"] for mod in z_dict}
        h = self.fusion(z_dict, out_mask_dict)
        return self.head(h)


class Runner:
    """
    Training runner — builds everything from config, runs train/val loop.

    训练运行器 — 根据配置构建所有模块，执行训练/验证循环。
    """

    def __init__(self, cfg: Config, resume: Optional[str] = None):
        self.cfg = cfg
        device_cfg = cfg.get("device", "auto")
        self.device = "cuda" if device_cfg == "auto" and torch.cuda.is_available() else (
            "cpu" if device_cfg == "auto" else device_cfg
        )
        self.resume_ckpt_path, self.resume_run_dir = self._resolve_resume_path(resume)
        self.start_epoch = 1
        self.history: Dict[str, List] = {}
        self.best_val_metric = float("-inf")
        self.best_epoch = 0
        self.patience_counter = 0
        set_seed(cfg.get("train", {}).get("seed", 42))
        self._build()

    def _resolve_resume_path(self, resume: Optional[str]) -> tuple[Optional[Path], Optional[Path]]:
        if not resume:
            return None, None

        path = Path(resume).expanduser()
        if path.is_dir():
            ckpt_path = path / "ckpt_last.pt"
            run_dir = path
        else:
            ckpt_path = path
            run_dir = path.parent

        if not ckpt_path.exists():
            raise FileNotFoundError(f"Resume checkpoint not found: {ckpt_path}")

        return ckpt_path, run_dir

    def _build(self):
        cfg = self.cfg

        # 1. Data
        data_cfg = cfg.get("data", {})
        self.modalities = data_cfg.get("modalities", ["video", "km"])
        dm_name = data_cfg.get("name", "amucs")
        dm_cfg = dict(data_cfg)
        dm_cfg["batch_size"] = cfg.get("train", {}).get("batch_size", 8)
        self.dm = DATAMODULES.build(dm_name, dm_cfg)

        # 2. Encoders
        model_cfg = cfg.get("model", {})
        d_model = model_cfg.get("d_model", 256)
        encoders_cfg = model_cfg.get("encoders", {})

        encoder_modules = {}
        for mod in self.modalities:
            enc_cfg = dict(encoders_cfg.get(mod, {}))
            enc_name = enc_cfg.pop("name", "stat" if mod == "km" else "resnet2d")
            enc_cfg["d_model"] = d_model
            registry = get_encoder_registry(mod)
            encoder_modules[mod] = registry.build(enc_name, enc_cfg)

        self.encoders = nn.ModuleDict(encoder_modules)

        # 3. Fusion
        fusion_cfg = dict(model_cfg.get("fusion", {}))
        fusion_name = fusion_cfg.pop("name", "lft")
        fusion_cfg["d_model"] = d_model
        self.fusion = FUSIONS.build(fusion_name, fusion_cfg)

        # 4. Head
        head_cfg = dict(model_cfg.get("head", {}))
        head_name = head_cfg.pop("name", "regression")
        head_cfg["d_model"] = d_model
        self.head = HEADS.build(head_name, head_cfg)

        # Compose into a single model
        self.model = MultimodalModel(self.encoders, self.fusion, self.head)
        self.model.to(self.device)

        # 5. Loss
        train_cfg = cfg.get("train", {})
        loss_name = train_cfg.get("loss", "ccc")
        self.loss_fn = LOSSES.build(loss_name)

        # 6. Metrics
        eval_cfg = cfg.get("eval", {})
        metric_names = eval_cfg.get("metrics", ["ccc"])
        self.metric_fns = {name: METRICS.build(name) for name in metric_names}

        # 7. Optimizer
        opt_cfg = train_cfg.get("optimizer", {})
        opt_name = opt_cfg.get("name", "adamw")
        lr = opt_cfg.get("lr", train_cfg.get("lr", 1e-4))
        weight_decay = opt_cfg.get("weight_decay", 0.01)

        if opt_name == "adam":
            self.optimizer = torch.optim.Adam(self.model.parameters(), lr=lr, weight_decay=weight_decay)
        else:
            self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=lr, weight_decay=weight_decay)

        # 8. Training params
        self.epochs = train_cfg.get("epochs", 50)
        self.seed = train_cfg.get("seed", 42)
        self.modality_dropout = train_cfg.get("modality_dropout", 0.0)

        # Early stopping
        es_cfg = train_cfg.get("early_stopping", {})
        self.es_patience = es_cfg.get("patience", 0)  # 0 = disabled
        self.es_metric = es_cfg.get("metric", "val_ccc")
        self.es_mode = es_cfg.get("mode", "max")

        # Run directory
        if self.resume_run_dir is not None:
            self.run_dir = self.resume_run_dir
            self.run_dir.mkdir(parents=True, exist_ok=True)
        else:
            runs_dir = Path(cfg.get("runs_dir", "runs"))
            fusion_name_for_dir = cfg.get("model", {}).get("fusion", {}).get("name", "lft")
            self.run_dir = create_run_dir(
                runs_dir,
                dataset=data_cfg.get("name", "amucs"),
                fusion=fusion_name_for_dir,
                modalities=self.modalities,
                seed=self.seed,
            )
            save_run_metadata(self.run_dir, dict(cfg), self.seed)

        if self.resume_ckpt_path is not None:
            self._load_checkpoint(self.resume_ckpt_path)

    def _checkpoint_payload(self, epoch: int) -> Dict[str, Any]:
        return {
            "model": self.model.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "epoch": epoch,
            "config": dict(self.cfg),
            "history": self.history,
            "best_val_metric": self.best_val_metric,
            "best_epoch": self.best_epoch,
            "patience_counter": self.patience_counter,
        }

    def _save_checkpoint(self, path: Path, epoch: int) -> None:
        tmp_path = path.with_suffix(path.suffix + ".tmp")
        torch.save(self._checkpoint_payload(epoch), tmp_path)
        tmp_path.replace(path)

    def _load_checkpoint(self, ckpt_path: Path) -> None:
        ckpt = torch.load(ckpt_path, map_location=self.device)
        model_state = ckpt.get("model")
        if model_state is None:
            raise KeyError(f"Checkpoint missing 'model' state: {ckpt_path}")
        self.model.load_state_dict(model_state)

        optimizer_state = ckpt.get("optimizer")
        if optimizer_state is not None:
            self.optimizer.load_state_dict(optimizer_state)

        ckpt_epoch = int(ckpt.get("epoch", 0))
        self.start_epoch = ckpt_epoch + 1
        self.history = ckpt.get("history", {}) or {}
        self.best_val_metric = ckpt.get("best_val_metric", self.best_val_metric)
        self.best_epoch = int(ckpt.get("best_epoch", 0))
        self.patience_counter = int(ckpt.get("patience_counter", 0))

        print(f"Resumed from checkpoint: {ckpt_path} (epoch {ckpt_epoch})")

    def _run_epoch(self, loader, phase: str = "train"):
        is_train = phase == "train"
        self.model.train(is_train)

        total_loss = 0.0
        count = 0
        all_preds = []
        all_targets = []

        with torch.set_grad_enabled(is_train):
            for batch in loader:
                x_dict = {mod: batch["x"][mod].to(self.device) for mod in batch["x"]}
                mask_dict = {mod: batch["mask"][mod].to(self.device) for mod in batch["mask"]}
                y = batch["y"].to(self.device)

                # Modality dropout (training only)
                if is_train and self.modality_dropout > 0 and len(mask_dict) > 1:
                    for mod in list(mask_dict.keys()):
                        if torch.rand(1).item() < self.modality_dropout:
                            mask_dict[mod] = torch.zeros_like(mask_dict[mod])

                y_hat = self.model(x_dict, mask_dict)

                if is_train:
                    loss = self.loss_fn(y_hat, y)
                    self.optimizer.zero_grad()
                    loss.backward()
                    self.optimizer.step()
                    bs = y.shape[0]
                    total_loss += loss.item() * bs
                    count += bs

                all_preds.append(y_hat.detach().cpu())
                all_targets.append(y.detach().cpu())

        preds = torch.cat(all_preds, dim=0)
        targets = torch.cat(all_targets, dim=0)

        metrics = {}
        for name, fn in self.metric_fns.items():
            metrics[f"{phase}_{name}"] = fn(preds, targets)

        if is_train and count > 0:
            metrics[f"{phase}_loss"] = total_loss / count

        return metrics

    def fit(self):
        """Run the full training loop."""
        train_loader = self.dm.train_dataloader()
        val_loader = self.dm.val_dataloader()

        total_params = sum(p.numel() for p in self.model.parameters())
        trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        print(f"Model parameters: {total_params:,} (trainable: {trainable_params:,})")
        print(f"Train samples: {len(train_loader.dataset)}, Val samples: {len(val_loader.dataset)}")
        print(f"Run directory: {self.run_dir}")

        history: Dict[str, List] = self.history if isinstance(self.history, dict) else {}
        if self.es_mode == "max":
            best_val_metric = self.best_val_metric
            if best_val_metric == float("inf"):
                best_val_metric = float("-inf")
        else:
            best_val_metric = self.best_val_metric
            if best_val_metric == float("-inf"):
                best_val_metric = float("inf")
        best_epoch = self.best_epoch
        patience_counter = self.patience_counter

        start_time = time.time()
        last_completed_epoch = self.start_epoch - 1
        early_stopped = False

        for epoch in range(self.start_epoch, self.epochs + 1):
            train_metrics = self._run_epoch(train_loader, "train")
            val_metrics = self._run_epoch(val_loader, "val")
            last_completed_epoch = epoch

            all_metrics = {**train_metrics, **val_metrics}
            for k, v in all_metrics.items():
                history.setdefault(k, []).append(v)

            # Print progress
            parts = [f"Epoch {epoch}/{self.epochs}"]
            for k, v in sorted(all_metrics.items()):
                parts.append(f"{k}: {v:.4f}")
            print(" | ".join(parts))

            # Check improvement
            current_metric = val_metrics.get(self.es_metric, val_metrics.get("val_ccc", 0))
            improved = (
                (self.es_mode == "max" and current_metric > best_val_metric) or
                (self.es_mode == "min" and current_metric < best_val_metric)
            )

            if improved:
                best_val_metric = current_metric
                best_epoch = epoch
                patience_counter = 0
                self.best_val_metric = best_val_metric
                self.best_epoch = best_epoch
                self.patience_counter = patience_counter
                self.history = history
                self._save_checkpoint(self.run_dir / "ckpt_best.pt", epoch)
            else:
                patience_counter += 1

            self.best_val_metric = best_val_metric
            self.best_epoch = best_epoch
            self.patience_counter = patience_counter
            self.history = history
            self._save_checkpoint(self.run_dir / "ckpt_last.pt", epoch)

            # Early stopping
            if self.es_patience > 0 and patience_counter >= self.es_patience:
                print(f"Early stopping at epoch {epoch} (patience={self.es_patience})")
                early_stopped = True
                break

        elapsed = time.time() - start_time

        if last_completed_epoch < self.start_epoch:
            print("No training epochs executed. Check --resume checkpoint epoch and configured train.epochs.")

        # Test evaluation
        test_metrics = {}
        test_loader = self.dm.test_dataloader()
        if test_loader is not None:
            test_metrics = self._run_epoch(test_loader, "test")

        # Save final metrics
        final_metrics = {
            "best_val_metric": best_val_metric,
            "best_epoch": best_epoch,
            "total_epochs": last_completed_epoch,
            "early_stopped": early_stopped,
            "total_params": total_params,
            "train_time_s": round(elapsed, 1),
            **{k: v[-1] for k, v in history.items() if isinstance(v, list) and v},
            **test_metrics,
        }
        save_metrics(self.run_dir, final_metrics)

        print(f"\nTraining complete. Best {self.es_metric}: {best_val_metric:.4f} (epoch {best_epoch})")
        print(f"Outputs saved to: {self.run_dir}")

        return final_metrics

