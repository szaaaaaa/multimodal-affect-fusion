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
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR

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
import src.models.encoders.telem   # noqa: F401
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

    def __init__(
        self,
        encoders: nn.ModuleDict,
        fusion: nn.Module,
        head: nn.Module,
        temporal_encoder: Optional[nn.Module] = None,
        has_align_loss: bool = False,
    ):
        super().__init__()
        self.encoders = encoders
        self.fusion = fusion
        self.head = head
        self.temporal_encoder = temporal_encoder
        self.has_align_loss = has_align_loss
        self._encoder_outputs: Optional[Dict] = None

    def forward(self, x_dict, mask_dict):
        z_dict = {}
        for mod, encoder in self.encoders.items():
            z = encoder(x_dict[mod], mask_dict.get(mod))
            if self.temporal_encoder is not None:
                z["tokens"] = self.temporal_encoder(z["tokens"], z.get("mask"))
            z_dict[mod] = z

        # Only clone during training (controlled by caller via self.training)
        if self.has_align_loss and self.training:
            self._encoder_outputs = {
                mod: {"tokens": z_dict[mod]["tokens"].clone(),
                      "mask": z_dict[mod]["mask"]}
                for mod in z_dict
            }

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
        self.start_batch_in_epoch = 0
        self.history: Dict[str, List] = {}
        self.best_val_metric = float("-inf")
        self.best_epoch = 0
        self.patience_counter = 0
        self.ckpt_every_batches = 1
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
        fusion_name = fusion_cfg.pop("name", "eft")
        fusion_cfg["d_model"] = d_model
        self.fusion = FUSIONS.build(fusion_name, fusion_cfg)

        # 4. Head
        head_cfg = dict(model_cfg.get("head", {}))
        head_name = head_cfg.pop("name", "regression")
        head_cfg["d_model"] = d_model
        self.head = HEADS.build(head_name, head_cfg)

        # 4c. Optional multi-scale temporal encoder (Direction D)
        temporal_encoder = None
        temporal_cfg = model_cfg.get("temporal_encoder", {})
        if temporal_cfg.get("enabled", False):
            from src.models.components.multiscale_temporal import MultiScaleTemporalEncoder
            temporal_encoder = MultiScaleTemporalEncoder(
                d_model=d_model,
                scales=temporal_cfg.get("scales", [1, 5, 25]),
                kernel_size=temporal_cfg.get("kernel_size", 3),
                dropout=temporal_cfg.get("dropout", 0.1),
            )

        # 4d. Optional contrastive alignment loss (Direction C)
        train_cfg = cfg.get("train", {})
        align_cfg = train_cfg.get("contrastive_alignment", {})
        self.align_loss = None
        if align_cfg.get("enabled", False):
            from src.losses.contrastive_alignment import ContrastiveAlignmentLoss
            align_cfg_with_d = dict(align_cfg)
            align_cfg_with_d.setdefault("d_model", d_model)
            self.align_loss = ContrastiveAlignmentLoss(align_cfg_with_d)
            self.align_loss.init_projs(self.modalities)  # eager init before optimizer

        # Compose into a single model
        self.model = MultimodalModel(
            self.encoders, self.fusion, self.head,
            temporal_encoder=temporal_encoder,
            has_align_loss=(self.align_loss is not None),
        )
        self.model.to(self.device)
        if self.align_loss is not None:
            self.align_loss.to(self.device)

        # 4b. Pre-allocate per-modality fusion modules (before optimizer)
        self.fusion.init_for_modalities(self.modalities, self.device)

        # 5. Loss
        train_cfg = cfg.get("train", {})
        loss_name = train_cfg.get("loss", "ccc")
        loss_cfg = train_cfg.get("loss_cfg", None)
        self.loss_fn = LOSSES.build(loss_name, loss_cfg)

        # Task type (regression or classification)
        self.task_type = cfg.get("task_type", "regression")

        # 6. Metrics
        eval_cfg = cfg.get("eval", {})
        metric_names = eval_cfg.get("metrics", ["ccc"])
        self.metric_fns = {name: METRICS.build(name) for name in metric_names}
        task_metrics_cfg = eval_cfg.get("task_metrics", None)
        self.task_metric_fns: Dict[str, Dict[str, Any]] = {}
        if isinstance(task_metrics_cfg, dict):
            for task, task_metric_names in task_metrics_cfg.items():
                if not task_metric_names:
                    continue
                self.task_metric_fns[str(task)] = {
                    name: METRICS.build(name)
                    for name in task_metric_names
                }

        multitask_metric_weights_cfg = eval_cfg.get("multitask_metric_weights", {})
        if isinstance(multitask_metric_weights_cfg, dict):
            self.multitask_metric_weights = {
                str(k): float(v) for k, v in multitask_metric_weights_cfg.items()
            }
        else:
            self.multitask_metric_weights = {}

        # 7. Optimizer
        opt_cfg = train_cfg.get("optimizer", {})
        opt_name = opt_cfg.get("name", "adamw")
        lr = opt_cfg.get("lr", train_cfg.get("lr", 1e-4))
        weight_decay = opt_cfg.get("weight_decay", 0.01)

        opt_params = list(self.model.parameters())
        if self.align_loss is not None:
            opt_params.extend(self.align_loss.parameters())

        if opt_name == "adam":
            self.optimizer = torch.optim.Adam(opt_params, lr=lr, weight_decay=weight_decay)
        else:
            self.optimizer = torch.optim.AdamW(opt_params, lr=lr, weight_decay=weight_decay)

        # 7b. Gradient clipping
        self.grad_clip = float(train_cfg.get("grad_clip", 0.0))  # 0 = disabled

        # 7c. LR Scheduler (cosine with optional warmup)
        sched_cfg = train_cfg.get("scheduler", {})
        sched_name = sched_cfg.get("name", "none")
        self.scheduler = None
        if sched_name != "none":
            warmup_epochs = int(sched_cfg.get("warmup_epochs", 0))
            total_epochs = train_cfg.get("epochs", 50)
            if sched_name == "cosine":
                cosine_sched = CosineAnnealingLR(
                    self.optimizer,
                    T_max=max(total_epochs - warmup_epochs, 1),
                )
                if warmup_epochs > 0:
                    warmup_sched = LinearLR(
                        self.optimizer,
                        start_factor=1e-2,
                        end_factor=1.0,
                        total_iters=warmup_epochs,
                    )
                    self.scheduler = SequentialLR(
                        self.optimizer,
                        schedulers=[warmup_sched, cosine_sched],
                        milestones=[warmup_epochs],
                    )
                else:
                    self.scheduler = cosine_sched

        # 8. Training params
        self.epochs = train_cfg.get("epochs", 50)
        self.seed = train_cfg.get("seed", 42)
        self.modality_dropout = train_cfg.get("modality_dropout", 0.0)
        self.ckpt_every_batches = max(int(train_cfg.get("ckpt_every_batches", 1)), 1)

        # AMP (bfloat16) — no GradScaler needed for bf16
        self.use_amp = bool(train_cfg.get("amp", False))
        self._amp_dtype = torch.bfloat16
        self._amp_device = self.device.split(":")[0]  # "cuda" or "cpu"

        # torch.compile (PyTorch >= 2.0)
        # Disabled when contrastive alignment is active — compile traces
        # forward() and drops the _encoder_outputs side-effect assignment.
        if train_cfg.get("compile", False) and hasattr(torch, "compile"):
            if self.align_loss is not None:
                print("torch.compile skipped (incompatible with contrastive alignment)")
            else:
                self.model = torch.compile(self.model)
                print("torch.compile enabled")

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
            fusion_name_for_dir = cfg.get("model", {}).get("fusion", {}).get("name", "eft")
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

    def _checkpoint_payload(
        self,
        epoch: int,
        batch_in_epoch: Optional[int] = None,
        num_batches_in_epoch: Optional[int] = None,
    ) -> Dict[str, Any]:
        payload = {
            "model": self.model.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "epoch": epoch,
            "batch_in_epoch": batch_in_epoch,
            "num_batches_in_epoch": num_batches_in_epoch,
            "config": dict(self.cfg),
            "history": self.history,
            "best_val_metric": self.best_val_metric,
            "best_epoch": self.best_epoch,
            "patience_counter": self.patience_counter,
        }
        if self.scheduler is not None:
            payload["scheduler"] = self.scheduler.state_dict()
        if self.align_loss is not None:
            payload["align_loss"] = self.align_loss.state_dict()
        return payload

    def _save_checkpoint(
        self,
        path: Path,
        epoch: int,
        batch_in_epoch: Optional[int] = None,
        num_batches_in_epoch: Optional[int] = None,
    ) -> None:
        # Some environments allow writing files but block rename/delete in-place.
        # Save directly to target path to keep checkpointing robust there.
        torch.save(
            self._checkpoint_payload(
                epoch,
                batch_in_epoch=batch_in_epoch,
                num_batches_in_epoch=num_batches_in_epoch,
            ),
            path,
        )

    def _load_checkpoint(self, ckpt_path: Path) -> None:
        ckpt = torch.load(ckpt_path, map_location=self.device)
        model_state = ckpt.get("model")
        if model_state is None:
            raise KeyError(f"Checkpoint missing 'model' state: {ckpt_path}")
        fusion = getattr(self.model, "fusion", None)
        if fusion is not None and hasattr(fusion, "prepare_lazy_layers_from_state_dict"):
            fusion.prepare_lazy_layers_from_state_dict(model_state)
        self.model.load_state_dict(model_state)

        optimizer_state = ckpt.get("optimizer")
        if optimizer_state is not None:
            self.optimizer.load_state_dict(optimizer_state)

        scheduler_state = ckpt.get("scheduler")
        if scheduler_state is not None and self.scheduler is not None:
            self.scheduler.load_state_dict(scheduler_state)

        align_loss_state = ckpt.get("align_loss")
        if align_loss_state is not None and self.align_loss is not None:
            self.align_loss.load_state_dict(align_loss_state)

        ckpt_epoch = int(ckpt.get("epoch", 0))
        ckpt_batch = ckpt.get("batch_in_epoch", None)
        ckpt_num_batches = ckpt.get("num_batches_in_epoch", None)

        if (
            ckpt_batch is not None
            and ckpt_num_batches is not None
            and int(ckpt_batch) < int(ckpt_num_batches) - 1
        ):
            self.start_epoch = ckpt_epoch
            self.start_batch_in_epoch = int(ckpt_batch) + 1
        else:
            self.start_epoch = ckpt_epoch + 1
            self.start_batch_in_epoch = 0

        self.history = ckpt.get("history", {}) or {}
        self.best_val_metric = ckpt.get("best_val_metric", self.best_val_metric)
        self.best_epoch = int(ckpt.get("best_epoch", 0))
        self.patience_counter = int(ckpt.get("patience_counter", 0))

        if self.start_batch_in_epoch > 0:
            print(
                f"Resumed from checkpoint: {ckpt_path} "
                f"(epoch {self.start_epoch}, batch {self.start_batch_in_epoch})"
            )
        else:
            print(f"Resumed from checkpoint: {ckpt_path} (epoch {ckpt_epoch})")

    def _run_epoch(
        self,
        loader,
        phase: str = "train",
        epoch: Optional[int] = None,
        start_batch_in_epoch: int = 0,
    ):
        is_train = phase == "train"
        self.model.train(is_train)

        total_loss = 0.0
        count = 0
        all_preds = []
        all_targets = []
        all_target_masks = []
        multitask_preds: Dict[str, List[torch.Tensor]] = {}
        multitask_targets: Dict[str, List[torch.Tensor]] = {}
        multitask_target_masks: Dict[str, List[torch.Tensor]] = {}
        total_batches = len(loader)

        with torch.set_grad_enabled(is_train):
            for batch_idx, batch in enumerate(loader):
                if batch_idx < start_batch_in_epoch:
                    continue

                x_dict = {mod: batch["x"][mod].to(self.device) for mod in batch["x"]}
                y_batch = batch["y"]
                is_multitask = isinstance(y_batch, dict)

                if is_multitask and "mod_mask" in batch:
                    mod_mask_source = batch["mod_mask"]
                else:
                    mod_mask_source = batch["mask"]
                mask_dict = {mod: mod_mask_source[mod].to(self.device) for mod in mod_mask_source}

                if is_multitask:
                    y = {task: y_batch[task].to(self.device) for task in y_batch}
                    if "mask" in batch and isinstance(batch["mask"], dict):
                        y_mask_source = batch["mask"]
                    elif "y_mask" in batch and isinstance(batch["y_mask"], dict):
                        y_mask_source = batch["y_mask"]
                    else:
                        y_mask_source = None

                    if y_mask_source is not None:
                        y_mask = {
                            task: y_mask_source[task].to(self.device).bool()
                            for task in y
                        }
                    else:
                        y_mask = None
                else:
                    y = y_batch.to(self.device)
                    y_mask = batch["y_mask"].to(self.device) if "y_mask" in batch else None

                # Modality dropout (training only)
                if is_train and self.modality_dropout > 0 and len(mask_dict) > 1:
                    for mod in list(mask_dict.keys()):
                        if torch.rand(1).item() < self.modality_dropout:
                            mask_dict[mod] = torch.zeros_like(mask_dict[mod])

                with torch.autocast(
                    device_type=self._amp_device,
                    dtype=self._amp_dtype,
                    enabled=self.use_amp,
                ):
                    y_hat = self.model(x_dict, mask_dict)

                    if is_train:
                        if is_multitask:
                            loss = self.loss_fn(y_hat, y, y_mask)
                        else:
                            if y_mask is not None:
                                try:
                                    loss = self.loss_fn(y_hat, y, y_mask)
                                except TypeError:
                                    loss = self.loss_fn(y_hat, y)
                            else:
                                loss = self.loss_fn(y_hat, y)

                        # Add contrastive alignment loss (Direction C)
                        if self.align_loss is not None and self.model._encoder_outputs is not None:
                            loss = loss + self.align_loss(self.model._encoder_outputs)

                if is_train:
                    self.optimizer.zero_grad()
                    loss.backward()
                    if self.grad_clip > 0:
                        clip_params = list(self.model.parameters())
                        if self.align_loss is not None:
                            clip_params.extend(self.align_loss.parameters())
                        torch.nn.utils.clip_grad_norm_(clip_params, self.grad_clip)
                    self.optimizer.step()
                    if is_multitask:
                        first_task = next(iter(y))
                        bs = y[first_task].shape[0]
                    else:
                        bs = y.shape[0]
                    total_loss += loss.item() * bs
                    count += bs

                if is_multitask:
                    for task, pred_task in y_hat.items():
                        if task not in y:
                            raise KeyError(f"Missing task label '{task}' in batch['y']")
                        multitask_preds.setdefault(task, []).append(pred_task.detach().cpu())
                        multitask_targets.setdefault(task, []).append(y[task].detach().cpu())
                        if y_mask is not None and task in y_mask:
                            multitask_target_masks.setdefault(task, []).append(y_mask[task].detach().cpu())
                else:
                    all_preds.append(y_hat.detach().cpu())
                    all_targets.append(y.detach().cpu())
                    if y_mask is not None:
                        all_target_masks.append(y_mask.detach().cpu())

                if (
                    is_train
                    and epoch is not None
                    and self.ckpt_every_batches > 0
                    and ((batch_idx + 1) % self.ckpt_every_batches == 0)
                ):
                    self._save_checkpoint(
                        self.run_dir / "ckpt_last.pt",
                        epoch,
                        batch_in_epoch=batch_idx,
                        num_batches_in_epoch=total_batches,
                    )

        if multitask_preds:
            metrics: Dict[str, float] = {}
            metric_values: Dict[str, List[float]] = {}

            for task in sorted(multitask_preds.keys()):
                preds = torch.cat(multitask_preds[task], dim=0)
                targets = torch.cat(multitask_targets[task], dim=0)

                if task in multitask_target_masks and multitask_target_masks[task]:
                    target_mask = torch.cat(multitask_target_masks[task], dim=0).bool()
                else:
                    target_mask = torch.ones_like(targets, dtype=torch.bool)

                preds_valid = preds[target_mask]
                targets_valid = targets[target_mask]
                metric_fns = self.task_metric_fns.get(task, self.metric_fns)

                if preds_valid.numel() == 0:
                    for name in metric_fns:
                        metrics[f"{phase}_{name}_{task}"] = 0.0
                        metric_values.setdefault(name, []).append(0.0)
                        if name == "macro_f1":
                            metrics[f"{phase}_f1_{task}"] = 0.0
                            metric_values.setdefault("f1", []).append(0.0)
                    continue

                for name, fn in metric_fns.items():
                    score = fn(preds_valid, targets_valid)
                    metrics[f"{phase}_{name}_{task}"] = score
                    metric_values.setdefault(name, []).append(score)

                    # Keep short aliases for easier early stopping setup.
                    if name == "macro_f1":
                        metrics[f"{phase}_f1_{task}"] = score
                        metric_values.setdefault("f1", []).append(score)

            for metric_name, vals in metric_values.items():
                if vals:
                    metrics[f"{phase}_{metric_name}_mean"] = float(sum(vals) / len(vals))

            # Optional weighted composite metric for mixed-task early stopping.
            # Example config keys:
            #   eval.multitask_metric_weights:
            #     ccc_arousal: 0.5
            #     macro_f1_trend: 0.5
            if self.multitask_metric_weights:
                weighted_sum = 0.0
                weight_denom = 0.0
                for metric_key, weight in self.multitask_metric_weights.items():
                    key = metric_key
                    if not key.startswith(f"{phase}_"):
                        key = f"{phase}_{key}"
                    if key not in metrics:
                        continue
                    weighted_sum += weight * metrics[key]
                    weight_denom += abs(weight)
                if weight_denom > 0:
                    metrics[f"{phase}_score_mixed"] = float(weighted_sum / weight_denom)

            if is_train and count > 0:
                metrics[f"{phase}_loss"] = total_loss / count
            return metrics

        preds = torch.cat(all_preds, dim=0)
        targets = torch.cat(all_targets, dim=0)

        if all_target_masks:
            target_masks = torch.cat(all_target_masks, dim=0).bool()

            if self.task_type == "classification":
                # preds: [B, T, C], targets: [B, T]
                preds_valid = preds[target_masks]       # [N, C]
                targets_valid = targets[target_masks]    # [N]
                preds_for_metric = preds_valid
                targets_for_metric = targets_valid
            else:
                # regression: squeeze out trailing dim-1
                preds_for_metric = preds
                targets_for_metric = targets

                if preds_for_metric.ndim == 3 and preds_for_metric.shape[-1] == 1:
                    preds_for_metric = preds_for_metric.squeeze(-1)
                if targets_for_metric.ndim == 3 and targets_for_metric.shape[-1] == 1:
                    targets_for_metric = targets_for_metric.squeeze(-1)

                preds_valid = preds_for_metric[target_masks]
                targets_valid = targets_for_metric[target_masks]

                preds_for_metric = preds_valid.unsqueeze(1)
                targets_for_metric = targets_valid.unsqueeze(1)
        else:
            preds_for_metric = preds
            targets_for_metric = targets

        metrics = {}
        for name, fn in self.metric_fns.items():
            metrics[f"{phase}_{name}"] = fn(preds_for_metric, targets_for_metric)

        if is_train and count > 0:
            metrics[f"{phase}_loss"] = total_loss / count

        return metrics

    def fit(self):
        """Run the full training loop."""
        val_loader = self.dm.val_dataloader()

        all_params = list(self.model.parameters())
        if self.align_loss is not None:
            all_params.extend(self.align_loss.parameters())
        total_params = sum(p.numel() for p in all_params)
        trainable_params = sum(p.numel() for p in all_params if p.requires_grad)
        print(f"Model parameters: {total_params:,} (trainable: {trainable_params:,})")
        train_loader_for_stats = self.dm.train_dataloader()
        print(f"Train samples: {len(train_loader_for_stats.dataset)}, Val samples: {len(val_loader.dataset)}")
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
            torch.manual_seed(self.seed + epoch)
            train_loader = self.dm.train_dataloader()
            start_batch_in_epoch = self.start_batch_in_epoch if epoch == self.start_epoch else 0
            train_metrics = self._run_epoch(
                train_loader,
                "train",
                epoch=epoch,
                start_batch_in_epoch=start_batch_in_epoch,
            )
            if self.scheduler is not None:
                self.scheduler.step()
            val_metrics = self._run_epoch(val_loader, "val")
            last_completed_epoch = epoch
            self.start_batch_in_epoch = 0

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
