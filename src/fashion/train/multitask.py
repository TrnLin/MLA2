"""Masked auxiliary-task optimisation with Season-only model selection."""

from __future__ import annotations

import math
import time
from collections.abc import Mapping, Sized
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader

from fashion.train.engine import (
    FoldResult,
    TrainConfig,
    _as_id_list,
    _learning_rate_factor,
    _resolve_device,
    _save_checkpoint,
)
from fashion.train.metrics import SEASON_LABELS, multiclass_metrics
from fashion.train.reproducibility import seed_everything


@dataclass(frozen=True)
class MultiTaskLossTerms:
    """Differentiable loss terms for one masked multi-task batch."""

    total: torch.Tensor
    season: torch.Tensor
    auxiliary: torch.Tensor
    auxiliary_count: int


@dataclass(frozen=True)
class RefitTrainConfig:
    """Fixed-epoch optimiser settings after model selection has been frozen."""

    seed: int
    epochs: int
    batch_size: int
    effective_batch_size: int
    learning_rate: float
    weight_decay: float
    gradient_clip_norm: float
    warmup_epochs: float
    use_amp: bool = True
    device: str = "auto"

    def validate(self) -> None:
        """Reject settings that would make the declared refit ambiguous."""
        if type(self.seed) is not int or self.seed < 0:
            raise ValueError("seed must be a non-negative integer")
        if type(self.epochs) is not int or self.epochs < 1:
            raise ValueError("epochs must be a positive integer")
        if type(self.batch_size) is not int or self.batch_size < 1:
            raise ValueError("batch_size must be a positive integer")
        if (
            type(self.effective_batch_size) is not int
            or self.effective_batch_size < self.batch_size
            or self.effective_batch_size % self.batch_size
        ):
            raise ValueError(
                "effective_batch_size must be an integer multiple of batch_size"
            )
        if not math.isfinite(self.learning_rate) or self.learning_rate <= 0:
            raise ValueError("learning_rate must be finite and positive")
        if not math.isfinite(self.weight_decay) or self.weight_decay < 0:
            raise ValueError("weight_decay must be finite and non-negative")
        if not math.isfinite(self.gradient_clip_norm) or self.gradient_clip_norm <= 0:
            raise ValueError("gradient_clip_norm must be finite and positive")
        if (
            not math.isfinite(self.warmup_epochs)
            or self.warmup_epochs < 0
            or self.warmup_epochs >= self.epochs
        ):
            raise ValueError("warmup_epochs must be finite, non-negative, and below epochs")
        if type(self.use_amp) is not bool:
            raise ValueError("use_amp must be a boolean")
        if self.device not in {"auto", "cpu", "cuda"}:
            raise ValueError("device must be 'auto', 'cpu', or 'cuda'")


@dataclass(frozen=True)
class RefitResult:
    """Training-only diagnostics for the declared final epoch state."""

    seed: int
    final_epoch: int
    epochs_completed: int
    history: list[dict[str, float | int]]
    parameter_count: int
    runtime_seconds: float
    peak_vram_mb: float | None
    device: str
    metadata: dict[str, Any]


def masked_multitask_cross_entropy(
    outputs: Mapping[str, torch.Tensor],
    season_targets: torch.Tensor,
    auxiliary_targets: torch.Tensor,
    auxiliary_mask: torch.Tensor,
    *,
    auxiliary_weight: float,
) -> MultiTaskLossTerms:
    """Apply auxiliary cross-entropy only where its training label is available."""
    if not math.isfinite(auxiliary_weight) or auxiliary_weight <= 0:
        raise ValueError("auxiliary_weight must be a finite positive value")
    missing = sorted({"season_logits", "article_type_logits"} - set(outputs))
    if missing:
        raise KeyError(f"multi-task model output is missing keys: {missing}")
    season_logits = outputs["season_logits"]
    auxiliary_logits = outputs["article_type_logits"]
    batch_size = int(season_targets.shape[0])
    if season_targets.ndim != 1 or auxiliary_targets.shape != season_targets.shape:
        raise ValueError("main and auxiliary targets must be matching [batch] tensors")
    if auxiliary_mask.shape != season_targets.shape:
        raise ValueError("auxiliary_mask must match the [batch] target shape")
    if season_logits.ndim != 2 or season_logits.shape[0] != batch_size:
        raise ValueError("season_logits must have shape [batch, class]")
    if auxiliary_logits.ndim != 2 or auxiliary_logits.shape[0] != batch_size:
        raise ValueError("article_type_logits must have shape [batch, class]")

    mask = auxiliary_mask.to(dtype=torch.bool)
    season_loss = F.cross_entropy(season_logits, season_targets)
    auxiliary_count = int(mask.sum().item())
    if auxiliary_count:
        auxiliary_loss = F.cross_entropy(auxiliary_logits[mask], auxiliary_targets[mask])
    else:
        # Keep the zero connected to the auxiliary graph so backward remains uniform.
        auxiliary_loss = auxiliary_logits.sum() * 0.0
    return MultiTaskLossTerms(
        total=season_loss + auxiliary_weight * auxiliary_loss,
        season=season_loss,
        auxiliary=auxiliary_loss,
        auxiliary_count=auxiliary_count,
    )


def _unpack_multitask_batch(
    batch: Any,
    *,
    require_ids: bool,
) -> tuple[Any, Any, Any, Any, Any | None]:
    if not isinstance(batch, Mapping):
        raise TypeError("multi-task batches must be mappings")
    required = {"image", "target", "auxiliary_target", "auxiliary_mask"}
    missing = sorted(required - set(batch))
    if missing:
        raise KeyError(f"multi-task batch is missing keys: {missing}")
    identifiers = batch.get("id")
    if require_ids and identifiers is None:
        raise ValueError("validation batches require stable sample IDs")
    return (
        batch["image"],
        batch["target"],
        batch["auxiliary_target"],
        batch["auxiliary_mask"],
        identifiers,
    )


def _forward_terms(
    model: nn.Module,
    batch: Any,
    *,
    device: torch.device,
    auxiliary_weight: float,
    require_ids: bool,
) -> tuple[
    torch.Tensor,
    torch.Tensor,
    MultiTaskLossTerms,
    Any | None,
]:
    images, season_targets, auxiliary_targets, auxiliary_mask, identifiers = (
        _unpack_multitask_batch(batch, require_ids=require_ids)
    )
    images = images.to(device, non_blocking=True)
    season_targets = season_targets.to(device, dtype=torch.long, non_blocking=True)
    auxiliary_targets = auxiliary_targets.to(device, dtype=torch.long, non_blocking=True)
    auxiliary_mask = auxiliary_mask.to(device, dtype=torch.bool, non_blocking=True)
    outputs = model(images)
    if not isinstance(outputs, Mapping):
        raise TypeError("multi-task model must return a mapping of logits")
    terms = masked_multitask_cross_entropy(
        outputs,
        season_targets,
        auxiliary_targets,
        auxiliary_mask,
        auxiliary_weight=auxiliary_weight,
    )
    return outputs["season_logits"], season_targets, terms, identifiers


def _evaluate_multitask(
    model: nn.Module,
    loader: DataLoader[Any],
    *,
    device: torch.device,
    labels: tuple[str, ...],
    auxiliary_weight: float,
    amp_enabled: bool,
) -> tuple[dict[str, float], dict[str, Any], list[Any], np.ndarray, np.ndarray]:
    model.eval()
    season_loss_sum = 0.0
    auxiliary_loss_sum = 0.0
    sample_count = 0
    auxiliary_count = 0
    identifiers: list[Any] = []
    target_parts: list[np.ndarray] = []
    probability_parts: list[np.ndarray] = []
    with torch.inference_mode():
        for batch in loader:
            with torch.amp.autocast(device_type=device.type, enabled=amp_enabled):
                logits, targets, terms, batch_ids = _forward_terms(
                    model,
                    batch,
                    device=device,
                    auxiliary_weight=auxiliary_weight,
                    require_ids=True,
                )
            batch_size = int(targets.shape[0])
            season_loss_sum += float(terms.season.detach()) * batch_size
            auxiliary_loss_sum += float(terms.auxiliary.detach()) * terms.auxiliary_count
            sample_count += batch_size
            auxiliary_count += terms.auxiliary_count
            identifiers.extend(_as_id_list(batch_ids))
            target_parts.append(targets.detach().cpu().numpy())
            probability_parts.append(torch.softmax(logits.float(), dim=1).cpu().numpy())
    if sample_count == 0:
        raise ValueError("validation loader produced no samples")
    targets_array = np.concatenate(target_parts).astype(np.int64, copy=False)
    probabilities = np.concatenate(probability_parts).astype(np.float64, copy=False)
    if len(identifiers) != sample_count:
        raise ValueError("validation IDs do not align with batch samples")
    if probabilities.shape[1] != len(labels):
        raise ValueError(
            f"model returned {probabilities.shape[1]} classes but labels define {len(labels)}"
        )
    if ((targets_array < 0) | (targets_array >= len(labels))).any():
        raise ValueError("validation targets contain an out-of-range class index")
    season_loss = season_loss_sum / sample_count
    auxiliary_loss = auxiliary_loss_sum / auxiliary_count if auxiliary_count else 0.0
    losses = {
        "total_loss": season_loss + auxiliary_weight * auxiliary_loss,
        "season_loss": season_loss,
        "auxiliary_loss": auxiliary_loss,
        "auxiliary_labeled_samples": float(auxiliary_count),
    }
    true_labels = np.asarray(labels, dtype=object)[targets_array]
    metrics = multiclass_metrics(true_labels, probabilities=probabilities, labels=labels)
    return losses, metrics, identifiers, targets_array, probabilities


def train_masked_multitask_fold(
    model: nn.Module,
    train_loader: DataLoader[Any],
    validation_loader: DataLoader[Any],
    *,
    config: TrainConfig,
    checkpoint_path: str | Path,
    auxiliary_weight: float,
    labels: tuple[str, ...] = SEASON_LABELS,
) -> FoldResult:
    """Train one fold and select epochs only by validation Season macro-F1."""
    config.validate()
    if not math.isfinite(auxiliary_weight) or auxiliary_weight <= 0:
        raise ValueError("auxiliary_weight must be a finite positive value")
    ordered_labels = tuple(str(label) for label in labels)
    if len(ordered_labels) < 2 or len(set(ordered_labels)) != len(ordered_labels):
        raise ValueError("labels must contain at least two unique values")
    if not isinstance(train_loader, Sized) or len(train_loader) == 0:
        raise ValueError("train_loader must contain at least one batch")
    if not isinstance(validation_loader, Sized) or len(validation_loader) == 0:
        raise ValueError("validation_loader must contain at least one batch")
    loader_batch_size = getattr(train_loader, "batch_size", None)
    if loader_batch_size is not None and loader_batch_size != config.batch_size:
        raise ValueError(
            "train_loader batch_size must match TrainConfig.batch_size; "
            f"got {loader_batch_size} and {config.batch_size}"
        )

    seed_everything(config.seed)
    device = _resolve_device(config.device)
    amp_enabled = config.use_amp and device.type == "cuda"
    model = model.to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
    )
    accumulation_steps = config.effective_batch_size // config.batch_size
    updates_per_epoch = math.ceil(len(train_loader) / accumulation_steps)
    total_updates = config.epochs * updates_per_epoch
    warmup_updates = round(config.warmup_epochs * updates_per_epoch)
    scaler = torch.amp.GradScaler(device.type, enabled=amp_enabled)

    output = Path(checkpoint_path)
    history: list[dict[str, float | int]] = []
    best_epoch = 0
    best_metric = -math.inf
    best_metrics: dict[str, Any] = {}
    best_ids: list[Any] = []
    best_targets = np.empty(0, dtype=np.int64)
    best_probabilities = np.empty((0, len(ordered_labels)), dtype=np.float64)
    best_checkpoint_sha256 = ""
    stale_epochs = 0
    update_index = 0
    started = time.perf_counter()
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)

    for epoch in range(1, config.epochs + 1):
        model.train()
        optimizer.zero_grad(set_to_none=True)
        season_loss_sum = 0.0
        auxiliary_loss_sum = 0.0
        epoch_samples = 0
        epoch_auxiliary_count = 0
        for batch_index, batch in enumerate(train_loader):
            group_start = (batch_index // accumulation_steps) * accumulation_steps
            group_size = min(accumulation_steps, len(train_loader) - group_start)
            with torch.amp.autocast(device_type=device.type, enabled=amp_enabled):
                _, targets, terms, _ = _forward_terms(
                    model,
                    batch,
                    device=device,
                    auxiliary_weight=auxiliary_weight,
                    require_ids=False,
                )
                scaled_loss = terms.total / group_size
            scaler.scale(scaled_loss).backward()
            batch_size = int(targets.shape[0])
            season_loss_sum += float(terms.season.detach()) * batch_size
            auxiliary_loss_sum += float(terms.auxiliary.detach()) * terms.auxiliary_count
            epoch_samples += batch_size
            epoch_auxiliary_count += terms.auxiliary_count

            end_of_group = (batch_index + 1) % accumulation_steps == 0
            end_of_epoch = batch_index + 1 == len(train_loader)
            if end_of_group or end_of_epoch:
                factor = _learning_rate_factor(update_index, total_updates, warmup_updates)
                for parameter_group in optimizer.param_groups:
                    parameter_group["lr"] = config.learning_rate * factor
                scaler.unscale_(optimizer)
                nn.utils.clip_grad_norm_(model.parameters(), config.gradient_clip_norm)
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)
                update_index += 1

        if epoch_samples == 0:
            raise ValueError("train loader produced no samples")
        train_season_loss = season_loss_sum / epoch_samples
        train_auxiliary_loss = (
            auxiliary_loss_sum / epoch_auxiliary_count if epoch_auxiliary_count else 0.0
        )
        validation_losses, metrics, ids, targets_array, probabilities = (
            _evaluate_multitask(
                model,
                validation_loader,
                device=device,
                labels=ordered_labels,
                auxiliary_weight=auxiliary_weight,
                amp_enabled=amp_enabled,
            )
        )
        macro_f1 = float(metrics["macro_f1"])
        train_total_loss = train_season_loss + auxiliary_weight * train_auxiliary_loss
        history.append(
            {
                "epoch": epoch,
                "train_loss": train_total_loss,
                "train_total_loss": train_total_loss,
                "train_season_loss": train_season_loss,
                "train_auxiliary_loss": train_auxiliary_loss,
                "train_auxiliary_labeled_samples": epoch_auxiliary_count,
                "validation_loss": validation_losses["total_loss"],
                "validation_total_loss": validation_losses["total_loss"],
                "validation_season_loss": validation_losses["season_loss"],
                "validation_auxiliary_loss": validation_losses["auxiliary_loss"],
                "validation_auxiliary_labeled_samples": int(
                    validation_losses["auxiliary_labeled_samples"]
                ),
                "validation_macro_f1": macro_f1,
                "validation_accuracy": float(metrics["accuracy"]),
                "learning_rate": float(optimizer.param_groups[0]["lr"]),
            }
        )
        if macro_f1 > best_metric + config.min_delta:
            best_epoch = epoch
            best_metric = macro_f1
            best_metrics = metrics
            best_ids = ids
            best_targets = targets_array
            best_probabilities = probabilities
            stale_epochs = 0
            best_checkpoint_sha256 = _save_checkpoint(
                output,
                {
                    "format_version": 1,
                    "model_state_dict": {
                        name: tensor.detach().cpu()
                        for name, tensor in model.state_dict().items()
                    },
                    "optimizer_state_dict": optimizer.state_dict(),
                    "train_config": asdict(config),
                    "labels": ordered_labels,
                    "auxiliary_weight": auxiliary_weight,
                    "best_epoch": best_epoch,
                    "best_macro_f1": best_metric,
                },
            )
        else:
            stale_epochs += 1
            if stale_epochs >= config.patience:
                break

    checkpoint = torch.load(output, map_location=device, weights_only=True)
    model.load_state_dict(checkpoint["model_state_dict"])
    runtime_seconds = time.perf_counter() - started
    peak_vram_mb = None
    if device.type == "cuda":
        peak_vram_mb = torch.cuda.max_memory_allocated(device) / (1024**2)
    return FoldResult(
        fold=config.fold,
        seed=config.seed,
        labels=ordered_labels,
        best_epoch=best_epoch,
        epochs_completed=len(history),
        best_macro_f1=best_metric,
        best_metrics=best_metrics,
        history=history,
        validation_ids=best_ids,
        targets=best_targets,
        probabilities=best_probabilities,
        checkpoint_path=str(output),
        checkpoint_sha256=best_checkpoint_sha256,
        parameter_count=sum(parameter.numel() for parameter in model.parameters()),
        runtime_seconds=runtime_seconds,
        peak_vram_mb=peak_vram_mb,
        stopped_early=len(history) < config.epochs,
        device=str(device),
        metadata={
            "amp_enabled": amp_enabled,
            "accumulation_steps": accumulation_steps,
            "updates_completed": update_index,
            "auxiliary_weight": auxiliary_weight,
            "selection_metric": "season_macro_f1",
        },
    )


def train_masked_multitask_refit(
    model: nn.Module,
    train_loader: DataLoader[Any],
    *,
    config: RefitTrainConfig,
    auxiliary_weight: float,
) -> RefitResult:
    """Train every declared epoch without validation, selection, or early stopping."""
    config.validate()
    if not math.isfinite(auxiliary_weight) or auxiliary_weight <= 0:
        raise ValueError("auxiliary_weight must be a finite positive value")
    if not isinstance(train_loader, Sized) or len(train_loader) == 0:
        raise ValueError("train_loader must contain at least one batch")
    loader_batch_size = getattr(train_loader, "batch_size", None)
    if loader_batch_size is not None and loader_batch_size != config.batch_size:
        raise ValueError(
            "train_loader batch_size must match RefitTrainConfig.batch_size; "
            f"got {loader_batch_size} and {config.batch_size}"
        )

    seed_everything(config.seed)
    device = _resolve_device(config.device)
    amp_enabled = config.use_amp and device.type == "cuda"
    model.to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
    )
    accumulation_steps = config.effective_batch_size // config.batch_size
    updates_per_epoch = math.ceil(len(train_loader) / accumulation_steps)
    total_updates = config.epochs * updates_per_epoch
    warmup_updates = round(config.warmup_epochs * updates_per_epoch)
    scaler = torch.amp.GradScaler(device.type, enabled=amp_enabled)
    history: list[dict[str, float | int]] = []
    update_index = 0
    started = time.perf_counter()
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)

    for epoch in range(1, config.epochs + 1):
        model.train()
        optimizer.zero_grad(set_to_none=True)
        season_loss_sum = 0.0
        auxiliary_loss_sum = 0.0
        epoch_samples = 0
        epoch_correct = 0
        epoch_auxiliary_count = 0
        for batch_index, batch in enumerate(train_loader):
            group_start = (batch_index // accumulation_steps) * accumulation_steps
            group_size = min(accumulation_steps, len(train_loader) - group_start)
            with torch.amp.autocast(device_type=device.type, enabled=amp_enabled):
                logits, targets, terms, _ = _forward_terms(
                    model,
                    batch,
                    device=device,
                    auxiliary_weight=auxiliary_weight,
                    require_ids=False,
                )
                scaled_loss = terms.total / group_size
            scaler.scale(scaled_loss).backward()
            batch_size = int(targets.shape[0])
            season_loss_sum += float(terms.season.detach()) * batch_size
            auxiliary_loss_sum += float(terms.auxiliary.detach()) * terms.auxiliary_count
            epoch_samples += batch_size
            epoch_correct += int(logits.detach().argmax(dim=1).eq(targets).sum().item())
            epoch_auxiliary_count += terms.auxiliary_count

            end_of_group = (batch_index + 1) % accumulation_steps == 0
            end_of_epoch = batch_index + 1 == len(train_loader)
            if end_of_group or end_of_epoch:
                factor = _learning_rate_factor(update_index, total_updates, warmup_updates)
                for parameter_group in optimizer.param_groups:
                    parameter_group["lr"] = config.learning_rate * factor
                scaler.unscale_(optimizer)
                nn.utils.clip_grad_norm_(model.parameters(), config.gradient_clip_norm)
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)
                update_index += 1

        if epoch_samples == 0:
            raise ValueError("train loader produced no samples")
        train_season_loss = season_loss_sum / epoch_samples
        train_auxiliary_loss = (
            auxiliary_loss_sum / epoch_auxiliary_count if epoch_auxiliary_count else 0.0
        )
        history.append(
            {
                "epoch": epoch,
                "train_loss": train_season_loss
                + auxiliary_weight * train_auxiliary_loss,
                "train_season_loss": train_season_loss,
                "train_auxiliary_loss": train_auxiliary_loss,
                "train_accuracy": epoch_correct / epoch_samples,
                "train_samples": epoch_samples,
                "train_auxiliary_labeled_samples": epoch_auxiliary_count,
                "learning_rate": float(optimizer.param_groups[0]["lr"]),
            }
        )

    runtime_seconds = time.perf_counter() - started
    peak_vram_mb = None
    if device.type == "cuda":
        peak_vram_mb = torch.cuda.max_memory_allocated(device) / (1024**2)
    return RefitResult(
        seed=config.seed,
        final_epoch=config.epochs,
        epochs_completed=len(history),
        history=history,
        parameter_count=sum(parameter.numel() for parameter in model.parameters()),
        runtime_seconds=runtime_seconds,
        peak_vram_mb=peak_vram_mb,
        device=str(device),
        metadata={
            "amp_enabled": amp_enabled,
            "accumulation_steps": accumulation_steps,
            "updates_completed": update_index,
            "selection_metric": None,
            "validation_used": False,
            "early_stopping_used": False,
            "checkpoint_rule": "save_the_declared_final_epoch_state",
            "auxiliary_weight": auxiliary_weight,
        },
    )
