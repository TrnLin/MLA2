"""One-fold scratch classifier training with reproducible best-model evidence."""

from __future__ import annotations

import io
import math
import time
from collections.abc import Mapping, Sequence, Sized
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader

from fashion.data.hashing import compute_sha256
from fashion.train.artifacts import atomic_write_bytes
from fashion.train.metrics import SEASON_LABELS, multiclass_metrics
from fashion.train.reproducibility import seed_everything


@dataclass(frozen=True)
class TrainConfig:
    """Budget and optimisation settings shared by comparable fold runs."""

    fold: int
    seed: int
    epochs: int = 30
    learning_rate: float = 3e-4
    weight_decay: float = 1e-4
    batch_size: int = 32
    effective_batch_size: int = 128
    gradient_clip_norm: float = 1.0
    warmup_epochs: float = 1.0
    patience: int = 5
    min_delta: float = 1e-4
    use_amp: bool = True
    device: str = "auto"

    def validate(self) -> None:
        """Reject settings that would make budgets ambiguous across runs."""
        if self.fold < 0 or self.seed < 0:
            raise ValueError("fold and seed must be non-negative")
        if self.epochs < 1 or self.patience < 1:
            raise ValueError("epochs and patience must be positive")
        if self.learning_rate <= 0 or self.weight_decay < 0:
            raise ValueError("learning_rate must be positive and weight_decay non-negative")
        if self.batch_size < 1 or self.effective_batch_size < self.batch_size:
            raise ValueError("effective_batch_size must be at least batch_size")
        if self.effective_batch_size % self.batch_size:
            raise ValueError("effective_batch_size must be divisible by batch_size")
        if self.gradient_clip_norm <= 0:
            raise ValueError("gradient_clip_norm must be positive")
        if self.warmup_epochs < 0 or self.warmup_epochs >= self.epochs:
            raise ValueError("warmup_epochs must be non-negative and smaller than epochs")
        if self.min_delta < 0:
            raise ValueError("min_delta must be non-negative")
        if self.device not in {"auto", "cpu", "cuda"}:
            raise ValueError("device must be 'auto', 'cpu', or 'cuda'")


@dataclass
class FoldResult:
    """Best validation predictions plus learning and resource evidence for one fold."""

    fold: int
    seed: int
    labels: tuple[str, ...]
    best_epoch: int
    epochs_completed: int
    best_macro_f1: float
    best_metrics: dict[str, Any]
    history: list[dict[str, float | int]]
    validation_ids: list[Any]
    targets: np.ndarray
    probabilities: np.ndarray
    checkpoint_path: str
    checkpoint_sha256: str
    parameter_count: int
    runtime_seconds: float
    peak_vram_mb: float | None
    stopped_early: bool
    device: str
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_oof_frame(self) -> pd.DataFrame:
        """Return the selected epoch as the standard exactly-once OOF schema."""
        predicted_indices = self.probabilities.argmax(axis=1)
        frame = pd.DataFrame(
            {
                "id": self.validation_ids,
                "fold": self.fold,
                "seed": self.seed,
                "y_true": [self.labels[index] for index in self.targets],
                "y_pred": [self.labels[index] for index in predicted_indices],
            }
        )
        for index, label in enumerate(self.labels):
            frame[f"prob_{label}"] = self.probabilities[:, index]
        return frame


def _resolve_device(requested: str) -> torch.device:
    if requested == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if requested == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available")
    return torch.device(requested)


def _unpack_batch(batch: Any, *, require_ids: bool) -> tuple[Any, Any, Any | None]:
    if isinstance(batch, Mapping):
        try:
            images = batch["image"]
            targets = batch["target"]
        except KeyError as error:
            raise KeyError("mapping batches require 'image' and 'target'") from error
        identifiers = batch.get("id")
    elif isinstance(batch, Sequence) and not isinstance(batch, (str, bytes)):
        if len(batch) not in {2, 3}:
            raise ValueError("sequence batches require (image, target[, id])")
        images, targets = batch[:2]
        identifiers = batch[2] if len(batch) == 3 else None
    else:
        raise TypeError("batch must be a mapping or a 2/3-item sequence")
    if require_ids and identifiers is None:
        raise ValueError("validation batches require stable sample IDs")
    return images, targets, identifiers


def _as_id_list(identifiers: Any) -> list[Any]:
    if isinstance(identifiers, torch.Tensor):
        return identifiers.detach().cpu().tolist()
    if isinstance(identifiers, np.ndarray):
        return identifiers.tolist()
    if isinstance(identifiers, (str, bytes)):
        return [identifiers]
    return list(identifiers)


def _learning_rate_factor(step: int, total_steps: int, warmup_steps: int) -> float:
    if warmup_steps and step < warmup_steps:
        return (step + 1) / warmup_steps
    cosine_steps = max(total_steps - warmup_steps, 1)
    progress = min(max((step - warmup_steps) / cosine_steps, 0.0), 1.0)
    return 0.5 * (1.0 + math.cos(math.pi * progress))


def _save_checkpoint(path: Path, payload: dict[str, Any]) -> str:
    buffer = io.BytesIO()
    torch.save(payload, buffer)
    atomic_write_bytes(path, buffer.getvalue())
    return compute_sha256(path)


def _evaluate(
    model: nn.Module,
    loader: DataLoader[Any],
    criterion: nn.Module,
    device: torch.device,
    labels: tuple[str, ...],
    amp_enabled: bool,
) -> tuple[float, dict[str, Any], list[Any], np.ndarray, np.ndarray]:
    model.eval()
    total_loss = 0.0
    total_samples = 0
    identifiers: list[Any] = []
    target_parts: list[np.ndarray] = []
    probability_parts: list[np.ndarray] = []
    with torch.inference_mode():
        for batch in loader:
            images, targets, batch_ids = _unpack_batch(batch, require_ids=True)
            images = images.to(device, non_blocking=True)
            targets = targets.to(device, dtype=torch.long, non_blocking=True)
            with torch.amp.autocast(device_type=device.type, enabled=amp_enabled):
                logits = model(images)
                if not isinstance(logits, torch.Tensor) or logits.ndim != 2:
                    raise TypeError("classifier model must return a [batch, class] tensor")
                loss = criterion(logits, targets)
            batch_size = int(targets.shape[0])
            total_loss += float(loss.detach()) * batch_size
            total_samples += batch_size
            identifiers.extend(_as_id_list(batch_ids))
            target_parts.append(targets.detach().cpu().numpy())
            probability_parts.append(torch.softmax(logits.float(), dim=1).cpu().numpy())
    if total_samples == 0:
        raise ValueError("validation loader produced no samples")
    targets_array = np.concatenate(target_parts).astype(np.int64, copy=False)
    probabilities = np.concatenate(probability_parts).astype(np.float64, copy=False)
    if len(identifiers) != total_samples:
        raise ValueError("validation IDs do not align with batch samples")
    if probabilities.shape[1] != len(labels):
        raise ValueError(
            f"model returned {probabilities.shape[1]} classes but labels define {len(labels)}"
        )
    if ((targets_array < 0) | (targets_array >= len(labels))).any():
        raise ValueError("validation targets contain an out-of-range class index")
    true_labels = np.asarray(labels, dtype=object)[targets_array]
    metrics = multiclass_metrics(true_labels, probabilities=probabilities, labels=labels)
    return total_loss / total_samples, metrics, identifiers, targets_array, probabilities


def train_fold(
    model: nn.Module,
    train_loader: DataLoader[Any],
    validation_loader: DataLoader[Any],
    *,
    config: TrainConfig,
    checkpoint_path: str | Path,
    labels: Sequence[str] = SEASON_LABELS,
    criterion: nn.Module | None = None,
) -> FoldResult:
    """Train one fold, restore its best macro-F1 epoch, and return its OOF rows."""
    config.validate()
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
    loss_function = criterion or nn.CrossEntropyLoss()
    loss_function = loss_function.to(device)
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
        epoch_loss = 0.0
        epoch_samples = 0
        for batch_index, batch in enumerate(train_loader):
            images, targets, _ = _unpack_batch(batch, require_ids=False)
            images = images.to(device, non_blocking=True)
            targets = targets.to(device, dtype=torch.long, non_blocking=True)
            group_start = (batch_index // accumulation_steps) * accumulation_steps
            group_size = min(accumulation_steps, len(train_loader) - group_start)
            with torch.amp.autocast(device_type=device.type, enabled=amp_enabled):
                logits = model(images)
                if not isinstance(logits, torch.Tensor) or logits.ndim != 2:
                    raise TypeError("classifier model must return a [batch, class] tensor")
                loss = loss_function(logits, targets)
                scaled_loss = loss / group_size
            scaler.scale(scaled_loss).backward()
            batch_size = int(targets.shape[0])
            epoch_loss += float(loss.detach()) * batch_size
            epoch_samples += batch_size

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
        validation_loss, metrics, ids, targets_array, probabilities = _evaluate(
            model,
            validation_loader,
            loss_function,
            device,
            ordered_labels,
            amp_enabled,
        )
        macro_f1 = float(metrics["macro_f1"])
        history.append(
            {
                "epoch": epoch,
                "train_loss": epoch_loss / epoch_samples,
                "validation_loss": validation_loss,
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
                        name: tensor.detach().cpu() for name, tensor in model.state_dict().items()
                    },
                    "optimizer_state_dict": optimizer.state_dict(),
                    "train_config": asdict(config),
                    "labels": ordered_labels,
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
        },
    )
