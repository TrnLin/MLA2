"""CNN optimization engine for Task 1 fold training."""

from __future__ import annotations

import copy
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as functional
from torch import nn
from torch.optim import Adam
from torch.optim.lr_scheduler import OneCycleLR
from torch.utils.data import DataLoader

from fashion.task1.dataset import Task1TorchDataset
from fashion.task1.evaluation import (
    TASK1_NUM_CLASSES,
    build_prediction_frame,
    classification_metrics,
)
from fashion.task1.models import Task1SmallCNN
from fashion.task1.preprocessing import (
    Task1Normalization,
    Task1PreprocessingConfig,
    build_task1_training_transform,
    build_task1_validation_transform,
)
from fashion.train.reproducibility import make_torch_generator, seed_worker

if TYPE_CHECKING:
    from fashion.task1.training import Task1TrainConfig


@dataclass(frozen=True)
class Task1CnnEngineResult:
    """Best CNN state and fixed-class evidence from one training run."""

    model: nn.Module
    best_epoch: int
    history: pd.DataFrame
    metrics: dict[str, float]
    predictions: pd.DataFrame


def _limited_batches(loader: DataLoader[dict[str, torch.Tensor]], maximum: int | None) -> int:
    """Return the exact number of loader batches that this run will consume."""
    batches = len(loader)
    return batches if maximum is None else min(batches, maximum)


def _evaluate(
    model: nn.Module,
    loader: DataLoader[dict[str, torch.Tensor]],
    *,
    device: torch.device,
    max_batches: int | None,
    class_names: list[str],
) -> tuple[dict[str, float], pd.DataFrame]:
    """Return fixed-label validation metrics and prediction evidence."""
    probabilities: list[np.ndarray] = []
    labels: list[np.ndarray] = []
    product_ids: list[np.ndarray] = []
    losses: list[float] = []
    model.eval()
    with torch.no_grad():
        for batch_number, batch in enumerate(loader):
            if max_batches is not None and batch_number >= max_batches:
                break
            images = batch["image"].to(device)
            target = batch["label"].to(device)
            logits = model(images)
            loss = functional.cross_entropy(logits, target)
            if not torch.isfinite(loss):
                raise ValueError("validation loss must be finite")
            scores = torch.softmax(logits, dim=1).detach().cpu().numpy()
            if not np.isfinite(scores).all():
                raise ValueError("validation probabilities must be finite")
            probabilities.append(scores)
            labels.append(target.detach().cpu().numpy())
            product_ids.append(batch["id"].detach().cpu().numpy())
            losses.append(float(loss.item()))
    if not probabilities:
        raise ValueError("validation loader produced no batches")
    matrix = np.concatenate(probabilities)
    y_true = np.concatenate(labels)
    ids = np.concatenate(product_ids)
    metrics = classification_metrics(y_true, matrix)
    metrics["validation_loss"] = float(np.mean(losses))
    return metrics, build_prediction_frame(ids, y_true, matrix, class_names)


def _training_loader(
    rows: pd.DataFrame,
    label_to_index: Mapping[str, int],
    *,
    normalization: Task1Normalization,
    preprocessing: Task1PreprocessingConfig,
    config: Task1TrainConfig,
    epoch: int,
    root: Path,
) -> DataLoader[dict[str, torch.Tensor]]:
    """Build an epoch-specific deterministic augmented loader."""
    transform = build_task1_training_transform(
        normalization,
        seed=config.seed,
        epoch=epoch,
        config=preprocessing,
    )
    dataset = Task1TorchDataset(rows, transform, label_to_index, root=root)
    return DataLoader(
        dataset,
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=config.num_workers,
        generator=make_torch_generator(config.seed + epoch),
        worker_init_fn=seed_worker if config.num_workers else None,
    )


def train_task1_cnn(
    training_rows: pd.DataFrame,
    validation_rows: pd.DataFrame,
    label_to_index: Mapping[str, int],
    class_names: list[str],
    *,
    normalization: Task1Normalization,
    preprocessing: Task1PreprocessingConfig,
    config: Task1TrainConfig,
    root: Path,
    device: torch.device,
    model_factory: Callable[[int], nn.Module] = Task1SmallCNN,
) -> Task1CnnEngineResult:
    """Train a CNN and return its selected model with validation evidence."""
    validation_dataset = Task1TorchDataset(
        validation_rows,
        build_task1_validation_transform(normalization, config=preprocessing),
        label_to_index,
        root=root,
    )
    validation_loader = DataLoader(
        validation_dataset,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=config.num_workers,
        worker_init_fn=seed_worker if config.num_workers else None,
    )
    initial_loader = _training_loader(
        training_rows,
        label_to_index,
        normalization=normalization,
        preprocessing=preprocessing,
        config=config,
        epoch=1,
        root=root,
    )
    steps_per_epoch = _limited_batches(initial_loader, config.max_train_batches)
    if steps_per_epoch == 0:
        raise ValueError("training loader produced no batches")

    model = model_factory(TASK1_NUM_CLASSES).to(device)
    optimizer = Adam(model.parameters(), lr=config.max_lr, weight_decay=config.weight_decay)
    scheduler = OneCycleLR(
        optimizer,
        max_lr=config.max_lr,
        epochs=config.epochs,
        steps_per_epoch=steps_per_epoch,
    )
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)

    history: list[dict[str, float | int]] = []
    best_state: dict[str, torch.Tensor] | None = None
    best_metrics: dict[str, float] | None = None
    best_epoch = 0
    for epoch in range(1, config.epochs + 1):
        training_loader = _training_loader(
            training_rows,
            label_to_index,
            normalization=normalization,
            preprocessing=preprocessing,
            config=config,
            epoch=epoch,
            root=root,
        )
        model.train()
        training_losses: list[float] = []
        for batch_number, batch in enumerate(training_loader):
            if batch_number >= steps_per_epoch:
                break
            images = batch["image"].to(device)
            target = batch["label"].to(device)
            optimizer.zero_grad(set_to_none=True)
            logits = model(images)
            loss = functional.cross_entropy(logits, target)
            if not torch.isfinite(loss):
                raise ValueError("training loss must be finite")
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), config.grad_clip_norm)
            optimizer.step()
            scheduler.step()
            training_losses.append(float(loss.item()))
        if len(training_losses) != steps_per_epoch:
            raise ValueError("training loader produced fewer batches than expected")
        epoch_metrics, _ = _evaluate(
            model,
            validation_loader,
            device=device,
            max_batches=config.max_validation_batches,
            class_names=class_names,
        )
        history.append(
            {"epoch": epoch, "train_loss": float(np.mean(training_losses)), **epoch_metrics}
        )
        if best_metrics is None or epoch_metrics["macro_f1"] > best_metrics["macro_f1"]:
            best_metrics = epoch_metrics
            best_epoch = epoch
            best_state = copy.deepcopy(model.state_dict())

    if best_state is None or best_metrics is None:
        raise RuntimeError("training did not produce a best validation state")
    model.load_state_dict(best_state)
    final_metrics, predictions = _evaluate(
        model,
        validation_loader,
        device=device,
        max_batches=config.max_validation_batches,
        class_names=class_names,
    )
    return Task1CnnEngineResult(
        model=model,
        best_epoch=best_epoch,
        history=pd.DataFrame(history),
        metrics=final_metrics,
        predictions=predictions,
    )
