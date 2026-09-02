"""Deterministic one-fold training and evidence writing for Task 1."""

from __future__ import annotations

import io
from collections.abc import Callable, Mapping
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Literal

import pandas as pd
import torch
from torch import nn

from fashion.config import RANDOM_SEED, ROOT, SPLITS_CSV, TASK1_RESULT_DIR
from fashion.data.dataset import load_splits
from fashion.data.hashing import compute_sha256
from fashion.task1.cnn_engine import train_task1_cnn
from fashion.task1.dataset import get_task1_fold_rows
from fashion.task1.evaluation import validate_task1_label_map
from fashion.task1.models import Task1SmallCNN, count_trainable_parameters
from fashion.task1.preprocessing import (
    DEFAULT_TASK1_PREPROCESSING,
    TASK1_CONTROL_PREPROCESSING,
    Task1PreprocessingConfig,
    fit_task1_normalization,
)
from fashion.train.artifacts import atomic_write_bytes, atomic_write_csv, canonical_sha256
from fashion.train.registry import RunRecord, RunRegistry, new_run_id, tracked_run
from fashion.train.reproducibility import seed_everything


@dataclass(frozen=True)
class Task1TrainConfig:
    """Frozen optimization settings for smoke and reportable Task 1 runs."""

    stage: Literal["smoke", "experiment"]
    epochs: int
    batch_size: int
    max_train_batches: int | None
    max_validation_batches: int | None
    final_eligible: bool
    seed: int = RANDOM_SEED
    max_lr: float = 1e-3
    weight_decay: float = 1e-5
    grad_clip_norm: float = 1.0
    num_workers: int = 0

    @classmethod
    def smoke(cls) -> Task1TrainConfig:
        """Return the short non-reportable integration configuration."""
        return cls(
            stage="smoke",
            epochs=1,
            batch_size=16,
            max_train_batches=2,
            max_validation_batches=2,
            final_eligible=False,
        )

    @classmethod
    def full(cls) -> Task1TrainConfig:
        """Return the fixed reportable CNN experiment configuration."""
        return cls(
            stage="experiment",
            epochs=20,
            batch_size=128,
            max_train_batches=None,
            max_validation_batches=None,
            final_eligible=True,
        )


@dataclass(frozen=True)
class Task1FoldResult:
    """Completed one-fold artifact locations and fixed-class metrics."""

    run_id: str
    fold: int
    preprocessing_id: str
    status: Literal["completed"]
    metrics: dict[str, float]
    checkpoint_path: Path
    history_path: Path
    prediction_path: Path


def select_training_device() -> torch.device:
    """Choose CUDA, then Apple MPS, then CPU without hiding the selected device."""
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def _smoke_rows(rows: pd.DataFrame) -> pd.DataFrame:
    """Keep a stable small development-only subset without changing saved splits."""
    return rows.sort_values("id", kind="stable").head(32).copy()


def _artifact_path(path: Path) -> str:
    """Store a project-relative artifact path when the target belongs to this checkout."""
    try:
        return path.resolve().relative_to(ROOT.resolve()).as_posix()
    except ValueError:
        return str(path.resolve())


def _implementation_hashes() -> dict[str, str]:
    """Hash all checked-in Task 1 code that defines one trained artifact."""
    relative_paths = (
        "src/fashion/task1/dataset.py",
        "src/fashion/task1/cnn_engine.py",
        "src/fashion/task1/evaluation.py",
        "src/fashion/task1/image_contract.py",
        "src/fashion/task1/models.py",
        "src/fashion/task1/preprocessing.py",
        "src/fashion/task1/training.py",
    )
    return {relative: compute_sha256(ROOT / relative) for relative in relative_paths}


def train_task1_fold(
    splits: pd.DataFrame,
    label_map: Mapping[str, object],
    *,
    validation_fold: int,
    preprocessing: Task1PreprocessingConfig,
    config: Task1TrainConfig,
    registry: RunRegistry | None = None,
    root: str | Path = ROOT,
    result_root: str | Path = TASK1_RESULT_DIR,
    device: torch.device | None = None,
    split_path: str | Path = SPLITS_CSV,
    model_factory: Callable[[int], nn.Module] = Task1SmallCNN,
) -> Task1FoldResult:
    """Train, select, reload, and register one sealed-development Task 1 fold."""
    if config.epochs <= 0 or config.batch_size <= 0:
        raise ValueError("epochs and batch_size must be positive")
    if config.max_train_batches is not None and config.max_train_batches <= 0:
        raise ValueError("max_train_batches must be positive or None")
    if config.max_validation_batches is not None and config.max_validation_batches <= 0:
        raise ValueError("max_validation_batches must be positive or None")
    if config.num_workers < 0:
        raise ValueError("num_workers must be non-negative")
    if config.final_eligible:
        if (
            config.stage != "experiment"
            or config.epochs != 20
            or config.seed != RANDOM_SEED
            or config.max_lr != 1e-3
            or config.weight_decay != 1e-5
            or config.grad_clip_norm != 1.0
            or config.max_train_batches is not None
            or config.max_validation_batches is not None
        ):
            raise ValueError(
                "final-eligible Task 1 runs require stage='experiment', "
                "fixed full-run optimization settings, and no batch limits"
            )
        if preprocessing not in (TASK1_CONTROL_PREPROCESSING, DEFAULT_TASK1_PREPROCESSING):
            raise ValueError("final-eligible Task 1 runs require approved preprocessing")
        if model_factory is not Task1SmallCNN:
            raise ValueError("final-eligible Task 1 runs require Task1SmallCNN")
        if Path(split_path).resolve() != SPLITS_CSV.resolve():
            raise ValueError("final-eligible Task 1 runs require the canonical split path")
        canonical_splits = load_splits(split_path)
        if not splits.equals(canonical_splits):
            raise ValueError("supplied splits must match the canonical split file")

    experiment_id = f"task1-cnn-{preprocessing.preprocessing_id}"
    run_config = {
        "validation_fold": validation_fold,
        "preprocessing": preprocessing.to_dict(),
        "training": asdict(config),
        "model_family": "task1_small_cnn_v1",
    }
    source_hashes = _implementation_hashes()
    record = RunRecord(
        run_id=new_run_id(experiment_id, validation_fold, config.seed),
        task="task1",
        stage=config.stage,
        experiment_id=experiment_id,
        model_family="task1_small_cnn_v1",
        benchmark_only=False,
        final_eligible=config.final_eligible,
        scratch=True,
        fold=validation_fold,
        seed=config.seed,
        transform_id=preprocessing.preprocessing_id,
        loss_id="cross_entropy_unweighted_v1",
        epochs_requested=config.epochs,
        primary_metric_name="macro_f1_124",
        config_sha256=canonical_sha256(run_config),
        split_sha256=compute_sha256(split_path),
        label_map_sha256=canonical_sha256(label_map),
        implementation_sha256=canonical_sha256(dict(sorted(source_hashes.items()))),
    )
    active_registry = registry or RunRegistry()
    project_root = Path(root)
    selected_device = device or select_training_device()

    with tracked_run(active_registry, record) as run:
        seed_everything(config.seed)
        label_to_index, class_names = validate_task1_label_map(label_map)
        training_rows, validation_rows = get_task1_fold_rows(splits, validation_fold)
        if config.stage == "smoke":
            training_rows = _smoke_rows(training_rows)
            validation_rows = _smoke_rows(validation_rows)
        normalization = fit_task1_normalization(
            training_rows,
            validation_fold=validation_fold,
            root=project_root,
            config=preprocessing,
        )
        engine_result = train_task1_cnn(
            training_rows,
            validation_rows,
            label_to_index,
            class_names,
            normalization=normalization,
            preprocessing=preprocessing,
            config=config,
            root=project_root,
            device=selected_device,
            model_factory=model_factory,
        )
        model = engine_result.model
        best_epoch = engine_result.best_epoch
        final_metrics = engine_result.metrics
        history = engine_result.history
        predictions = engine_result.predictions
        run_dir = Path(result_root) / run.run_id
        checkpoint_path = run_dir / "checkpoint.pt"
        history_path = run_dir / "history.csv"
        prediction_path = run_dir / "predictions.csv"
        model_config = asdict(model.config) if hasattr(model, "config") else {"num_classes": 124}
        checkpoint = {
            "model_state_dict": model.state_dict(),
            "model_config": model_config,
            "train_config": asdict(config),
            "preprocessing": preprocessing.to_dict(),
            "normalization": normalization.to_dict(),
            "class_names": class_names,
            "label_map_sha256": run.label_map_sha256,
            "validation_fold": validation_fold,
            "seed": config.seed,
            "best_epoch": best_epoch,
            "metrics": final_metrics,
        }
        checkpoint_buffer = io.BytesIO()
        torch.save(checkpoint, checkpoint_buffer)
        atomic_write_bytes(checkpoint_path, checkpoint_buffer.getvalue())
        atomic_write_csv(history_path, pd.DataFrame(history))
        atomic_write_csv(prediction_path, predictions)

        run.epochs_completed = config.epochs
        run.best_epoch = best_epoch
        run.primary_metric_value = final_metrics["macro_f1"]
        run.metrics = final_metrics
        run.parameter_count = count_trainable_parameters(model)
        if selected_device.type == "cuda":
            run.peak_vram_mb = torch.cuda.max_memory_allocated(selected_device) / (1024**2)
        run.checkpoint_path = _artifact_path(checkpoint_path)
        run.checkpoint_sha256 = compute_sha256(checkpoint_path)
        run.history_path = _artifact_path(history_path)
        run.history_sha256 = compute_sha256(history_path)
        run.prediction_path = _artifact_path(prediction_path)
        run.prediction_sha256 = compute_sha256(prediction_path)

    return Task1FoldResult(
        run_id=record.run_id,
        fold=validation_fold,
        preprocessing_id=preprocessing.preprocessing_id,
        status="completed",
        metrics=final_metrics,
        checkpoint_path=checkpoint_path,
        history_path=history_path,
        prediction_path=prediction_path,
    )
