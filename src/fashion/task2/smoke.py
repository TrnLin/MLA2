"""Deterministic G0 overfit and registry/checkpoint smoke gate for Task 2."""

from __future__ import annotations

import hashlib
import json
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Literal

import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader

from fashion.config import (
    LABEL_MAPS_JSON,
    RANDOM_SEED,
    ROOT,
    RUNS_CSV,
    SPLITS_CSV,
    TASK2_CHECKPOINT_DIR,
    TASK2_RUN_DIR,
)
from fashion.data.dataset import get_cv_split, get_samples, load_label_maps, load_splits
from fashion.data.hashing import compute_sha256
from fashion.data.torch import (
    EncodedClassificationDataset,
    ImageTransformSpec,
    build_image_transform,
    fit_fold_stats,
)
from fashion.models.season import SeasonModelSpec, assert_final_model, build_season_model
from fashion.train.artifacts import atomic_write_csv, atomic_write_json
from fashion.train.cache import RunCacheKey, build_run_cache_key, find_cached_run
from fashion.train.engine import TrainConfig, train_fold
from fashion.train.metrics import validate_oof
from fashion.train.registry import RunRecord, RunRegistry, new_run_id, tracked_run
from fashion.train.reproducibility import make_torch_generator, seed_everything, seed_worker

SmokeMode = Literal["run_or_load", "run", "load"]


class SmokeGateError(RuntimeError):
    """Raised when G0 proves that the learning pipeline is not ready for experiments."""


@dataclass(frozen=True)
class G0SmokeConfig:
    """Immutable sample, optimisation, and pass rules for the non-comparison G0 gate."""

    experiment_id: str = "g0-pipeline-smoke"
    stage: str = "g0_smoke"
    schema_version: str = "1.0.0"
    validation_fold: int = 0
    seed: int = RANDOM_SEED
    model_family: str = "smallcnn"
    image_size: tuple[int, int] = (80, 60)
    augmentation: str = "a0"
    tiny_per_class: int = 16
    tiny_steps: int = 100
    tiny_learning_rate: float = 3e-3
    minimum_tiny_accuracy: float = 0.95
    maximum_tiny_loss_ratio: float = 0.20
    integration_train_per_class: int = 128
    integration_validation_per_class: int = 32
    integration_epochs: int = 2
    batch_size: int = 32
    validation_batch_size: int = 64
    effective_batch_size: int = 128
    learning_rate: float = 1e-3
    weight_decay: float = 1e-4
    gradient_clip_norm: float = 1.0
    num_workers: int = 0
    use_amp: bool = True
    device: str = "auto"

    def validate(self) -> None:
        """Fail before touching data when the smoke question is ambiguous."""
        if self.schema_version != "1.0.0":
            raise ValueError(f"unsupported G0 schema: {self.schema_version}")
        if not self.experiment_id.strip() or self.stage != "g0_smoke":
            raise ValueError("G0 requires a non-empty ID and stage='g0_smoke'")
        if self.validation_fold not in range(5) or self.seed < 0:
            raise ValueError("G0 fold must be 0-4 and seed must be non-negative")
        if self.model_family != "smallcnn":
            raise ValueError("G0 uses the declared smallcnn pipeline only")
        if len(self.image_size) != 2 or any(value < 1 for value in self.image_size):
            raise ValueError("G0 image_size must contain two positive integers")
        if self.augmentation not in {"a0", "a1"}:
            raise ValueError("G0 integration augmentation must be a0 or a1")
        positive_integers = (
            self.tiny_per_class,
            self.tiny_steps,
            self.integration_train_per_class,
            self.integration_validation_per_class,
            self.integration_epochs,
            self.batch_size,
            self.validation_batch_size,
            self.effective_batch_size,
        )
        if any(value < 1 for value in positive_integers):
            raise ValueError("G0 sample counts, steps, epochs, and batch sizes must be positive")
        if self.tiny_per_class > self.integration_train_per_class:
            raise ValueError("tiny_per_class cannot exceed integration_train_per_class")
        if self.effective_batch_size < self.batch_size:
            raise ValueError("effective_batch_size must be at least batch_size")
        if self.effective_batch_size % self.batch_size:
            raise ValueError("effective_batch_size must be divisible by batch_size")
        if self.tiny_learning_rate <= 0 or self.learning_rate <= 0 or self.weight_decay < 0:
            raise ValueError("G0 learning rates must be positive and weight decay non-negative")
        if self.gradient_clip_norm <= 0:
            raise ValueError("G0 gradient_clip_norm must be positive")
        if not 0 <= self.minimum_tiny_accuracy <= 1:
            raise ValueError("minimum_tiny_accuracy must be in [0,1]")
        if self.maximum_tiny_loss_ratio <= 0:
            raise ValueError("maximum_tiny_loss_ratio must be positive")
        if self.num_workers < 0:
            raise ValueError("num_workers must be non-negative")
        if self.device not in {"auto", "cpu", "cuda"}:
            raise ValueError("device must be auto, cpu, or cuda")

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, raw: dict[str, Any]) -> G0SmokeConfig:
        allowed = set(cls.__dataclass_fields__)
        if unknown := sorted(set(raw) - allowed):
            raise ValueError(f"unknown G0 config fields: {unknown}")
        values = dict(raw)
        if "image_size" in values:
            values["image_size"] = tuple(int(value) for value in values["image_size"])
        config = cls(**values)
        config.validate()
        return config


@dataclass(frozen=True)
class G0SmokeResult:
    """New or cache-verified evidence that the G0 gate passed."""

    source: Literal["run", "cache"]
    run_id: str
    passed: bool
    tiny_overfit: dict[str, Any]
    integration: dict[str, Any]
    oof: pd.DataFrame
    cache_key: RunCacheKey
    artifacts: dict[str, str]


def load_g0_config(path: str | Path) -> G0SmokeConfig:
    """Load a strict UTF-8 G0 JSON declaration."""
    source = Path(path)
    with source.open(encoding="utf-8") as handle:
        raw = json.load(handle)
    if not isinstance(raw, dict):
        raise ValueError(f"G0 config must be a JSON object: {source}")
    return G0SmokeConfig.from_dict(raw)


def select_balanced_smoke_rows(
    frame: pd.DataFrame,
    *,
    labels: tuple[str, ...],
    per_class: int,
    seed: int,
    target: str = "season",
) -> pd.DataFrame:
    """Select stable ID-ranked rows without changing canonical fold membership."""
    if per_class < 1:
        raise ValueError("per_class must be positive")
    if frame.empty or frame["id"].duplicated().any():
        raise ValueError("smoke candidates must be non-empty with unique IDs")
    if not frame["partition"].eq("development").all():
        raise ValueError("smoke candidates must contain development rows only")
    unknown = sorted(set(frame[target].astype(str)) - set(labels))
    if unknown:
        raise ValueError(f"smoke candidates contain unknown labels: {unknown}")

    selected = []
    for class_index, label in enumerate(labels):
        rows = frame.loc[frame[target].astype(str).eq(label)].copy()
        if len(rows) < per_class:
            raise ValueError(
                f"G0 needs {per_class} {label} rows but the canonical side has {len(rows)}"
            )
        rows["_smoke_rank"] = rows["id"].map(
            lambda item_id: hashlib.sha256(f"{seed}:{int(item_id)}".encode()).hexdigest()
        )
        rows = rows.sort_values(["_smoke_rank", "id"], kind="stable").head(per_class)
        rows["_class_order"] = class_index
        selected.append(rows)
    return (
        pd.concat(selected, ignore_index=True)
        .sort_values(["_class_order", "_smoke_rank", "id"], kind="stable")
        .drop(columns=["_class_order", "_smoke_rank"])
        .reset_index(drop=True)
    )


def _resolve_device(requested: str) -> torch.device:
    if requested == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if requested == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested for G0 but is unavailable")
    return torch.device(requested)


def overfit_tiny_batch(
    images: torch.Tensor,
    targets: torch.Tensor,
    *,
    config: G0SmokeConfig,
    num_classes: int,
) -> dict[str, Any]:
    """Memorise one fixed balanced batch and return the predeclared pass evidence."""
    if images.ndim != 4 or targets.ndim != 1 or len(images) != len(targets):
        raise ValueError("tiny overfit requires aligned NCHW images and 1D targets")
    seed_everything(config.seed)
    device = _resolve_device(config.device)
    model = build_season_model(
        SeasonModelSpec(family=config.model_family, num_classes=num_classes)
    ).to(device)
    assert_final_model(model)
    batch_images = images.to(device)
    batch_targets = targets.to(device, dtype=torch.long)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config.tiny_learning_rate,
        weight_decay=0.0,
    )

    def measure() -> tuple[float, float]:
        model.eval()
        with torch.inference_mode():
            logits = model(batch_images)
            loss = float(criterion(logits, batch_targets))
            accuracy = float(logits.argmax(dim=1).eq(batch_targets).float().mean())
        return loss, accuracy

    initial_loss, initial_accuracy = measure()
    loss_trace = []
    gradients_finite = True
    for step in range(1, config.tiny_steps + 1):
        model.train()
        optimizer.zero_grad(set_to_none=True)
        logits = model(batch_images)
        loss = criterion(logits, batch_targets)
        loss.backward()
        gradients_finite = gradients_finite and all(
            parameter.grad is None or torch.isfinite(parameter.grad).all().item()
            for parameter in model.parameters()
        )
        nn.utils.clip_grad_norm_(model.parameters(), config.gradient_clip_norm)
        optimizer.step()
        if step == 1 or step % 10 == 0 or step == config.tiny_steps:
            loss_trace.append({"step": step, "train_loss": float(loss.detach())})
    final_loss, final_accuracy = measure()
    loss_ratio = final_loss / initial_loss
    passed = (
        gradients_finite
        and final_accuracy >= config.minimum_tiny_accuracy
        and loss_ratio <= config.maximum_tiny_loss_ratio
    )
    return {
        "products": len(images),
        "steps": config.tiny_steps,
        "device": str(device),
        "initial_loss": initial_loss,
        "initial_accuracy": initial_accuracy,
        "final_loss": final_loss,
        "final_accuracy": final_accuracy,
        "loss_ratio": loss_ratio,
        "minimum_accuracy": config.minimum_tiny_accuracy,
        "maximum_loss_ratio": config.maximum_tiny_loss_ratio,
        "gradients_finite": gradients_finite,
        "passed": passed,
        "loss_trace": loss_trace,
    }


def _worker_options(num_workers: int) -> dict[str, Any]:
    return {"persistent_workers": True, "prefetch_factor": 2} if num_workers else {}


def _build_smoke_loaders(
    *,
    training: pd.DataFrame,
    validation: pd.DataFrame,
    tiny: pd.DataFrame,
    labels: tuple[str, ...],
    label_to_index: dict[str, int],
    config: G0SmokeConfig,
    data_root: Path,
) -> tuple[DataLoader[Any], DataLoader[Any], DataLoader[Any], dict[str, Any]]:
    stats = fit_fold_stats(
        training,
        validation_fold=config.validation_fold,
        image_size=config.image_size,
        root=data_root,
    )
    static_transform = build_image_transform(stats, training=False)
    training_transform = build_image_transform(
        stats,
        training=True,
        augmentation=config.augmentation,
    )
    tiny_dataset = EncodedClassificationDataset(
        tiny,
        transform=static_transform,
        target="season",
        label_to_index=label_to_index,
        root=data_root,
    )
    training_dataset = EncodedClassificationDataset(
        training,
        transform=training_transform,
        target="season",
        label_to_index=label_to_index,
        root=data_root,
    )
    validation_dataset = EncodedClassificationDataset(
        validation,
        transform=static_transform,
        target="season",
        label_to_index=label_to_index,
        root=data_root,
    )
    options = _worker_options(config.num_workers)
    tiny_loader = DataLoader(
        tiny_dataset,
        batch_size=len(tiny_dataset),
        shuffle=False,
        num_workers=config.num_workers,
        pin_memory=torch.cuda.is_available(),
        worker_init_fn=seed_worker,
        generator=make_torch_generator(config.seed),
        **options,
    )
    training_loader = DataLoader(
        training_dataset,
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=config.num_workers,
        pin_memory=torch.cuda.is_available(),
        worker_init_fn=seed_worker,
        generator=make_torch_generator(config.seed + 1),
        **options,
    )
    validation_loader = DataLoader(
        validation_dataset,
        batch_size=config.validation_batch_size,
        shuffle=False,
        num_workers=config.num_workers,
        pin_memory=torch.cuda.is_available(),
        worker_init_fn=seed_worker,
        generator=make_torch_generator(config.seed + 2),
        **options,
    )
    return tiny_loader, training_loader, validation_loader, stats.to_dict()


def _implementation_paths() -> tuple[str, ...]:
    return (
        "src/fashion/task2/smoke.py",
        "src/fashion/data/dataset.py",
        "src/fashion/data/images.py",
        "src/fashion/data/torch.py",
        "src/fashion/models/season.py",
        "src/fashion/train/artifacts.py",
        "src/fashion/train/cache.py",
        "src/fashion/train/engine.py",
        "src/fashion/train/metrics.py",
        "src/fashion/train/registry.py",
        "src/fashion/train/reproducibility.py",
    )


def _registry_path(path: Path, root: Path) -> str:
    try:
        return path.resolve().relative_to(root.resolve()).as_posix()
    except ValueError:
        return str(path.resolve())


def _resolve_artifact(path: str, root: Path) -> Path:
    candidate = Path(path)
    return candidate if candidate.is_absolute() else root / candidate


def _load_cached_result(
    *,
    row: dict[str, str],
    artifacts: dict[str, str],
    key: RunCacheKey,
    expected_validation_ids: list[int],
    labels: tuple[str, ...],
    data_root: Path,
) -> G0SmokeResult:
    history_path = _resolve_artifact(row["history_path"], data_root)
    prediction_path = _resolve_artifact(row["prediction_path"], data_root)
    with history_path.open(encoding="utf-8") as handle:
        history = json.load(handle)
    oof = pd.read_csv(prediction_path)
    validate_oof(oof, expected_ids=expected_validation_ids, labels=labels)
    if history.get("g0_passed") is not True:
        raise SmokeGateError(f"cached G0 history is not a pass: {history_path}")
    return G0SmokeResult(
        source="cache",
        run_id=row["run_id"],
        passed=True,
        tiny_overfit=dict(history["tiny_overfit"]),
        integration=dict(history["integration"]),
        oof=oof,
        cache_key=key,
        artifacts=artifacts,
    )


def run_or_load_g0_smoke(
    config: G0SmokeConfig | str | Path,
    *,
    mode: SmokeMode = "run_or_load",
    data_root: str | Path = ROOT,
    source_root: str | Path = ROOT,
    splits_path: str | Path = SPLITS_CSV,
    label_map_path: str | Path = LABEL_MAPS_JSON,
    registry_path: str | Path = RUNS_CSV,
    checkpoint_directory: str | Path = TASK2_CHECKPOINT_DIR,
    run_directory: str | Path = TASK2_RUN_DIR,
) -> G0SmokeResult:
    """Run G0 once or reuse only hash-verified checkpoint, OOF, and history bytes."""
    if mode not in {"run_or_load", "run", "load"}:
        raise ValueError(f"unknown G0 mode: {mode}")
    resolved = load_g0_config(config) if isinstance(config, (str, Path)) else config
    resolved.validate()
    project_root = Path(data_root).resolve()
    source_repository = Path(source_root).resolve()
    split_file = Path(splits_path)
    label_file = Path(label_map_path)
    checkpoint_root = Path(checkpoint_directory)
    run_root = Path(run_directory)

    mappings = load_label_maps(label_file)
    season_mapping = mappings["season"]
    labels = tuple(str(label) for label in season_mapping["classes"])
    label_to_index = {
        str(label): int(index)
        for label, index in dict(season_mapping["label_to_index"]).items()
    }
    if label_to_index != {label: index for index, label in enumerate(labels)}:
        raise ValueError("G0 label order and indices disagree")

    splits = load_splits(split_file)
    training_all, validation_all = get_cv_split(splits, resolved.validation_fold)
    training_all = get_samples(training_all, target="season").reset_index(drop=True)
    validation_all = get_samples(validation_all, target="season").reset_index(drop=True)
    integration_training = select_balanced_smoke_rows(
        training_all,
        labels=labels,
        per_class=resolved.integration_train_per_class,
        seed=resolved.seed,
    )
    integration_validation = select_balanced_smoke_rows(
        validation_all,
        labels=labels,
        per_class=resolved.integration_validation_per_class,
        seed=resolved.seed + 1,
    )
    tiny_training = select_balanced_smoke_rows(
        integration_training,
        labels=labels,
        per_class=resolved.tiny_per_class,
        seed=resolved.seed + 2,
    )
    expected_validation_ids = integration_validation["id"].astype(int).tolist()

    key = build_run_cache_key(
        resolved.to_dict(),
        fold=resolved.validation_fold,
        seed=resolved.seed,
        implementation_paths=_implementation_paths(),
        split_path=split_file,
        label_map_path=label_file,
        root=source_repository,
    )
    registry = RunRegistry(registry_path)
    if mode in {"run_or_load", "load"}:
        cached = find_cached_run(
            registry,
            key,
            required_artifacts=("checkpoint", "prediction", "history"),
            artifact_root=project_root,
        )
        if cached is not None:
            return _load_cached_result(
                row=cached.row,
                artifacts=dict(cached.verified_artifacts),
                key=key,
                expected_validation_ids=expected_validation_ids,
                labels=labels,
                data_root=project_root,
            )
    if mode == "load":
        raise FileNotFoundError("no valid cached G0 smoke run")

    run_id = new_run_id(resolved.experiment_id, resolved.validation_fold, resolved.seed)
    record = RunRecord(
        run_id=run_id,
        experiment_id=resolved.experiment_id,
        fold=resolved.validation_fold,
        seed=resolved.seed,
        config_sha256=key.config_sha256,
        split_sha256=key.split_sha256,
        label_map_sha256=key.label_map_sha256,
        implementation_sha256=key.implementation_sha256,
        stage=resolved.stage,
        model_family=resolved.model_family,
        benchmark_only=False,
        final_eligible=False,
        scratch=True,
        transform_id=ImageTransformSpec(
            image_size=resolved.image_size,
            augmentation=resolved.augmentation,
        ).transform_id,
        loss_id="cross_entropy",
        epochs_requested=resolved.integration_epochs,
        primary_metric_name="smoke_validation_macro_f1",
    )
    checkpoint_path = checkpoint_root / f"{run_id}.pt"
    artifact_directory = run_root / run_id
    prediction_path = artifact_directory / "oof.csv"
    history_path = artifact_directory / "history.json"
    started = time.perf_counter()
    with tracked_run(registry, record) as run:
        tiny_loader, training_loader, validation_loader, stats = _build_smoke_loaders(
            training=integration_training,
            validation=integration_validation,
            tiny=tiny_training,
            labels=labels,
            label_to_index=label_to_index,
            config=resolved,
            data_root=project_root,
        )
        tiny_batch = next(iter(tiny_loader))
        tiny_evidence = overfit_tiny_batch(
            tiny_batch["image"],
            tiny_batch["target"],
            config=resolved,
            num_classes=len(labels),
        )
        if not tiny_evidence["passed"]:
            raise SmokeGateError(
                "tiny overfit failed: "
                f"accuracy={tiny_evidence['final_accuracy']:.4f}, "
                f"loss_ratio={tiny_evidence['loss_ratio']:.4f}"
            )

        integration_model = build_season_model(
            SeasonModelSpec(family=resolved.model_family, num_classes=len(labels))
        )
        assert_final_model(integration_model)
        fold_result = train_fold(
            integration_model,
            training_loader,
            validation_loader,
            config=TrainConfig(
                fold=resolved.validation_fold,
                seed=resolved.seed,
                epochs=resolved.integration_epochs,
                learning_rate=resolved.learning_rate,
                weight_decay=resolved.weight_decay,
                batch_size=resolved.batch_size,
                effective_batch_size=resolved.effective_batch_size,
                gradient_clip_norm=resolved.gradient_clip_norm,
                warmup_epochs=0,
                patience=resolved.integration_epochs,
                use_amp=resolved.use_amp,
                device=resolved.device,
            ),
            checkpoint_path=checkpoint_path,
            labels=labels,
        )
        oof = fold_result.to_oof_frame()
        validate_oof(oof, expected_ids=expected_validation_ids, labels=labels)
        output_oof = oof.copy()
        output_oof.insert(0, "run_id", run_id)
        output_oof.insert(1, "experiment_id", resolved.experiment_id)
        integration_evidence = {
            "training_products": len(integration_training),
            "validation_products": len(integration_validation),
            "epochs_completed": fold_result.epochs_completed,
            "best_epoch": fold_result.best_epoch,
            "best_metrics": fold_result.best_metrics,
            "history": fold_result.history,
            "device": fold_result.device,
            "peak_vram_mb": fold_result.peak_vram_mb,
            "runtime_seconds": fold_result.runtime_seconds,
        }
        atomic_write_csv(prediction_path, output_oof)
        atomic_write_json(
            history_path,
            {
                "schema_version": "1.0.0",
                "g0_passed": True,
                "run_id": run_id,
                "config": resolved.to_dict(),
                "cache_key": asdict(key),
                "selected_ids": {
                    "tiny_training": tiny_training["id"].astype(int).tolist(),
                    "integration_training": integration_training["id"].astype(int).tolist(),
                    "integration_validation": expected_validation_ids,
                },
                "fold_stats": stats,
                "tiny_overfit": tiny_evidence,
                "integration": integration_evidence,
            },
        )
        run.epochs_completed = fold_result.epochs_completed
        run.best_epoch = fold_result.best_epoch
        run.primary_metric_value = float(fold_result.best_metrics["macro_f1"])
        run.metrics = {
            "g0_passed": True,
            "tiny_overfit": tiny_evidence,
            "integration": fold_result.best_metrics,
        }
        run.parameter_count = fold_result.parameter_count
        run.peak_vram_mb = fold_result.peak_vram_mb
        run.checkpoint_path = _registry_path(checkpoint_path, project_root)
        run.checkpoint_sha256 = fold_result.checkpoint_sha256
        run.prediction_path = _registry_path(prediction_path, project_root)
        run.prediction_sha256 = compute_sha256(prediction_path)
        run.history_path = _registry_path(history_path, project_root)
        run.history_sha256 = compute_sha256(history_path)
        run.runtime_seconds = time.perf_counter() - started

    return G0SmokeResult(
        source="run",
        run_id=run_id,
        passed=True,
        tiny_overfit=tiny_evidence,
        integration=integration_evidence,
        oof=output_oof,
        cache_key=key,
        artifacts={
            "checkpoint": run.checkpoint_sha256,
            "prediction": run.prediction_sha256,
            "history": run.history_sha256,
        },
    )
