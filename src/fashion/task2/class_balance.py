"""Isolated run-or-load orchestration for the frozen I1 class-balance experiment."""

from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path
from typing import Any

import pandas as pd

from fashion.config import (
    LABEL_MAPS_JSON,
    ROOT,
    RUNS_CSV,
    SPLITS_CSV,
    TASK2_CHECKPOINT_DIR,
    TASK2_RUN_DIR,
)
from fashion.data.dataset import get_cv_split, get_samples, load_label_maps, load_splits
from fashion.data.hashing import compute_sha256
from fashion.data.torch import ImageTransformSpec, TaskLoaders, build_task_loaders
from fashion.models.season import (
    SeasonModelSpec,
    assert_final_model,
    build_season_model,
    model_boundary_audit,
)
from fashion.task2.experiments import (
    DataRunConfig,
    ExecutionMode,
    ExperimentConfig,
    ExperimentFoldOutput,
    OptimisationRunConfig,
    load_experiment_config,
)
from fashion.train.artifacts import atomic_write_csv, atomic_write_json
from fashion.train.cache import RunCacheKey, build_run_cache_key, find_cached_run
from fashion.train.engine import FoldResult, TrainConfig, train_fold
from fashion.train.losses import (
    EFFECTIVE_NUMBER_BETA,
    EFFECTIVE_NUMBER_LOSS_ID,
    EffectiveNumberAudit,
    build_effective_number_cross_entropy,
    fit_effective_number_weights,
)
from fashion.train.metrics import validate_oof
from fashion.train.registry import RunRecord, RunRegistry, new_run_id, tracked_run
from fashion.train.reproducibility import seed_everything

I1_EXPERIMENT_ID = "g4-i1-effective-number-c1"
I1_STAGE = "g4_i1_class_balanced"

# Keep the legacy deep list byte-for-byte stable. A test checks this prefix against
# experiments._implementation_paths("deep") without editing that cache-critical file.
I1_IMPLEMENTATION_PATHS = (
    "src/fashion/task2/experiments.py",
    "src/fashion/train/artifacts.py",
    "src/fashion/train/cache.py",
    "src/fashion/train/metrics.py",
    "src/fashion/train/registry.py",
    "src/fashion/data/dataset.py",
    "src/fashion/data/images.py",
    "src/fashion/data/torch.py",
    "src/fashion/models/season.py",
    "src/fashion/train/engine.py",
    "src/fashion/train/reproducibility.py",
    "src/fashion/train/losses.py",
    "src/fashion/task2/class_balance.py",
)

I1_DATA_CONFIG = DataRunConfig(
    image_size=(80, 60),
    augmentation="a0",
    batch_size=32,
    validation_batch_size=128,
    num_workers=4,
    pin_memory=True,
)
I1_OPTIMISATION_CONFIG = OptimisationRunConfig(
    epochs=30,
    learning_rate=1e-3,
    weight_decay=1e-4,
    effective_batch_size=128,
    gradient_clip_norm=1.0,
    warmup_epochs=1.0,
    patience=5,
    min_delta=1e-4,
    use_amp=True,
    device="auto",
)


def validate_i1_config(config: ExperimentConfig) -> None:
    """Reject any change beyond G3 C1 identity, stage, and effective-number loss."""
    config.validate()
    expected = {
        "experiment_id": I1_EXPERIMENT_ID,
        "method": "deep",
        "model_family": "smallcnn",
        "stage": I1_STAGE,
        "target": "season",
        "folds": tuple(range(5)),
        "seeds": (2753,),
        "loss_id": EFFECTIVE_NUMBER_LOSS_ID,
        "data": I1_DATA_CONFIG,
        "optimisation": I1_OPTIMISATION_CONFIG,
    }
    observed = {
        "experiment_id": config.experiment_id,
        "method": config.method,
        "model_family": config.model_family,
        "stage": config.stage,
        "target": config.target,
        "folds": config.folds,
        "seeds": config.seeds,
        "loss_id": config.loss_id,
        "data": config.data,
        "optimisation": config.optimisation,
    }
    mismatches = [name for name, value in expected.items() if observed[name] != value]
    if mismatches:
        raise ValueError(f"I1 config violates the frozen protocol: {mismatches}")


def _registry_path(path: Path, root: Path) -> str:
    try:
        return path.resolve().relative_to(root.resolve()).as_posix()
    except ValueError:
        return str(path.resolve())


def _expected_validation(
    *,
    fold: int,
    splits_path: Path,
    target: str,
) -> pd.DataFrame:
    splits = load_splits(splits_path)
    _, validation = get_cv_split(splits, fold)
    return get_samples(validation, target=target).reset_index(drop=True)


def _validate_output_oof(
    oof: pd.DataFrame,
    *,
    expected: pd.DataFrame,
    labels: tuple[str, ...],
) -> dict[str, Any]:
    return validate_oof(oof, expected_ids=expected["id"], labels=labels)


def fit_i1_fold_balance(
    loaders: TaskLoaders,
    *,
    validation_fold: int,
    splits_path: str | Path = SPLITS_CSV,
) -> EffectiveNumberAudit:
    """Fit I1 weights from the loader's training side and reject ID drift."""
    splits = load_splits(splits_path)
    training, _ = get_cv_split(splits, validation_fold)
    training = get_samples(training, target="season").reset_index(drop=True)
    if training.empty or not training["partition"].eq("development").all():
        raise ValueError("I1 class weights require development training rows")
    training_ids = tuple(int(value) for value in training["id"])
    if training_ids != tuple(loaders.training_ids):
        raise ValueError("I1 class-weight IDs do not match the training loader")
    audit = fit_effective_number_weights(
        training["season"].astype(str),
        loaders.labels,
        training_ids=training_ids,
        beta=EFFECTIVE_NUMBER_BETA,
    )
    if audit.training_id_sha256 != loaders.stats.training_id_sha256:
        raise ValueError("I1 class weights and fold statistics used different training IDs")
    return audit


def _load_cached_output(
    *,
    cached: Any,
    config: ExperimentConfig,
    fold: int,
    seed: int,
    key: RunCacheKey,
    data_root: Path,
    expected: pd.DataFrame,
    labels: tuple[str, ...],
) -> ExperimentFoldOutput:
    prediction_path = Path(cached.row["prediction_path"])
    if not prediction_path.is_absolute():
        prediction_path = data_root / prediction_path
    oof = pd.read_csv(prediction_path)
    _validate_output_oof(oof, expected=expected, labels=labels)
    return ExperimentFoldOutput(
        experiment_id=config.experiment_id,
        fold=fold,
        seed=seed,
        run_id=cached.run_id,
        source="cache",
        oof=oof,
        metrics=json.loads(cached.row["metrics"] or "{}"),
        cache_key=key,
        artifacts=dict(cached.verified_artifacts),
    )


def _write_run_artifacts(
    *,
    run_id: str,
    config: ExperimentConfig,
    fold: int,
    seed: int,
    oof: pd.DataFrame,
    metrics: dict[str, Any],
    history_payload: dict[str, Any],
    run_directory: Path,
    data_root: Path,
) -> dict[str, str]:
    directory = run_directory / run_id
    prediction_path = directory / "oof.csv"
    history_path = directory / "history.json"
    output = oof.copy()
    output.insert(0, "run_id", run_id)
    output.insert(1, "experiment_id", config.experiment_id)
    if "seed" not in output:
        output.insert(3, "seed", seed)
    atomic_write_csv(prediction_path, output)
    atomic_write_json(
        history_path,
        {
            "schema_version": "1.0.0",
            "run_id": run_id,
            "experiment_id": config.experiment_id,
            "fold": fold,
            "seed": seed,
            "config": config.to_dict(),
            "metrics": metrics,
            **history_payload,
        },
    )
    return {
        "prediction_path": _registry_path(prediction_path, data_root),
        "prediction_sha256": compute_sha256(prediction_path),
        "history_path": _registry_path(history_path, data_root),
        "history_sha256": compute_sha256(history_path),
    }


def _execute_i1(
    *,
    config: ExperimentConfig,
    fold: int,
    seed: int,
    checkpoint_path: Path,
    data_root: Path,
    splits_path: Path,
    label_map_path: Path,
) -> tuple[FoldResult, dict[str, Any]]:
    # Match the legacy deep path: the run seed owns initialization and training.
    seed_everything(seed)
    loaders = build_task_loaders(
        validation_fold=fold,
        image_size=config.data.image_size,
        batch_size=config.data.batch_size,
        target=config.target,
        augmentation=config.data.augmentation,
        seed=seed,
        num_workers=config.data.num_workers,
        validation_batch_size=config.data.validation_batch_size,
        pin_memory=config.data.pin_memory,
        root=data_root,
        splits_path=splits_path,
        label_map_path=label_map_path,
    )
    balance_audit = fit_i1_fold_balance(
        loaders,
        validation_fold=fold,
        splits_path=splits_path,
    )
    if balance_audit.loss_id != config.loss_id:
        raise ValueError("fitted I1 loss does not match the declared config")
    model = build_season_model(
        SeasonModelSpec(family=config.model_family, num_classes=len(loaders.labels))
    )
    assert_final_model(model)
    result = train_fold(
        model,
        loaders.train,
        loaders.validation,
        config=TrainConfig(
            fold=fold,
            seed=seed,
            batch_size=config.data.batch_size,
            **asdict(config.optimisation),
        ),
        checkpoint_path=checkpoint_path,
        labels=loaders.labels,
        criterion=build_effective_number_cross_entropy(balance_audit),
    )
    oof = result.to_oof_frame()
    _validate_output_oof(
        oof,
        expected=pd.DataFrame({"id": loaders.validation_ids}),
        labels=loaders.labels,
    )
    return result, {
        "model_boundary": model_boundary_audit(model),
        "loader_audit": loaders.audit(),
        "class_balance": balance_audit.to_dict(),
        "loss_note": (
            "Training and validation loss use fold-fitted effective-number weights; "
            "compare I1 with G3 using OOF macro-F1 and per-class metrics, not loss scale."
        ),
        "epoch_history": result.history,
        "engine_metadata": result.metadata,
    }


def _run_one(
    config: ExperimentConfig,
    *,
    fold: int,
    seed: int,
    mode: ExecutionMode,
    registry: RunRegistry,
    data_root: Path,
    source_root: Path,
    splits_path: Path,
    label_map_path: Path,
    checkpoint_directory: Path,
    run_directory: Path,
) -> ExperimentFoldOutput:
    key = build_run_cache_key(
        config.to_dict(),
        fold=fold,
        seed=seed,
        implementation_paths=I1_IMPLEMENTATION_PATHS,
        split_path=splits_path,
        label_map_path=label_map_path,
        root=source_root,
    )
    expected = _expected_validation(
        fold=fold,
        splits_path=splits_path,
        target=config.target,
    )
    mappings = load_label_maps(label_map_path)
    labels = tuple(str(value) for value in mappings[config.target]["classes"])
    cached = None
    if mode in {"run_or_load", "load"}:
        cached = find_cached_run(
            registry,
            key,
            required_artifacts=("checkpoint", "prediction", "history"),
            artifact_root=data_root,
        )
    if cached is not None:
        return _load_cached_output(
            cached=cached,
            config=config,
            fold=fold,
            seed=seed,
            key=key,
            data_root=data_root,
            expected=expected,
            labels=labels,
        )
    if mode == "load":
        raise FileNotFoundError(
            f"no valid cached run for {config.experiment_id}, fold={fold}, seed={seed}"
        )

    run_id = new_run_id(config.experiment_id, fold, seed)
    transform_id = ImageTransformSpec(
        image_size=config.data.image_size,
        augmentation=config.data.augmentation,
    ).transform_id
    record = RunRecord(
        run_id=run_id,
        experiment_id=config.experiment_id,
        fold=fold,
        seed=seed,
        config_sha256=key.config_sha256,
        split_sha256=key.split_sha256,
        label_map_sha256=key.label_map_sha256,
        implementation_sha256=key.implementation_sha256,
        stage=config.stage,
        model_family=config.model_family,
        benchmark_only=False,
        final_eligible=True,
        scratch=True,
        transform_id=transform_id,
        loss_id=config.loss_id,
        epochs_requested=config.optimisation.epochs,
        primary_metric_name="macro_f1",
    )
    with tracked_run(registry, record) as run:
        checkpoint_path = checkpoint_directory / f"{run_id}.pt"
        result, history = _execute_i1(
            config=config,
            fold=fold,
            seed=seed,
            checkpoint_path=checkpoint_path,
            data_root=data_root,
            splits_path=splits_path,
            label_map_path=label_map_path,
        )
        oof = result.to_oof_frame()
        _validate_output_oof(oof, expected=expected, labels=labels)
        metrics = result.best_metrics
        run.epochs_completed = result.epochs_completed
        run.best_epoch = result.best_epoch
        run.checkpoint_path = _registry_path(checkpoint_path, data_root)
        run.checkpoint_sha256 = result.checkpoint_sha256
        run.parameter_count = result.parameter_count
        run.peak_vram_mb = result.peak_vram_mb
        run.runtime_seconds = result.runtime_seconds
        artifact_fields = _write_run_artifacts(
            run_id=run_id,
            config=config,
            fold=fold,
            seed=seed,
            oof=oof,
            metrics=metrics,
            history_payload=history,
            run_directory=run_directory,
            data_root=data_root,
        )
        run.primary_metric_value = float(metrics["macro_f1"])
        run.metrics = metrics
        run.prediction_path = artifact_fields["prediction_path"]
        run.prediction_sha256 = artifact_fields["prediction_sha256"]
        run.history_path = artifact_fields["history_path"]
        run.history_sha256 = artifact_fields["history_sha256"]

    artifacts = {
        "checkpoint": run.checkpoint_sha256,
        "prediction": run.prediction_sha256,
        "history": run.history_sha256,
    }
    output_oof = oof.copy()
    output_oof.insert(0, "run_id", run_id)
    output_oof.insert(1, "experiment_id", config.experiment_id)
    return ExperimentFoldOutput(
        experiment_id=config.experiment_id,
        fold=fold,
        seed=seed,
        run_id=run_id,
        source="run",
        oof=output_oof,
        metrics=metrics,
        cache_key=key,
        artifacts=artifacts,
    )


def run_or_load_i1_experiment(
    config: ExperimentConfig | str | Path,
    *,
    mode: ExecutionMode = "run_or_load",
    data_root: str | Path = ROOT,
    source_root: str | Path = ROOT,
    splits_path: str | Path = SPLITS_CSV,
    label_map_path: str | Path = LABEL_MAPS_JSON,
    registry_path: str | Path = RUNS_CSV,
    checkpoint_directory: str | Path = TASK2_CHECKPOINT_DIR,
    run_directory: str | Path = TASK2_RUN_DIR,
) -> list[ExperimentFoldOutput]:
    """Execute or verify the five frozen I1 folds without changing legacy cache keys."""
    if mode not in {"run_or_load", "run", "load"}:
        raise ValueError(f"unknown execution mode: {mode}")
    resolved = load_experiment_config(config) if isinstance(config, (str, Path)) else config
    validate_i1_config(resolved)
    registry = RunRegistry(registry_path)
    return [
        _run_one(
            resolved,
            fold=fold,
            seed=seed,
            mode=mode,
            registry=registry,
            data_root=Path(data_root).resolve(),
            source_root=Path(source_root).resolve(),
            splits_path=Path(splits_path),
            label_map_path=Path(label_map_path),
            checkpoint_directory=Path(checkpoint_directory),
            run_directory=Path(run_directory),
        )
        for seed in resolved.seeds
        for fold in resolved.folds
    ]
