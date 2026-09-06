"""Run-or-load orchestration for frozen I2 ArticleType auxiliary experiments."""

from __future__ import annotations

import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from fashion.config import (
    LABEL_MAPS_JSON,
    ROOT,
    RUNS_CSV,
    SPLITS_CSV,
    TASK2_CHECKPOINT_DIR,
    TASK2_RUN_DIR,
)
from fashion.data.dataset import load_label_maps
from fashion.data.multitask import build_multitask_loaders
from fashion.data.torch import ImageTransformSpec
from fashion.models.season import (
    SeasonModelSpec,
    assert_final_model,
    build_multitask_season_model,
    model_boundary_audit,
)
from fashion.task2.experiments import (
    DataRunConfig,
    ExecutionMode,
    ExperimentConfig,
    ExperimentFoldOutput,
    OptimisationRunConfig,
    _expected_validation,
    _load_cached_output,
    _registry_path,
    _validate_output_oof,
    _write_run_artifacts,
)
from fashion.train.cache import RunCacheKey, build_run_cache_key, find_cached_run
from fashion.train.engine import FoldResult, TrainConfig
from fashion.train.multitask import train_masked_multitask_fold
from fashion.train.registry import RunRecord, RunRegistry, new_run_id, tracked_run
from fashion.train.reproducibility import seed_everything

I2_STAGE = "g4_i2_multitask"
I2_AUXILIARY_TARGET = "articleType"
I2_MASK_POLICY = "available_labels_only"
I2_PROTOCOLS = {
    "g4-i2-article-type-lambda-0-1-c1": {
        "auxiliary_weight": 0.1,
        "loss_id": "season_ce_plus_masked_article_type_ce_lambda_0_1",
    },
    "g4-i2-article-type-lambda-0-3-c1": {
        "auxiliary_weight": 0.3,
        "loss_id": "season_ce_plus_masked_article_type_ce_lambda_0_3",
    },
}
I2_DATA_CONFIG = DataRunConfig(
    image_size=(80, 60),
    augmentation="a0",
    batch_size=32,
    validation_batch_size=128,
    num_workers=4,
    pin_memory=True,
)
I2_OPTIMISATION_CONFIG = OptimisationRunConfig(
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
I2_IMPLEMENTATION_PATHS = (
    "src/fashion/task2/experiments.py",
    "src/fashion/config.py",
    "src/fashion/data/hashing.py",
    "src/fashion/data/metadata.py",
    "src/fashion/data/splits.py",
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
    "src/fashion/data/multitask.py",
    "src/fashion/train/multitask.py",
    "src/fashion/task2/multitask.py",
)


@dataclass(frozen=True)
class AuxiliaryRunConfig:
    """Training-only auxiliary label and its contribution to the total loss."""

    target: str
    loss_weight: float
    mask_policy: str = I2_MASK_POLICY

    def validate(self) -> None:
        if self.target != I2_AUXILIARY_TARGET:
            raise ValueError(f"I2 auxiliary target must be {I2_AUXILIARY_TARGET}")
        if not math.isfinite(self.loss_weight) or self.loss_weight <= 0:
            raise ValueError("auxiliary loss_weight must be a finite positive value")
        if self.mask_policy != I2_MASK_POLICY:
            raise ValueError(f"I2 mask_policy must be {I2_MASK_POLICY}")


@dataclass(frozen=True)
class I2ExperimentConfig:
    """A standard Task 2 experiment plus the explicit auxiliary-loss contract."""

    base: ExperimentConfig
    auxiliary: AuxiliaryRunConfig

    @property
    def experiment_id(self) -> str:
        return self.base.experiment_id

    @property
    def model_family(self) -> str:
        return self.base.model_family

    @property
    def stage(self) -> str:
        return self.base.stage

    @property
    def target(self) -> str:
        return self.base.target

    @property
    def folds(self) -> tuple[int, ...]:
        return self.base.folds

    @property
    def seeds(self) -> tuple[int, ...]:
        return self.base.seeds

    @property
    def loss_id(self) -> str:
        return self.base.loss_id

    @property
    def data(self) -> DataRunConfig:
        return self.base.data

    @property
    def optimisation(self) -> OptimisationRunConfig:
        return self.base.optimisation

    def to_dict(self) -> dict[str, Any]:
        payload = self.base.to_dict()
        payload["auxiliary"] = asdict(self.auxiliary)
        return payload


def load_i2_config(path: str | Path) -> I2ExperimentConfig:
    """Parse an I2 JSON declaration without weakening the shared strict schema."""
    source = Path(path)
    with source.open(encoding="utf-8") as handle:
        raw = json.load(handle)
    if not isinstance(raw, dict):
        raise ValueError(f"I2 config must be a JSON object: {source}")
    base_raw = dict(raw)
    if "auxiliary" not in base_raw:
        raise ValueError("I2 config is missing the auxiliary block")
    auxiliary_raw = base_raw.pop("auxiliary")
    if not isinstance(auxiliary_raw, dict):
        raise ValueError("I2 auxiliary block must be a JSON object")
    unknown = sorted(set(auxiliary_raw) - set(AuxiliaryRunConfig.__dataclass_fields__))
    if unknown:
        raise ValueError(f"unknown I2 auxiliary fields: {unknown}")
    required = {"target", "loss_weight"}
    missing = sorted(required - set(auxiliary_raw))
    if missing:
        raise ValueError(f"I2 auxiliary block is missing fields: {missing}")
    config = I2ExperimentConfig(
        base=ExperimentConfig.from_dict(base_raw),
        auxiliary=AuxiliaryRunConfig(**auxiliary_raw),
    )
    validate_i2_config(config)
    return config


def validate_i2_config(config: I2ExperimentConfig) -> None:
    """Reject any I2 change beyond identity, loss weight, and auxiliary head."""
    config.base.validate()
    config.auxiliary.validate()
    if config.experiment_id not in I2_PROTOCOLS:
        raise ValueError(f"unknown frozen I2 experiment_id: {config.experiment_id}")
    protocol = I2_PROTOCOLS[config.experiment_id]
    expected = {
        "method": "deep",
        "model_family": "smallcnn",
        "stage": I2_STAGE,
        "target": "season",
        "folds": tuple(range(5)),
        "seeds": (2753,),
        "loss_id": protocol["loss_id"],
        "data": I2_DATA_CONFIG,
        "optimisation": I2_OPTIMISATION_CONFIG,
        "auxiliary_target": I2_AUXILIARY_TARGET,
        "auxiliary_weight": protocol["auxiliary_weight"],
        "mask_policy": I2_MASK_POLICY,
    }
    observed = {
        "method": config.base.method,
        "model_family": config.model_family,
        "stage": config.stage,
        "target": config.target,
        "folds": config.folds,
        "seeds": config.seeds,
        "loss_id": config.loss_id,
        "data": config.data,
        "optimisation": config.optimisation,
        "auxiliary_target": config.auxiliary.target,
        "auxiliary_weight": config.auxiliary.loss_weight,
        "mask_policy": config.auxiliary.mask_policy,
    }
    mismatches = [name for name, value in expected.items() if observed[name] != value]
    if mismatches:
        raise ValueError(f"I2 config violates the frozen protocol: {mismatches}")


def _execute_i2(
    *,
    config: I2ExperimentConfig,
    fold: int,
    seed: int,
    checkpoint_path: Path,
    data_root: Path,
    splits_path: Path,
    label_map_path: Path,
) -> tuple[FoldResult, dict[str, Any]]:
    seed_everything(seed)
    loaders = build_multitask_loaders(
        validation_fold=fold,
        image_size=config.data.image_size,
        batch_size=config.data.batch_size,
        main_target=config.target,
        auxiliary_target=config.auxiliary.target,
        augmentation=config.data.augmentation,
        seed=seed,
        num_workers=config.data.num_workers,
        validation_batch_size=config.data.validation_batch_size,
        pin_memory=config.data.pin_memory,
        root=data_root,
        splits_path=splits_path,
        label_map_path=label_map_path,
    )
    model = build_multitask_season_model(
        SeasonModelSpec(family=config.model_family, num_classes=len(loaders.labels)),
        article_type_classes=len(loaders.auxiliary_labels),
    )
    assert_final_model(model)
    result = train_masked_multitask_fold(
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
        auxiliary_weight=config.auxiliary.loss_weight,
        labels=loaders.labels,
    )
    oof = result.to_oof_frame()
    _validate_output_oof(
        oof,
        expected=_expected_validation(
            fold=fold,
            splits_path=splits_path,
            target=config.target,
        ),
        labels=loaders.labels,
    )
    return result, {
        "model_boundary": model_boundary_audit(model),
        "loader_audit": loaders.audit(),
        "auxiliary": asdict(config.auxiliary),
        "loss_note": (
            "I2 total-loss scales differ across lambda values; compare models using "
            "Season OOF metrics and the declared aligned/conflict slices."
        ),
        "inference_boundary": "image_only_predict_season_logits",
        "epoch_history": result.history,
        "engine_metadata": result.metadata,
    }


def _run_one(
    config: I2ExperimentConfig,
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
    key: RunCacheKey = build_run_cache_key(
        config.to_dict(),
        fold=fold,
        seed=seed,
        implementation_paths=I2_IMPLEMENTATION_PATHS,
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
        result, history = _execute_i2(
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


def run_or_load_i2_experiment(
    config: I2ExperimentConfig | str | Path,
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
    """Execute or verify every fold in one frozen I2 lambda declaration."""
    if mode not in {"run_or_load", "run", "load"}:
        raise ValueError(f"unknown execution mode: {mode}")
    resolved = load_i2_config(config) if isinstance(config, (str, Path)) else config
    validate_i2_config(resolved)
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


def run_i2_matrix(
    configs: list[I2ExperimentConfig | str | Path],
    **kwargs: Any,
) -> list[ExperimentFoldOutput]:
    """Run the two declared lambda questions in order without duplicate identities."""
    resolved = [
        load_i2_config(config) if isinstance(config, (str, Path)) else config
        for config in configs
    ]
    identifiers = [config.experiment_id for config in resolved]
    if len(set(identifiers)) != len(identifiers):
        raise ValueError("I2 matrix contains duplicate experiment_id values")
    outputs: list[ExperimentFoldOutput] = []
    for config in resolved:
        outputs.extend(run_or_load_i2_experiment(config, **kwargs))
    return outputs
