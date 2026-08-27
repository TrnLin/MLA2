"""Immutable config parsing and hash-validated run-or-load experiment orchestration."""

from __future__ import annotations

import json
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Literal

import pandas as pd

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
from fashion.data.torch import ImageTransformSpec, build_task_loaders
from fashion.models.season import (
    BenchmarkModelSpec,
    SeasonModelSpec,
    assert_final_model,
    build_benchmark_model,
    build_season_model,
    model_boundary_audit,
)
from fashion.task2.baselines import evaluate_majority_fold
from fashion.task2.classical import HogHsvSpec, evaluate_hog_hsv_svm_fold
from fashion.train.artifacts import atomic_write_csv, atomic_write_json
from fashion.train.cache import RunCacheKey, build_run_cache_key, find_cached_run
from fashion.train.engine import FoldResult, TrainConfig, train_fold
from fashion.train.metrics import (
    validate_metrics_match_oof,
    validate_oof,
    validate_oof_identity,
)
from fashion.train.registry import RunRecord, RunRegistry, new_run_id, tracked_run
from fashion.train.reproducibility import seed_everything

ExperimentMethod = Literal["majority", "hog_hsv_svm", "deep"]
ExecutionMode = Literal["run_or_load", "run", "load"]
FINAL_DEEP_FAMILIES = frozenset({"smallcnn", "resnet18_small_stem", "mobilenet_v3_small"})
BENCHMARK_FAMILIES = frozenset(
    {"resnet18_standard_scratch", "resnet18_standard_pretrained"}
)


@dataclass(frozen=True)
class DataRunConfig:
    """Image and DataLoader settings shared within one comparison row."""

    image_size: tuple[int, int] = (80, 60)
    augmentation: str = "a0"
    batch_size: int = 32
    validation_batch_size: int = 64
    num_workers: int = 4
    pin_memory: bool | None = None

    def validate(self) -> None:
        if len(self.image_size) != 2 or any(int(value) < 1 for value in self.image_size):
            raise ValueError("data.image_size must contain two positive integers")
        if self.augmentation not in {"none", "a0", "a1"}:
            raise ValueError("data.augmentation must be none, a0, or a1")
        if self.batch_size < 1 or self.validation_batch_size < 1:
            raise ValueError("data batch sizes must be positive")
        if self.num_workers < 0:
            raise ValueError("data.num_workers must be non-negative")


@dataclass(frozen=True)
class OptimisationRunConfig:
    """Equal-budget optimisation settings converted into ``TrainConfig`` per fold."""

    epochs: int = 8
    learning_rate: float = 3e-4
    weight_decay: float = 1e-4
    effective_batch_size: int = 128
    gradient_clip_norm: float = 1.0
    warmup_epochs: float = 1.0
    patience: int = 5
    min_delta: float = 1e-4
    use_amp: bool = True
    device: str = "auto"


@dataclass(frozen=True)
class ExperimentConfig:
    """One immutable scientific question, expanded over declared folds and seeds."""

    experiment_id: str
    method: ExperimentMethod
    model_family: str
    stage: str
    schema_version: str = "1.0.0"
    target: str = "season"
    folds: tuple[int, ...] = (0, 1, 2, 3, 4)
    seeds: tuple[int, ...] = (RANDOM_SEED,)
    loss_id: str = "cross_entropy"
    data: DataRunConfig = field(default_factory=DataRunConfig)
    optimisation: OptimisationRunConfig = field(default_factory=OptimisationRunConfig)
    hog_hsv: HogHsvSpec = field(default_factory=HogHsvSpec)

    def validate(self) -> None:
        if self.schema_version != "1.0.0":
            raise ValueError(f"unsupported Task 2 experiment schema: {self.schema_version}")
        if not self.experiment_id.strip() or not self.stage.strip():
            raise ValueError("experiment_id and stage must be non-empty")
        if self.target != "season":
            raise ValueError("Task 2 experiment target must be season")
        if self.method not in {"majority", "hog_hsv_svm", "deep"}:
            raise ValueError(f"unknown experiment method: {self.method}")
        expected_families = {
            "majority": {"majority"},
            "hog_hsv_svm": {"hog_hsv_svm"},
            "deep": FINAL_DEEP_FAMILIES | BENCHMARK_FAMILIES,
        }
        if self.model_family not in expected_families[self.method]:
            raise ValueError(
                f"model_family {self.model_family!r} is invalid for method {self.method!r}"
            )
        if not self.folds or len(set(self.folds)) != len(self.folds):
            raise ValueError("folds must be non-empty and unique")
        if set(self.folds) - set(range(5)):
            raise ValueError("folds must use only canonical values 0 through 4")
        if not self.seeds or len(set(self.seeds)) != len(self.seeds):
            raise ValueError("seeds must be non-empty and unique")
        if any(seed < 0 for seed in self.seeds):
            raise ValueError("seeds must be non-negative")
        if not self.loss_id.strip():
            raise ValueError("loss_id must be non-empty")
        self.data.validate()
        self.hog_hsv.validate()
        if self.method == "deep":
            for fold in self.folds:
                TrainConfig(
                    fold=fold,
                    seed=self.seeds[0],
                    batch_size=self.data.batch_size,
                    **asdict(self.optimisation),
                ).validate()

    def to_dict(self) -> dict[str, Any]:
        """Return the canonical hash payload written into run history."""
        return asdict(self)

    @classmethod
    def from_dict(cls, raw: dict[str, Any]) -> ExperimentConfig:
        """Parse strict nested JSON so misspelled options fail before a run starts."""
        allowed = {
            "schema_version",
            "experiment_id",
            "method",
            "model_family",
            "stage",
            "target",
            "folds",
            "seeds",
            "loss_id",
            "data",
            "optimisation",
            "hog_hsv",
        }
        unknown = sorted(set(raw) - allowed)
        if unknown:
            raise ValueError(f"unknown experiment config fields: {unknown}")
        required = {"experiment_id", "method", "model_family", "stage"}
        missing = sorted(required - set(raw))
        if missing:
            raise ValueError(f"experiment config is missing fields: {missing}")

        data_raw = dict(raw.get("data", {}))
        optimisation_raw = dict(raw.get("optimisation", {}))
        hog_raw = dict(raw.get("hog_hsv", {}))
        _reject_unknown_dataclass_fields(DataRunConfig, data_raw, "data")
        _reject_unknown_dataclass_fields(
            OptimisationRunConfig,
            optimisation_raw,
            "optimisation",
        )
        _reject_unknown_dataclass_fields(HogHsvSpec, hog_raw, "hog_hsv")
        if "image_size" in data_raw:
            data_raw["image_size"] = tuple(data_raw["image_size"])
        for name in (
            "hog_pixels_per_cell",
            "hog_cells_per_block",
            "hsv_bins",
            "image_size",
        ):
            if name in hog_raw:
                hog_raw[name] = tuple(hog_raw[name])
        values = dict(raw)
        values["folds"] = tuple(values.get("folds", (0, 1, 2, 3, 4)))
        values["seeds"] = tuple(values.get("seeds", (RANDOM_SEED,)))
        values["data"] = DataRunConfig(**data_raw)
        values["optimisation"] = OptimisationRunConfig(**optimisation_raw)
        values["hog_hsv"] = HogHsvSpec(**hog_raw)
        config = cls(**values)
        config.validate()
        return config


@dataclass(frozen=True)
class ExperimentFoldOutput:
    """A newly executed or verified cached fold result."""

    experiment_id: str
    fold: int
    seed: int
    run_id: str
    source: Literal["run", "cache"]
    oof: pd.DataFrame
    metrics: dict[str, Any]
    cache_key: RunCacheKey
    artifacts: dict[str, str]


def _reject_unknown_dataclass_fields(kind: type[Any], values: dict[str, Any], scope: str) -> None:
    allowed = set(kind.__dataclass_fields__)
    unknown = sorted(set(values) - allowed)
    if unknown:
        raise ValueError(f"unknown {scope} fields: {unknown}")


def load_experiment_config(path: str | Path) -> ExperimentConfig:
    """Load one UTF-8 JSON experiment declaration without modifying it."""
    source = Path(path)
    with source.open(encoding="utf-8") as handle:
        raw = json.load(handle)
    if not isinstance(raw, dict):
        raise ValueError(f"experiment config must be a JSON object: {source}")
    return ExperimentConfig.from_dict(raw)


def _implementation_paths(method: ExperimentMethod) -> tuple[str, ...]:
    common = (
        "src/fashion/task2/experiments.py",
        "src/fashion/train/artifacts.py",
        "src/fashion/train/cache.py",
        "src/fashion/train/metrics.py",
        "src/fashion/train/registry.py",
    )
    if method == "majority":
        return (*common, "src/fashion/task2/baselines.py")
    if method == "hog_hsv_svm":
        return (
            *common,
            "src/fashion/data/images.py",
            "src/fashion/task2/baselines.py",
            "src/fashion/task2/classical.py",
        )
    return (
        *common,
        "src/fashion/data/dataset.py",
        "src/fashion/data/images.py",
        "src/fashion/data/torch.py",
        "src/fashion/models/season.py",
        "src/fashion/train/engine.py",
        "src/fashion/train/reproducibility.py",
    )


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
    expected_targets = None
    if "season" in expected:
        expected_targets = dict(
            zip(expected["id"], expected["season"].astype(str), strict=True)
        )
    return validate_oof(
        oof,
        expected_ids=expected["id"],
        labels=labels,
        expected_targets=expected_targets,
    )


def _load_cached_output(
    *,
    cached,
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
    metrics = json.loads(cached.row["metrics"] or "{}")
    validate_oof_identity(
        oof,
        run_id=cached.run_id,
        experiment_id=config.experiment_id,
        fold=fold,
        seed=seed,
    )
    validate_metrics_match_oof(oof, metrics, labels=labels)
    return ExperimentFoldOutput(
        experiment_id=config.experiment_id,
        fold=fold,
        seed=seed,
        run_id=cached.run_id,
        source="cache",
        oof=oof,
        metrics=metrics,
        cache_key=key,
        artifacts=dict(cached.verified_artifacts),
    )


def _model_flags(config: ExperimentConfig) -> dict[str, bool]:
    pretrained = config.model_family == "resnet18_standard_pretrained"
    benchmark = config.model_family in BENCHMARK_FAMILIES
    return {
        "scratch": not pretrained,
        "benchmark_only": benchmark,
        "final_eligible": not benchmark,
    }


def _build_deep_model(config: ExperimentConfig, *, num_classes: int):
    if config.model_family in FINAL_DEEP_FAMILIES:
        model = build_season_model(
            SeasonModelSpec(
                family=config.model_family,
                num_classes=num_classes,
            )
        )
        assert_final_model(model)
        return model
    return build_benchmark_model(
        BenchmarkModelSpec(
            family=config.model_family,
            num_classes=num_classes,
        )
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


def _execute_baseline(
    *,
    config: ExperimentConfig,
    fold: int,
    seed: int,
    expected: pd.DataFrame,
    splits_path: Path,
    data_root: Path,
) -> tuple[pd.DataFrame, dict[str, Any], dict[str, Any], int]:
    splits = load_splits(splits_path)
    training, validation = get_cv_split(splits, fold)
    if config.method == "majority":
        result = evaluate_majority_fold(
            training,
            validation,
            validation_fold=fold,
        )
        history = {
            "model": {
                "labels": result.model.labels,
                "class_counts": result.model.class_counts,
                "class_probabilities": result.model.class_probabilities,
                "majority_label": result.model.majority_label,
                "training_product_count": result.model.training_product_count,
                "training_id_sha256": result.model.training_id_sha256,
            }
        }
        parameter_count = 0
    else:
        result = evaluate_hog_hsv_svm_fold(
            training,
            validation,
            validation_fold=fold,
            spec=config.hog_hsv,
            seed=seed,
            root=data_root,
        )
        classifier = result.model.pipeline.named_steps["svm"]
        parameter_count = int(classifier.coef_.size + classifier.intercept_.size)
        history = {
            "model": {
                "spec": asdict(result.model.spec),
                "training_product_count": result.model.training_product_count,
                "training_id_sha256": result.model.training_id_sha256,
                "feature_count": result.model.feature_count,
                "probability_method": result.model.probability_method,
            }
        }
    _validate_output_oof(result.oof, expected=expected, labels=result.model.labels)
    return result.oof, result.metrics, history, parameter_count


def _execute_deep(
    *,
    config: ExperimentConfig,
    fold: int,
    seed: int,
    checkpoint_path: Path,
    data_root: Path,
    splits_path: Path,
    label_map_path: Path,
) -> tuple[FoldResult, dict[str, Any]]:
    # Model parameters are initialized before train_fold receives the model, so the
    # declared run seed must be applied here as well as inside the training engine.
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
    model = _build_deep_model(config, num_classes=len(loaders.labels))
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
        implementation_paths=_implementation_paths(config.method),
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
    required_artifacts = (
        ("checkpoint", "prediction", "history")
        if config.method == "deep"
        else ("prediction", "history")
    )
    cached = None
    if mode in {"run_or_load", "load"}:
        cached = find_cached_run(
            registry,
            key,
            required_artifacts=required_artifacts,
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
    flags = _model_flags(config)
    transform_id = "none"
    if config.method == "hog_hsv_svm":
        transform_id = config.hog_hsv.feature_id
    elif config.method == "deep":
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
        benchmark_only=flags["benchmark_only"],
        final_eligible=flags["final_eligible"],
        scratch=flags["scratch"],
        transform_id=transform_id,
        loss_id=config.loss_id,
        epochs_requested=(config.optimisation.epochs if config.method == "deep" else 0),
        primary_metric_name="macro_f1",
    )
    started = time.perf_counter()
    with tracked_run(registry, record) as run:
        if config.method == "deep":
            checkpoint_path = checkpoint_directory / f"{run_id}.pt"
            result, history = _execute_deep(
                config=config,
                fold=fold,
                seed=seed,
                checkpoint_path=checkpoint_path,
                data_root=data_root,
                splits_path=splits_path,
                label_map_path=label_map_path,
            )
            oof = result.to_oof_frame()
            metrics = result.best_metrics
            run.epochs_completed = result.epochs_completed
            run.best_epoch = result.best_epoch
            run.checkpoint_path = _registry_path(checkpoint_path, data_root)
            run.checkpoint_sha256 = result.checkpoint_sha256
            run.parameter_count = result.parameter_count
            run.peak_vram_mb = result.peak_vram_mb
            run.runtime_seconds = result.runtime_seconds
        else:
            oof, metrics, history, parameter_count = _execute_baseline(
                config=config,
                fold=fold,
                seed=seed,
                expected=expected,
                splits_path=splits_path,
                data_root=data_root,
            )
            run.epochs_completed = 0
            run.parameter_count = parameter_count
            run.runtime_seconds = time.perf_counter() - started
        _validate_output_oof(oof, expected=expected, labels=labels)
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
        "prediction": run.prediction_sha256,
        "history": run.history_sha256,
    }
    if run.checkpoint_sha256:
        artifacts["checkpoint"] = run.checkpoint_sha256
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


def run_or_load_experiment(
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
    """Expand one immutable config and execute or verify every fold/seed run."""
    if mode not in {"run_or_load", "run", "load"}:
        raise ValueError(f"unknown execution mode: {mode}")
    resolved = load_experiment_config(config) if isinstance(config, (str, Path)) else config
    resolved.validate()
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


def run_matrix(
    configs: list[ExperimentConfig | str | Path],
    **kwargs: Any,
) -> list[ExperimentFoldOutput]:
    """Run declared scientific questions in order and reject duplicate experiment IDs."""
    resolved = [
        load_experiment_config(config) if isinstance(config, (str, Path)) else config
        for config in configs
    ]
    identifiers = [config.experiment_id for config in resolved]
    if len(set(identifiers)) != len(identifiers):
        raise ValueError("experiment matrix contains duplicate experiment_id values")
    outputs: list[ExperimentFoldOutput] = []
    for config in resolved:
        outputs.extend(run_or_load_experiment(config, **kwargs))
    return outputs
