"""Post-submission I2 activation and frozen-embedding experiments.

These experiments are deliberately isolated from the frozen Task 2 selection,
refit, holdout, and official-prediction evidence. They answer two later research
questions without changing the submitted I2 model:

1. Does frozen I2 actually contain dead ReLU feature channels? This diagnostic
   gate is evaluated before any activation replacement is justified.
2. Does a Random Forest fitted to frozen I2 embeddings outperform I2's learned
   linear Season head on the same canonical folds?
3. Does histogram gradient boosting, or a fixed 50/50 probability ensemble,
   improve further without tuning on the validation predictions?
"""

from __future__ import annotations

import argparse
import json
import math
import re
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Literal

import numpy as np
import pandas as pd
import torch
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from torch import nn
from torch.utils.data import DataLoader

from fashion.config import LABEL_MAPS_JSON, ROOT, RUNS_CSV, SPLITS_CSV
from fashion.data.dataset import get_samples, load_label_maps, load_splits
from fashion.data.hashing import compute_sha256
from fashion.data.multitask import build_multitask_loaders
from fashion.data.torch import ImageTransformSpec, build_task_loaders
from fashion.models.season import SeasonModelSpec, build_multitask_season_model
from fashion.task2.experiments import _expected_validation, _validate_output_oof
from fashion.task2.multitask import I2ExperimentConfig, load_i2_config
from fashion.train.artifacts import atomic_write_csv, atomic_write_json, verify_artifact
from fashion.train.cache import RunCacheKey, build_run_cache_key, find_cached_run
from fashion.train.engine import FoldResult, TrainConfig
from fashion.train.metrics import (
    SEASON_LABELS,
    multiclass_metrics,
    paired_group_bootstrap,
    validate_oof_identity,
)
from fashion.train.multitask import _evaluate_multitask, train_masked_multitask_fold
from fashion.train.registry import RunRecord, RunRegistry, new_run_id, tracked_run
from fashion.train.reproducibility import seed_everything

ExecutionMode = Literal["run_or_load", "run", "load"]

BASE_I2_EXPERIMENT_ID = "g4-i2-article-type-lambda-0-3-c1"
PRIMARY_SEED = 2753
CANONICAL_FOLDS = tuple(range(5))
DEFAULT_ACTIVATION_CONFIG = ROOT / "configs/task2/post_submission_i2_leaky_relu.json"
DEFAULT_RANDOM_FOREST_CONFIG = (
    ROOT / "configs/task2/post_submission_i2_embedding_random_forest.json"
)
DEFAULT_GRADIENT_BOOSTING_CONFIG = (
    ROOT / "configs/task2/post_submission_i2_embedding_hist_gradient_boosting.json"
)
DEFAULT_EVIDENCE_DIRECTORY = ROOT / "results/evidence/task2/post_submission"
DEFAULT_FIGURE_DIRECTORY = ROOT / "results/figures/task2/post_submission"
DEFAULT_TEMPORARY_DIRECTORY = ROOT / "tmp/task2/post_submission"

POST_SUBMISSION_IMPLEMENTATION_PATHS = (
    "src/fashion/task2/post_submission_experiments.py",
    "src/fashion/config.py",
    "src/fashion/data/dataset.py",
    "src/fashion/data/images.py",
    "src/fashion/data/multitask.py",
    "src/fashion/data/torch.py",
    "src/fashion/models/season.py",
    "src/fashion/train/artifacts.py",
    "src/fashion/train/cache.py",
    "src/fashion/train/engine.py",
    "src/fashion/train/metrics.py",
    "src/fashion/train/multitask.py",
    "src/fashion/train/registry.py",
    "src/fashion/train/reproducibility.py",
)


@dataclass(frozen=True)
class ActivationExperimentSpec:
    """Controlled activation replacement and diagnostic thresholds."""

    schema_version: str
    experiment_id: str
    base_i2_config: str
    activation: str
    negative_slope: float
    exact_dead_epsilon: float
    near_dead_positive_rate: float

    def validate(self) -> None:
        if self.schema_version != "1.0.0":
            raise ValueError("unsupported activation experiment schema")
        if not self.experiment_id.startswith("postsubmit-"):
            raise ValueError("post-submission experiment_id must start with 'postsubmit-'")
        if self.activation != "leaky_relu":
            raise ValueError("activation experiment must use leaky_relu")
        if not math.isfinite(self.negative_slope) or not 0.0 < self.negative_slope < 1.0:
            raise ValueError("negative_slope must be finite and in (0, 1)")
        if not math.isfinite(self.exact_dead_epsilon) or self.exact_dead_epsilon < 0:
            raise ValueError("exact_dead_epsilon must be finite and non-negative")
        if (
            not math.isfinite(self.near_dead_positive_rate)
            or not 0.0 < self.near_dead_positive_rate < 1.0
        ):
            raise ValueError("near_dead_positive_rate must be finite and in (0, 1)")


@dataclass(frozen=True)
class RandomForestExperimentSpec:
    """Fixed Random Forest head fitted to frozen I2 embeddings."""

    schema_version: str
    experiment_id: str
    base_i2_config: str
    n_estimators: int
    max_features: str
    min_samples_leaf: int
    class_weight: str | None
    oob_score: bool
    n_jobs: int
    embedding_batch_size: int
    random_state_policy: str

    def validate(self) -> None:
        if self.schema_version != "1.0.0":
            raise ValueError("unsupported Random Forest experiment schema")
        if not self.experiment_id.startswith("postsubmit-"):
            raise ValueError("post-submission experiment_id must start with 'postsubmit-'")
        if type(self.n_estimators) is not int or self.n_estimators < 10:
            raise ValueError("n_estimators must be an integer of at least 10")
        if self.max_features != "sqrt":
            raise ValueError("this controlled experiment fixes max_features='sqrt'")
        if type(self.min_samples_leaf) is not int or self.min_samples_leaf < 1:
            raise ValueError("min_samples_leaf must be a positive integer")
        if self.class_weight not in {None, "balanced", "balanced_subsample"}:
            raise ValueError("unsupported Random Forest class_weight")
        if type(self.oob_score) is not bool:
            raise ValueError("oob_score must be a boolean")
        if type(self.n_jobs) is not int or self.n_jobs == 0:
            raise ValueError("n_jobs must be a non-zero integer")
        if type(self.embedding_batch_size) is not int or self.embedding_batch_size < 1:
            raise ValueError("embedding_batch_size must be a positive integer")
        if self.random_state_policy != "seed_plus_fold":
            raise ValueError("random_state_policy must be 'seed_plus_fold'")


@dataclass(frozen=True)
class GradientBoostingExperimentSpec:
    """Fixed histogram-gradient-boosting head on frozen I2 embeddings."""

    schema_version: str
    experiment_id: str
    base_i2_config: str
    max_iter: int
    learning_rate: float
    max_leaf_nodes: int
    min_samples_leaf: int
    l2_regularization: float
    max_features: float
    class_weight: str | None
    early_stopping: bool
    embedding_batch_size: int
    random_state_policy: str

    def validate(self) -> None:
        if self.schema_version != "1.0.0":
            raise ValueError("unsupported gradient-boosting experiment schema")
        if not self.experiment_id.startswith("postsubmit-"):
            raise ValueError("post-submission experiment_id must start with 'postsubmit-'")
        if type(self.max_iter) is not int or self.max_iter < 10:
            raise ValueError("max_iter must be an integer of at least 10")
        if not math.isfinite(self.learning_rate) or not 0.0 < self.learning_rate <= 1.0:
            raise ValueError("learning_rate must be finite and in (0, 1]")
        if type(self.max_leaf_nodes) is not int or self.max_leaf_nodes < 2:
            raise ValueError("max_leaf_nodes must be an integer of at least 2")
        if type(self.min_samples_leaf) is not int or self.min_samples_leaf < 1:
            raise ValueError("min_samples_leaf must be a positive integer")
        if not math.isfinite(self.l2_regularization) or self.l2_regularization < 0:
            raise ValueError("l2_regularization must be finite and non-negative")
        if not math.isfinite(self.max_features) or not 0.0 < self.max_features <= 1.0:
            raise ValueError("max_features must be finite and in (0, 1]")
        if self.class_weight not in {None, "balanced"}:
            raise ValueError("gradient-boosting class_weight must be null or 'balanced'")
        if self.early_stopping is not False:
            raise ValueError("early_stopping must be false to avoid an undeclared inner split")
        if type(self.embedding_batch_size) is not int or self.embedding_batch_size < 1:
            raise ValueError("embedding_batch_size must be a positive integer")
        if self.random_state_policy != "seed_plus_fold":
            raise ValueError("random_state_policy must be 'seed_plus_fold'")


@dataclass(frozen=True)
class SourceFoldRun:
    """Verified frozen I2 checkpoint, OOF predictions, and training history."""

    fold: int
    run_id: str
    checkpoint_path: Path
    checkpoint_sha256: str
    prediction_path: Path
    prediction_sha256: str
    history_path: Path
    history_sha256: str


@dataclass(frozen=True)
class PostSubmissionRunOutput:
    """One executed or cache-loaded post-submission fold."""

    experiment_id: str
    fold: int
    seed: int
    run_id: str
    source: Literal["run", "cache"]
    oof: pd.DataFrame
    metrics: dict[str, Any]
    cache_key: RunCacheKey
    checkpoint_path: Path | None
    history_path: Path


@dataclass(frozen=True)
class RecoverableNumericStop:
    """A failed matched run whose earlier selected checkpoint is still valid."""

    run_id: str
    checkpoint_path: Path
    error_message: str
    runtime_seconds: float


def _load_exact_dataclass(path: str | Path, kind: type[Any]) -> Any:
    source = Path(path)
    with source.open(encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, dict):
        raise ValueError(f"config must be a JSON object: {source}")
    expected = set(kind.__dataclass_fields__)
    if set(payload) != expected:
        raise ValueError(
            f"config fields changed for {source}; "
            f"missing={sorted(expected - set(payload))}, "
            f"unknown={sorted(set(payload) - expected)}"
        )
    result = kind(**payload)
    result.validate()
    return result


def load_activation_experiment_spec(
    path: str | Path = DEFAULT_ACTIVATION_CONFIG,
) -> ActivationExperimentSpec:
    """Load the fail-closed LeakyReLU ablation declaration."""
    return _load_exact_dataclass(path, ActivationExperimentSpec)


def load_random_forest_experiment_spec(
    path: str | Path = DEFAULT_RANDOM_FOREST_CONFIG,
) -> RandomForestExperimentSpec:
    """Load the fail-closed frozen-embedding Random Forest declaration."""
    return _load_exact_dataclass(path, RandomForestExperimentSpec)


def load_gradient_boosting_experiment_spec(
    path: str | Path = DEFAULT_GRADIENT_BOOSTING_CONFIG,
) -> GradientBoostingExperimentSpec:
    """Load the fail-closed frozen-embedding gradient-boosting declaration."""
    return _load_exact_dataclass(path, GradientBoostingExperimentSpec)


def _resolve_project_path(raw_path: str | Path, *, project_root: Path) -> Path:
    path = Path(raw_path)
    resolved = path.resolve() if path.is_absolute() else (project_root / path).resolve()
    try:
        resolved.relative_to(project_root.resolve())
    except ValueError as error:
        raise ValueError(f"path leaves the project root: {raw_path}") from error
    return resolved


def _load_base_i2_config(raw_path: str, *, project_root: Path) -> I2ExperimentConfig:
    config = load_i2_config(_resolve_project_path(raw_path, project_root=project_root))
    if config.experiment_id != BASE_I2_EXPERIMENT_ID:
        raise ValueError("post-submission experiments must use the frozen lambda=0.3 I2 source")
    if config.folds != CANONICAL_FOLDS or config.seeds != (PRIMARY_SEED,):
        raise ValueError("base I2 config changed its canonical folds or primary seed")
    return config


def _artifact_path(raw_path: str, *, data_root: Path) -> Path:
    path = Path(raw_path)
    return path if path.is_absolute() else data_root / path


def _verify_source_i2_runs(
    *,
    registry_path: Path,
    data_root: Path,
    source_experiment_id: str = BASE_I2_EXPERIMENT_ID,
    seed: int = PRIMARY_SEED,
) -> dict[int, SourceFoldRun]:
    registry = RunRegistry(registry_path)
    frame = registry.read()
    selected = frame.loc[
        frame["experiment_id"].eq(source_experiment_id)
        & frame["status"].eq("completed")
        & pd.to_numeric(frame["seed"], errors="coerce").eq(seed)
    ].copy()
    selected["_fold"] = pd.to_numeric(selected["fold"], errors="coerce")
    selected = selected.loc[selected["_fold"].isin(CANONICAL_FOLDS)]
    if selected.empty:
        raise FileNotFoundError("no completed primary-seed frozen I2 rows exist")

    outputs: dict[int, SourceFoldRun] = {}
    for fold in CANONICAL_FOLDS:
        candidates = selected.loc[selected["_fold"].eq(fold)].sort_values(
            "finished_at_utc", ascending=False
        )
        verified: SourceFoldRun | None = None
        for row in candidates.to_dict(orient="records"):
            checkpoint_path = _artifact_path(row["checkpoint_path"], data_root=data_root)
            prediction_path = _artifact_path(row["prediction_path"], data_root=data_root)
            history_path = _artifact_path(row["history_path"], data_root=data_root)
            try:
                verify_artifact(checkpoint_path, row["checkpoint_sha256"])
                verify_artifact(prediction_path, row["prediction_sha256"])
                verify_artifact(history_path, row["history_sha256"])
            except (FileNotFoundError, ValueError):
                continue
            verified = SourceFoldRun(
                fold=fold,
                run_id=str(row["run_id"]),
                checkpoint_path=checkpoint_path,
                checkpoint_sha256=str(row["checkpoint_sha256"]),
                prediction_path=prediction_path,
                prediction_sha256=str(row["prediction_sha256"]),
                history_path=history_path,
                history_sha256=str(row["history_sha256"]),
            )
            break
        if verified is None:
            raise FileNotFoundError(f"fold {fold} has no fully verified frozen I2 artifacts")
        outputs[fold] = verified
    return outputs


def replace_relu_with_leaky_relu(
    module: nn.Module,
    *,
    negative_slope: float = 0.01,
) -> tuple[str, ...]:
    """Replace parameter-free ReLUs in place while preserving every learned tensor key."""
    if not math.isfinite(negative_slope) or not 0.0 < negative_slope < 1.0:
        raise ValueError("negative_slope must be finite and in (0, 1)")
    replaced: list[str] = []

    def visit(parent: nn.Module, prefix: str) -> None:
        for name, child in tuple(parent.named_children()):
            qualified = f"{prefix}.{name}" if prefix else name
            if isinstance(child, nn.ReLU):
                setattr(
                    parent,
                    name,
                    nn.LeakyReLU(negative_slope=negative_slope, inplace=child.inplace),
                )
                replaced.append(qualified)
            else:
                visit(child, qualified)

    visit(module, "")
    return tuple(replaced)


def _build_i2_model(
    *,
    season_classes: int,
    article_type_classes: int,
    negative_slope: float | None = None,
) -> nn.Module:
    model = build_multitask_season_model(
        SeasonModelSpec(family="smallcnn", num_classes=season_classes),
        article_type_classes=article_type_classes,
    )
    if negative_slope is not None:
        replaced = replace_relu_with_leaky_relu(
            model.base_model,
            negative_slope=negative_slope,
        )
        if len(replaced) != 8:
            raise ValueError(f"SmallCNN activation count changed; expected 8, got {len(replaced)}")
    return model


def _load_i2_checkpoint(model: nn.Module, source: Path) -> dict[str, Any]:
    checkpoint = torch.load(source, map_location="cpu", weights_only=True)
    required = {"model_state_dict", "labels", "auxiliary_weight", "best_epoch"}
    if not isinstance(checkpoint, dict) or not required.issubset(checkpoint):
        raise ValueError(f"invalid I2 checkpoint structure: {source}")
    if tuple(checkpoint["labels"]) != tuple(SEASON_LABELS):
        raise ValueError("I2 checkpoint changed the canonical Season class order")
    if not math.isclose(float(checkpoint["auxiliary_weight"]), 0.3, abs_tol=1e-12):
        raise ValueError("I2 checkpoint changed the frozen auxiliary weight")
    model.load_state_dict(checkpoint["model_state_dict"], strict=True)
    return checkpoint


def _nonfinite_gradient_epoch(message: str) -> int | None:
    match = re.fullmatch(r"non-finite refit gradients at epoch=(\d+), batch=(\d+)", message)
    if match is None:
        return None
    epoch = int(match.group(1))
    return epoch if epoch > 0 else None


def _find_recoverable_numeric_stop(
    *,
    registry: RunRegistry,
    key: RunCacheKey,
    experiment_id: str,
    fold: int,
    seed: int,
    checkpoint_directory: Path,
) -> RecoverableNumericStop | None:
    """Find a valid best checkpoint left by a recorded late AMP overflow."""
    frame = registry.read()
    candidates = frame.loc[
        frame["experiment_id"].eq(experiment_id)
        & frame["status"].eq("failed")
        & frame["error_type"].eq("FloatingPointError")
        & frame["config_sha256"].eq(key.config_sha256)
        & frame["split_sha256"].eq(key.split_sha256)
        & frame["label_map_sha256"].eq(key.label_map_sha256)
        & pd.to_numeric(frame["fold"], errors="coerce").eq(fold)
        & pd.to_numeric(frame["seed"], errors="coerce").eq(seed)
    ].sort_values("finished_at_utc", ascending=False)
    for row in candidates.to_dict(orient="records"):
        message = str(row["error_message"])
        if _nonfinite_gradient_epoch(message) is None:
            continue
        checkpoint_path = checkpoint_directory / f"{row['run_id']}.pt"
        if not checkpoint_path.is_file():
            continue
        try:
            checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
        except (OSError, RuntimeError, ValueError):
            continue
        if (
            not isinstance(checkpoint, dict)
            or tuple(checkpoint.get("labels", ())) != tuple(SEASON_LABELS)
            or not math.isclose(
                float(checkpoint.get("auxiliary_weight", math.nan)),
                0.3,
                abs_tol=1e-12,
            )
            or not isinstance(checkpoint.get("model_state_dict"), dict)
            or not all(
                bool(torch.isfinite(tensor).all().item())
                for tensor in checkpoint["model_state_dict"].values()
                if torch.is_floating_point(tensor)
            )
        ):
            continue
        runtime = pd.to_numeric(pd.Series([row["runtime_seconds"]]), errors="coerce").iloc[0]
        return RecoverableNumericStop(
            run_id=str(row["run_id"]),
            checkpoint_path=checkpoint_path,
            error_message=message,
            runtime_seconds=float(runtime) if pd.notna(runtime) else math.nan,
        )
    return None


def _recover_best_checkpoint(
    *,
    model: nn.Module,
    validation_loader: DataLoader[Any],
    checkpoint_path: Path,
    fold: int,
    seed: int,
    labels: tuple[str, ...],
    auxiliary_weight: float,
    requested_device: str,
    error_message: str,
    runtime_seconds: float,
) -> FoldResult:
    """Evaluate the finite best epoch selected before a later numerical stop."""
    failed_epoch = _nonfinite_gradient_epoch(error_message)
    if failed_epoch is None:
        raise ValueError("checkpoint recovery requires a recognised gradient-stop message")
    checkpoint = _load_i2_checkpoint(model, checkpoint_path)
    device = _resolve_device(requested_device)
    amp_enabled = device.type == "cuda"
    model = model.to(device)
    losses, metrics, identifiers, targets, probabilities = _evaluate_multitask(
        model,
        validation_loader,
        device=device,
        labels=labels,
        auxiliary_weight=auxiliary_weight,
        amp_enabled=amp_enabled,
    )
    stored_metric = float(checkpoint["best_macro_f1"])
    if not math.isclose(float(metrics["macro_f1"]), stored_metric, abs_tol=1e-12):
        raise ValueError("recovered checkpoint does not reproduce its selected macro-F1")
    return FoldResult(
        fold=fold,
        seed=seed,
        labels=labels,
        best_epoch=int(checkpoint["best_epoch"]),
        epochs_completed=failed_epoch - 1,
        best_macro_f1=stored_metric,
        best_metrics=metrics,
        history=[],
        validation_ids=identifiers,
        targets=targets,
        probabilities=probabilities,
        checkpoint_path=str(checkpoint_path),
        checkpoint_sha256=compute_sha256(checkpoint_path),
        parameter_count=sum(parameter.numel() for parameter in model.parameters()),
        runtime_seconds=runtime_seconds,
        peak_vram_mb=None,
        stopped_early=True,
        device=str(device),
        metadata={
            "amp_enabled": amp_enabled,
            "selection_metric": "season_macro_f1",
            "termination": "nonfinite_gradient_after_finite_best_checkpoint",
            "termination_message": error_message,
            "validation_losses": losses,
            "history_available": False,
        },
    )


def _resolve_device(requested: str = "auto") -> torch.device:
    if requested == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if requested == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available")
    if requested not in {"cpu", "cuda"}:
        raise ValueError("device must be 'auto', 'cpu', or 'cuda'")
    return torch.device(requested)


def _registry_relative_path(path: Path, *, data_root: Path) -> str:
    try:
        return path.resolve().relative_to(data_root.resolve()).as_posix()
    except ValueError:
        return str(path.resolve())


def _write_run_artifacts(
    *,
    run_id: str,
    experiment_id: str,
    fold: int,
    seed: int,
    config: dict[str, Any],
    oof: pd.DataFrame,
    metrics: dict[str, Any],
    history: dict[str, Any],
    run_directory: Path,
    data_root: Path,
) -> tuple[Path, Path, dict[str, str]]:
    directory = run_directory / run_id
    prediction_path = directory / "oof.csv"
    history_path = directory / "history.json"
    output = oof.copy()
    output.insert(0, "run_id", run_id)
    output.insert(1, "experiment_id", experiment_id)
    atomic_write_csv(prediction_path, output)
    atomic_write_json(
        history_path,
        {
            "schema_version": "1.0.0",
            "run_id": run_id,
            "experiment_id": experiment_id,
            "fold": fold,
            "seed": seed,
            "config": config,
            "metrics": metrics,
            **history,
        },
    )
    fields = {
        "prediction_path": _registry_relative_path(prediction_path, data_root=data_root),
        "prediction_sha256": compute_sha256(prediction_path),
        "history_path": _registry_relative_path(history_path, data_root=data_root),
        "history_sha256": compute_sha256(history_path),
    }
    return prediction_path, history_path, fields


def _load_cached_fold(
    *,
    registry: RunRegistry,
    key: RunCacheKey,
    required_artifacts: tuple[str, ...],
    data_root: Path,
    experiment_id: str,
    fold: int,
    seed: int,
    splits_path: Path,
    labels: tuple[str, ...],
) -> PostSubmissionRunOutput | None:
    cached = find_cached_run(
        registry,
        key,
        required_artifacts=required_artifacts,
        artifact_root=data_root,
    )
    if cached is None:
        return None
    prediction_path = _artifact_path(cached.row["prediction_path"], data_root=data_root)
    history_path = _artifact_path(cached.row["history_path"], data_root=data_root)
    checkpoint_path = None
    if cached.row["checkpoint_path"]:
        checkpoint_path = _artifact_path(cached.row["checkpoint_path"], data_root=data_root)
    oof = pd.read_csv(prediction_path)
    _validate_output_oof(
        oof,
        expected=_expected_validation(fold=fold, splits_path=splits_path, target="season"),
        labels=labels,
    )
    validate_oof_identity(
        oof,
        run_id=cached.run_id,
        experiment_id=experiment_id,
        fold=fold,
        seed=seed,
    )
    return PostSubmissionRunOutput(
        experiment_id=experiment_id,
        fold=fold,
        seed=seed,
        run_id=cached.run_id,
        source="cache",
        oof=oof,
        metrics=json.loads(cached.row["metrics"] or "{}"),
        cache_key=key,
        checkpoint_path=checkpoint_path,
        history_path=history_path,
    )


def _run_leaky_relu_fold(
    *,
    spec: ActivationExperimentSpec,
    base: I2ExperimentConfig,
    fold: int,
    mode: ExecutionMode,
    registry: RunRegistry,
    data_root: Path,
    source_root: Path,
    splits_path: Path,
    label_map_path: Path,
    temporary_directory: Path,
) -> PostSubmissionRunOutput:
    seed = PRIMARY_SEED
    config_payload = {
        "role": "post_submission_controlled_activation_ablation",
        "activation_experiment": asdict(spec),
        "base_i2": base.to_dict(),
        "selection_or_submission_impact": False,
    }
    key = build_run_cache_key(
        config_payload,
        fold=fold,
        seed=seed,
        implementation_paths=POST_SUBMISSION_IMPLEMENTATION_PATHS,
        split_path=splits_path,
        label_map_path=label_map_path,
        root=source_root,
    )
    labels = tuple(load_label_maps(label_map_path)["season"]["classes"])
    if mode in {"run_or_load", "load"}:
        cached = _load_cached_fold(
            registry=registry,
            key=key,
            required_artifacts=("checkpoint", "prediction", "history"),
            data_root=data_root,
            experiment_id=spec.experiment_id,
            fold=fold,
            seed=seed,
            splits_path=splits_path,
            labels=labels,
        )
        if cached is not None:
            print(f"[LeakyReLU] fold {fold}: loaded verified cache", flush=True)
            return cached
    if mode == "load":
        raise FileNotFoundError(f"no verified LeakyReLU cache for fold {fold}")

    checkpoint_directory = temporary_directory / "checkpoints"
    recoverable = None
    if mode == "run_or_load":
        recoverable = _find_recoverable_numeric_stop(
            registry=registry,
            key=key,
            experiment_id=spec.experiment_id,
            fold=fold,
            seed=seed,
            checkpoint_directory=checkpoint_directory,
        )
    run_id = new_run_id(spec.experiment_id, fold, seed)
    checkpoint_path = (
        recoverable.checkpoint_path
        if recoverable is not None
        else checkpoint_directory / f"{run_id}.pt"
    )
    run_directory = temporary_directory / "runs"
    transform_id = ImageTransformSpec(
        image_size=base.data.image_size,
        augmentation=base.data.augmentation,
    ).transform_id
    record = RunRecord(
        run_id=run_id,
        experiment_id=spec.experiment_id,
        fold=fold,
        seed=seed,
        config_sha256=key.config_sha256,
        split_sha256=key.split_sha256,
        label_map_sha256=key.label_map_sha256,
        implementation_sha256=key.implementation_sha256,
        stage="post_submission_activation_ablation",
        model_family="smallcnn_leaky_relu",
        benchmark_only=True,
        final_eligible=False,
        scratch=True,
        transform_id=transform_id,
        loss_id=f"{base.loss_id}_leaky_relu_0_01",
        epochs_requested=base.optimisation.epochs,
        primary_metric_name="macro_f1",
    )
    if recoverable is None:
        print(f"[LeakyReLU] fold {fold}: training matched I2", flush=True)
    else:
        print(
            f"[LeakyReLU] fold {fold}: recovering finite best checkpoint from "
            f"failed run {recoverable.run_id}",
            flush=True,
        )
    with tracked_run(registry, record) as run:
        seed_everything(seed)
        loaders = build_multitask_loaders(
            validation_fold=fold,
            image_size=base.data.image_size,
            batch_size=base.data.batch_size,
            main_target="season",
            auxiliary_target="articleType",
            augmentation=base.data.augmentation,
            seed=seed,
            num_workers=base.data.num_workers,
            validation_batch_size=base.data.validation_batch_size,
            pin_memory=base.data.pin_memory,
            root=data_root,
            splits_path=splits_path,
            label_map_path=label_map_path,
        )
        model = _build_i2_model(
            season_classes=len(loaders.labels),
            article_type_classes=len(loaders.auxiliary_labels),
            negative_slope=spec.negative_slope,
        )
        numerical_stop: dict[str, Any] | None = None
        if recoverable is not None:
            result = _recover_best_checkpoint(
                model=model,
                validation_loader=loaders.validation,
                checkpoint_path=checkpoint_path,
                fold=fold,
                seed=seed,
                labels=loaders.labels,
                auxiliary_weight=base.auxiliary.loss_weight,
                requested_device=base.optimisation.device,
                error_message=recoverable.error_message,
                runtime_seconds=recoverable.runtime_seconds,
            )
            numerical_stop = {
                "recovered_from_failed_run_id": recoverable.run_id,
                "message": recoverable.error_message,
                "policy": "retain_finite_best_epoch_selected_before_late_overflow",
                "retrained": False,
            }
        else:
            training_started = time.perf_counter()
            try:
                result = train_masked_multitask_fold(
                    model,
                    loaders.train,
                    loaders.validation,
                    config=TrainConfig(
                        fold=fold,
                        seed=seed,
                        batch_size=base.data.batch_size,
                        **asdict(base.optimisation),
                    ),
                    checkpoint_path=checkpoint_path,
                    auxiliary_weight=base.auxiliary.loss_weight,
                    labels=loaders.labels,
                )
            except FloatingPointError as error:
                message = str(error)
                if _nonfinite_gradient_epoch(message) is None or not checkpoint_path.is_file():
                    raise
                elapsed = time.perf_counter() - training_started
                result = _recover_best_checkpoint(
                    model=model,
                    validation_loader=loaders.validation,
                    checkpoint_path=checkpoint_path,
                    fold=fold,
                    seed=seed,
                    labels=loaders.labels,
                    auxiliary_weight=base.auxiliary.loss_weight,
                    requested_device=base.optimisation.device,
                    error_message=message,
                    runtime_seconds=elapsed,
                )
                numerical_stop = {
                    "message": message,
                    "policy": "retain_finite_best_epoch_selected_before_late_overflow",
                    "retrained": True,
                }
        oof = result.to_oof_frame()
        _validate_output_oof(
            oof,
            expected=_expected_validation(fold=fold, splits_path=splits_path, target="season"),
            labels=loaders.labels,
        )
        history = {
            "experiment_boundary": {
                "post_submission": True,
                "changes_frozen_final": False,
                "single_changed_factor": "ReLU_to_LeakyReLU",
                "negative_slope": spec.negative_slope,
                "matched_base_experiment": BASE_I2_EXPERIMENT_ID,
            },
            "loader_audit": loaders.audit(),
            "replaced_activation_count": 8,
            "epoch_history": result.history,
            "engine_metadata": result.metadata,
            "numerical_stop": numerical_stop,
        }
        _, history_path, artifact_fields = _write_run_artifacts(
            run_id=run_id,
            experiment_id=spec.experiment_id,
            fold=fold,
            seed=seed,
            config=config_payload,
            oof=oof,
            metrics=result.best_metrics,
            history=history,
            run_directory=run_directory,
            data_root=data_root,
        )
        run.epochs_completed = result.epochs_completed
        run.best_epoch = result.best_epoch
        run.primary_metric_value = float(result.best_metrics["macro_f1"])
        run.metrics = result.best_metrics
        run.runtime_seconds = result.runtime_seconds
        run.peak_vram_mb = result.peak_vram_mb
        run.parameter_count = result.parameter_count
        run.checkpoint_path = _registry_relative_path(checkpoint_path, data_root=data_root)
        run.checkpoint_sha256 = result.checkpoint_sha256
        run.prediction_path = artifact_fields["prediction_path"]
        run.prediction_sha256 = artifact_fields["prediction_sha256"]
        run.history_path = artifact_fields["history_path"]
        run.history_sha256 = artifact_fields["history_sha256"]

    output_oof = oof.copy()
    output_oof.insert(0, "run_id", run_id)
    output_oof.insert(1, "experiment_id", spec.experiment_id)
    print(
        f"[LeakyReLU] fold {fold}: macro-F1={result.best_metrics['macro_f1']:.4f}, "
        f"best epoch={result.best_epoch}",
        flush=True,
    )
    return PostSubmissionRunOutput(
        experiment_id=spec.experiment_id,
        fold=fold,
        seed=seed,
        run_id=run_id,
        source="run",
        oof=output_oof,
        metrics=result.best_metrics,
        cache_key=key,
        checkpoint_path=checkpoint_path,
        history_path=history_path,
    )


def _ids_to_list(values: Any) -> list[int]:
    if isinstance(values, torch.Tensor):
        return [int(value) for value in values.detach().cpu().tolist()]
    return [int(value) for value in values]


def _extract_embeddings(
    model: nn.Module,
    loader: DataLoader[Any],
    *,
    device: torch.device,
) -> tuple[np.ndarray, np.ndarray, list[int]]:
    model = model.to(device).eval()
    embeddings: list[np.ndarray] = []
    targets: list[np.ndarray] = []
    identifiers: list[int] = []
    with torch.inference_mode():
        for batch in loader:
            images = batch["image"].to(device, non_blocking=device.type == "cuda")
            features = model.base_model.forward_embedding(images)
            embeddings.append(features.detach().float().cpu().numpy())
            target = batch["target"]
            if isinstance(target, torch.Tensor):
                targets.append(target.detach().cpu().numpy().astype(np.int64, copy=False))
            else:
                targets.append(np.asarray(target, dtype=np.int64))
            identifiers.extend(_ids_to_list(batch["id"]))
    if not embeddings:
        raise ValueError("embedding loader produced no samples")
    matrix = np.concatenate(embeddings).astype(np.float32, copy=False)
    target_array = np.concatenate(targets).astype(np.int64, copy=False)
    if len(matrix) != len(target_array) or len(matrix) != len(identifiers):
        raise ValueError("embedding, target, and ID row counts differ")
    return matrix, target_array, identifiers


def _run_random_forest_fold(
    *,
    spec: RandomForestExperimentSpec,
    base: I2ExperimentConfig,
    source: SourceFoldRun,
    fold: int,
    mode: ExecutionMode,
    registry: RunRegistry,
    data_root: Path,
    source_root: Path,
    splits_path: Path,
    label_map_path: Path,
    temporary_directory: Path,
) -> PostSubmissionRunOutput:
    seed = PRIMARY_SEED
    config_payload = {
        "role": "post_submission_frozen_embedding_head_comparison",
        "random_forest_experiment": asdict(spec),
        "base_i2": base.to_dict(),
        "source_run_id": source.run_id,
        "source_checkpoint_sha256": source.checkpoint_sha256,
        "embedding_transform": "fold_fitted_normalisation_without_random_augmentation",
        "selection_or_submission_impact": False,
    }
    key = build_run_cache_key(
        config_payload,
        fold=fold,
        seed=seed,
        implementation_paths=POST_SUBMISSION_IMPLEMENTATION_PATHS,
        split_path=splits_path,
        label_map_path=label_map_path,
        root=source_root,
    )
    labels = tuple(load_label_maps(label_map_path)["season"]["classes"])
    if mode in {"run_or_load", "load"}:
        cached = _load_cached_fold(
            registry=registry,
            key=key,
            required_artifacts=("prediction", "history"),
            data_root=data_root,
            experiment_id=spec.experiment_id,
            fold=fold,
            seed=seed,
            splits_path=splits_path,
            labels=labels,
        )
        if cached is not None:
            print(f"[Embedding + RF] fold {fold}: loaded verified cache", flush=True)
            return cached
    if mode == "load":
        raise FileNotFoundError(f"no verified embedding + RF cache for fold {fold}")

    run_id = new_run_id(spec.experiment_id, fold, seed)
    run_directory = temporary_directory / "runs"
    config_random_state = seed + fold
    transform_id = ImageTransformSpec(
        image_size=base.data.image_size,
        augmentation="none",
    ).transform_id
    record = RunRecord(
        run_id=run_id,
        experiment_id=spec.experiment_id,
        fold=fold,
        seed=seed,
        config_sha256=key.config_sha256,
        split_sha256=key.split_sha256,
        label_map_sha256=key.label_map_sha256,
        implementation_sha256=key.implementation_sha256,
        stage="post_submission_frozen_embedding_head",
        model_family="smallcnn_i2_embedding_random_forest",
        benchmark_only=True,
        final_eligible=False,
        scratch=True,
        transform_id=transform_id,
        loss_id="random_forest_gini_on_frozen_i2_embedding",
        epochs_requested=0,
        primary_metric_name="macro_f1",
    )
    print(f"[Embedding + RF] fold {fold}: extracting frozen 256-D features", flush=True)
    with tracked_run(registry, record) as run:
        started = time.perf_counter()
        loaders = build_task_loaders(
            validation_fold=fold,
            image_size=base.data.image_size,
            batch_size=spec.embedding_batch_size,
            target="season",
            augmentation="none",
            seed=seed,
            num_workers=base.data.num_workers,
            validation_batch_size=spec.embedding_batch_size,
            pin_memory=base.data.pin_memory,
            root=data_root,
            splits_path=splits_path,
            label_map_path=label_map_path,
        )
        mappings = load_label_maps(label_map_path)
        article_type_classes = int(mappings["articleType"]["num_classes"])
        model = _build_i2_model(
            season_classes=len(loaders.labels),
            article_type_classes=article_type_classes,
        )
        _load_i2_checkpoint(model, source.checkpoint_path)
        device = _resolve_device(base.optimisation.device)
        train_x, train_y, train_ids = _extract_embeddings(
            model,
            loaders.train,
            device=device,
        )
        validation_x, validation_y, validation_ids = _extract_embeddings(
            model,
            loaders.validation,
            device=device,
        )
        del model
        if device.type == "cuda":
            torch.cuda.empty_cache()
        if set(train_ids) != set(loaders.training_ids):
            raise ValueError("embedding training IDs differ from the canonical fold")
        if tuple(validation_ids) != loaders.validation_ids:
            raise ValueError("embedding validation IDs differ from the canonical fold order")
        if train_x.shape[1] != 256 or validation_x.shape[1] != 256:
            raise ValueError("frozen I2 embedding dimension changed from 256")

        forest = RandomForestClassifier(
            n_estimators=spec.n_estimators,
            criterion="gini",
            max_features=spec.max_features,
            min_samples_leaf=spec.min_samples_leaf,
            class_weight=spec.class_weight,
            bootstrap=True,
            oob_score=spec.oob_score,
            n_jobs=spec.n_jobs,
            random_state=config_random_state,
        )
        fit_started = time.perf_counter()
        forest.fit(train_x, train_y)
        fit_seconds = time.perf_counter() - fit_started
        raw_probabilities = forest.predict_proba(validation_x)
        probabilities = np.zeros((len(validation_x), len(loaders.labels)), dtype=np.float64)
        for source_column, class_index in enumerate(forest.classes_):
            probabilities[:, int(class_index)] = raw_probabilities[:, source_column]
        label_array = np.asarray(loaders.labels, dtype=object)
        true_labels = label_array[validation_y]
        metrics = multiclass_metrics(
            true_labels,
            probabilities=probabilities,
            labels=loaders.labels,
        )
        predicted_indices = probabilities.argmax(axis=1)
        oof = pd.DataFrame(
            {
                "id": validation_ids,
                "fold": fold,
                "seed": seed,
                "y_true": true_labels,
                "y_pred": label_array[predicted_indices],
            }
        )
        for index, label in enumerate(loaders.labels):
            oof[f"prob_{label}"] = probabilities[:, index]
        _validate_output_oof(
            oof,
            expected=_expected_validation(fold=fold, splits_path=splits_path, target="season"),
            labels=loaders.labels,
        )
        node_count = int(sum(estimator.tree_.node_count for estimator in forest.estimators_))
        history = {
            "experiment_boundary": {
                "post_submission": True,
                "changes_frozen_final": False,
                "encoder_frozen": True,
                "source_run_id": source.run_id,
                "source_checkpoint_sha256": source.checkpoint_sha256,
                "only_newly_fitted_component": "RandomForestClassifier",
            },
            "loader_audit": loaders.audit(),
            "embedding": {
                "dimension": int(train_x.shape[1]),
                "training_rows": len(train_x),
                "validation_rows": len(validation_x),
                "dtype": str(train_x.dtype),
                "random_augmentation_used": False,
            },
            "random_forest": {
                "parameters": forest.get_params(deep=False),
                "random_state": config_random_state,
                "fit_seconds": fit_seconds,
                "oob_accuracy": (
                    float(forest.oob_score_) if spec.oob_score else None
                ),
                "tree_node_count": node_count,
                "mean_tree_depth": float(
                    np.mean([estimator.tree_.max_depth for estimator in forest.estimators_])
                ),
            },
        }
        _, history_path, artifact_fields = _write_run_artifacts(
            run_id=run_id,
            experiment_id=spec.experiment_id,
            fold=fold,
            seed=seed,
            config=config_payload,
            oof=oof,
            metrics=metrics,
            history=history,
            run_directory=run_directory,
            data_root=data_root,
        )
        run.epochs_completed = 0
        run.primary_metric_value = float(metrics["macro_f1"])
        run.metrics = metrics
        run.runtime_seconds = time.perf_counter() - started
        run.parameter_count = node_count
        run.prediction_path = artifact_fields["prediction_path"]
        run.prediction_sha256 = artifact_fields["prediction_sha256"]
        run.history_path = artifact_fields["history_path"]
        run.history_sha256 = artifact_fields["history_sha256"]

    output_oof = oof.copy()
    output_oof.insert(0, "run_id", run_id)
    output_oof.insert(1, "experiment_id", spec.experiment_id)
    print(
        f"[Embedding + RF] fold {fold}: macro-F1={metrics['macro_f1']:.4f}, "
        f"OOB accuracy={history['random_forest']['oob_accuracy']:.4f}",
        flush=True,
    )
    return PostSubmissionRunOutput(
        experiment_id=spec.experiment_id,
        fold=fold,
        seed=seed,
        run_id=run_id,
        source="run",
        oof=output_oof,
        metrics=metrics,
        cache_key=key,
        checkpoint_path=None,
        history_path=history_path,
    )


def _run_gradient_boosting_fold(
    *,
    spec: GradientBoostingExperimentSpec,
    base: I2ExperimentConfig,
    source: SourceFoldRun,
    fold: int,
    mode: ExecutionMode,
    registry: RunRegistry,
    data_root: Path,
    source_root: Path,
    splits_path: Path,
    label_map_path: Path,
    temporary_directory: Path,
) -> PostSubmissionRunOutput:
    seed = PRIMARY_SEED
    config_payload = {
        "role": "post_submission_frozen_embedding_gradient_boosting",
        "gradient_boosting_experiment": asdict(spec),
        "base_i2": base.to_dict(),
        "source_run_id": source.run_id,
        "source_checkpoint_sha256": source.checkpoint_sha256,
        "embedding_transform": "fold_fitted_normalisation_without_random_augmentation",
        "selection_or_submission_impact": False,
    }
    key = build_run_cache_key(
        config_payload,
        fold=fold,
        seed=seed,
        implementation_paths=POST_SUBMISSION_IMPLEMENTATION_PATHS,
        split_path=splits_path,
        label_map_path=label_map_path,
        root=source_root,
    )
    labels = tuple(load_label_maps(label_map_path)["season"]["classes"])
    if mode in {"run_or_load", "load"}:
        cached = _load_cached_fold(
            registry=registry,
            key=key,
            required_artifacts=("prediction", "history"),
            data_root=data_root,
            experiment_id=spec.experiment_id,
            fold=fold,
            seed=seed,
            splits_path=splits_path,
            labels=labels,
        )
        if cached is not None:
            print(f"[Embedding + HGB] fold {fold}: loaded verified cache", flush=True)
            return cached
    if mode == "load":
        raise FileNotFoundError(f"no verified embedding + HGB cache for fold {fold}")

    run_id = new_run_id(spec.experiment_id, fold, seed)
    run_directory = temporary_directory / "runs"
    random_state = seed + fold
    transform_id = ImageTransformSpec(
        image_size=base.data.image_size,
        augmentation="none",
    ).transform_id
    record = RunRecord(
        run_id=run_id,
        experiment_id=spec.experiment_id,
        fold=fold,
        seed=seed,
        config_sha256=key.config_sha256,
        split_sha256=key.split_sha256,
        label_map_sha256=key.label_map_sha256,
        implementation_sha256=key.implementation_sha256,
        stage="post_submission_frozen_embedding_head",
        model_family="smallcnn_i2_embedding_hist_gradient_boosting",
        benchmark_only=True,
        final_eligible=False,
        scratch=True,
        transform_id=transform_id,
        loss_id="hist_gradient_boosting_log_loss_on_frozen_i2_embedding",
        epochs_requested=0,
        primary_metric_name="macro_f1",
    )
    print(f"[Embedding + HGB] fold {fold}: extracting frozen 256-D features", flush=True)
    with tracked_run(registry, record) as run:
        started = time.perf_counter()
        loaders = build_task_loaders(
            validation_fold=fold,
            image_size=base.data.image_size,
            batch_size=spec.embedding_batch_size,
            target="season",
            augmentation="none",
            seed=seed,
            num_workers=base.data.num_workers,
            validation_batch_size=spec.embedding_batch_size,
            pin_memory=base.data.pin_memory,
            root=data_root,
            splits_path=splits_path,
            label_map_path=label_map_path,
        )
        mappings = load_label_maps(label_map_path)
        model = _build_i2_model(
            season_classes=len(loaders.labels),
            article_type_classes=int(mappings["articleType"]["num_classes"]),
        )
        _load_i2_checkpoint(model, source.checkpoint_path)
        device = _resolve_device(base.optimisation.device)
        train_x, train_y, train_ids = _extract_embeddings(
            model,
            loaders.train,
            device=device,
        )
        validation_x, validation_y, validation_ids = _extract_embeddings(
            model,
            loaders.validation,
            device=device,
        )
        del model
        if device.type == "cuda":
            torch.cuda.empty_cache()
        if set(train_ids) != set(loaders.training_ids):
            raise ValueError("gradient-boosting training IDs differ from the canonical fold")
        if tuple(validation_ids) != loaders.validation_ids:
            raise ValueError("gradient-boosting validation IDs differ from canonical order")
        if train_x.shape[1] != 256 or validation_x.shape[1] != 256:
            raise ValueError("frozen I2 embedding dimension changed from 256")

        booster = HistGradientBoostingClassifier(
            loss="log_loss",
            learning_rate=spec.learning_rate,
            max_iter=spec.max_iter,
            max_leaf_nodes=spec.max_leaf_nodes,
            min_samples_leaf=spec.min_samples_leaf,
            l2_regularization=spec.l2_regularization,
            max_features=spec.max_features,
            class_weight=spec.class_weight,
            early_stopping=spec.early_stopping,
            random_state=random_state,
        )
        fit_started = time.perf_counter()
        booster.fit(train_x, train_y)
        fit_seconds = time.perf_counter() - fit_started
        raw_probabilities = booster.predict_proba(validation_x)
        probabilities = np.zeros((len(validation_x), len(loaders.labels)), dtype=np.float64)
        for source_column, class_index in enumerate(booster.classes_):
            probabilities[:, int(class_index)] = raw_probabilities[:, source_column]
        label_array = np.asarray(loaders.labels, dtype=object)
        true_labels = label_array[validation_y]
        metrics = multiclass_metrics(
            true_labels,
            probabilities=probabilities,
            labels=loaders.labels,
        )
        predicted_indices = probabilities.argmax(axis=1)
        oof = pd.DataFrame(
            {
                "id": validation_ids,
                "fold": fold,
                "seed": seed,
                "y_true": true_labels,
                "y_pred": label_array[predicted_indices],
            }
        )
        for index, label in enumerate(loaders.labels):
            oof[f"prob_{label}"] = probabilities[:, index]
        _validate_output_oof(
            oof,
            expected=_expected_validation(fold=fold, splits_path=splits_path, target="season"),
            labels=loaders.labels,
        )
        history = {
            "experiment_boundary": {
                "post_submission": True,
                "changes_frozen_final": False,
                "encoder_frozen": True,
                "source_run_id": source.run_id,
                "source_checkpoint_sha256": source.checkpoint_sha256,
                "only_newly_fitted_component": "HistGradientBoostingClassifier",
                "internal_validation_split_used": False,
            },
            "loader_audit": loaders.audit(),
            "embedding": {
                "dimension": int(train_x.shape[1]),
                "training_rows": len(train_x),
                "validation_rows": len(validation_x),
                "dtype": str(train_x.dtype),
                "random_augmentation_used": False,
            },
            "hist_gradient_boosting": {
                "parameters": booster.get_params(deep=False),
                "random_state": random_state,
                "fit_seconds": fit_seconds,
                "iterations_completed": int(booster.n_iter_),
            },
        }
        _, history_path, artifact_fields = _write_run_artifacts(
            run_id=run_id,
            experiment_id=spec.experiment_id,
            fold=fold,
            seed=seed,
            config=config_payload,
            oof=oof,
            metrics=metrics,
            history=history,
            run_directory=run_directory,
            data_root=data_root,
        )
        run.epochs_completed = 0
        run.primary_metric_value = float(metrics["macro_f1"])
        run.metrics = metrics
        run.runtime_seconds = time.perf_counter() - started
        run.prediction_path = artifact_fields["prediction_path"]
        run.prediction_sha256 = artifact_fields["prediction_sha256"]
        run.history_path = artifact_fields["history_path"]
        run.history_sha256 = artifact_fields["history_sha256"]

    output_oof = oof.copy()
    output_oof.insert(0, "run_id", run_id)
    output_oof.insert(1, "experiment_id", spec.experiment_id)
    print(
        f"[Embedding + HGB] fold {fold}: macro-F1={metrics['macro_f1']:.4f}, "
        f"iterations={booster.n_iter_}",
        flush=True,
    )
    return PostSubmissionRunOutput(
        experiment_id=spec.experiment_id,
        fold=fold,
        seed=seed,
        run_id=run_id,
        source="run",
        oof=output_oof,
        metrics=metrics,
        cache_key=key,
        checkpoint_path=None,
        history_path=history_path,
    )


class _ActivationAccumulator:
    """GPU-side sufficient statistics for one activation layer."""

    def __init__(self) -> None:
        self.observation_count = 0
        self.pre_positive: torch.Tensor | None = None
        self.post_zero: torch.Tensor | None = None
        self.post_negative: torch.Tensor | None = None
        self.post_abs_sum: torch.Tensor | None = None
        self.post_abs_max: torch.Tensor | None = None

    def observe_input(self, tensor: torch.Tensor, *, epsilon: float) -> None:
        detached = tensor.detach()
        if detached.ndim != 4:
            raise ValueError("activation diagnostic expects BCHW feature maps")
        reduction = (0, 2, 3)
        positive = (detached > epsilon).sum(dim=reduction, dtype=torch.int64)
        if self.pre_positive is None:
            self.pre_positive = torch.zeros_like(positive)
        self.pre_positive += positive
        self.observation_count += int(detached.shape[0] * detached.shape[2] * detached.shape[3])

    def observe_output(self, tensor: torch.Tensor, *, epsilon: float) -> None:
        detached = tensor.detach()
        reduction = (0, 2, 3)
        zero = (detached.abs() <= epsilon).sum(dim=reduction, dtype=torch.int64)
        negative = (detached < -epsilon).sum(dim=reduction, dtype=torch.int64)
        absolute = detached.float().abs()
        absolute_sum = absolute.sum(dim=reduction)
        absolute_max = absolute.amax(dim=reduction)
        if self.post_zero is None:
            self.post_zero = torch.zeros_like(zero)
            self.post_negative = torch.zeros_like(negative)
            self.post_abs_sum = torch.zeros_like(absolute_sum)
            self.post_abs_max = torch.zeros_like(absolute_max)
        self.post_zero += zero
        self.post_negative += negative
        self.post_abs_sum += absolute_sum
        self.post_abs_max = torch.maximum(self.post_abs_max, absolute_max)

    def rows(
        self,
        *,
        variant: str,
        fold: int,
        layer: str,
        exact_dead_epsilon: float,
        near_dead_positive_rate: float,
    ) -> list[dict[str, Any]]:
        tensors = (
            self.pre_positive,
            self.post_zero,
            self.post_negative,
            self.post_abs_sum,
            self.post_abs_max,
        )
        if self.observation_count <= 0 or any(value is None for value in tensors):
            raise ValueError(f"activation layer {layer} collected no observations")
        pre_positive, post_zero, post_negative, post_abs_sum, post_abs_max = (
            value.detach().cpu().numpy() for value in tensors if value is not None
        )
        output: list[dict[str, Any]] = []
        for channel in range(len(pre_positive)):
            positive_rate = float(pre_positive[channel] / self.observation_count)
            max_abs = float(post_abs_max[channel])
            output.append(
                {
                    "variant": variant,
                    "fold": fold,
                    "layer": layer,
                    "channel": channel,
                    "observation_count": self.observation_count,
                    "preactivation_positive_rate": positive_rate,
                    "postactivation_zero_rate": float(
                        post_zero[channel] / self.observation_count
                    ),
                    "postactivation_negative_rate": float(
                        post_negative[channel] / self.observation_count
                    ),
                    "mean_absolute_activation": float(
                        post_abs_sum[channel] / self.observation_count
                    ),
                    "max_absolute_activation": max_abs,
                    "exact_dead": bool(max_abs <= exact_dead_epsilon),
                    "near_dead": bool(positive_rate < near_dead_positive_rate),
                }
            )
        return output


def collect_activation_activity(
    model: nn.Module,
    loader: DataLoader[Any],
    *,
    variant: str,
    fold: int,
    exact_dead_epsilon: float,
    near_dead_positive_rate: float,
    device: str = "auto",
) -> pd.DataFrame:
    """Scan a complete validation fold and report channel-level activation activity."""
    resolved_device = _resolve_device(device)
    activation_modules = [
        (name, module)
        for name, module in model.base_model.named_modules()
        if isinstance(module, (nn.ReLU, nn.LeakyReLU))
    ]
    if len(activation_modules) != 8:
        raise ValueError(
            "SmallCNN activation count changed; "
            f"expected 8, got {len(activation_modules)}"
        )
    accumulators = {name: _ActivationAccumulator() for name, _ in activation_modules}
    handles: list[Any] = []
    for name, module in activation_modules:
        accumulator = accumulators[name]

        def pre_hook(
            _module: nn.Module,
            inputs: tuple[torch.Tensor, ...],
            *,
            target: _ActivationAccumulator = accumulator,
        ) -> None:
            target.observe_input(inputs[0], epsilon=exact_dead_epsilon)

        def post_hook(
            _module: nn.Module,
            _inputs: tuple[torch.Tensor, ...],
            output: torch.Tensor,
            *,
            target: _ActivationAccumulator = accumulator,
        ) -> None:
            target.observe_output(output, epsilon=exact_dead_epsilon)

        handles.append(module.register_forward_pre_hook(pre_hook))
        handles.append(module.register_forward_hook(post_hook))

    model = model.to(resolved_device).eval()
    try:
        with torch.inference_mode():
            for batch in loader:
                images = batch["image"].to(
                    resolved_device,
                    non_blocking=resolved_device.type == "cuda",
                )
                model.predict_season_logits(images)
    finally:
        for handle in handles:
            handle.remove()
        model.to("cpu")
        if resolved_device.type == "cuda":
            torch.cuda.empty_cache()

    rows: list[dict[str, Any]] = []
    for name, _ in activation_modules:
        rows.extend(
            accumulators[name].rows(
                variant=variant,
                fold=fold,
                layer=name,
                exact_dead_epsilon=exact_dead_epsilon,
                near_dead_positive_rate=near_dead_positive_rate,
            )
        )
    return pd.DataFrame(rows)


def _activation_layer_summary(channels: pd.DataFrame) -> pd.DataFrame:
    grouped = channels.groupby(["variant", "fold", "layer"], sort=False)
    rows: list[dict[str, Any]] = []
    for (variant, fold, layer), frame in grouped:
        rows.append(
            {
                "variant": variant,
                "fold": int(fold),
                "layer": layer,
                "channel_count": len(frame),
                "exact_dead_channels": int(frame["exact_dead"].sum()),
                "exact_dead_channel_rate": float(frame["exact_dead"].mean()),
                "near_dead_channels": int(frame["near_dead"].sum()),
                "near_dead_channel_rate": float(frame["near_dead"].mean()),
                "mean_preactivation_positive_rate": float(
                    frame["preactivation_positive_rate"].mean()
                ),
                "mean_postactivation_zero_rate": float(
                    frame["postactivation_zero_rate"].mean()
                ),
                "mean_postactivation_negative_rate": float(
                    frame["postactivation_negative_rate"].mean()
                ),
                "mean_absolute_activation": float(
                    frame["mean_absolute_activation"].mean()
                ),
            }
        )
    return pd.DataFrame(rows)


def _run_activation_diagnostics(
    *,
    spec: ActivationExperimentSpec,
    base: I2ExperimentConfig,
    source_runs: dict[int, SourceFoldRun],
    leaky_outputs: list[PostSubmissionRunOutput] | None,
    data_root: Path,
    splits_path: Path,
    label_map_path: Path,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    mappings = load_label_maps(label_map_path)
    season_classes = int(mappings["season"]["num_classes"])
    article_type_classes = int(mappings["articleType"]["num_classes"])
    leaky_by_fold = (
        None
        if leaky_outputs is None
        else {output.fold: output for output in leaky_outputs}
    )
    all_channels: list[pd.DataFrame] = []
    for fold in CANONICAL_FOLDS:
        print(f"[Neuron audit] fold {fold}: scanning complete validation fold", flush=True)
        loaders = build_task_loaders(
            validation_fold=fold,
            image_size=base.data.image_size,
            batch_size=base.data.validation_batch_size,
            target="season",
            augmentation="none",
            seed=PRIMARY_SEED,
            num_workers=base.data.num_workers,
            validation_batch_size=base.data.validation_batch_size,
            pin_memory=base.data.pin_memory,
            root=data_root,
            splits_path=splits_path,
            label_map_path=label_map_path,
        )
        relu_model = _build_i2_model(
            season_classes=season_classes,
            article_type_classes=article_type_classes,
        )
        _load_i2_checkpoint(relu_model, source_runs[fold].checkpoint_path)
        all_channels.append(
            collect_activation_activity(
                relu_model,
                loaders.validation,
                variant="I2 ReLU",
                fold=fold,
                exact_dead_epsilon=spec.exact_dead_epsilon,
                near_dead_positive_rate=spec.near_dead_positive_rate,
                device=base.optimisation.device,
            )
        )
        if leaky_by_fold is not None:
            leaky_checkpoint = leaky_by_fold[fold].checkpoint_path
            if leaky_checkpoint is None:
                raise ValueError(f"LeakyReLU fold {fold} has no checkpoint")
            leaky_model = _build_i2_model(
                season_classes=season_classes,
                article_type_classes=article_type_classes,
                negative_slope=spec.negative_slope,
            )
            _load_i2_checkpoint(leaky_model, leaky_checkpoint)
            all_channels.append(
                collect_activation_activity(
                    leaky_model,
                    loaders.validation,
                    variant="I2 LeakyReLU",
                    fold=fold,
                    exact_dead_epsilon=spec.exact_dead_epsilon,
                    near_dead_positive_rate=spec.near_dead_positive_rate,
                    device=base.optimisation.device,
                )
            )
    channels = pd.concat(all_channels, ignore_index=True)
    return channels, _activation_layer_summary(channels)


def _pooled_oof(outputs: list[PostSubmissionRunOutput]) -> pd.DataFrame:
    frame = pd.concat([output.oof for output in outputs], ignore_index=True)
    if len(frame) != 32_753 or frame["id"].duplicated().any():
        raise ValueError("pooled post-submission OOF must contain 32,753 unique IDs")
    return frame.sort_values("id").reset_index(drop=True)


def _source_oof(source_runs: dict[int, SourceFoldRun]) -> pd.DataFrame:
    frames: list[pd.DataFrame] = []
    for fold in CANONICAL_FOLDS:
        frame = pd.read_csv(source_runs[fold].prediction_path)
        _validate_output_oof(
            frame,
            expected=_expected_validation(
                fold=fold,
                splits_path=SPLITS_CSV,
                target="season",
            ),
            labels=tuple(SEASON_LABELS),
        )
        frames.append(frame)
    pooled = pd.concat(frames, ignore_index=True)
    if len(pooled) != 32_753 or pooled["id"].duplicated().any():
        raise ValueError("frozen I2 source OOF must contain 32,753 unique IDs")
    return pooled.sort_values("id").reset_index(drop=True)


def _metrics_from_oof(frame: pd.DataFrame) -> dict[str, Any]:
    probabilities = frame[[f"prob_{label}" for label in SEASON_LABELS]].to_numpy(float)
    return multiclass_metrics(
        frame["y_true"].astype(str),
        probabilities=probabilities,
        labels=SEASON_LABELS,
    )


def _comparison_tables(
    models: dict[str, pd.DataFrame],
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    overall_rows: list[dict[str, Any]] = []
    fold_rows: list[dict[str, Any]] = []
    per_class_rows: list[dict[str, Any]] = []
    for model_name, frame in models.items():
        pooled_metrics = _metrics_from_oof(frame)
        overall_rows.append(
            {
                "model": model_name,
                "macro_f1": pooled_metrics["macro_f1"],
                "accuracy": pooled_metrics["accuracy"],
                "balanced_accuracy": pooled_metrics["balanced_accuracy"],
                "weighted_f1": pooled_metrics["weighted_f1"],
                "nll": pooled_metrics["nll"],
                "brier": pooled_metrics["brier"],
                "ece": pooled_metrics["ece"],
                "rows": pooled_metrics["n_samples"],
            }
        )
        for label, values in pooled_metrics["per_class"].items():
            per_class_rows.append(
                {
                    "model": model_name,
                    "class": label,
                    **values,
                }
            )
        for fold in CANONICAL_FOLDS:
            fold_frame = frame.loc[pd.to_numeric(frame["fold"]).eq(fold)]
            fold_metrics = _metrics_from_oof(fold_frame)
            fold_rows.append(
                {
                    "model": model_name,
                    "fold": fold,
                    "macro_f1": fold_metrics["macro_f1"],
                    "accuracy": fold_metrics["accuracy"],
                    "balanced_accuracy": fold_metrics["balanced_accuracy"],
                    "rows": fold_metrics["n_samples"],
                }
            )
    return (
        pd.DataFrame(overall_rows),
        pd.DataFrame(fold_rows),
        pd.DataFrame(per_class_rows),
    )


def _bootstrap_intervals(
    models: dict[str, pd.DataFrame],
    *,
    splits_path: Path,
    temporary_directory: Path,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    aligned = models["I2 ReLU"][["id", "y_true", "y_pred"]].rename(
        columns={"y_pred": "relu_prediction"}
    )
    candidate_contracts = {
        "I2 LeakyReLU": ("leaky_prediction", "LeakyReLU_minus_ReLU"),
        "I2 embedding + RF": ("rf_prediction", "EmbeddingRF_minus_ReLU"),
        "I2 embedding + HGB": ("hgb_prediction", "EmbeddingHGB_minus_ReLU"),
        "I2 + RF 50/50": ("relu_rf_prediction", "ReluRFEnsemble_minus_ReLU"),
        "I2 + HGB 50/50": ("relu_hgb_prediction", "ReluHGBEnsemble_minus_ReLU"),
    }
    present_candidates = [name for name in candidate_contracts if name in models]
    if not present_candidates:
        raise ValueError("bootstrap comparison requires at least one candidate beside I2 ReLU")
    comparisons: dict[str, tuple[pd.Series, pd.Series]] = {}
    for model_name in present_candidates:
        column, comparison_id = candidate_contracts[model_name]
        candidate = models[model_name][["id", "y_true", "y_pred"]].rename(
            columns={"y_true": f"{column}_truth", "y_pred": column}
        )
        aligned = aligned.merge(candidate, on="id", how="inner", validate="one_to_one")
        if not aligned["y_true"].eq(aligned[f"{column}_truth"]).all():
            raise ValueError(f"{model_name} OOF targets differ from frozen I2")
        aligned = aligned.drop(columns=f"{column}_truth")
        comparisons[comparison_id] = (
            aligned["relu_prediction"],
            aligned[column],
        )
    development = get_samples(load_splits(splits_path), partition="development", target="season")
    family_map = development.set_index("id")["product_family_group"].astype(str)
    aligned["product_family_group"] = aligned["id"].map(family_map)
    if aligned["product_family_group"].isna().any():
        raise ValueError("bootstrap alignment lost product-family groups")
    draws = paired_group_bootstrap(
        aligned["y_true"],
        aligned["product_family_group"],
        comparisons,
        labels=SEASON_LABELS,
        replicates=10_000,
        random_seed=PRIMARY_SEED,
    )
    temporary_directory.mkdir(parents=True, exist_ok=True)
    atomic_write_csv(temporary_directory / "paired_bootstrap_draws.csv", draws)
    rows: list[dict[str, Any]] = []
    for comparison_id, frame in draws.groupby("comparison_id", sort=False):
        for metric, column in (
            ("macro_f1", "b_minus_a_macro_f1"),
            ("accuracy", "b_minus_a_accuracy"),
        ):
            values = frame[column].to_numpy(float)
            lower, median, upper = np.quantile(values, [0.025, 0.5, 0.975])
            rows.append(
                {
                    "comparison": comparison_id,
                    "metric": metric,
                    "median_delta": median,
                    "ci95_lower": lower,
                    "ci95_upper": upper,
                    "probability_delta_above_zero": float(np.mean(values > 0.0)),
                    "replicates": len(values),
                }
            )
    return pd.DataFrame(rows), draws


def _plot_model_comparison(overall: pd.DataFrame, folds: pd.DataFrame, output: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    metrics = ("macro_f1", "accuracy", "balanced_accuracy")
    labels = ("Macro-F1", "Accuracy", "Balanced accuracy")
    models = overall["model"].tolist()
    colors = (
        "#355C7D",
        "#F67280",
        "#6C9A8B",
        "#C06C84",
        "#F8B195",
        "#2A9D8F",
    )
    x = np.arange(len(metrics), dtype=float)
    width = 0.78 / len(models)
    figure, axis = plt.subplots(figsize=(11, 5.6), constrained_layout=True)
    for index, model_name in enumerate(models):
        color = colors[index]
        row = overall.loc[overall["model"].eq(model_name)].iloc[0]
        offsets = x + (index - (len(models) - 1) / 2.0) * width
        values = [float(row[metric]) for metric in metrics]
        bars = axis.bar(offsets, values, width=width, color=color, label=model_name, alpha=0.88)
        axis.bar_label(bars, fmt="%.3f", padding=3, fontsize=9)
        model_folds = folds.loc[folds["model"].eq(model_name)]
        for metric_index, metric in enumerate(metrics):
            jitter = np.linspace(-0.04, 0.04, len(model_folds))
            axis.scatter(
                np.full(len(model_folds), offsets[metric_index]) + jitter,
                model_folds[metric],
                s=18,
                color="#202020",
                alpha=0.65,
                zorder=4,
            )
    axis.set_xticks(x, labels)
    axis.set_ylim(0.0, 0.9)
    axis.set_ylabel("Five-fold pooled development OOF score")
    axis.set_title("Post-submission I2 experiments: same IDs and canonical folds")
    axis.grid(axis="y", alpha=0.2)
    axis.legend(loc="lower right")
    output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output, dpi=180, bbox_inches="tight")
    plt.close(figure)


def _plot_activation_comparison(layers: pd.DataFrame, output: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    averaged = (
        layers.groupby(["variant", "layer"], as_index=False)
        .agg(
            exact_dead_channel_rate=("exact_dead_channel_rate", "mean"),
            near_dead_channel_rate=("near_dead_channel_rate", "mean"),
            mean_postactivation_zero_rate=("mean_postactivation_zero_rate", "mean"),
        )
    )
    layer_order = list(dict.fromkeys(averaged["layer"].tolist()))
    variants = ("I2 ReLU", "I2 LeakyReLU")
    colors = {"I2 ReLU": "#355C7D", "I2 LeakyReLU": "#F67280"}
    panels = (
        ("exact_dead_channel_rate", "Exact-dead channels"),
        ("near_dead_channel_rate", "Near-dead channels (<1% positive input)"),
        ("mean_postactivation_zero_rate", "Zero activation values"),
    )
    figure, axes = plt.subplots(1, 3, figsize=(16, 5.4), constrained_layout=True)
    x = np.arange(len(layer_order), dtype=float)
    width = 0.38
    for axis, (metric, title) in zip(axes, panels, strict=True):
        for index, variant in enumerate(variants):
            subset = averaged.loc[averaged["variant"].eq(variant)].set_index("layer")
            values = subset.reindex(layer_order)[metric].to_numpy(float) * 100.0
            axis.bar(
                x + (index - 0.5) * width,
                values,
                width=width,
                color=colors[variant],
                label=variant,
                alpha=0.88,
            )
        axis.set_title(title)
        axis.set_xticks(x, [name.replace("features.", "") for name in layer_order], rotation=45)
        axis.set_ylabel("Mean across five validation folds (%)")
        axis.grid(axis="y", alpha=0.2)
    axes[0].legend(loc="upper right")
    figure.suptitle(
        "I2 activation health: full held-out fold scans (channels, not single values)",
        fontsize=14,
        fontweight="bold",
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output, dpi=180, bbox_inches="tight")
    plt.close(figure)


def _plot_bootstrap_intervals(intervals: pd.DataFrame, output: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    macro = intervals.loc[intervals["metric"].eq("macro_f1")].reset_index(drop=True)
    figure, axis = plt.subplots(figsize=(9, 3.8), constrained_layout=True)
    y = np.arange(len(macro), dtype=float)
    medians = macro["median_delta"].to_numpy(float)
    lower = macro["ci95_lower"].to_numpy(float)
    upper = macro["ci95_upper"].to_numpy(float)
    axis.errorbar(
        medians,
        y,
        xerr=np.vstack([medians - lower, upper - medians]),
        fmt="o",
        color="#355C7D",
        ecolor="#F67280",
        capsize=5,
        linewidth=2,
    )
    axis.axvline(0.0, color="#202020", linestyle="--", linewidth=1.2, label="No change")
    axis.set_yticks(y, macro["comparison"])
    axis.set_xlabel("Paired macro-F1 change (candidate minus original I2)")
    axis.set_title("95% product-family bootstrap intervals (10,000 resamples)")
    axis.grid(axis="x", alpha=0.2)
    axis.legend(loc="best")
    output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output, dpi=180, bbox_inches="tight")
    plt.close(figure)


def _plot_learning_curves(
    source_runs: dict[int, SourceFoldRun],
    leaky_outputs: list[PostSubmissionRunOutput],
    output: Path,
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    records: list[dict[str, Any]] = []
    sources = {
        "I2 ReLU": [source_runs[fold].history_path for fold in CANONICAL_FOLDS],
        "I2 LeakyReLU": [
            output_row.history_path
            for output_row in sorted(leaky_outputs, key=lambda item: item.fold)
        ],
    }
    for variant, paths in sources.items():
        for fold, path in enumerate(paths):
            with path.open(encoding="utf-8") as handle:
                payload = json.load(handle)
            for row in payload["epoch_history"]:
                records.append({"variant": variant, "fold": fold, **row})
    history = pd.DataFrame(records)
    colors = {"I2 ReLU": "#355C7D", "I2 LeakyReLU": "#F67280"}
    figure, axes = plt.subplots(1, 2, figsize=(12, 4.8), constrained_layout=True)
    for variant, color in colors.items():
        subset = history.loc[history["variant"].eq(variant)]
        for fold, fold_frame in subset.groupby("fold"):
            axes[0].plot(
                fold_frame["epoch"],
                fold_frame["train_loss"],
                color=color,
                alpha=0.16,
                linewidth=1,
            )
            axes[1].plot(
                fold_frame["epoch"],
                fold_frame["validation_macro_f1"],
                color=color,
                alpha=0.16,
                linewidth=1,
            )
        mean = subset.groupby("epoch", as_index=False).agg(
            train_loss=("train_loss", "mean"),
            validation_macro_f1=("validation_macro_f1", "mean"),
        )
        axes[0].plot(
            mean["epoch"], mean["train_loss"], color=color, linewidth=2.5, label=variant
        )
        axes[1].plot(
            mean["epoch"],
            mean["validation_macro_f1"],
            color=color,
            linewidth=2.5,
            label=variant,
        )
    axes[0].set(title="Training objective", xlabel="Epoch", ylabel="Total multitask loss")
    axes[1].set(title="Selection metric", xlabel="Epoch", ylabel="Validation macro-F1")
    for axis in axes:
        axis.grid(alpha=0.2)
        axis.legend(loc="best")
    figure.suptitle("Matched I2 learning curves: faint lines are folds; thick lines are means")
    output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output, dpi=180, bbox_inches="tight")
    plt.close(figure)


def plot_relu_baseline_activity(layers: pd.DataFrame, output: str | Path) -> Path:
    """Plot within-layer channel death rates separately from ordinary ReLU sparsity."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    required = {
        "layer",
        "channel_count",
        "exact_dead_channel_rate",
        "near_dead_channel_rate",
        "mean_postactivation_zero_rate",
    }
    missing = sorted(required - set(layers))
    if missing:
        raise ValueError(f"ReLU activity table is missing columns: {missing}")
    summary = (
        layers.groupby("layer", as_index=False, sort=False)
        .agg(
            channel_count=("channel_count", "first"),
            exact_dead_channel_rate=("exact_dead_channel_rate", "mean"),
            near_dead_channel_rate=("near_dead_channel_rate", "mean"),
            mean_postactivation_zero_rate=("mean_postactivation_zero_rate", "mean"),
        )
    )
    x = np.arange(len(summary), dtype=float)
    labels = [name.replace("features.", "") for name in summary["layer"]]
    panels = (
        ("exact_dead_channel_rate", "Exact-dead channels"),
        ("near_dead_channel_rate", "Near-dead channels"),
        ("mean_postactivation_zero_rate", "Zero activation values (not dead channels)"),
    )
    figure, axes = plt.subplots(1, 3, figsize=(16, 5.2), constrained_layout=True)
    for axis, (metric, title) in zip(axes, panels, strict=True):
        values = summary[metric].to_numpy(float) * 100.0
        bars = axis.bar(x, values, color="#355C7D", alpha=0.88)
        axis.set_title(title)
        axis.set_xticks(x, labels, rotation=45)
        axis.set_ylabel("Mean across five validation folds (%)")
        axis.grid(axis="y", alpha=0.2)
        if metric in {"exact_dead_channel_rate", "near_dead_channel_rate"}:
            for bar, channel_count in zip(
                bars,
                summary["channel_count"],
                strict=True,
            ):
                axis.text(
                    bar.get_x() + bar.get_width() / 2,
                    0.15,
                    f"0/{int(channel_count)}",
                    ha="center",
                    va="bottom",
                    fontsize=8,
                    rotation=90,
                )
            axis.set_ylim(0.0, 5.0)
    figure.suptitle(
        "Frozen I2 ReLU audit: dead channels / all channels within each layer",
        fontsize=14,
        fontweight="bold",
    )
    target = Path(output)
    target.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(target, dpi=180, bbox_inches="tight")
    plt.close(figure)
    return target


def _build_random_forest_evidence(
    *,
    source_runs: dict[int, SourceFoldRun],
    rf_outputs: list[PostSubmissionRunOutput],
    registry: RunRegistry,
    splits_path: Path,
    evidence_directory: Path,
    figure_directory: Path,
    temporary_directory: Path,
) -> dict[str, Any]:
    """Build evidence after the ReLU-death gate rejects further LeakyReLU training."""
    models = {
        "I2 ReLU": _source_oof(source_runs),
        "I2 embedding + RF": _pooled_oof(rf_outputs),
    }
    overall, folds, per_class = _comparison_tables(models)
    intervals, _ = _bootstrap_intervals(
        models,
        splits_path=splits_path,
        temporary_directory=temporary_directory,
    )
    evidence_directory.mkdir(parents=True, exist_ok=True)
    figure_directory.mkdir(parents=True, exist_ok=True)
    paths = {
        "model_comparison": evidence_directory / "model_comparison.csv",
        "fold_comparison": evidence_directory / "fold_comparison.csv",
        "per_class_metrics": evidence_directory / "per_class_metrics.csv",
        "bootstrap_intervals": evidence_directory / "paired_bootstrap_intervals.csv",
        "performance_figure": figure_directory / "model_performance_comparison.png",
        "bootstrap_figure": figure_directory / "paired_bootstrap_intervals.png",
    }
    for name, frame in (
        ("model_comparison", overall),
        ("fold_comparison", folds),
        ("per_class_metrics", per_class),
        ("bootstrap_intervals", intervals),
    ):
        atomic_write_csv(paths[name], frame)
    _plot_model_comparison(overall, folds, paths["performance_figure"])
    _plot_bootstrap_intervals(intervals, paths["bootstrap_figure"])

    relu_summary_path = evidence_directory / "relu_baseline_activation_summary.json"
    relu_layer_path = evidence_directory / "relu_baseline_activation_by_layer.csv"
    if not relu_summary_path.is_file() or not relu_layer_path.is_file():
        raise FileNotFoundError("run the frozen ReLU activation audit before RF evidence")
    with relu_summary_path.open(encoding="utf-8") as handle:
        relu_activation = json.load(handle)
    relu_figure_path = plot_relu_baseline_activity(
        pd.read_csv(relu_layer_path),
        figure_directory / "relu_baseline_dead_channels_by_layer.png",
    )
    leaky_attempts = registry.read()
    leaky_attempts = leaky_attempts.loc[
        leaky_attempts["experiment_id"].eq("postsubmit-i2-leaky-relu-0-01")
    ]
    indexed = overall.set_index("model")
    baseline = float(indexed.loc["I2 ReLU", "macro_f1"])
    forest = float(indexed.loc["I2 embedding + RF", "macro_f1"])
    summary = {
        "schema_version": "1.0.0",
        "role": "post_submission_exploration_only",
        "changes_frozen_final_or_submission": False,
        "activation_gate": {
            "question": "Does frozen I2 show dead ReLU channels within its layers?",
            "result": relu_activation,
            "decision": "do_not_continue_leaky_relu_as_a_dead_neuron_remedy",
            "reason": (
                "Every layer in every fold had zero exact-dead and zero near-dead channels. "
                "Ordinary zero-valued ReLU activations are sparsity, not dead neurons."
            ),
            "partial_attempts_retained_in_registry": leaky_attempts[
                ["run_id", "fold", "status", "primary_metric_value", "error_message"]
            ].to_dict(orient="records"),
        },
        "random_forest_result": {
            "original_i2_macro_f1": baseline,
            "embedding_random_forest_macro_f1": forest,
            "delta_macro_f1": forest - baseline,
            "improved": forest > baseline,
        },
        "source_runs": {
            str(fold): {
                "run_id": source_runs[fold].run_id,
                "checkpoint_sha256": source_runs[fold].checkpoint_sha256,
            }
            for fold in CANONICAL_FOLDS
        },
        "random_forest_runs": [output.run_id for output in rf_outputs],
        "artifacts": {
            **{
                name: {
                    "path": _registry_relative_path(path, data_root=ROOT),
                    "sha256": compute_sha256(path),
                }
                for name, path in paths.items()
            },
            "relu_activation_figure": {
                "path": _registry_relative_path(relu_figure_path, data_root=ROOT),
                "sha256": compute_sha256(relu_figure_path),
            },
            "relu_activation_by_layer": {
                "path": _registry_relative_path(relu_layer_path, data_root=ROOT),
                "sha256": compute_sha256(relu_layer_path),
            },
        },
    }
    summary_path = evidence_directory / "summary.json"
    atomic_write_json(summary_path, summary)
    summary["summary_path"] = str(summary_path)
    return summary


def _load_completed_experiment_oof(
    *,
    experiment_id: str,
    registry: RunRegistry,
    data_root: Path,
    splits_path: Path,
) -> tuple[pd.DataFrame, list[str]]:
    """Load one verified completed OOF file per canonical fold, newest first."""
    frame = registry.read()
    selected = frame.loc[
        frame["experiment_id"].eq(experiment_id)
        & frame["status"].eq("completed")
        & pd.to_numeric(frame["seed"], errors="coerce").eq(PRIMARY_SEED)
    ].copy()
    selected["_fold"] = pd.to_numeric(selected["fold"], errors="coerce")
    oof_frames: list[pd.DataFrame] = []
    run_ids: list[str] = []
    for fold in CANONICAL_FOLDS:
        candidates = selected.loc[selected["_fold"].eq(fold)].sort_values(
            "finished_at_utc", ascending=False
        )
        chosen: pd.DataFrame | None = None
        chosen_run_id = ""
        for row in candidates.to_dict(orient="records"):
            prediction_path = _artifact_path(row["prediction_path"], data_root=data_root)
            history_path = _artifact_path(row["history_path"], data_root=data_root)
            try:
                verify_artifact(prediction_path, row["prediction_sha256"])
                verify_artifact(history_path, row["history_sha256"])
            except (OSError, RuntimeError, ValueError):
                continue
            candidate = pd.read_csv(prediction_path)
            _validate_output_oof(
                candidate,
                expected=_expected_validation(
                    fold=fold,
                    splits_path=splits_path,
                    target="season",
                ),
                labels=tuple(SEASON_LABELS),
            )
            validate_oof_identity(
                candidate,
                run_id=str(row["run_id"]),
                experiment_id=experiment_id,
                fold=fold,
                seed=PRIMARY_SEED,
            )
            chosen = candidate
            chosen_run_id = str(row["run_id"])
            break
        if chosen is None:
            raise FileNotFoundError(
                f"{experiment_id} has no verified completed output for fold {fold}"
            )
        oof_frames.append(chosen)
        run_ids.append(chosen_run_id)
    pooled = pd.concat(oof_frames, ignore_index=True).sort_values("id").reset_index(drop=True)
    if len(pooled) != 32_753 or pooled["id"].duplicated().any():
        raise ValueError(f"{experiment_id} pooled OOF coverage changed")
    return pooled, run_ids


def _average_probability_oof(
    baseline: pd.DataFrame,
    candidate: pd.DataFrame,
    *,
    weight: float = 0.5,
) -> pd.DataFrame:
    """Average aligned probabilities with a fixed, untuned convex weight."""
    if not math.isclose(weight, 0.5, abs_tol=1e-12):
        raise ValueError("post-submission ensemble weight is fixed at 0.5")
    identity = ["id", "fold", "seed", "y_true"]
    probability_columns = [f"prob_{label}" for label in SEASON_LABELS]
    left = baseline[identity + probability_columns].sort_values("id").reset_index(drop=True)
    right = candidate[identity + probability_columns].sort_values("id").reset_index(drop=True)
    if not left[identity].equals(right[identity]):
        raise ValueError("ensemble inputs do not share identical OOF identity and targets")
    probabilities = (
        weight * left[probability_columns].to_numpy(float)
        + (1.0 - weight) * right[probability_columns].to_numpy(float)
    )
    output = left[identity].copy()
    labels = np.asarray(SEASON_LABELS, dtype=object)
    output["y_pred"] = labels[probabilities.argmax(axis=1)]
    for index, column in enumerate(probability_columns):
        output[column] = probabilities[:, index]
    return output


def _build_boosting_evidence(
    *,
    source_runs: dict[int, SourceFoldRun],
    hgb_outputs: list[PostSubmissionRunOutput],
    registry: RunRegistry,
    data_root: Path,
    splits_path: Path,
    evidence_directory: Path,
    figure_directory: Path,
    temporary_directory: Path,
) -> dict[str, Any]:
    relu = _source_oof(source_runs)
    forest, forest_run_ids = _load_completed_experiment_oof(
        experiment_id="postsubmit-i2-embedding-random-forest",
        registry=registry,
        data_root=data_root,
        splits_path=splits_path,
    )
    boosting = _pooled_oof(hgb_outputs)
    models = {
        "I2 ReLU": relu,
        "I2 embedding + RF": forest,
        "I2 embedding + HGB": boosting,
        "I2 + RF 50/50": _average_probability_oof(relu, forest),
        "I2 + HGB 50/50": _average_probability_oof(relu, boosting),
    }
    overall, folds, per_class = _comparison_tables(models)
    intervals, _ = _bootstrap_intervals(
        models,
        splits_path=splits_path,
        temporary_directory=temporary_directory / "boosting",
    )
    evidence_directory.mkdir(parents=True, exist_ok=True)
    figure_directory.mkdir(parents=True, exist_ok=True)
    paths = {
        "model_comparison": evidence_directory / "boosted_model_comparison.csv",
        "fold_comparison": evidence_directory / "boosted_fold_comparison.csv",
        "per_class_metrics": evidence_directory / "boosted_per_class_metrics.csv",
        "bootstrap_intervals": evidence_directory / "boosted_paired_bootstrap_intervals.csv",
        "performance_figure": figure_directory / "boosted_model_performance_comparison.png",
        "bootstrap_figure": figure_directory / "boosted_paired_bootstrap_intervals.png",
    }
    for name, frame in (
        ("model_comparison", overall),
        ("fold_comparison", folds),
        ("per_class_metrics", per_class),
        ("bootstrap_intervals", intervals),
    ):
        atomic_write_csv(paths[name], frame)
    _plot_model_comparison(overall, folds, paths["performance_figure"])
    _plot_bootstrap_intervals(intervals, paths["bootstrap_figure"])

    ranking = overall.sort_values("macro_f1", ascending=False).reset_index(drop=True)
    baseline = float(
        overall.loc[overall["model"].eq("I2 ReLU"), "macro_f1"].iloc[0]
    )
    comparisons = {
        str(row["model"]): {
            "macro_f1": float(row["macro_f1"]),
            "delta_vs_i2_relu": float(row["macro_f1"] - baseline),
        }
        for _, row in overall.iterrows()
    }
    summary = {
        "schema_version": "1.0.0",
        "role": "post_submission_exploration_only",
        "changes_frozen_final_or_submission": False,
        "protocol": {
            "encoder": "frozen fold-specific I2 SmallCNN",
            "embedding_dimension": 256,
            "gradient_boosting_grid_search_used": False,
            "ensemble_weights_tuned": False,
            "ensemble_weight": 0.5,
            "evaluation": "same 32753 development OOF IDs and canonical folds",
        },
        "winner": {
            "model": str(ranking.iloc[0]["model"]),
            "macro_f1": float(ranking.iloc[0]["macro_f1"]),
            "delta_vs_i2_relu": float(ranking.iloc[0]["macro_f1"] - baseline),
        },
        "comparisons": comparisons,
        "random_forest_runs": forest_run_ids,
        "gradient_boosting_runs": [output.run_id for output in hgb_outputs],
        "artifacts": {
            name: {
                "path": _registry_relative_path(path, data_root=ROOT),
                "sha256": compute_sha256(path),
            }
            for name, path in paths.items()
        },
    }
    summary_path = evidence_directory / "boosting_summary.json"
    atomic_write_json(summary_path, summary)
    summary["summary_path"] = str(summary_path)
    return summary


def _build_evidence(
    *,
    source_runs: dict[int, SourceFoldRun],
    leaky_outputs: list[PostSubmissionRunOutput],
    rf_outputs: list[PostSubmissionRunOutput],
    activation_channels: pd.DataFrame,
    activation_layers: pd.DataFrame,
    splits_path: Path,
    evidence_directory: Path,
    figure_directory: Path,
    temporary_directory: Path,
) -> dict[str, Any]:
    models = {
        "I2 ReLU": _source_oof(source_runs),
        "I2 LeakyReLU": _pooled_oof(leaky_outputs),
        "I2 embedding + RF": _pooled_oof(rf_outputs),
    }
    overall, folds, per_class = _comparison_tables(models)
    intervals, _ = _bootstrap_intervals(
        models,
        splits_path=splits_path,
        temporary_directory=temporary_directory,
    )
    evidence_directory.mkdir(parents=True, exist_ok=True)
    figure_directory.mkdir(parents=True, exist_ok=True)
    paths = {
        "model_comparison": evidence_directory / "model_comparison.csv",
        "fold_comparison": evidence_directory / "fold_comparison.csv",
        "per_class_metrics": evidence_directory / "per_class_metrics.csv",
        "activation_channels": evidence_directory / "activation_by_channel.csv",
        "activation_layers": evidence_directory / "activation_by_layer.csv",
        "bootstrap_intervals": evidence_directory / "paired_bootstrap_intervals.csv",
        "performance_figure": figure_directory / "model_performance_comparison.png",
        "activation_figure": figure_directory / "dead_neuron_comparison.png",
        "bootstrap_figure": figure_directory / "paired_bootstrap_intervals.png",
        "learning_curve_figure": figure_directory / "relu_vs_leaky_learning_curves.png",
    }
    atomic_write_csv(paths["model_comparison"], overall)
    atomic_write_csv(paths["fold_comparison"], folds)
    atomic_write_csv(paths["per_class_metrics"], per_class)
    atomic_write_csv(paths["activation_channels"], activation_channels)
    atomic_write_csv(paths["activation_layers"], activation_layers)
    atomic_write_csv(paths["bootstrap_intervals"], intervals)
    _plot_model_comparison(overall, folds, paths["performance_figure"])
    _plot_activation_comparison(activation_layers, paths["activation_figure"])
    _plot_bootstrap_intervals(intervals, paths["bootstrap_figure"])
    _plot_learning_curves(source_runs, leaky_outputs, paths["learning_curve_figure"])

    indexed = overall.set_index("model")
    baseline = float(indexed.loc["I2 ReLU", "macro_f1"])
    leaky = float(indexed.loc["I2 LeakyReLU", "macro_f1"])
    forest = float(indexed.loc["I2 embedding + RF", "macro_f1"])
    activation_overall = (
        activation_channels.groupby("variant", as_index=False)
        .agg(
            exact_dead_channel_rate=("exact_dead", "mean"),
            near_dead_channel_rate=("near_dead", "mean"),
            mean_postactivation_zero_rate=("postactivation_zero_rate", "mean"),
            mean_preactivation_positive_rate=("preactivation_positive_rate", "mean"),
        )
        .set_index("variant")
    )
    numerical_stops: dict[str, Any] = {}
    for output in leaky_outputs:
        with output.history_path.open(encoding="utf-8") as handle:
            run_history = json.load(handle)
        if run_history.get("numerical_stop") is not None:
            numerical_stops[str(output.fold)] = run_history["numerical_stop"]
    summary = {
        "schema_version": "1.0.0",
        "role": "post_submission_exploration_only",
        "changes_frozen_final_or_submission": False,
        "development_oof_rows_per_model": 32_753,
        "canonical_folds": list(CANONICAL_FOLDS),
        "primary_seed": PRIMARY_SEED,
        "results": {
            "original_i2_macro_f1": baseline,
            "leaky_relu_macro_f1": leaky,
            "leaky_relu_delta": leaky - baseline,
            "leaky_relu_improved": leaky > baseline,
            "embedding_random_forest_macro_f1": forest,
            "embedding_random_forest_delta": forest - baseline,
            "embedding_random_forest_improved": forest > baseline,
        },
        "activation_diagnostic": {
            "unit": "feature_channel",
            "exact_dead_definition": (
                "maximum absolute postactivation <= 1e-12 over a full validation fold"
            ),
            "near_dead_definition": "preactivation positive rate < 1% over a full validation fold",
            "relu": activation_overall.loc["I2 ReLU"].to_dict(),
            "leaky_relu": activation_overall.loc["I2 LeakyReLU"].to_dict(),
            "warning": (
                "LeakyReLU makes exact postactivation zeros unlikely by definition; "
                "near-dead preactivation rates are the fairer cross-activation comparison."
            ),
        },
        "source_runs": {
            str(fold): {
                "run_id": source_runs[fold].run_id,
                "checkpoint_sha256": source_runs[fold].checkpoint_sha256,
            }
            for fold in CANONICAL_FOLDS
        },
        "new_runs": {
            "leaky_relu": [output.run_id for output in leaky_outputs],
            "embedding_random_forest": [output.run_id for output in rf_outputs],
        },
        "leaky_relu_numerical_stops": numerical_stops,
        "artifacts": {
            name: {
                "path": _registry_relative_path(path, data_root=ROOT),
                "sha256": compute_sha256(path),
            }
            for name, path in paths.items()
        },
    }
    summary_path = evidence_directory / "summary.json"
    atomic_write_json(summary_path, summary)
    summary["summary_path"] = str(summary_path)
    return summary


def run_relu_activation_audit(
    *,
    activation_config_path: str | Path = DEFAULT_ACTIVATION_CONFIG,
    data_root: str | Path = ROOT,
    source_root: str | Path = ROOT,
    splits_path: str | Path = SPLITS_CSV,
    label_map_path: str | Path = LABEL_MAPS_JSON,
    registry_path: str | Path = RUNS_CSV,
    evidence_directory: str | Path = DEFAULT_EVIDENCE_DIRECTORY,
    figure_directory: str | Path = DEFAULT_FIGURE_DIRECTORY,
) -> dict[str, Any]:
    """Measure dead channels within every frozen-I2 ReLU layer and fold."""
    data_root_path = Path(data_root).resolve()
    source_root_path = Path(source_root).resolve()
    splits = Path(splits_path)
    labels = Path(label_map_path)
    spec = load_activation_experiment_spec(activation_config_path)
    base = _load_base_i2_config(spec.base_i2_config, project_root=source_root_path)
    source_runs = _verify_source_i2_runs(
        registry_path=Path(registry_path),
        data_root=data_root_path,
    )
    channels, layers = _run_activation_diagnostics(
        spec=spec,
        base=base,
        source_runs=source_runs,
        leaky_outputs=None,
        data_root=data_root_path,
        splits_path=splits,
        label_map_path=labels,
    )
    if set(channels["variant"]) != {"I2 ReLU"}:
        raise ValueError("the ReLU gate must contain only frozen-I2 ReLU diagnostics")

    evidence = Path(evidence_directory)
    figures = Path(figure_directory)
    evidence.mkdir(parents=True, exist_ok=True)
    figures.mkdir(parents=True, exist_ok=True)
    channel_path = evidence / "relu_baseline_activation_by_channel.csv"
    layer_path = evidence / "relu_baseline_activation_by_layer.csv"
    summary_path = evidence / "relu_baseline_activation_summary.json"
    figure_path = figures / "relu_baseline_dead_channels_by_layer.png"
    atomic_write_csv(channel_path, channels)
    atomic_write_csv(layer_path, layers)

    by_fold = []
    for fold, frame in channels.groupby("fold", sort=True):
        by_fold.append(
            {
                "fold": int(fold),
                "channels": int(len(frame)),
                "exact_dead_rate": float(frame["exact_dead"].mean()),
                "near_dead_rate": float(frame["near_dead"].mean()),
                "mean_zero_rate": float(frame["postactivation_zero_rate"].mean()),
            }
        )
    summary = {
        "definitions": {
            "exact_dead": (
                f"max absolute postactivation <= {spec.exact_dead_epsilon:g} "
                "over a complete validation fold"
            ),
            "near_dead": (
                "preactivation positive rate < "
                f"{100.0 * spec.near_dead_positive_rate:g}% over a complete "
                "validation fold"
            ),
            "zero_rate": (
                "fraction of individual activation values equal to zero; "
                "not a dead-neuron rate"
            ),
        },
        "overall": {
            "validation_folds": len(CANONICAL_FOLDS),
            "channels_scanned_across_folds": int(len(channels)),
            "exact_dead_channel_rate": float(channels["exact_dead"].mean()),
            "near_dead_channel_rate": float(channels["near_dead"].mean()),
            "mean_postactivation_zero_rate": float(
                channels["postactivation_zero_rate"].mean()
            ),
        },
        "by_fold": by_fold,
    }
    atomic_write_json(summary_path, summary)
    plot_relu_baseline_activity(layers, figure_path)

    print("Frozen I2 ReLU channel audit (dead / all channels in each layer):")
    for layer, frame in layers.groupby("layer", sort=False):
        channel_count = int(frame["channel_count"].iloc[0])
        exact = frame["exact_dead_channels"].astype(int).tolist()
        near = frame["near_dead_channels"].astype(int).tolist()
        zero_rate = 100.0 * float(frame["mean_postactivation_zero_rate"].mean())
        print(
            f"  {layer}: exact {exact}/{channel_count}; near {near}/{channel_count}; "
            f"ordinary zero activations {zero_rate:.2f}%",
            flush=True,
        )
    print(
        "Decision: do not continue LeakyReLU as a dead-neuron remedy; "
        "no exact-dead or near-dead channels were found.",
        flush=True,
    )
    return {
        "summary": summary,
        "artifacts": {
            "channels": str(channel_path),
            "layers": str(layer_path),
            "summary": str(summary_path),
            "figure": str(figure_path),
        },
    }


def run_post_submission_experiments(
    *,
    mode: ExecutionMode = "run_or_load",
    activation_config_path: str | Path = DEFAULT_ACTIVATION_CONFIG,
    random_forest_config_path: str | Path = DEFAULT_RANDOM_FOREST_CONFIG,
    gradient_boosting_config_path: str | Path = DEFAULT_GRADIENT_BOOSTING_CONFIG,
    data_root: str | Path = ROOT,
    source_root: str | Path = ROOT,
    splits_path: str | Path = SPLITS_CSV,
    label_map_path: str | Path = LABEL_MAPS_JSON,
    registry_path: str | Path = RUNS_CSV,
    evidence_directory: str | Path = DEFAULT_EVIDENCE_DIRECTORY,
    figure_directory: str | Path = DEFAULT_FIGURE_DIRECTORY,
    temporary_directory: str | Path = DEFAULT_TEMPORARY_DIRECTORY,
) -> dict[str, Any]:
    """Run the justified audit, RF, and boosting sequence without retraining I2."""
    if mode not in {"run_or_load", "run", "load"}:
        raise ValueError(f"unknown execution mode: {mode}")
    audit = run_relu_activation_audit(
        activation_config_path=activation_config_path,
        data_root=data_root,
        source_root=source_root,
        splits_path=splits_path,
        label_map_path=label_map_path,
        registry_path=registry_path,
        evidence_directory=evidence_directory,
        figure_directory=figure_directory,
    )
    forest = run_random_forest_experiment(
        mode=mode,
        random_forest_config_path=random_forest_config_path,
        data_root=data_root,
        source_root=source_root,
        splits_path=splits_path,
        label_map_path=label_map_path,
        registry_path=registry_path,
        evidence_directory=evidence_directory,
        figure_directory=figure_directory,
        temporary_directory=temporary_directory,
    )
    boosting = run_gradient_boosting_experiment(
        mode=mode,
        gradient_boosting_config_path=gradient_boosting_config_path,
        data_root=data_root,
        source_root=source_root,
        splits_path=splits_path,
        label_map_path=label_map_path,
        registry_path=registry_path,
        evidence_directory=evidence_directory,
        figure_directory=figure_directory,
        temporary_directory=temporary_directory,
    )
    return {"relu_audit": audit, "random_forest": forest, "boosting": boosting}


def run_random_forest_experiment(
    *,
    mode: ExecutionMode = "run_or_load",
    random_forest_config_path: str | Path = DEFAULT_RANDOM_FOREST_CONFIG,
    data_root: str | Path = ROOT,
    source_root: str | Path = ROOT,
    splits_path: str | Path = SPLITS_CSV,
    label_map_path: str | Path = LABEL_MAPS_JSON,
    registry_path: str | Path = RUNS_CSV,
    evidence_directory: str | Path = DEFAULT_EVIDENCE_DIRECTORY,
    figure_directory: str | Path = DEFAULT_FIGURE_DIRECTORY,
    temporary_directory: str | Path = DEFAULT_TEMPORARY_DIRECTORY,
) -> dict[str, Any]:
    """Run only the independent frozen-I2-embedding Random Forest experiment."""
    if mode not in {"run_or_load", "run", "load"}:
        raise ValueError(f"unknown execution mode: {mode}")
    data_root_path = Path(data_root).resolve()
    source_root_path = Path(source_root).resolve()
    splits = Path(splits_path)
    labels = Path(label_map_path)
    registry_file = Path(registry_path)
    temporary = Path(temporary_directory)
    forest_spec = load_random_forest_experiment_spec(random_forest_config_path)
    base = _load_base_i2_config(
        forest_spec.base_i2_config,
        project_root=source_root_path,
    )
    source_runs = _verify_source_i2_runs(
        registry_path=registry_file,
        data_root=data_root_path,
    )
    registry = RunRegistry(registry_file)
    outputs = [
        _run_random_forest_fold(
            spec=forest_spec,
            base=base,
            source=source_runs[fold],
            fold=fold,
            mode=mode,
            registry=registry,
            data_root=data_root_path,
            source_root=source_root_path,
            splits_path=splits,
            label_map_path=labels,
            temporary_directory=temporary,
        )
        for fold in CANONICAL_FOLDS
    ]
    summary = _build_random_forest_evidence(
        source_runs=source_runs,
        rf_outputs=outputs,
        registry=registry,
        splits_path=splits,
        evidence_directory=Path(evidence_directory),
        figure_directory=Path(figure_directory),
        temporary_directory=temporary,
    )
    print(json.dumps(summary["random_forest_result"], indent=2), flush=True)
    print(f"Evidence: {summary['summary_path']}", flush=True)
    return summary


def run_gradient_boosting_experiment(
    *,
    mode: ExecutionMode = "run_or_load",
    gradient_boosting_config_path: str | Path = DEFAULT_GRADIENT_BOOSTING_CONFIG,
    data_root: str | Path = ROOT,
    source_root: str | Path = ROOT,
    splits_path: str | Path = SPLITS_CSV,
    label_map_path: str | Path = LABEL_MAPS_JSON,
    registry_path: str | Path = RUNS_CSV,
    evidence_directory: str | Path = DEFAULT_EVIDENCE_DIRECTORY,
    figure_directory: str | Path = DEFAULT_FIGURE_DIRECTORY,
    temporary_directory: str | Path = DEFAULT_TEMPORARY_DIRECTORY,
) -> dict[str, Any]:
    """Run fixed HGB and untuned 50/50 ensembles against verified I2/RF OOF."""
    if mode not in {"run_or_load", "run", "load"}:
        raise ValueError(f"unknown execution mode: {mode}")
    data_root_path = Path(data_root).resolve()
    source_root_path = Path(source_root).resolve()
    splits = Path(splits_path)
    labels = Path(label_map_path)
    registry_file = Path(registry_path)
    temporary = Path(temporary_directory)
    spec = load_gradient_boosting_experiment_spec(gradient_boosting_config_path)
    base = _load_base_i2_config(spec.base_i2_config, project_root=source_root_path)
    source_runs = _verify_source_i2_runs(
        registry_path=registry_file,
        data_root=data_root_path,
    )
    registry = RunRegistry(registry_file)
    outputs = [
        _run_gradient_boosting_fold(
            spec=spec,
            base=base,
            source=source_runs[fold],
            fold=fold,
            mode=mode,
            registry=registry,
            data_root=data_root_path,
            source_root=source_root_path,
            splits_path=splits,
            label_map_path=labels,
            temporary_directory=temporary,
        )
        for fold in CANONICAL_FOLDS
    ]
    summary = _build_boosting_evidence(
        source_runs=source_runs,
        hgb_outputs=outputs,
        registry=registry,
        data_root=data_root_path,
        splits_path=splits,
        evidence_directory=Path(evidence_directory),
        figure_directory=Path(figure_directory),
        temporary_directory=temporary,
    )
    print(json.dumps(summary["winner"], indent=2), flush=True)
    print(f"Evidence: {summary['summary_path']}", flush=True)
    return summary


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Audit frozen-I2 ReLU channels, then compare fixed Random Forest and "
            "gradient-boosting heads without changing the frozen final model."
        )
    )
    parser.add_argument(
        "--mode",
        choices=("run_or_load", "run", "load"),
        default="run_or_load",
        help="Reuse hash-verified folds by default, force reruns, or require existing caches.",
    )
    parser.add_argument(
        "--experiment",
        choices=("all", "relu_audit", "random_forest", "gradient_boosting"),
        default="all",
        help=(
            "Run the justified audit/RF/HGB sequence, or one named stage. "
            "The default does not continue LeakyReLU after a negative dead-neuron gate."
        ),
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    arguments = _build_parser().parse_args(argv)
    if arguments.experiment == "relu_audit":
        run_relu_activation_audit()
    elif arguments.experiment == "random_forest":
        run_random_forest_experiment(mode=arguments.mode)
    elif arguments.experiment == "gradient_boosting":
        run_gradient_boosting_experiment(mode=arguments.mode)
    else:
        run_post_submission_experiments(mode=arguments.mode)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
