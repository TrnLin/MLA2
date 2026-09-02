"""Classic HOG feature and model candidates for Task 1."""

from __future__ import annotations

import json
import pickle
from collections.abc import Mapping
from dataclasses import asdict, dataclass
from importlib.metadata import version
from pathlib import Path
from typing import Literal

import numpy as np
import pandas as pd

from fashion.config import RANDOM_SEED, ROOT, SPLITS_CSV, TASK1_HOG_CACHE_DIR, TASK1_RESULT_DIR
from fashion.data.dataset import load_splits
from fashion.data.hashing import compute_sha256
from fashion.task1.classical_features import (
    HOG_CACHE_SCHEMA_VERSION,  # noqa: F401
    TASK1_HOG_COARSE,  # noqa: F401
    TASK1_HOG_FINE,  # noqa: F401
    TASK1_HOG_SPECS,
    Task1HogFeatureSet,
    Task1HogSpec,
    extract_task1_hog,
    load_or_build_task1_hog_features,
)
from fashion.task1.classical_models import (
    TASK1_DEFAULT_KNN,
    TASK1_DEFAULT_SVM,
    TASK1_KNN_GRID,
    TASK1_SVM_GRID,
    Task1ClassicalModelConfig,
    Task1KNNConfig,
    Task1LinearSVMConfig,
    fit_predict_task1_knn,
    fit_predict_task1_linear_svm,
    knn_probabilities_from_neighbors,
    query_task1_neighbors,
)
from fashion.task1.dataset import get_task1_fold_rows
from fashion.task1.evaluation import (
    TASK1_NUM_CLASSES,
    build_prediction_frame,
    classification_metrics,
    validate_task1_label_map,
)
from fashion.train.artifacts import (
    atomic_write_bytes,
    atomic_write_csv,
    canonical_sha256,
    verify_artifact,
)
from fashion.train.registry import RunRecord, RunRegistry, new_run_id, tracked_run

__all__ = [
    "Task1KNNConfig",
    "Task1LinearSVMConfig",
    "Task1ClassicalModelConfig",
    "TASK1_KNN_GRID",
    "TASK1_SVM_GRID",
    "TASK1_DEFAULT_KNN",
    "TASK1_DEFAULT_SVM",
    "query_task1_neighbors",
    "knn_probabilities_from_neighbors",
    "fit_predict_task1_knn",
    "fit_predict_task1_linear_svm",
    "Task1ClassicalRunConfig",
    "Task1ClassicalFoldResult",
    "run_task1_classical_fold",
]


@dataclass(frozen=True)
class Task1ClassicalRunConfig:
    """Execution controls for a small smoke run or reportable classic fold."""

    stage: Literal["smoke", "experiment"]
    final_eligible: bool
    seed: int = RANDOM_SEED
    validation_batch_size: int = 512

    @classmethod
    def smoke(cls) -> "Task1ClassicalRunConfig":
        return cls(stage="smoke", final_eligible=False)

    @classmethod
    def full(cls) -> "Task1ClassicalRunConfig":
        return cls(stage="experiment", final_eligible=True)


@dataclass(frozen=True)
class Task1ClassicalFoldResult:
    """Completed classical fold evidence and persisted artifact paths."""

    run_id: str
    fold: int
    candidate_id: str
    hog_id: str
    model_family: str
    status: Literal["completed"]
    metrics: dict[str, float]
    model_path: Path
    prediction_path: Path


def _classical_model_family(config: Task1ClassicalModelConfig) -> str:
    if isinstance(config, Task1KNNConfig):
        return "task1_hog_knn_v1"
    if isinstance(config, Task1LinearSVMConfig):
        return "task1_hog_linear_svm_v1"
    raise TypeError("model_config must be an approved Task 1 classic model configuration")


def _classical_artifact_path(path: Path) -> str:
    try:
        return path.resolve().relative_to(ROOT.resolve()).as_posix()
    except ValueError:
        return str(path.resolve())


def _resolve_classical_artifact_path(path: str) -> Path:
    stored = Path(path)
    return stored if stored.is_absolute() else ROOT / stored


def _classical_implementation_sha256() -> str:
    relative_paths = (
        "src/fashion/task1/classical.py",
        "src/fashion/task1/classical_models.py",
        "src/fashion/task1/dataset.py",
        "src/fashion/task1/evaluation.py",
    )
    identity = {
        "source_sha256": {
            relative: compute_sha256(ROOT / relative) for relative in relative_paths
        },
        "library_versions": {
            "numpy": version("numpy"),
            "Pillow": version("Pillow"),
            "scikit-image": version("scikit-image"),
            "scikit-learn": version("scikit-learn"),
        },
    }
    return canonical_sha256(identity)


def _smoke_hog_features(
    rows: pd.DataFrame,
    spec: Task1HogSpec,
    *,
    root: Path,
) -> tuple[np.ndarray, np.ndarray]:
    ordered = rows.sort_values("id", kind="stable").copy()
    ids = ordered["id"].to_numpy(dtype=np.int64)
    features = np.stack(
        [extract_task1_hog(root / str(row.path), spec) for row in ordered.itertuples()]
    ).astype(np.float32, copy=False)
    return ids, features


def _aligned_hog_features(feature_set: Task1HogFeatureSet, rows: pd.DataFrame) -> np.ndarray:
    ids = rows["id"].to_numpy(dtype=np.int64)
    positions = {int(product_id): index for index, product_id in enumerate(feature_set.ids)}
    try:
        indexes = np.asarray([positions[int(product_id)] for product_id in ids], dtype=np.int64)
    except KeyError as error:
        raise ValueError("fold rows are missing from the development HOG cache") from error
    return feature_set.features[indexes]


def _completed_classical_result(
    row: pd.Series,
    *,
    validation_fold: int,
    candidate_id: str,
    hog_id: str,
    model_family: str,
) -> Task1ClassicalFoldResult | None:
    try:
        model_path = _resolve_classical_artifact_path(str(row["checkpoint_path"]))
        prediction_path = _resolve_classical_artifact_path(str(row["prediction_path"]))
        verify_artifact(model_path, str(row["checkpoint_sha256"]))
        verify_artifact(prediction_path, str(row["prediction_sha256"]))
        predictions = pd.read_csv(prediction_path)
        probabilities = predictions.loc[
            :, [f"prob_{index:03d}" for index in range(TASK1_NUM_CLASSES)]
        ].to_numpy(dtype=np.float64)
        metrics = classification_metrics(
            predictions["true_index"].to_numpy(dtype=np.int64), probabilities
        )
        registered_metrics = json.loads(str(row["metrics"]))
        if canonical_sha256(metrics) != canonical_sha256(registered_metrics):
            return None
    except (KeyError, TypeError, ValueError, json.JSONDecodeError, OSError):
        return None
    except Exception:
        return None
    return Task1ClassicalFoldResult(
        run_id=str(row["run_id"]),
        fold=validation_fold,
        candidate_id=candidate_id,
        hog_id=hog_id,
        model_family=model_family,
        status="completed",
        metrics=metrics,
        model_path=model_path,
        prediction_path=prediction_path,
    )


def run_task1_classical_fold(
    splits: pd.DataFrame,
    label_map: Mapping[str, object],
    *,
    validation_fold: int,
    hog_spec: Task1HogSpec,
    model_config: Task1ClassicalModelConfig,
    run_config: Task1ClassicalRunConfig,
    registry: RunRegistry | None = None,
    root: str | Path = ROOT,
    result_root: str | Path = TASK1_RESULT_DIR,
    cache_root: str | Path = TASK1_HOG_CACHE_DIR,
    split_path: str | Path = SPLITS_CSV,
) -> Task1ClassicalFoldResult:
    """Fit, register, and safely resume one fixed-class HOG classic fold."""
    if validation_fold not in range(5):
        raise ValueError("validation_fold must be in range(5)")
    if run_config.validation_batch_size <= 0:
        raise ValueError("validation_batch_size must be positive")
    is_smoke = run_config.stage == "smoke" and not run_config.final_eligible
    is_full = run_config.stage == "experiment" and run_config.final_eligible
    if not (is_smoke or is_full):
        raise ValueError(
            "Task 1 classic run config must be either smoke with final_eligible=False "
            "or experiment with final_eligible=True"
        )
    if is_full and hog_spec not in TASK1_HOG_SPECS:
        raise ValueError("final-eligible runs require one of the frozen Task 1 HOG specs")
    split_file = Path(split_path)
    if is_full:
        if run_config.seed != RANDOM_SEED:
            raise ValueError(
                "final-eligible Task 1 classic runs require stage='experiment' and seed 2753"
            )
        if split_file.resolve() != SPLITS_CSV.resolve():
            raise ValueError("final-eligible Task 1 classic runs require the canonical split path")
        if not splits.equals(load_splits(split_file)):
            raise ValueError("supplied splits must match the canonical split file")

    label_to_index, class_names = validate_task1_label_map(label_map)
    label_map_sha256 = canonical_sha256(label_map)
    model_family = _classical_model_family(model_config)
    candidate_id = f"{hog_spec.hog_id}-{model_config.config_id}"
    experiment_id = f"task1-classical-{candidate_id}"
    split_sha256 = compute_sha256(split_file)
    implementation_sha256 = _classical_implementation_sha256()
    config_payload = {
        "validation_fold": validation_fold,
        "hog": hog_spec.to_dict(),
        "model_config": asdict(model_config),
        "run_config": asdict(run_config),
        "model_family": model_family,
        "split_sha256": split_sha256,
        "label_map_sha256": label_map_sha256,
        "implementation_sha256": implementation_sha256,
    }
    config_sha256 = canonical_sha256(config_payload)
    active_registry = registry or RunRegistry()
    matches = active_registry.find(
        task="task1",
        experiment_id=experiment_id,
        fold=validation_fold,
        config_sha256=config_sha256,
        label_map_sha256=label_map_sha256,
        implementation_sha256=implementation_sha256,
        status="completed",
    )
    if not matches.empty:
        resumed = _completed_classical_result(
            matches.iloc[-1],
            validation_fold=validation_fold,
            candidate_id=candidate_id,
            hog_id=hog_spec.hog_id,
            model_family=model_family,
        )
        if resumed is not None:
            return resumed

    record = RunRecord(
        run_id=new_run_id(experiment_id, validation_fold, run_config.seed),
        task="task1",
        stage=run_config.stage,
        experiment_id=experiment_id,
        model_family=model_family,
        benchmark_only=False,
        final_eligible=run_config.final_eligible,
        scratch=True,
        fold=validation_fold,
        seed=run_config.seed,
        transform_id=hog_spec.hog_id,
        loss_id="not_applicable",
        primary_metric_name="macro_f1_124",
        config_sha256=config_sha256,
        split_sha256=split_sha256,
        label_map_sha256=label_map_sha256,
        implementation_sha256=implementation_sha256,
    )
    project_root = Path(root)

    with tracked_run(active_registry, record) as run:
        training_rows, validation_rows = get_task1_fold_rows(splits, validation_fold)
        if run_config.stage == "smoke":
            training_rows = training_rows.sort_values("id", kind="stable").head(32).copy()
            validation_rows = validation_rows.sort_values("id", kind="stable").head(32).copy()
            _, x_train = _smoke_hog_features(training_rows, hog_spec, root=project_root)
            validation_ids, x_validation = _smoke_hog_features(
                validation_rows, hog_spec, root=project_root
            )
        else:
            development_rows = splits.loc[splits["partition"].eq("development")].copy()
            feature_set = load_or_build_task1_hog_features(
                development_rows,
                hog_spec,
                root=project_root,
                cache_root=cache_root,
                split_sha256=split_sha256,
            )
            validation_ids = validation_rows["id"].to_numpy(dtype=np.int64)
            x_train = _aligned_hog_features(feature_set, training_rows)
            x_validation = _aligned_hog_features(feature_set, validation_rows)

        y_train = np.asarray(
            [label_to_index[str(label)] for label in training_rows["articleType"]], dtype=np.int64
        )
        y_validation = np.asarray(
            [label_to_index[str(label)] for label in validation_rows["articleType"]], dtype=np.int64
        )
        if isinstance(model_config, Task1KNNConfig):
            model, probabilities = fit_predict_task1_knn(
                x_train,
                y_train,
                x_validation,
                model_config,
                batch_size=run_config.validation_batch_size,
            )
        else:
            model, probabilities = fit_predict_task1_linear_svm(
                x_train, y_train, x_validation, model_config
            )
        metrics = classification_metrics(y_validation, probabilities)
        predictions = build_prediction_frame(
            validation_ids, y_validation, probabilities, class_names
        )
        run_dir = Path(result_root) / run.run_id
        model_path = run_dir / "model.pkl"
        prediction_path = run_dir / "predictions.csv"
        model_bundle = {
            "model": model,
            "hog": hog_spec.to_dict(),
            "model_config": asdict(model_config),
            "class_names": class_names,
            "validation_fold": validation_fold,
            "seed": run_config.seed,
            "metrics": metrics,
        }
        atomic_write_bytes(
            model_path, pickle.dumps(model_bundle, protocol=pickle.HIGHEST_PROTOCOL)
        )
        atomic_write_csv(prediction_path, predictions)

        run.primary_metric_value = metrics["macro_f1"]
        run.metrics = metrics
        run.checkpoint_path = _classical_artifact_path(model_path)
        run.checkpoint_sha256 = compute_sha256(model_path)
        run.prediction_path = _classical_artifact_path(prediction_path)
        run.prediction_sha256 = compute_sha256(prediction_path)

    return Task1ClassicalFoldResult(
        run_id=record.run_id,
        fold=validation_fold,
        candidate_id=candidate_id,
        hog_id=hog_spec.hog_id,
        model_family=model_family,
        status="completed",
        metrics=metrics,
        model_path=model_path,
        prediction_path=prediction_path,
    )
