"""Frozen Usage U2 screen: full-RGB HOG with a calibrated linear SVM."""

from __future__ import annotations

import time
import uuid
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.calibration import CalibratedClassifierCV
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC

from fashion.config import LABEL_MAPS_JSON, RANDOM_SEED, ROOT, RUNS_CSV, SPLITS_CSV
from fashion.data import get_cv_split, load_label_maps, load_splits
from fashion.data.hashing import compute_sha256, write_deterministic_csv
from fashion.data.task3_clean_slate_eda import build_clean_slate_audit_contract
from fashion.train.metrics import classification_metrics
from fashion.train.registry import RunRegistry
from fashion.train.task3_clean_slate import (
    CLEAN_SLATE_SCREEN_FOLDS,
    HOST_MEMORY_LIMIT_BYTES,
    SCREEN_SECONDS_PER_FOLD_LIMIT,
    _aggregate_screen,
    _canonical_inner_splits,
    _classes,
    _completed_fold,
    _expand_probabilities,
    _fold_paths,
    _json_dump,
    _matrix_rows,
    _peak_memory_bytes,
    _pickle_dump,
    _prediction_frame,
    _registry_start,
    _relative,
    _screen_folds,
    _stable_digest,
    _valid,
    build_fixed_feature_cache,
    fixed_feature_vector,
)
from fashion.train.task3_experiments import effective_number_class_weights

USAGE_HOG_EXPERIMENT_ID = "t3_usage_v2_u2_full_rgb_hog_svm"
USAGE_HOG_HYPOTHESIS_ID = "full_rgb_hog_linear_margin_reduces_usage_overfit"
USAGE_HOG_ARTIFACT_ROOT = "experiments/t3_usage_v2_u2_full_rgb_hog_svm"


@dataclass(frozen=True)
class UsageHogSvmConfig:
    """One frozen screen; there is no feature or C search."""

    target: str = "usage"
    feature_view: str = "full_rgb_hog"
    c: float = 1.0
    inner_folds: int = 4
    calibration: str = "sigmoid"
    class_weight_strategy: str = "effective_number_sample_weight"
    class_weight_beta: float = 0.999
    class_weight_cap: float = 5.0
    max_iterations: int = 20_000
    seed: int = RANDOM_SEED
    scratch: bool = True
    submission_eligible: bool = True


class WeightedScaledLinearSVC(ClassifierMixin, BaseEstimator):
    """Fit weighted scaling and a one-vs-rest linear SVM inside each CV split."""

    def __init__(self, *, c: float = 1.0, max_iterations: int = 20_000, seed: int = 2753):
        self.c = c
        self.max_iterations = max_iterations
        self.seed = seed

    def fit(
        self,
        features: np.ndarray,
        labels: np.ndarray,
        sample_weight: np.ndarray | None = None,
    ) -> WeightedScaledLinearSVC:
        self.scaler_ = StandardScaler()
        self.scaler_.fit(features, sample_weight=sample_weight)
        self.estimator_ = LinearSVC(
            C=self.c,
            dual="auto",
            max_iter=self.max_iterations,
            random_state=self.seed,
        )
        self.estimator_.fit(self.scaler_.transform(features), labels, sample_weight=sample_weight)
        self.classes_ = self.estimator_.classes_
        return self

    def decision_function(self, features: np.ndarray) -> np.ndarray:
        return self.estimator_.decision_function(self.scaler_.transform(features))

    def predict(self, features: np.ndarray) -> np.ndarray:
        return self.estimator_.predict(self.scaler_.transform(features))


def _calibrated_parameter_count(model: CalibratedClassifierCV) -> int:
    count = 0
    for calibrated in model.calibrated_classifiers_:
        estimator = calibrated.estimator.estimator_
        count += int(estimator.coef_.size + estimator.intercept_.size)
        count += 2 * len(calibrated.calibrators)
    return count


def _sample_weights(labels: Sequence[str], classes: Sequence[str], config: UsageHogSvmConfig):
    counts = pd.Series(labels).value_counts()
    class_weights = effective_number_class_weights(
        [int(counts.get(name, 0)) for name in classes],
        beta=config.class_weight_beta,
        cap=config.class_weight_cap,
    )
    lookup = dict(zip(classes, class_weights, strict=True))
    return np.asarray([lookup[str(label)] for label in labels], dtype=np.float64), class_weights


def prepare_usage_hog_features(
    *,
    root: str | Path = ROOT,
    output_root: str | Path,
    workers: int | None = None,
    local_work_dir: str | Path | None = None,
) -> dict[str, Any]:
    """Build or reuse a label-blind pure full-RGB HOG cache."""
    root = Path(root)
    splits = load_splits(root / SPLITS_CSV.relative_to(ROOT))
    audit = build_clean_slate_audit_contract(splits, root=root)
    cache = build_fixed_feature_cache(
        splits,
        view="full_rgb_hog",
        audit_contract_hash=str(audit["audit_contract_hash"]),
        output_dir=Path(output_root) / USAGE_HOG_ARTIFACT_ROOT / "feature_cache",
        root=root,
        workers=workers,
        local_work_dir=local_work_dir,
    )
    return {"audit_contract": audit, "usage": cache}


def _run_fold(
    fold: int,
    *,
    splits: pd.DataFrame,
    label_maps: Mapping[str, Mapping[str, Any]],
    cache: Mapping[str, Any],
    audit_hash: str,
    parent_run_ids: Sequence[str],
    output_root: Path,
    registry_path: Path,
    registry_mirrors: Sequence[str | Path],
    root: Path,
    reuse_completed: bool,
) -> dict[str, Any]:
    config = UsageHogSvmConfig()
    training_all, validation_all = get_cv_split(splits, fold)
    training = _valid(training_all, "usage").reset_index(drop=True)
    validation = _valid(validation_all, "usage").reset_index(drop=True)
    if set(training["product_family_group"]).intersection(validation["product_family_group"]):
        raise ValueError("an outer product family crosses the usage fold")
    classes = _classes(label_maps, "usage")
    config_payload = {
        **asdict(config),
        "validation_fold": fold,
        "audit_contract_hash": audit_hash,
        "screen_folds": list(CLEAN_SLATE_SCREEN_FOLDS),
        "feature_contract": Path(str(cache["contract_path"])).name,
        "parent_run_ids": list(parent_run_ids),
        "single_changed_factor": "cnn_pixels_to_fixed_full_rgb_hog_linear_svm",
        "augmentation": "none",
    }
    config_hash = _stable_digest(config_payload)
    target_dir = output_root / USAGE_HOG_ARTIFACT_ROOT / "usage"
    if reuse_completed and (
        completed := _completed_fold(
            target_dir=target_dir,
            fold=fold,
            config_hash=config_hash,
            registry_path=registry_path,
        )
    ):
        return completed

    timestamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    run_id = (
        f"t3_usage_v2_u2_hogsvm_usage_hogsvm_f{fold}_s{config.seed}_"
        f"{config_hash[:12]}_{timestamp}{uuid.uuid4().hex[:6]}"
    )
    paths = _fold_paths(target_dir, run_id)
    _json_dump(config_payload, paths["config"])
    registry = RunRegistry(registry_path, mirrors=registry_mirrors)
    _registry_start(
        registry,
        run_id=run_id,
        experiment_id=USAGE_HOG_EXPERIMENT_ID,
        hypothesis_id=USAGE_HOG_HYPOTHESIS_ID,
        target="usage",
        fold=fold,
        config_hash=config_hash,
        config_path=paths["config"],
        history_path=paths["history"],
        training=training,
        validation=validation,
        model_family="full_rgb_hog_calibrated_linear_svm",
        root=root,
        parent_run_ids=parent_run_ids,
    )
    started = time.perf_counter()
    last_stage = "registered_before_first_model_fit"
    try:
        x_train = _matrix_rows(training, cache)
        x_validation = _matrix_rows(validation, cache)
        y_train = training["usage"].astype(str).to_numpy()
        sample_weight, class_weights = _sample_weights(y_train, classes, config)
        inner_splits = _canonical_inner_splits(training, outer_fold=fold)
        estimator = WeightedScaledLinearSVC(
            c=config.c,
            max_iterations=config.max_iterations,
            seed=config.seed + fold,
        )
        model = CalibratedClassifierCV(
            estimator=estimator,
            method=config.calibration,
            cv=inner_splits,
            ensemble=True,
        )
        print(
            f"[task3-usage-hog] fold={fold}: fitting {config.inner_folds} calibrated SVMs",
            flush=True,
        )
        model.fit(x_train, y_train, sample_weight=sample_weight)
        validation_probabilities = _expand_probabilities(
            model.predict_proba(x_validation), model.classes_, classes
        )
        training_probabilities = _expand_probabilities(
            model.predict_proba(x_train), model.classes_, classes
        )
        last_stage = "outer_model_fit_complete"
        registry.update(run_id, {"last_completed_stage": last_stage})

        predictions = _prediction_frame(
            validation,
            target="usage",
            classes=classes,
            probabilities=validation_probabilities,
            run_id=run_id,
        )
        write_deterministic_csv(predictions, paths["predictions"], index=False)
        class_lookup = {name: index for index, name in enumerate(classes)}
        validation_labels = validation["usage"].astype(str).map(class_lookup).to_numpy(dtype=int)
        training_labels = training["usage"].astype(str).map(class_lookup).to_numpy(dtype=int)
        metrics = classification_metrics(validation_labels, validation_probabilities, classes)
        train_metrics = classification_metrics(training_labels, training_probabilities, classes)
        parameter_count = _calibrated_parameter_count(model)
        write_deterministic_csv(
            pd.DataFrame(
                {
                    "class_name": classes,
                    "training_count": [int((y_train == name).sum()) for name in classes],
                    "effective_number_weight": class_weights,
                }
            ),
            paths["history"],
            index=False,
        )
        _pickle_dump(
            {
                "run_id": run_id,
                "config": config_payload,
                "class_names": classes,
                "model": model,
                "feature_view": config.feature_view,
            },
            paths["checkpoint"],
        )
        train_seconds = time.perf_counter() - started
        metrics.update(
            {
                "run_id": run_id,
                "target": "usage",
                "validation_fold": fold,
                "experiment_id": USAGE_HOG_EXPERIMENT_ID,
                "hypothesis_id": USAGE_HOG_HYPOTHESIS_ID,
                "model_family": "full_rgb_hog_calibrated_linear_svm",
                "config_hash": config_hash,
                "audit_contract_hash": audit_hash,
                "final_train_macro_f1": train_metrics["macro_f1"],
                "final_train_validation_macro_f1_gap": float(
                    train_metrics["macro_f1"] - metrics["macro_f1"]
                ),
                "parameter_count": parameter_count,
                "train_seconds": train_seconds,
                "peak_memory_bytes": _peak_memory_bytes(),
                "screen_scope": "canonical_outer_folds_0_and_4",
                "robustness_scope": "deferred_to_full_five_fold_stage",
                "class_weights": class_weights.tolist(),
            }
        )
        metrics["prediction_sha256"] = compute_sha256(paths["predictions"])
        metrics["checkpoint_sha256"] = compute_sha256(paths["checkpoint"])
        _json_dump(metrics, paths["metrics"])
        registry.complete(
            run_id,
            {
                "parameter_count": parameter_count,
                "checkpoint_path": _relative(paths["checkpoint"], root),
                "checkpoint_sha256": metrics["checkpoint_sha256"],
                "prediction_path": _relative(paths["predictions"], root),
                "prediction_sha256": metrics["prediction_sha256"],
                "metrics_json": metrics,
                "train_seconds": train_seconds,
                "peak_memory_bytes": metrics["peak_memory_bytes"],
                "checkpoint_bytes": paths["checkpoint"].stat().st_size,
                "last_completed_stage": "screen_fold_complete",
            },
        )
        return {
            "run_id": run_id,
            "run_dir": str(paths["run_dir"]),
            "prediction_path": str(paths["predictions"]),
            "metrics_path": str(paths["metrics"]),
            "metrics": metrics,
        }
    except BaseException as error:
        registry.fail(run_id, error, last_completed_stage=last_stage)
        raise


def check_usage_hog_svm_setup(
    *,
    root: str | Path = ROOT,
    folds: Iterable[int] = CLEAN_SLATE_SCREEN_FOLDS,
) -> dict[str, Any]:
    """Perform a zero-fit preflight for Usage U2."""
    fold_list = _screen_folds(folds)
    root = Path(root)
    splits = load_splits(root / SPLITS_CSV.relative_to(ROOT))
    label_maps = load_label_maps(root / LABEL_MAPS_JSON.relative_to(ROOT))
    for fold in fold_list:
        training, validation = get_cv_split(splits, fold)
        if set(_valid(training, "usage")["product_family_group"]).intersection(
            _valid(validation, "usage")["product_family_group"]
        ):
            raise ValueError(f"a usage family crosses fold {fold}")
    descriptor = fixed_feature_vector(
        np.full((80, 60, 3), 255, dtype=np.uint8), view="full_rgb_hog"
    )
    estimated_bytes = int(splits["partition"].eq("development").sum()) * descriptor.nbytes
    if estimated_bytes > HOST_MEMORY_LIMIT_BYTES:
        raise RuntimeError("estimated full-RGB HOG cache exceeds the host memory limit")
    return {
        "ready": True,
        "training_blockers": [],
        "folds": list(fold_list),
        "usage_classes": _classes(label_maps, "usage"),
        "model": "full-RGB HOG + weighted StandardScaler + calibrated LinearSVC(C=1)",
        "feature_columns": len(descriptor),
        "estimated_cache_bytes": estimated_bytes,
        "host_memory_limit_bytes": HOST_MEMORY_LIMIT_BYTES,
        "model_fits": 0,
        "optimizer_steps": 0,
        "execution_device": "cpu",
    }


def run_usage_hog_svm_screen(
    *,
    prepared_features: Mapping[str, Any],
    parent_run_ids: Sequence[str],
    output_root: str | Path,
    folds: Iterable[int] = CLEAN_SLATE_SCREEN_FOLDS,
    registry_path: str | Path = RUNS_CSV,
    registry_mirrors: Sequence[str | Path] = (),
    root: str | Path = ROOT,
    anchor_prediction_path: str | Path | None = None,
    reuse_completed: bool = True,
) -> dict[str, Any]:
    """Run the frozen two-fold Usage U2 screen and pool matched OOF rows."""
    if len(parent_run_ids) != 5 or len(set(parent_run_ids)) != 5:
        raise ValueError("Usage U2 requires five distinct E2 parent run IDs")
    fold_list = _screen_folds(folds)
    root = Path(root)
    output_root = Path(output_root)
    splits = load_splits(root / SPLITS_CSV.relative_to(ROOT))
    label_maps = load_label_maps(root / LABEL_MAPS_JSON.relative_to(ROOT))
    audit_hash = str(prepared_features["audit_contract"]["audit_contract_hash"])
    results = [
        _run_fold(
            fold,
            splits=splits,
            label_maps=label_maps,
            cache=prepared_features["usage"],
            audit_hash=audit_hash,
            parent_run_ids=parent_run_ids,
            output_root=output_root,
            registry_path=Path(registry_path),
            registry_mirrors=registry_mirrors,
            root=root,
            reuse_completed=reuse_completed,
        )
        for fold in fold_list
    ]
    aggregate = _aggregate_screen(
        "usage",
        results,
        output_root=output_root,
        root=root,
        model_family="full_rgb_hog_calibrated_linear_svm",
        experiment_id=USAGE_HOG_EXPERIMENT_ID,
        hypothesis_id=USAGE_HOG_HYPOTHESIS_ID,
        anchor_prediction_path=anchor_prediction_path,
        artifact_root=USAGE_HOG_ARTIFACT_ROOT,
        seconds_per_fold_limit=SCREEN_SECONDS_PER_FOLD_LIMIT,
        memory_limit_bytes=HOST_MEMORY_LIMIT_BYTES,
    )
    metrics = dict(aggregate["metrics"])
    metrics["frozen_decision_routes"] = {
        "route_a": "macro_f1 >= 0.417319 and paired family-bootstrap lower 95% bound > 0",
        "route_b": (
            "macro_f1 >= 0.402319 and either clean-gap reduction >= 25% "
            "or NLL/Brier improvement >= 10%"
        ),
        "calibration": "ECE <= 0.050",
        "class_safety": "no supported class F1 drop > 0.030",
    }
    _json_dump(metrics, Path(str(aggregate["metrics_path"])))
    aggregate["metrics"] = metrics
    return aggregate
